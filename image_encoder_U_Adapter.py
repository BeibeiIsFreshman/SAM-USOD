# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type, List

from Mine.encoder_adapter.common import LayerNorm2d, MLPBlock



class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Block(nn.Module):
    """带有提示微调的Transformer块"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            layer_idx: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
            use_prompt_tuning: bool = True,
    ) -> None:
        """
        Args:
            dim (int): 输入通道数
            num_heads (int): 注意力头数量
            layer_idx (int): 当前层的索引，用于获取对应的提示向量
            mlp_ratio (float): MLP隐藏层维度与嵌入维度的比例
            qkv_bias (bool): 是否为query, key, value添加可学习的偏置
            norm_layer (nn.Module): 标准化层
            act_layer (nn.Module): 激活层
            use_rel_pos (bool): 是否在注意力图中添加相对位置编码
            rel_pos_zero_init (bool): 是否将相对位置参数初始化为零
            window_size (int): 窗口注意力块的窗口大小
            input_size (tuple(int, int) or None): 用于计算相对位置参数大小的输入分辨率
            use_prompt_tuning (bool): 是否使用提示微调
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.use_prompt_tuning = use_prompt_tuning

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        # 提示微调模块
        if use_prompt_tuning:
            self.prompt_tuning = PromptTuningModule(
                dim=dim,
                num_heads=num_heads // 2,  # 使用较少的头以减少计算量
                use_gate=True,
            )

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征 [B, H, W, C]
            prompt (torch.Tensor, optional): 提示向量 [num_prompts, C]

        Returns:
            torch.Tensor: 处理后的特征 [B, H, W, C]
        """
        shortcut = x
        x = self.norm1(x)

        # 窗口分割
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # 反向窗口分割
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        # 应用MLP之前应用提示微调
        if self.use_prompt_tuning and prompt is not None:
            # 重新整形为 [B, H*W, C] 用于提示微调
            B, H, W, C = x.shape
            x_flat = x.reshape(B, H * W, C)

            # 应用提示微调
            x_flat = self.prompt_tuning(x_flat, prompt)

            # 重新整形回 [B, H, W, C]
            x = x_flat.reshape(B, H, W, C)

        x = x + self.mlp(self.norm2(x))

        return x


class PromptGenerator(nn.Module):
    """增强版提示生成器，能够根据输入特征自适应生成提示向量"""

    def __init__(
            self,
            embed_dim: int = 768,
            num_prompts: int = 8,
            num_layers: int = 12,
            prompt_init_std: float = 0.02,
            use_adaptive_prompts: bool = True
    ) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.num_layers = num_layers
        self.use_adaptive_prompts = use_adaptive_prompts

        # 基础提示向量
        self.base_prompts = nn.Parameter(
            torch.randn(num_layers, num_prompts, embed_dim) * prompt_init_std
        )

        # 层适应器
        self.layer_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_layers)
        ])

        # 特征条件适应器 - 新增
        if use_adaptive_prompts:
            self.feature_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, num_prompts)
                ) for _ in range(num_layers)
            ])

    def forward(self, layer_idx: int, feature_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成提示向量

        Args:
            layer_idx (int): 当前层的索引
            feature_context (torch.Tensor, optional): 当前特征的上下文信息 [B, C]

        Returns:
            torch.Tensor: 形状为 [num_prompts, embed_dim] 的提示向量
        """
        # 获取基础提示向量
        prompt = self.base_prompts[layer_idx]  # [num_prompts, embed_dim]

        # 使用层适应器进行调整
        adapted_prompt = self.layer_adapters[layer_idx](prompt)  # [num_prompts, embed_dim]

        # 如果提供了特征上下文，则进行进一步调整
        if self.use_adaptive_prompts and feature_context is not None:
            # 生成提示向量的权重
            batch_size = feature_context.shape[0]
            weights = self.feature_adapters[layer_idx](feature_context)  # [B, num_prompts]
            weights = F.softmax(weights, dim=-1)

            # 对每个批次应用不同的权重
            weighted_prompts = []
            for b in range(batch_size):
                batch_weights = weights[b].unsqueeze(-1)  # [num_prompts, 1]
                weighted_prompt = adapted_prompt * batch_weights
                weighted_prompts.append(weighted_prompt)

            # 取批次平均
            if batch_size > 0:
                adapted_prompt = torch.stack(weighted_prompts).mean(dim=0)

        return adapted_prompt


class PromptTuningModule(nn.Module):
    """改进的提示微调模块，采用更高效的注意力机制和特征融合策略"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            use_gate: bool = True,
            dropout: float = 0.1,
            use_residual_ffn: bool = True,  # 新增残差FFN选项
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.use_residual_ffn = use_residual_ffn

        # 特征投影
        self.q_proj = nn.Linear(dim, dim)

        # 提示投影
        self.kv_proj = nn.Linear(dim, dim * 2)

        # 高效的注意力机制
        self.prompt_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 门控机制，控制提示向量的影响力
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        # 残差FFN
        if use_residual_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout),
            )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        融合特征与提示向量

        Args:
            x (torch.Tensor): 输入特征，形状为 [B, H*W, C]
            prompt (torch.Tensor): 提示向量，形状为 [num_prompts, C]

        Returns:
            torch.Tensor: 经过提示微调后的特征 [B, H*W, C]
        """
        B, L, C = x.shape

        # 保存原始输入用于残差连接
        identity = x

        # 特征投影为查询
        q = self.q_proj(x)  # [B, L, C]

        # 复制提示向量以匹配批次大小
        batch_prompts = prompt.unsqueeze(0).expand(B, -1, -1)  # [B, num_prompts, C]

        # 提示投影为键和值
        kv = self.kv_proj(batch_prompts)  # [B, num_prompts, 2*C]
        k, v = kv.chunk(2, dim=-1)  # 两个 [B, num_prompts, C]

        # 使用注意力机制融合提示向量与特征
        # 使用特征作为查询，提示作为键和值
        attn_output, _ = self.prompt_attn(
            query=q,
            key=k,
            value=v,
        )  # [B, L, C]

        # 应用归一化和残差连接
        attn_output = self.norm1(attn_output)

        if self.use_gate:
            # 计算上下文特征，用于门控
            context_features = torch.cat([identity, attn_output], dim=-1)  # [B, L, 2*C]
            gate_value = self.gate(context_features)  # [B, L, 1]

            # 应用门控
            x = identity + gate_value * attn_output
        else:
            # 直接应用残差连接
            x = identity + attn_output

        # 应用残差FFN
        if self.use_residual_ffn:
            ffn_output = self.ffn(self.norm2(x))
            x = x + ffn_output

        return x


class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            use_prompt_tuning: bool = True,
            num_prompts: int = 8,
            use_adaptive_prompts: bool = True,  # 新增
            use_residual_ffn: bool = True,  # 新增
    ) -> None:
        """
        Args:
            ...与原始保持一致...
            use_adaptive_prompts (bool): 是否使用自适应提示生成
            use_residual_ffn (bool): 是否在提示微调中使用残差FFN
        """
        super().__init__()
        self.img_size = img_size
        self.use_prompt_tuning = use_prompt_tuning

        # 保持原有的patch_embed和pos_embed实现
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        # 使用改进的提示生成器
        if use_prompt_tuning:
            self.prompt_generator = PromptGenerator(
                embed_dim=embed_dim,
                num_prompts=num_prompts,
                num_layers=depth,
                use_adaptive_prompts=use_adaptive_prompts
            )

        # Block实现基本保持不变，但内部使用改进的PromptTuningModule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                layer_idx=i,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_prompt_tuning=use_prompt_tuning,
            )
            # 替换Block中的PromptTuningModule为改进版
            if use_prompt_tuning:
                block.prompt_tuning = PromptTuningModule(
                    dim=embed_dim,
                    num_heads=num_heads // 2,
                    use_gate=True,
                    use_residual_ffn=use_residual_ffn
                )
            self.blocks.append(block)

        # 保持原有的neck实现
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        features_list = []

        # 获取全局特征上下文，用于自适应提示生成
        if self.use_prompt_tuning and hasattr(self.prompt_generator, 'use_adaptive_prompts'):
            # 简单的全局池化作为特征上下文
            B, H, W, C = x.shape
            feature_context = x.mean(dim=[1, 2])  # [B, C]
        else:
            feature_context = None

        for i, blk in enumerate(self.blocks):
            # 使用改进的提示生成器
            if self.use_prompt_tuning:
                prompt = self.prompt_generator(i, feature_context)
                x = blk(x, prompt)
            else:
                x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


if __name__ == '__main__':
    model = ImageEncoderViT(256).cuda()
    right = torch.randn(1, 3, 256, 256).cuda()
    out = model(right)
    # print(out[0].shape)
    for i in out:
        print(i.shape)
    # for i in out:
    #     print(i.shape)