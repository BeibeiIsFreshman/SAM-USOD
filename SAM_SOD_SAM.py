import torch
from torch import nn
import torch.nn.functional as F
from Mine.modeling.image_encoder import ImageEncoderViT
from Mine.modeling.mask_decoder_ import MaskDecoder
from Mine.modeling.prompt_encoder_ import PromptEncoder
from Mine.modeling import TwoWayTransformer
from typing import Any, Dict, List, Tuple
from functools import partial
from torch.nn.functional import interpolate
from Mine.PVTv2 import pvt_v2_b4
from torch.fft import fft2


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3 * 2, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )

        self.conv_d1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in2, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d3 = BasicConv2d(in2, in1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        f[3] = self.bcon4(f[3])
        f[2] = self.bcon3(f[2])
        f[1] = self.bcon2(f[1])
        f[0] = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f[3], f[2]), 1))
        d32 = self.bcon3_2(torch.cat((d43, f[1]), 1))
        d21 = self.bcon2_1(torch.cat((d32, f[0]), 1))
        out = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)

        return d21, out, d32, d43


class FeatureAlignmentModule(nn.Module):
    def __init__(self):
        super(FeatureAlignmentModule, self).__init__()

        # For [n,256,16,16] -> [n,64,64,64]
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        )

        # For [n,256,16,16] -> [n,128,32,32]
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        )

        # For [n,256,16,16] -> [n,320,16,16]
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        # For [n,256,16,16] -> [n,512,8,8]
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Residual connections to preserve information
        self.residual1 = nn.Conv2d(256, 64, kernel_size=1)
        self.residual2 = nn.Conv2d(256, 128, kernel_size=1)
        self.residual3 = nn.Conv2d(256, 320, kernel_size=1)
        self.residual4 = nn.Conv2d(256, 512, kernel_size=1)

        # Attention mechanisms to focus on important features
        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(128)
        self.channel_attention3 = ChannelAttention(320)
        self.channel_attention4 = ChannelAttention(512)

    def forward(self, x1, x2, x3, x4):
        # Assuming x1, x2, x3, x4 are the four input features with shape [n,256,16,16]

        # Process feature 1: [n,256,16,16] -> [n,64,64,64]
        out1 = self.up_conv1(x1)
        res1 = F.interpolate(self.residual1(x1), scale_factor=4, mode='bilinear', align_corners=True)
        out1 = out1 + res1
        out1 = self.channel_attention1(out1)

        # Process feature 2: [n,256,16,16] -> [n,128,32,32]
        out2 = self.up_conv2(x2)
        res2 = F.interpolate(self.residual2(x2), scale_factor=2, mode='bilinear', align_corners=True)
        out2 = out2 + res2
        out2 = self.channel_attention2(out2)

        # Process feature 3: [n,256,16,16] -> [n,320,16,16]
        out3 = self.conv3(x3)
        res3 = self.residual3(x3)
        out3 = out3 + res3
        out3 = self.channel_attention3(out3)

        # Process feature 4: [n,256,16,16] -> [n,512,8,8]
        out4 = self.down_conv4(x4)
        res4 = F.interpolate(self.residual4(x4), scale_factor=0.5, mode='bilinear', align_corners=True)
        out4 = out4 + res4
        out4 = self.channel_attention4(out4)

        return out1, out2, out3, out4


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ChannelFrequencyAttention(nn.Module):
    """通道和频域融合注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelFrequencyAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        # 频域注意力
        self.freq_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        # 空间通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)

        # 频域注意力
        fft_feature = fft2(x)
        magnitude = torch.abs(fft_feature)
        phase = torch.angle(fft_feature)

        # 归一化幅值
        magnitude = magnitude / (torch.max(magnitude) + 1e-8)

        # 合并幅值和相位信息
        freq_info = torch.cat([magnitude.mean(dim=1, keepdim=True),
                               phase.mean(dim=1, keepdim=True)], dim=1)
        freq_att = self.sigmoid(self.freq_conv(freq_info))

        # 组合两种注意力
        x_channel = x * channel_att
        x_freq = x * freq_att

        return x_channel + x_freq


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(self, in_channels):
        super(CrossModalAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rgb_feat, depth_feat):
        batch_size, C, height, width = rgb_feat.size()

        # 从RGB计算查询
        query = self.query_conv(rgb_feat).view(batch_size, -1, height * width).permute(0, 2, 1)

        # 从深度计算键值
        key = self.key_conv(depth_feat).view(batch_size, -1, height * width)
        value = self.value_conv(depth_feat).view(batch_size, -1, height * width)

        # 注意力图
        attention = self.softmax(torch.bmm(query, key))

        # 注意力加权深度特征
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        # 残差连接
        out = self.gamma * out + rgb_feat

        return out


class WaveletDecomposition(nn.Module):
    """小波分解模块"""

    def __init__(self):
        super(WaveletDecomposition, self).__init__()

    def forward(self, x):
        # 简化的小波分解：将特征分为低频和高频成分
        # 在实际应用中，可以使用PyWavelets库实现更复杂的小波变换

        # 低通滤波器（平均池化）
        low_freq = F.avg_pool2d(x, kernel_size=2, stride=2)
        low_freq = F.interpolate(low_freq, scale_factor=2, mode='bilinear', align_corners=False)

        # 高频部分（细节）
        high_freq = x - low_freq

        return low_freq, high_freq


class AdvancedFeatureFusion(nn.Module):
    """高级特征融合模块"""

    def __init__(self, in_channels):
        super(AdvancedFeatureFusion, self).__init__()

        # 注意力模块
        self.channel_freq_att_rgb = ChannelFrequencyAttention(in_channels)
        self.channel_freq_att_depth = ChannelFrequencyAttention(in_channels)

        # 跨模态注意力
        self.cross_att_rgb2depth = CrossModalAttention(in_channels)
        self.cross_att_depth2rgb = CrossModalAttention(in_channels)

        # 小波分解
        self.wavelet = WaveletDecomposition()

        # 融合卷积
        self.fusion_conv_low = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.fusion_conv_high = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 门控融合参数
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat):
        # 应用通道频率注意力
        rgb_att = self.channel_freq_att_rgb(rgb_feat)
        depth_att = self.channel_freq_att_depth(depth_feat)

        # 跨模态注意力
        rgb2depth = self.cross_att_rgb2depth(rgb_feat, depth_feat)
        depth2rgb = self.cross_att_depth2rgb(depth_feat, rgb_feat)

        # 小波分解
        rgb_low, rgb_high = self.wavelet(rgb_att)
        depth_low, depth_high = self.wavelet(depth_att)

        # 不同频率分量融合
        fused_low = self.fusion_conv_low(torch.cat([rgb_low, depth_low], dim=1))
        fused_high = self.fusion_conv_high(torch.cat([rgb_high, depth_high], dim=1))

        # 自适应门控机制
        gate_weights = self.gate(torch.cat([rgb_att, depth_att], dim=1))
        weighted_rgb = rgb2depth * gate_weights[:, 0:1, :, :]
        weighted_depth = depth2rgb * gate_weights[:, 1:2, :, :]

        # 最终融合
        fusion = self.final_fusion(torch.cat([fused_low, fused_high, weighted_rgb + weighted_depth], dim=1))

        return fusion


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, channel_list=[64, 128, 320, 512]):
        super(MultiScaleFeatureFusion, self).__init__()

        # 为每个尺度创建融合模块
        self.fusion_modules = nn.ModuleList([
            AdvancedFeatureFusion(channel_list[0]),
            AdvancedFeatureFusion(channel_list[1]),
            AdvancedFeatureFusion(channel_list[2]),
            AdvancedFeatureFusion(channel_list[3])
        ])

    def forward(self, rgb_list, depth_list):
        fused_features = []

        for i, fusion_module in enumerate(self.fusion_modules):
            fused_features.append(fusion_module(rgb_list[i], depth_list[i]))

        return fused_features


class MDSAM(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()

        self.pixel_mean: List[float] = [123.675, 116.28, 103.53]
        self.pixel_std: List[float] = [58.395, 57.12, 57.375]

        # 图像和深度编码器
        self.image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256
        )
        # self.prompt_encoder = PromptEncoder()
        image_embedding_size = img_size // 16
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # self.backbone = pvt_v2_b4(True)
        #
        # # 添加新的特征融合模块
        # self.feature_fusion = MultiScaleFeatureFusion([64, 128, 320, 512])
        #
        # self.decoder = Decoder(64, 128, 320, 512)
        #
        # self.out_best = nn.Sequential(
        #     Up(scale_factor=1 / 2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 1, 3, 1, 1)
        # )
        self.out_sam = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True)
        )
        # self.out1_best = nn.Sequential(
        #     Up(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 1, 3, 1, 1)
        # )
        # self.out2_best = nn.Sequential(
        #     Up(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 1, 3, 1, 1)
        # )
        # self.out3_best = nn.Sequential(
        #     Up(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 1, 3, 1, 1)
        # )
        # self.out4_best = nn.Sequential(
        #     Up(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(128, 1, 3, 1, 1)
        # )

    def forward(self, img, depth=None):
        rgb_list = self.image_encoder(img)

        # rgb_b = self.backbone(img)
        # depth_b = self.backbone(depth)
        # fusion = self.feature_fusion(rgb_b, depth_b)
        # out = self.decoder(fusion)


        dense_prompt = self.prompt_encoder(
            masks=None,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=rgb_list,
            image_pe=self.prompt_encoder.get_dense_pe(),
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
        )

        return (F.sigmoid(self.out_sam(low_res_masks)), F.sigmoid(self.out_sam(low_res_masks)))


def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed


def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]


def load(net, ckpt, img_size):
    ckpt = torch.load(ckpt, map_location='cpu')
    from collections import OrderedDict
    dict = OrderedDict()
    # for k, v in ckpt.items():
    #     if 'pe_layer' in k:
    #         dict[k[15:]] = v
    #         continue
    #     if 'pos_embed' in k:
    #         dict[k] = reshapePos(v, img_size)
    #         continue
    #     if 'rel_pos' in k:
    #         dict[k] = reshapeRel(k, v, img_size)
    #     else:
    #         dict[k] = v

    for k, v in ckpt.items():
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k:
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            dict[k] = v
        # if "prompt_encoder" in k:
        #     dict[k] = v
        # if "mask_decoder" in k:
        #     dict[k] = v

    # 加载权重
    state1, state2 = net.load_state_dict(dict, strict=False)
    print("----------------------:", state1)
    print("----------------------:", state2)

    return "TRUE"


if __name__ == '__main__':
    model = MDSAM().cuda()
    state = load(model,
                 "/media/tbb/9b281502-670b-4aec-957e-085adc101020/UAV/Fine_SAM/Backbone_pth/sam_vit_b_01ec64.pth", 256)
    right = torch.randn(4, 3, 256, 256).cuda()
    left = torch.randn(4, 3, 256, 256).cuda()
    out = model(right, left)
    for i in out:
        print(i.shape)