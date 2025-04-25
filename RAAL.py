import torch
from torch import nn
import torch.nn.functional as F

## RegionAwareAttentionLoss
class AdaptiveMultiScaleDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5, beta=0.3):
        super(AdaptiveMultiScaleDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 平衡蒸馏损失和监督损失的权重
        self.beta1 = nn.Parameter(torch.tensor(beta))  # 结构一致性权重
        self.beta2 = nn.Parameter(torch.tensor(1-beta))  # 结构一致性权重
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        # 自适应权重学习
        self.scale_weights = nn.Parameter(torch.ones(4) / 4)
        self.region_attention = RegionAwareAttention()

    def forward(self, teacher_output, student_outputs, targets=None):
        """
        自蒸馏损失计算
        Args:
            teacher_output: SAM的输出 [batch, 1, H, W]
            student_outputs: 其他四个分支输出的列表 [batch, 1, H, W] x 4
            targets: 可选的GT标签 [batch, 1, H, W]
        """
        batch_size = teacher_output.size(0)

        # 标准化权重
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        # 1. 软标签蒸馏损失
        distillation_losses = []
        structure_losses = []

        # 调整教师输出尺寸以适应不同学生输出
        for i, student_output in enumerate(student_outputs):
            # 调整教师输出尺寸以匹配学生输出
            if teacher_output.shape != student_output.shape:
                adapted_teacher = F.interpolate(
                    teacher_output,
                    size=student_output.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            else:
                adapted_teacher = teacher_output

            # 获取区域注意力图
            attention_map = self.region_attention(adapted_teacher, student_output)

            # 软标签知识蒸馏 (KL散度)
            teacher_soft = (adapted_teacher / self.temperature).sigmoid()
            student_soft = (student_output / self.temperature).sigmoid()

            # 应用对数以使用KL散度
            soft_loss = self.kl_div(
                F.log_softmax(student_soft.view(batch_size, -1), dim=1),
                F.softmax(teacher_soft.view(batch_size, -1), dim=1)
            ).sum(dim=1).mean() * (self.temperature ** 2)

            # 结构一致性损失 (梯度方向一致性)
            teacher_grad_x = self._spatial_gradient_x(adapted_teacher)
            teacher_grad_y = self._spatial_gradient_y(adapted_teacher)
            student_grad_x = self._spatial_gradient_x(student_output)
            student_grad_y = self._spatial_gradient_y(student_output)

            # 结构一致性使用余弦相似度
            grad_x_sim = self._cosine_similarity(teacher_grad_x, student_grad_x)
            grad_y_sim = self._cosine_similarity(teacher_grad_y, student_grad_y)
            struct_loss = (2 - grad_x_sim - grad_y_sim) / 2

            # 应用注意力权重
            soft_loss = (soft_loss * attention_map).mean()
            struct_loss = (struct_loss * attention_map).mean()

            distillation_losses.append(soft_loss)
            structure_losses.append(struct_loss)

        # 组合不同尺度的损失
        distillation_loss = sum([w * l for w, l in zip(normalized_weights, distillation_losses)])
        structure_loss = sum([w * l for w, l in zip(normalized_weights, structure_losses)])

        # 如果有GT标签，则添加监督损失
        if targets is not None:
            supervision_loss = self._compute_supervision_loss(student_outputs, targets)
            total_loss = (
                                     1 - self.alpha) * supervision_loss + self.alpha * distillation_loss + self.beta * structure_loss
        else:
            total_loss = self.beta1 * distillation_loss + self.beta2 * structure_loss

        return total_loss

    def _spatial_gradient_x(self, tensor):
        """计算x方向的梯度"""
        gradient_x = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        return F.pad(gradient_x, (0, 1, 0, 0), mode='replicate')

    def _spatial_gradient_y(self, tensor):
        """计算y方向的梯度"""
        gradient_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return F.pad(gradient_y, (0, 0, 0, 1), mode='replicate')

    def _cosine_similarity(self, x, y):
        """计算余弦相似度"""
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        x_norm = F.normalize(x_flat, p=2, dim=1)
        y_norm = F.normalize(y_flat, p=2, dim=1)
        return (x_norm * y_norm).sum(dim=1).mean()

    def _compute_supervision_loss(self, student_outputs, targets):
        """计算监督损失"""
        losses = []
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        for i, student_output in enumerate(student_outputs):
            # 调整目标尺寸以匹配学生输出
            if targets.shape != student_output.shape:
                adapted_targets = F.interpolate(
                    targets,
                    size=student_output.shape[2:],
                    mode='nearest'
                )
            else:
                adapted_targets = targets

            loss = self.bce_loss(student_output, adapted_targets).mean()
            losses.append(loss)

        return sum([w * l for w, l in zip(normalized_weights, losses)])


class RegionAwareAttention(nn.Module):
    """区域感知注意力机制，关注困难区域"""

    def __init__(self, epsilon=1e-6):
        super(RegionAwareAttention, self).__init__()
        self.epsilon = epsilon
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, teacher, student):
        # 计算教师和学生输出的差异
        diff = torch.abs(teacher - student)

        # 计算空间不确定度 (熵)
        entropy = -(teacher * torch.log(teacher + self.epsilon) +
                    (1 - teacher) * torch.log(1 - teacher + self.epsilon))

        # 拼接差异和熵
        attention_features = torch.cat([diff, entropy], dim=1)

        # 生成注意力权重
        attention = self.sigmoid(self.conv(attention_features))

        # 归一化确保总权重和为1
        attention = attention / (attention.sum() + self.epsilon)
        # print(attention)

        return attention


# right = torch.randn(4, 1, 256, 256).cuda()
# left = []
# left1 = torch.randn(4, 1, 256, 256).cuda()
# left2 = torch.randn(4, 1, 256, 256).cuda()
# left3 = torch.randn(4, 1, 256, 256).cuda()
# left4 = torch.randn(4, 1, 256, 256).cuda()
# left.append(left1)
# left.append(left2)
# left.append(left3)
# left.append(left4)
#
# distillation_criterion = AdaptiveMultiScaleDistillationLoss(
#         temperature=2.0,
#         alpha=0.5,
#         beta=0.3
#     ).cuda()
#
# loss = distillation_criterion(right, left)
# print(loss)