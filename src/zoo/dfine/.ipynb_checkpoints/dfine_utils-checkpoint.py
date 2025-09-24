"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_xyxy_to_cxcywh

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CosineAlignmentLossNoParams(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 无需可学习参数

#     def forward(self, fea, depth_fea, ):
#         """
#         通过通道池化实现无参数通道映射，将depth_fea通道数匹配到fea（256通道）
#         """
#         total_loss = 0.0
#         # 遍历fea的3个层级，对应depth_fea的后3个层级（index 1,2,3）
#         for i in range(3):
#             f = fea[i]  # 形状：(B, 256, H_fea, W_fea)
#             d = depth_fea[i + 1]  # 形状：(B, C_depth, H_d, W_d)，C分别为512、1024、1024

#             # --------------------------
#             # 1. 通道池化映射到256通道
#             # --------------------------
#             c_depth = d.shape[1]
#             pool_kernel = c_depth // 64  # 计算池化核大小：512→2，1024→4
            
#             # 通道维度池化（支持平均池化或最大池化）
#             # 先将通道维度reshape为(256, pool_kernel)，再池化
#             d_reshaped = d.view(d.shape[0], 256, pool_kernel, d.shape[2], d.shape[3])
#             d_mapped = torch.mean(d_reshaped, dim=2)  # 平均池化：融合通道信息
#             # d_mapped = torch.max(d_reshaped, dim=2)[0]  # 可选：最大池化，突出显著特征

#             # --------------------------
#             # 2. 空间对齐（插值到fea的尺寸）
#             # --------------------------
#             d_aligned = F.interpolate(
#                 d_mapped,
#                 size=(f.shape[2], f.shape[3]),
#                 mode="bilinear",
#                 align_corners=True
#             )  # 形状：(B, 256, H_fea, W_fea)

#             # --------------------------
#             # 3. 计算余弦损失
#             # --------------------------
#             cos_sim = F.cosine_similarity(f, d_aligned, dim=1)
#             loss_i = 1 - cos_sim.mean()
#             total_loss += loss_i

#         total_loss /= 3
#         return total_loss
class CosineAlignmentLossNoParams(nn.Module):
    def __init__(self):
        super().__init__()
        # fea 要映射到的目标通道数
        self.target_channels = [48, 96, 192, 384]

    def channel_interpolate(self, x, target_c):
        """
        仅对通道维度插值 (无参数版本)
        x: (B, C, H, W)
        return: (B, target_c, H, W)
        """
        B, C, H, W = x.shape
    
        # (B, C, H, W) -> (B*H*W, 1, C)
        x_reshape = x.permute(0, 2, 3, 1).reshape(-1, 1, C)
    
        # 对通道维度 C 做 1D 插值
        x_interp = F.interpolate(
            x_reshape, size=target_c, mode="linear", align_corners=True
        )  # (B*H*W, 1, target_c)
    
        # 还原回 (B, H, W, target_c)
        x_interp = x_interp.squeeze(1).reshape(B, H, W, target_c)
    
        # 调整到 (B, target_c, H, W)
        return x_interp.permute(0, 3, 1, 2).contiguous()

    def forward(self, fea, depth_fea):
        total_loss = 0.0

        for i in range(3):
            f = fea[i]        # (B, 256, Hf, Wf)
            d = depth_fea[i+1]  # (B, C_depth, Hd, Wd)
            target_c = self.target_channels[i]

            # 1. fea 和 depth_fea 仅在通道维度插值
            f_mapped = self.channel_interpolate(f, target_c)
            d_mapped = self.channel_interpolate(d, target_c)

            # 2. 空间对齐
            d_aligned = F.interpolate(
                d_mapped, size=(f_mapped.shape[2], f_mapped.shape[3]),
                mode="bilinear", align_corners=True
            )

            # 3. 计算余弦损失
            cos_sim = F.cosine_similarity(f_mapped, d_aligned, dim=1)
            loss_i = 1 - cos_sim.mean()
            total_loss += loss_i

        total_loss /= 3
        return total_loss

def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
