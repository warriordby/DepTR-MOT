"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def compute_depth_distributions_with_save(
    y, mask, bins=10, low_quantile=0.05, save_dir='./depth_distributions_normalized'
):
    """
    计算每个掩码目标的深度分布，去掉远小于主要深度范围的像素，并保存处理前后分布曲线图。
    每个目标保存两张图：
        1. Original vs Filtered
        2. Histogram vs Softmax-normalized Histogram

    参数:
        y (Tensor): 深度图，shape [1, N, H, W] 或 [N, H, W]
        mask (Tensor): 掩码矩阵，shape [N, H, W]，值为 0/1
        bins (int): 直方图 bin 数
        low_quantile (float): 去掉高分位数的深度点（暂未使用）
        save_dir (str): 保存图像的文件夹

    返回:
        distributions (Tensor): 每个目标的深度分布，shape [N, bins]
    """
    if y.dim() == 4:  # [1, N, H, W]
        y = y.squeeze(0)

    os.makedirs(save_dir, exist_ok=True)
    depths=[]
    distributions = []
    device = y.device 

    for i in range(mask.size(0)):  # 遍历每个目标
        cur_mask = mask[i].squeeze(dim=0)      # [H, W]

        # 掩码区域的深度值
        masked_depth = y[cur_mask >= 0.5]
        
        if masked_depth.numel() == 0:
            # mask为空时使用全0 tensor
            distributions.append(torch.zeros(bins, device=device))
            depths.append(torch.tensor(0., device=device))
            continue

        # 只保留主要深度范围（去掉 top quantile）
        depth_mean = masked_depth.mean()
        depth_std = masked_depth.std()
        
        # 3σ原则确定阈值
        if torch.isnan(depth_mean) or torch.isnan(depth_std):
            low_threshold = 0
            max_threshold = 25
        else:
            low_threshold = depth_mean - 3 * depth_std
            max_threshold = depth_mean + 3 * depth_std
            
        # 筛选出在正常范围内的深度值
        filtered_depth = masked_depth[(masked_depth >= low_threshold) & (masked_depth <= max_threshold)]
        
        # depths.append(filtered_depth.mean() if filtered_depth.numel() > 0 else masked_depth.mean())

        num=filtered_depth.numel()
        max_depth = filtered_depth.max() if num>0 else max_threshold
        min_depth = filtered_depth.min() if num>0 else low_threshold
        
        # 归一化处理后的深度
        norm_depth = (filtered_depth - min_depth) / (max_depth - min_depth + 1e-6)
        hist = torch.histc(norm_depth, bins=bins, min=0, max=1)

        # 归一化纵坐标 (直方图归一化到 [0,1])
        if hist.sum() > 0:
            hist_norm = hist / hist.sum()
        else:
            hist_norm = hist
            
        max_bin_idx = torch.argmax(hist)
        bin_width = (max_depth - min_depth) / bins
        # bin 中心对应的深度值
        mode_depth = min_depth + (max_bin_idx.float() + 0.5) * bin_width
        depths.append(mode_depth)
        distributions.append(hist_norm.to(device))
    
  
    return torch.stack(distributions, dim=0), torch.stack(depths)


# def compute_depth_distributions_with_save(
#     y, mask, bins=10, low_quantile=0.05, save_dir='./depth_distributions_normalized'
# ):
#     """
#     计算每个掩码目标的深度分布，去掉远小于主要深度范围的像素，并保存处理前后分布曲线图。
#     每个目标保存两张图：
#         1. Original vs Filtered
#         2. Histogram vs Softmax-normalized Histogram

#     参数:
#         y (Tensor): 深度图，shape [1, N, H, W] 或 [N, H, W]
#         mask (Tensor): 掩码矩阵，shape [N, H, W]，值为 0/1
#         bins (int): 直方图 bin 数
#         low_quantile (float): 去掉高分位数的深度点（暂未使用）
#         save_dir (str): 保存图像的文件夹

#     返回:
#         distributions (Tensor): 每个目标的深度分布，shape [N, bins]
#     """
#     if y.dim() == 4:  # [1, N, H, W]
#         y = y.squeeze(0)

#     os.makedirs(save_dir, exist_ok=True)
#     depths=[]
#     distributions = []
#     device = y.device 

#     for i in range(mask.size(0)):  # 遍历每个目标
#         cur_mask = mask[i].squeeze(dim=0)      # [H, W]

#         # 掩码区域的深度值
#         masked_depth = y[cur_mask >= 0.5]
        
#         if masked_depth.numel() == 0:
#             # mask为空时使用全0 tensor
#             distributions.append(torch.zeros(bins, device=device))
#             depths.append(torch.tensor(0., device=device))
#             continue

#         # 只保留主要深度范围（去掉 top quantile）
#         depth_mean = masked_depth.mean()
#         depth_std = masked_depth.std()
        
#         # 3σ原则确定阈值
#         if torch.isnan(depth_mean) or torch.isnan(depth_std):
#             low_threshold = 0
#             max_threshold = 25
#         else:
#             low_threshold = depth_mean - 3 * depth_std
#             max_threshold = depth_mean + 3 * depth_std
            
#         # 筛选出在正常范围内的深度值
#         filtered_depth = masked_depth[(masked_depth >= low_threshold) & (masked_depth <= max_threshold)]
        
#         depths.append(filtered_depth.mean() if filtered_depth.numel() > 0 else masked_depth.mean())

#         num=filtered_depth.numel()
#         max_depth = filtered_depth.max() if num>0 else max_threshold
#         min_depth = filtered_depth.min() if num>0 else low_threshold
        
#         # 归一化处理后的深度
#         norm_depth = (filtered_depth - min_depth) / (max_depth - min_depth + 1e-6)
#         hist = torch.histc(norm_depth, bins=bins, min=0, max=1)

#         # 归一化纵坐标 (直方图归一化到 [0,1])
#         if hist.sum() > 0:
#             hist_norm = hist / hist.sum()
#         else:
#             hist_norm = hist
        
#         distributions.append(hist_norm.to(device))
    
#         # 可视化 1: 原始 vs 过滤
#         plt.figure(figsize=(8,4))
#         # 原始深度分布
#         orig_norm_depth = (masked_depth - masked_depth.min()) / (masked_depth.max() - masked_depth.min() + 1e-6)
#         orig_hist = torch.histc(orig_norm_depth, bins=bins, min=0, max=1)
#         orig_hist = orig_hist / (orig_hist.sum() + 1e-6)  # 纵坐标归一化
        
#         plt.plot(range(bins), orig_hist.cpu().numpy(), label='Original', marker='o')
#         plt.plot(range(bins), hist_norm.cpu().numpy(), label='Filtered', marker='x')
#         plt.title(f'Target {i} Depth Distribution (Normalized)')
#         plt.xlabel('Bins')
#         plt.ylabel('Normalized Counts')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'target_{i}_depth_distribution.png'))
#         plt.close()

#         # 可视化 2: 直方图 vs Softmax
#         plt.figure(figsize=(8,4))
#         hist_softmax = F.softmax(hist, dim=-1).cpu().numpy()
#         orig_hist_softmax = F.softmax(orig_hist, dim=-1).cpu().numpy()
        
#         plt.plot(range(bins), orig_hist.cpu().numpy(), label='Original Norm', marker='o')
#         plt.plot(range(bins), orig_hist_softmax, label='Original Softmax', linestyle='--')
#         plt.plot(range(bins), hist_norm.cpu().numpy(), label='Filtered Norm', marker='x')
#         plt.plot(range(bins), hist_softmax, label='Filtered Softmax', linestyle='--')
#         plt.title(f'Target {i} Histogram vs Softmax')
#         plt.xlabel('Bins')
#         plt.ylabel('Probability')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'target_{i}_hist_vs_softmax.png'))
#         plt.close()

#     return torch.stack(distributions, dim=0), torch.stack(depths)



import matplotlib.patches as patches
from .box_ops import box_cxcywh_to_xyxy
import numpy as np
def visualize_small_masks(imgs, masks, targets, depths, min_pixels=50, save_dir='./min_pixels_mask_visualization'):
    """
    可视化像素点较少的掩码及对应的 bounding box 和深度值
    
    参数:
        imgs (Tensor or ndarray): 原始图像列表，shape [B, C, H, W]
        masks (Tensor): 掩码列表，shape [B, N, 1, H, W]
        boxes (list of Tensors): 每张图像的 bounding boxes，shape [N,4]
        depths (list of Tensors): 每个掩码对应的深度
        min_pixels (int): 小于该像素数的掩码会被可视化
        save_dir (str): 保存路径
    """

    os.makedirs(save_dir, exist_ok=True)

    for i, (img, mask_per_img, depths_per_img) in enumerate(zip(imgs, masks,  depths)):
        img_np = np.transpose(img.cpu().numpy(), (1, 2, 0))  # 转为 HWC
        boxes_per_img=targets[i]['boxes']
        for j, (m, box, depth) in enumerate(zip(mask_per_img, boxes_per_img, depths_per_img)):
            m_np = m[0].cpu().numpy()  # [H,W]
            num_pixels = (m_np > 0.5).sum()

            if num_pixels < min_pixels:  # 只显示像素点少的掩码
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_np)

                ax.imshow(m_np, cmap='Reds', alpha=0.4)  # alpha控制透明度

                # 绘制 bounding box
                box = box_cxcywh_to_xyxy(box.unsqueeze(0))[0].cpu().numpy()
                H, W = img_np.shape[:2]
                x1, y1, x2, y2 = box * np.array([W, H, W, H])
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)

                # 添加深度值
                ax.text(
                    x1, y1 - 5,
                    f"Depth: {depth.item():.2f}",
                    color="yellow", fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.5)
                )

                plt.axis("off")
                plt.savefig(os.path.join(save_dir, f"img{i}_mask{j}.png"), bbox_inches="tight", dpi=300)
                plt.close()


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def deformable_attention_core_func(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )

    return output.permute(0, 2, 1)



def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    # print('sampling_locations.shape',sampling_locations.shape)
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1

    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = torch.nn.functional.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)

def deformable_attention_core_func_4depth(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,  # [bs, query_length, n_head, n_levels * n_points, 3]
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
    depth_scale=1.0,  # 用于缩放第三维度的系数
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 3] (x,y,depth)
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    # 分离位置坐标(x,y)和第三维度信息(depth)
    xy_locations = sampling_locations[..., :2]  # [bs, query_length, n_head, n_levels * n_points, 2]
    depth_values = sampling_locations[..., 2:]  # [bs, query_length, n_head, n_levels * n_points, 1]
    
    # 处理位置坐标部分 (x,y)，保持原有逻辑不变
    if method == "default":
        sampling_grids = 2 * xy_locations - 1
    elif method == "discrete":
        sampling_grids = xy_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    
    # 处理第三维度信息 (depth)
    depth_values = depth_values.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [bs*n_head, Len_q, n_levels*n_points, 1]
    depth_list = depth_values.split(num_points_list, dim=-2)
    
    # 特征采样和深度信息融合
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]
        depth_l = depth_list[level].squeeze(-1)  # [bs*n_head, Len_q, num_points]
        
        if method == "default":
            sampling_value_l = torch.nn.functional.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            # 离散采样位置坐标
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)

            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )
        
        # 融合深度信息
        # 这里使用简单的乘法融合，可根据实际需求调整
        depth_factor = depth_l.unsqueeze(1) * depth_scale  # [bs*n_head, 1, Len_q, num_points]
        sampling_value_l = sampling_value_l * depth_factor.expand_as(sampling_value_l)
        
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)

def get_activation(act: str, inpace: bool = True):
    """get activation"""
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace

    return m
