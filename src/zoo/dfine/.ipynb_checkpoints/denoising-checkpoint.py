"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


# def get_contrastive_denoising_training_group(
#     targets,
#     num_classes,
#     num_queries,
#     class_embed,
#     num_denoising=100,
#     label_noise_ratio=0.5,
#     box_noise_scale=1.0,
# ):
#     """cnd"""
#     if num_denoising <= 0:
#         return None, None, None, None

#     num_gts = [len(t["labels"]) for t in targets]
#     device = targets[0]["labels"].device

#     max_gt_num = max(num_gts)
#     if max_gt_num == 0:
#         dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
#         return None, None, None, dn_meta

#     num_group = num_denoising // max_gt_num
#     num_group = 1 if num_group == 0 else num_group
#     # pad gt to max_num of a batch
#     bs = len(num_gts)

#     input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
#     input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
#     pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

#     for i in range(bs):
#         num_gt = num_gts[i]
#         if num_gt > 0:
#             input_query_class[i, :num_gt] = targets[i]["labels"]
#             input_query_bbox[i, :num_gt] = targets[i]["boxes"]
#             pad_gt_mask[i, :num_gt] = 1
#     # each group has positive and negative queries.
#     input_query_class = input_query_class.tile([1, 2 * num_group])
#     input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
#     pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
#     # positive and negative mask
#     negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
#     negative_gt_mask[:, max_gt_num:] = 1
#     negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
#     positive_gt_mask = 1 - negative_gt_mask
#     # contrastive denoising training positive index
#     positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
#     dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
#     dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
#     # total denoising queries
#     num_denoising = int(max_gt_num * 2 * num_group)

#     if label_noise_ratio > 0:
#         mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
#         # randomly put a new one here
#         new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
#         input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

#     if box_noise_scale > 0:
#         known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
#         diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
#         rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
#         rand_part = torch.rand_like(input_query_bbox)
#         rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
#         # shrink_mask = torch.zeros_like(rand_sign)
#         # shrink_mask[:, :, :2] = (rand_sign[:, :, :2] == 1)  # rand_sign == 1 → (x1, y1) ↘ →  smaller bbox
#         # shrink_mask[:, :, 2:] = (rand_sign[:, :, 2:] == -1)  # rand_sign == -1 →  (x2, y2) ↖ →  smaller bbox
#         # mask = rand_part > (upper_bound / (upper_bound+1))
#         # # this is to make sure the dn bbox can be reversed to the original bbox by dfine head.
#         # rand_sign = torch.where((shrink_mask * (1 - negative_gt_mask) * mask).bool(), \
#         #                         rand_sign * upper_bound / (upper_bound+1) / rand_part, rand_sign)
#         known_bbox += rand_sign * rand_part * diff
#         known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
#         input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
#         input_query_bbox[input_query_bbox < 0] *= -1
#         input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

#     input_query_logits = class_embed(input_query_class)

#     tgt_size = num_denoising + num_queries
#     attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
#     # match query cannot see the reconstruction
#     attn_mask[num_denoising:, :num_denoising] = True

#     # reconstruct cannot see each other
#     for i in range(num_group):
#         if i == 0:
#             attn_mask[
#                 max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
#                 max_gt_num * 2 * (i + 1) : num_denoising,
#             ] = True
#         if i == num_group - 1:
#             attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
#         else:
#             attn_mask[
#                 max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
#                 max_gt_num * 2 * (i + 1) : num_denoising,
#             ] = True
#             attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

#     dn_meta = {
#         "dn_positive_idx": dn_positive_idx,
#         "dn_num_group": num_group,
#         "dn_num_split": [num_denoising, num_queries],
#     }

#     # print(input_query_class.shape) # torch.Size([4, 196, 256])
#     # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
#     # print(attn_mask.shape) # torch.Size([496, 496])

#     return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    depth_noise_scale=0.3,  # 新增：深度噪声缩放因子
):
    """cnd with depth prediction"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return None, None, None, dn_meta

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)

    # 初始化边界框和深度预测
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)  # 四维边界框 (cx, cy, w, h)
    input_query_depth = torch.zeros([bs, max_gt_num, 1], device=device)  # 一维深度预测
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    # 填充真实标签、边界框和深度
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"][..., :4]  # 前四维是边界框
            # if "depth" in targets[i]["boxes"][0].keys():  # 检查是否有深度信息
                ###############################################

            input_query_depth[i, :num_gt] = targets[i]["depth"][...,-1]  # 第五维是深度
            pad_gt_mask[i, :num_gt] = 1

    # 复制多份以创建去噪组
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    input_query_depth = input_query_depth.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    # 创建正负样本掩码
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    # 计算正样本索引
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    # 总去噪查询数
    num_denoising = int(max_gt_num * 2 * num_group)

    # 添加标签噪声
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # 添加边界框噪声
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1

    # 添加深度噪声
    if depth_noise_scale > 0:
        depth_diff = input_query_depth * depth_noise_scale  # 根据深度值大小按比例添加噪声
        depth_rand_sign = torch.randint_like(input_query_depth, 0, 2) * 2.0 - 1.0  # 随机正负
        depth_rand_part = torch.rand_like(input_query_depth)
        depth_rand_part = (depth_rand_part + 1.0) * negative_gt_mask + depth_rand_part * (1 - negative_gt_mask)
        input_query_depth += depth_rand_sign * depth_rand_part * depth_diff
        input_query_depth = torch.clip(input_query_depth, min=0.0)  # 确保深度值非负

    # 将边界框和深度拼接为完整的预测
    input_query_bbox_depth = torch.cat([input_query_bbox, input_query_depth], dim=-1)  # [bs, num_queries, 5]
    input_query_bbox_depth_unact = inverse_sigmoid(input_query_bbox_depth)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True

    # 设置去噪组之间的注意力掩码
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    return input_query_logits, input_query_bbox_depth_unact, attn_mask, dn_meta