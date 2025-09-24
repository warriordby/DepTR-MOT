"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()

class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes, depths = outputs["pred_logits"], outputs["pred_boxes"][...,:-1], outputs["pred_boxes"][...,-1:]
        
        # 转换边界框格式并调整大小 (xyxy格式)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = torch.fmod(index, self.num_classes)
            index = index // self.num_classes
            
            # 对框和深度应用相同的索引
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            depths = depths.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, depths.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                
                # 对框和深度应用相同的索引
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                depths = torch.gather(
                    depths, dim=1, index=index.unsqueeze(-1).tile(1, 1, depths.shape[-1])
                )

        # 重新映射COCO类别
        if self.remap_mscoco_category and not self.deploy_mode:
            from ...data.dataset import mscoco_label2category
            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        # 准备适合bytesort的格式: [x1, y1, x2, y2, score, depth, label]
        # 合并所有信息到一个张量中
        batch_size = boxes.shape[0]
        num_queries = boxes.shape[1]
        
        # 调整深度和标签的形状以匹配
        depths_flat = depths.squeeze(-1)  # 移除多余维度
        labels_float = labels.float()     # 转换为float以便合并
        
        # 按bytesort要求的格式拼接: [x1, y1, x2, y2, score, depth, label]
        bytesort_input = torch.cat([
            boxes,  # 4个坐标值
            scores.unsqueeze(-1),  # 分数
            depths_flat.unsqueeze(-1),  # 深度值
            labels_float.unsqueeze(-1)  # 标签
        ], dim=-1)  # 形状: [batch_size, num_queries, 7]

        # 部署模式下直接返回整理好的张量
        if self.deploy_mode:
            return bytesort_input

        # 构建结果列表，同时保留原始格式和bytesort格式
        results = []
        for bbox, score, lab, dep, bs_input in zip(boxes, scores, labels, depths_flat, bytesort_input):
            result = {
                'labels': lab,
                'boxes': bbox,
                'scores': score,
                'depths': dep,
                'bytesort_input': bs_input  # 适合bytesort的格式
            }
            results.append(result)

        return results
    
    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
