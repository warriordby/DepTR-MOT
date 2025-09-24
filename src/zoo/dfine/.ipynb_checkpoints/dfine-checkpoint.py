"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn
import sys
import os
import torch
import matplotlib.pyplot as plt
from VideoDepthAnything import VideoDepthAnything
from SAM2.sam2.build_sam import build_sam2
from SAM2.sam2.sam2_image_predictor import SAM2ImagePredictor
from ...core import register
import numpy as np
__all__ = [
    "DFINE",
]
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # matplotlib 默认自带
import torch.nn.functional as F
from .box_ops import box_cxcywh_to_xyxy
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
from .utils import compute_depth_distributions_with_save, visualize_small_masks

# def compute_depth_distributions(y, mask, bins=10, max_depth=25.0):
#     """
#     计算每个掩码目标的深度分布。

#     参数:
#         y (Tensor): 深度图，shape [1, N, H, W] 或 [N, H, W]
#         mask (Tensor): 掩码矩阵，shape [N, H, W]，值为 0/1
#         bins (int): 直方图 bin 数
#         max_depth (float): 最大深度值，用于归一化到 [0,1]

#     返回:
#         distributions (list of Tensor): 每个目标的深度分布，shape [bins]
#         avg_depths (list of float): 每个目标的平均深度
#     """
#     if y.dim() == 4:  # [1, N, H, W]
#         y = y.squeeze(0)
        
#     distributions = []

#     for i in range(mask.size(0)):  # 遍历每个目标
#         cur_mask = mask[i].squeeze(dim=0)      # [H, W]

#         # 掩码区域的深度值
#         masked_depth = y[cur_mask == 1]

#         # 归一化到 [0,1]
#         norm_depth = (masked_depth / max_depth).clamp(0, 1)

#         # 计算直方图
#         hist = torch.histc(norm_depth, bins=bins, min=0, max=1)
#         # prob_dist = hist / (hist.sum() + 1e-6)
#         # 转成概率分布
#         prob_dist = F.softmax(hist, dim=0)

#         distributions.append(prob_dist)


#     return torch.stack(distributions, dim=0)



@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.depth_pred = VideoDepthAnything(**{'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}) #'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
        self.depth_pred.load_state_dict(torch.load(f'./VideoDepthAnything/video_depth_anything_vits.pth', map_location='cpu'), strict=True)
        checkpoint = "SAM2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.mask_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        # 冻结 depth_pred 的参数，使其不参与梯度更新
        for param in self.depth_pred.parameters():
            param.requires_grad = False  # 禁止参数计算梯度
        for param in self.mask_predictor.model.parameters():
            param.requires_grad = False  # 禁止参数计算梯度
        self.depth_pred.eval()
        self.mask_predictor.model.eval()


    def forward(self, x, targets=None):


        if targets is not None:
            print('x.shape', x.shape)
            b,c,h,w=x.shape
            imgs=x.detach() #.cpu().numpy()
            
            y, depth_fea=self.depth_pred(x.unsqueeze(dim=0).detach()) # tensor detach()
            # print('max_depth',y.max())
            # print('min_depth',y.min())
            x = self.backbone(x)
            # fea=x
            x = self.encoder(x)
            x = self.decoder(x, targets)
            masks=[]
            depths=[]
            gt_distributions=[]
            for i, img in enumerate(imgs):
                target_boxes = targets[i]['boxes'].detach()
                masks_per_img = []
                img = np.transpose(img.cpu().numpy(), (1, 2, 0)) 
                self.mask_predictor.set_image(img)
                for box in target_boxes:  # 🔑 每个box单独预测一次
                    box_xyxy = box_cxcywh_to_xyxy(box.unsqueeze(0)) * torch.tensor([w, h, w, h], device=x["pred_boxes"].device)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        mask, scores, logits = self.mask_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box_xyxy,   # 每次一个 box
                            multimask_output=False,
                        )
                    mask=torch.from_numpy(mask).float().cuda().detach()
                    masks_per_img.append(mask)
                # print('depth max',  y.max())
                # print('depth min',  y.min())
                if masks_per_img:  # 检查列表是否非空
                    masks_per_img = torch.stack(masks_per_img)
                    depth=(masks_per_img * y.squeeze(dim=0)[i, None, None, :, :]).sum(dim=(-2, -1)) / (masks_per_img.sum(dim=(-2, -1)) + 1e-6)
                else:
                    print('target_boxes',target_boxes)
                    print('len img',len(imgs))
                    masks_per_img = torch.empty((0, 1, h, w), device=x["pred_boxes"].device)  # (0个box, 1, h, w)
                    depth = torch.empty((0, 1), device=x["pred_boxes"].device)
                masks.append(masks_per_img)

                # gt_distribution , depth = compute_depth_distributions_with_save(y.squeeze(dim=0)[i] , torch.stack(masks_per_img), bins=25)
                # gt_distributions.append(gt_distribution)
                
                
                if depth != None:
                    depths.append(depth.detach().to(imgs.device))
                else:
                    depths.append([])
            # visualize_small_masks(imgs, masks, targets, depths, min_pixels=50)

            # for i, (img, mask) in enumerate(zip(imgs, masks)):
            #     # 原始图像
            #     img_np = np.transpose(img.cpu().detach(), (1, 2, 0)) 
            #     img_np = img_np / img_np.max()
            #     boxes=targets[i]['boxes']
            #     scne=targets[i]["image_path"].split('/')[-3]
            #     frame=targets[i]["image_path"].split('/')[-1]
            #     # 每个框和对应的mask
            #     for j, (m, box) in enumerate(zip(mask, boxes)):
            #         # 画mask轮廓
            #         fig, ax = plt.subplots(figsize=(8, 8))
                
            #         ax.imshow(img_np)
            #         # ax.contour(m[0].cpu().numpy(), colors="r", linewidths=2)
            #         ax.imshow(m[0].cpu().numpy(), cmap='Reds', alpha=0.4)
            #         # 画bbox
            #         box = box_cxcywh_to_xyxy(box[:4].unsqueeze(0))[0].cpu().detach().numpy()*np.array([w, h, w, h])
            #         x1, y1, x2, y2 = box
            #         rect = patches.Rectangle(
            #             (x1, y1), x2 - x1, y2 - y1,
            #             linewidth=2, edgecolor="lime", facecolor="none"
            #         )
            #         ax.add_patch(rect)
            
            #         # 对应的深度值
            #         depth_value = depths[i][j].item()
            #         ax.text(
            #             x1, y1 - 5,
            #             f"Depth: {depth_value:.2f}",
            #             color="yellow", fontsize=10, bbox=dict(facecolor="black", alpha=0.5)
            #         )
            #         os.makedirs(f"./mask/{scne}",exist_ok=True)
            #         plt.axis("off")
            #         plt.savefig(f"./mask/{scne}/{frame}", bbox_inches="tight", dpi=300)
            #         plt.close()
            return x, y.squeeze(dim=0).detach(), depth_fea, depths, None # torch.cat(gt_distributions, dim=0).detach()
        
        else:
            print('x.shape', x.shape)
            b,c,h,w=x.shape
            # y, depth_fea=self.depth_pred(x.unsqueeze(dim=0)) # tensor detach()

            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x, targets)

            return x, None, None#, y.squeeze(dim=0)

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
