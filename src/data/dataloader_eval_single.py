"""
Eval-only single-frame dataloader:
- 不做滑动窗口
- 每个样本是一张单帧图
- 保持与模型一致的输出格式: (images[B,C,H,W], targets[List[dict]])
"""

from __future__ import annotations
import random
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()

__all__ = [
    "EvalDataLoader",
    "EvalBatchImageCollateFunction",
    "eval_batch_image_collate_fn",
]


@register()
class EvalDataLoader(data.DataLoader):
    """只在 eval 阶段使用的 DataLoader（不做滑窗包装）"""
    __inject__ = ["dataset", "collate_fn"]

    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop("dataset", None)
        collate_fn = kwargs.pop("collate_fn", None)
        if dataset is None:
            raise ValueError("EvalDataLoader requires a dataset")

        if collate_fn is None:
            collate_fn = EvalBatchImageCollateFunction()

        super().__init__(dataset=dataset, collate_fn=collate_fn, **kwargs)
        self._shuffle = kwargs.get("shuffle", False)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += f"\n    {n}: {getattr(self, n)}"
        format_string += "\n    mode: single-frame (eval only)\n)"
        return format_string

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def eval_batch_image_collate_fn(items):
    """与训练版保持一致的 batch_image_collate_fn，但只针对单帧"""
    # items: List[(image[C,H,W], target:dict)]
    images = torch.cat([x[0][None] for x in items], dim=0)  # [B,C,H,W]
    targets = [x[1] for x in items]
    return images, targets


class _BaseCollate(object):
    def set_epoch(self, epoch: int):
        self._epoch = epoch
    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)


@register()

class EvalBatchImageCollateFunction(_BaseCollate):
    def __init__(self, base_size=None, keep_ratio=False, stop_epoch=None, **kwargs):
        super().__init__()
        self.base_size = base_size
        self.keep_ratio = keep_ratio

    def __call__(self, items):
        # 1) 取最后一帧 + 规范成 CHW
        imgs, tgts = [], []
        shapes = []  # 记录原始 H,W
        for img, tgt in items:
            if isinstance(img, (list, tuple)):
                assert len(img) > 0, "Empty image window in eval items"
                img = img[-1]
            if isinstance(tgt, (list, tuple)):
                assert len(tgt) > 0, "Empty target window in eval items"
                tgt = tgt[-1]

            assert torch.is_tensor(img), f"Expect Tensor image, got {type(img)}"
            if img.dim() == 2:
                img = img.unsqueeze(0)         # H,W -> 1,H,W
            elif img.dim() != 3:
                raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")

            img = img.contiguous()
            _, H, W = img.shape
            shapes.append((H, W))
            tgts.append(tgt if tgt is not None else {})
            imgs.append(img)

        if self.base_size is not None:
            if isinstance(self.base_size, int):
                tgtH, tgtW = self.base_size, self.base_size
            else:
                tgtH, tgtW = int(self.base_size[0]), int(self.base_size[1])
        else:
            tgtH = max(h for h, _ in shapes)
            tgtW = max(w for _, w in shapes)

        # 3) 按需 resize / pad 对齐
        aligned_imgs = []
        for img in imgs:
            _, H, W = img.shape
            if (H, W) == (tgtH, tgtW):
                aligned_imgs.append(img.unsqueeze(0))
                continue

            if self.keep_ratio:
                # 3a) 等比缩放 + 零填充到 (tgtH, tgtW)
                scale = min(tgtH / H, tgtW / W)
                newH = max(1, int(round(H * scale)))
                newW = max(1, int(round(W * scale)))
                img_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(newH, newW),
                    mode="bilinear", align_corners=False
                )[0]
                # pad 到目标大小（下右方向）
                padH = tgtH - newH
                padW = tgtW - newW
                img_padded = torch.nn.functional.pad(
                    img_resized, (0, padW, 0, padH), mode="constant", value=0
                )
                aligned_imgs.append(img_padded.unsqueeze(0))
            else:
                # 3b) 直接双线性 resize 到 (tgtH, tgtW)
                img_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(tgtH, tgtW),
                    mode="bilinear", align_corners=False
                )
                aligned_imgs.append(img_resized)

        images = torch.cat(aligned_imgs, dim=0)  # [B,C,H,W]

        # 4) 同步调整 masks
        if len(tgts) > 0 and "masks" in tgts[0]:
            for tg in tgts:
                m = tg.get("masks", None)
                if m is None:
                    continue
                if m.dim() == 2:
                    m = m.unsqueeze(0)  # H,W -> 1,H,W
                if self.keep_ratio:
                    _, Hm, Wm = m.shape
                    if (Hm, Wm) != (tgtH, tgtW):
                        scale = min(tgtH / Hm, tgtW / Wm)
                        newH = max(1, int(round(Hm * scale)))
                        newW = max(1, int(round(Wm * scale)))
                        m_resized = torch.nn.functional.interpolate(
                            m.unsqueeze(0).float(), size=(newH, newW),
                            mode="nearest"
                        )[0]
                        padH = tgtH - newH
                        padW = tgtW - newW
                        m_padded = torch.nn.functional.pad(
                            m_resized, (0, padW, 0, padH), mode="constant", value=0
                        )
                        tg["masks"] = m_padded.to(m.dtype)
                else:
                    if m.shape[-2:] != (tgtH, tgtW):
                        tg["masks"] = torch.nn.functional.interpolate(
                            m.unsqueeze(0).float(), size=(tgtH, tgtW),
                            mode="nearest"
                        )[0].to(m.dtype)

        return images, tgts