"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Modified: sliding-window only version
- Remove all single-frame paths
- Keep public names: DataLoader, BaseCollateFunction, BatchImageCollateFunction, batch_image_collate_fn
- Always return sequences [B, T, C, H, W]
"""
from __future__ import annotations
from torchvision.transforms.functional import to_tensor
import random
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from ..core import register

torchvision.disable_beta_transforms_warning()

__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]

# ===== 在文件顶部附近加一个小工具函数 =====
def _deep_base(ds):
    """unwrap to the bottom dataset (e.g., CocoDetection)"""
    while hasattr(ds, "base"):
        ds = ds.base
    return ds
def _unwrap_sliding(ds):
    while isinstance(ds, _SlidingWindowView):
        ds = ds.base
    return ds


# -------------------------
# 基础 Collate（保留名）
# -------------------------
class BaseCollateFunction(object):
    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)

    def __call__(self, items):
        raise NotImplementedError("BaseCollateFunction.__call__ not implemented")


# -------------------------
# 内部：序列 collate 实现
# items: List[(imgs[T,C,H,W] or list[T*Tensor[C,H,W]], targets[List[dict]]])]
# -> images[B,T,C,H,W], targets[List[List[dict]]]
# -------------------------
def _seq_collate(items):
    seqs, tgts = [], []
    for imgs, t in items:
        # imgs: list[T * Tensor[C,H,W]] or Tensor[T,C,H,W]
        if isinstance(imgs, (list, tuple)):
            imgs = torch.stack(imgs, dim=0)  # [T,C,H,W]
        assert torch.is_tensor(imgs) and imgs.dim() == 4, f"Expect [T,C,H,W], got {type(imgs)} {getattr(imgs, 'shape', None)}"
        seqs.append(imgs)   # List[T,C,H,W]
        tgts.append(t)      # List[List[dict]]

    # Stack images to [B, T, C, H, W] → flatten to [B*T, C, H, W]
    images = torch.stack(seqs, dim=0)      # [B,T,C,H,W]
    B, T, C, H, W = images.shape
    images = images.view(B * T, C, H, W)   

    # Flatten targets: List[List[dict]] → List[dict]
    targets = [frame for seq in tgts for frame in seq] 
    return images, targets


@register()
def batch_image_collate_fn(items):
    return _seq_collate(items)

# -------------------------
@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch: Optional[int] = None,
        base_size: Optional[int | Tuple[int, int]] = None,
        keep_ratio: bool = False,
        ema_restart_decay: float = 0.9999, 
        base_size_repeat: Optional[int] = None,  
        **kwargs, 
    ) -> None:
       
        params = kwargs.pop("params", None)
        if isinstance(params, dict):
            if "stop_epoch" in params: stop_epoch = params["stop_epoch"]
            if "base_size" in params: base_size = params["base_size"]
            if "keep_ratio" in params: keep_ratio = params["keep_ratio"]
            if "base_size_repeat" in params: base_size_repeat = params["base_size_repeat"]

        super().__init__()
        self.stop_epoch = stop_epoch if stop_epoch is not None else 10**8
        self.base_size = base_size
        self.keep_ratio = keep_ratio

    def __call__(self, items):
       
        images, targets = _seq_collate(items)  # images: [B*T,C,H,W] ; targets: List[dict]
        BT, C, H, W = images.shape
        newH, newW = H, W

        if self.base_size is not None and self.epoch < self.stop_epoch:
            if isinstance(self.base_size, int):
                newH, newW = self.base_size, self.base_size
            elif isinstance(self.base_size, (list, tuple)) and len(self.base_size) == 2:
                newH, newW = int(self.base_size[0]), int(self.base_size[1])
            else:
                raise ValueError(f"Invalid base_size: {self.base_size}")

            if (newH != H) or (newW != W):
                images = F.interpolate(images, size=(newH, newW), mode="bilinear", align_corners=False)

        try:
            from torchvision.ops import box_convert
        except Exception:
            box_convert = torchvision.ops.box_convert 

        def _scale_xyxy(boxes_xyxy: torch.Tensor, src_h: int, src_w: int, dst_h: int, dst_w: int):
            if (src_h == dst_h) and (src_w == dst_w):
                return boxes_xyxy
            sx = float(dst_w) / float(src_w)
            sy = float(dst_h) / float(src_h)
            out = boxes_xyxy.clone()
            out[..., [0, 2]] *= sx
            out[..., [1, 3]] *= sy
            return out

        for t in targets:
            if "boxes" not in t:
                continue
            boxes = t["boxes"]
            if boxes.numel() == 0:
                t["boxes_fmt"] = "cxcywh"
                continue

            if t.get("boxes_fmt", None) == "cxcywh":
                continue

            # 读取原图尺寸（注意 ConvertCocoPolysToMask 里 orig_size = [W, H]）
            if "orig_size" in t:
                src_w, src_h = int(t["orig_size"][0]), int(t["orig_size"][1])
            else:
                # 用当前张量尺寸
                src_h, src_w = H, W

            boxes_xyxy = _scale_xyxy(boxes, src_h, src_w, newH, newW)

            # 转为 cxcywh
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            # 归一化到 0~1
            boxes_cxcywh[..., [0, 2]] /= float(newW)  # cx, w
            boxes_cxcywh[..., [1, 3]] /= float(newH)  # cy, h
            t["boxes"] = boxes_cxcywh.clamp_(0, 1)
            t["boxes_fmt"] = "cxcywh"

        return images, targets


class _SlidingWindowView(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = _unwrap_sliding(base_dataset)
        needed = ["sample_begin_frames", "sample_frames_idx", "get_multi_frames"]
        for n in needed:
            if not hasattr(self.base, n):
                raise AttributeError(
                    f"SlidingWindowView requires dataset.{n}, but {type(self.base).__name__} lacks it."
                )

    def __len__(self) -> int:
        return len(self.base.sample_begin_frames)

    def set_epoch(self, epoch: int):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __getitem__(self, index: int):
        vid, begin = self.base.sample_begin_frames[index]
        idxs = self.base.sample_frames_idx(vid=vid, begin_frame=begin)

        if getattr(self.base, "_eval_window1", False):
            idxs = [idxs[-1]]
        imgs, infos = self.base.get_multi_frames(vid=vid, idxs=idxs)
        if isinstance(imgs, (list, tuple)) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
            imgs = [to_tensor(im) for im in imgs]

        if getattr(self.base, "transform", None) is not None:
            imgs, infos = self.base.transform(imgs, infos)

        return imgs, infos

    def load_item(self, index: int):
        if hasattr(self.base, "load_item"):
            return self.base.load_item(index)
        else:
            return self.__getitem__(index)



# =====  DataLoader.__init__，支持 window_len / window_interval =====
@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __init__(self, *args, **kwargs):
        params = kwargs.pop("params", None)
        if isinstance(params, dict):
            kwargs.update(params)

        dataset = kwargs.pop("dataset", None)
        collate_fn = kwargs.pop("collate_fn", None)

        pos = list(args)
        if dataset is None and len(pos) > 0:
            dataset = pos.pop(0)
        if collate_fn is None and len(pos) > 0:
            maybe_cf = pos[0]
            if callable(maybe_cf) or isinstance(maybe_cf, BaseCollateFunction):
                collate_fn = pos.pop(0)

        if dataset is None:
            raise ValueError("DataLoader requires a dataset")

        # 滑窗
        if not isinstance(dataset, _SlidingWindowView):
            print("[DataLoader] Wrapping dataset with SlidingWindowView")
            dataset = _SlidingWindowView(dataset)
        else:
            print("[DataLoader] Dataset is already SlidingWindowView")

        # 读取底层 window 配置并保存到 DataLoader 
        base = dataset
        while hasattr(base, "base"):
            base = base.base
        self._window_len = getattr(base, "window_len", None)
        self._window_interval = getattr(base, "window_interval", None)

        print(f"[DataLoader] window_len={self._window_len}, window_interval={self._window_interval}")

        if collate_fn is None:
            collate_fn = BatchImageCollateFunction()

        super().__init__(dataset=dataset, collate_fn=collate_fn, **kwargs)
        self._shuffle = kwargs.get("shuffle", False)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += f"    {n}: {getattr(self, n)}"

        format_string += f"\n    mode: window"
        format_string += f"\n    window_len: {getattr(self, '_window_len', None)}, window_interval: {getattr(self, '_window_interval', None)}\n)"
        return format_string

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)
    # ===== 在 DataLoader 类里加个便捷方法：运行期改窗口 =====
    def set_window(self, window_len: int | None = None, window_interval: int | None = None, rebuild: bool = True):
        base_ds = _deep_base(self.dataset)
        changed = False
        if window_len is not None and hasattr(base_ds, "window_len"):
            base_ds.window_len = int(window_len); changed = True
            self._window_len = int(window_len)
        if window_interval is not None and hasattr(base_ds, "window_interval"):
            base_ds.window_interval = int(window_interval); changed = True
            self._window_interval = int(window_interval)
        if changed and rebuild and hasattr(base_ds, "set_epoch"):
            base_ds.set_epoch(getattr(self, "epoch", 0))
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