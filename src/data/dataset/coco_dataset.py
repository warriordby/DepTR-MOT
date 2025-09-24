"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import faster_coco_eval.core.mask as coco_mask
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import torch
import torchvision
import os
from PIL import Image
from typing import Dict, List, Tuple
from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection"]

# ======================================================================
#                           Sliding-Window CocoDetection
# ======================================================================
@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self,
                 img_folder,
                 ann_file,
                 transforms,
                 return_masks=False,
                 remap_mscoco_category=False,
                 window_len=None,
                 window_interval=None,
                 **kwargs):
        super().__init__(img_folder, ann_file)

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        # 之前是读环境变量，改成：优先 YAML；否则回退 env；再否则用默认
        import os
        if window_len is None:
            window_len = int(os.getenv("DFINE_WINDOW_LEN", "2"))
        if window_interval is None:
            window_interval = int(os.getenv("DFINE_WINDOW_INTERVAL", "1"))

        self.window_len = int(window_len)
        self.window_interval = int(window_interval)
        

        # ----------------- 基于 file_name 构建 video/frame 索引 -----------------
        self._imgid2dsidx: Dict[int, int] = {img_id: i for i, img_id in enumerate(self.ids)}
        self._video2frames: Dict[str, List[int]] = {}         # vid -> sorted frame list (int)
        self._vf2idx: Dict[Tuple[str, int], int] = {}         # (vid, frame) -> dataset index
        self._video2frame_ids: Dict[str, List[int]] = {}      # vid -> sorted dataset indices

        for ds_idx, img_id in enumerate(self.ids):
            rec = self.coco.loadImgs(img_id)[0]
            file_name = rec["file_name"]  # 期待形如 'dancetrack0052/img1/00000896.jpg'
            vid, frm = self._parse_video_and_frame(file_name)
            self._video2frames.setdefault(vid, []).append(frm)
            self._vf2idx[(vid, frm)] = ds_idx

        # 排序，记录 dataset 索引的顺序序列（方便按位置滑窗，不依赖帧号是否连续）
        for vid, frames in self._video2frames.items():
            frames.sort()
            self._video2frames[vid] = frames
            ds_indices = [self._vf2idx[(vid, f)] for f in frames]
            self._video2frame_ids[vid] = ds_indices

        # 初始化滑窗起点
        self.set_epoch(0)
    def get_img_info(self, idx: int) -> dict:
            """
            返回图像的元信息（不加载图像本身）
            用于评估/COCO API 转换时避免读取图片
            """
            image_id = self.ids[idx]
            rec = self.coco.loadImgs(image_id)[0]  # coco 标注里已有 width/height
            return {
                "image_id": image_id,
                "file_name": rec["file_name"],
                "height": rec["height"],
                "width": rec["width"],
                "idx": idx,
                "image_path": os.path.join(self.img_folder, rec["file_name"])
            }
    # ----------------- 纯滑窗 DataLoader 依赖的 4 个接口 -----------------
    def set_epoch(self, epoch: int):
        """
        构建滑窗起点列表：sample_begin_frames = [(vid, pos0), ...]
        第二个元素是“帧位置索引”（在该视频内的下标），而非真实帧号。
        """
        self.sample_begin_frames: List[Tuple[str, int]] = []
        L, K = self.window_len, self.window_interval
        for vid, frames in self._video2frames.items():
            T = len(frames)
            max_start = T - (L - 1) * K
            if max_start <= 0:
                continue
            for start_pos in range(0, max_start):
                self.sample_begin_frames.append((vid, start_pos))

    def sample_frames_idx(self, vid: str, begin_frame: int) -> List[int]:
        """
        输入：vid、begin_frame(位置索引)
        输出：真实帧号列表（int）
        """
        frames = self._video2frames[vid]
        L, K = self.window_len, self.window_interval
        idxs = []
        for i in range(L):
            pos = begin_frame + i * K
            if pos >= len(frames):
                break
            idxs.append(frames[pos])
        return idxs

    def get_multi_frames(self, vid: str, idxs: List[int]):
        """
        输入一组“真实帧号”，返回 imgs(list of PIL.Image) 与 infos(list of target dict)
        复用本类的 load_item 流程，保持与单帧一致的 target 结构。
        """
        imgs, infos = [], []
        for frm in idxs:
            ds_idx = self._vf2idx[(vid, frm)]
            img, target = self.load_item(ds_idx)
            imgs.append(img)
            infos.append(target)
        return imgs, infos

    def transform(self, imgs: List[Image.Image], infos: List[dict]):
        """
        多帧版 transform：把原单帧 _transforms 逐帧应用。
        """
        if self._transforms is None:
            return imgs, infos
        out_imgs, out_infos = [], []
        for img, target in zip(imgs, infos):
            img2, tgt2, _ = self._transforms(img, target, self)
            out_imgs.append(img2)
            out_infos.append(tgt2)
        return out_imgs, out_infos

    # ----------------- 原有单帧接口（保留以兼容其他路径） -----------------
    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_path = os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"])
        target = {"image_id": image_id, "image_path": image_path, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], key="boxes", spatial_size=image.size[::-1]
            )
        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")
        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        s += f" window_len: {self.window_len}, window_interval: {self.window_interval}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    # ----------------- categories（保留） -----------------
    @property
    def categories(self):
        return self.coco.dataset["categories"]

    @property
    def category2name(self):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}

    # ----------------- helper: 解析 video / frame -----------------
    def _parse_video_and_frame(self, file_name: str) -> Tuple[str, int]:
        """
        期望路径类似: 'dancetrack0052/img1/00000896.jpg'
        - video: 上上级目录（img1 的父目录）
        - frame: 文件名（去扩展名）转 int
        若没有 'img1'，则 video='default'，frame 从文件名纯数字解析；若失败则回退为 0。
        """
        p = file_name.replace("\\", "/").split("/")
        vid = "default"
        frm = None
        # 优先匹配 */<video>/img1/<frame>.jpg
        if len(p) >= 3 and p[-2].lower() == "img1":
            vid = p[-3]
            base = os.path.splitext(p[-1])[0]
            try:
                frm = int(base)
            except Exception:
                frm = None
        if frm is None:
            base = os.path.splitext(p[-1])[0]
            try:
                frm = int(base)
            except Exception:
                frm = 0
        return vid, frm

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        image_path = target["image_path"]

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["image_path"] = image_path
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        depth= torch.tensor([-1 for obj in anno]) #torch.tensor([obj['depth']for obj in anno])
        target['depth'] = depth #(depth-depth.min()) / (-depth.min() + depth.max())
        
        # print('target[\'depth\']', target['depth'])
        return image, target


mscoco_category2name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
