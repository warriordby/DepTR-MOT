"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
"""
from ByteTrack.yolox.tracker.byte_tracker import BYTEDTracker
import math
import sys
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes
from .utils import *
from ..zoo.dfine.dfine_utils import plot_distributions

# ----------------------------
# 评估时临时把滑窗 window_len 设置为 1
# ----------------------------
@contextmanager
def _temp_window1(dataloader):
    """
    在 with 块中把滑窗长度临时改为 1；退出时恢复。
    仅当 dataloader.dataset 是 SlidingWindowView 并且其 base 有 window_len 时生效。
    """
    ds = getattr(dataloader, "dataset", None)
    base = getattr(ds, "base", None)  # _SlidingWindowView.base -> 你的 CocoDetection
    if base is None or not hasattr(base, "window_len"):
        # 非滑窗或拿不到 window_len，直接运行
        yield None
        return

    old_len = int(base.window_len)
    try:
        base.window_len = 1          # ✅ eval 只取单帧
        # 重新构建 sample_begin_frames
        if hasattr(base, "set_epoch"):
            # 用 dataloader 记录的 epoch，拿不到就用 0
            epoch = getattr(dataloader, "epoch", 0)
            base.set_epoch(epoch)
        yield old_len
    finally:
        # 恢复
        base.window_len = old_len
        if hasattr(base, "set_epoch"):
            epoch = getattr(dataloader, "epoch", 0)
            base.set_epoch(epoch)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # ===== 前向（兼容 AMP 与多返回）=====
        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True, dtype=torch.float16, enabled=True):
                outputs, pred_depth, depth_fea , depth,gt_distribution = model(samples, targets=targets)
                # = _split_model_outputs(raw_out)

                # # 训练首个 step 做一次调试打印
                # if i == 0 and (not dist_utils.is_dist_available_and_initialized() or dist_utils.is_main_process()):
                #     tb = targets[0]["boxes"]
                #     pb = outputs.get("pred_boxes", None) if isinstance(outputs, dict) else None
                #     print("[debug] image size (HxW):", samples.shape[-2], "x", samples.shape[-1])
                #     print("[debug] targets range:", float(tb.min()), float(tb.max()))
                #     if isinstance(outputs, dict) and pb is not None:
                #         print("[debug] preds   range:", float(pb.detach().min()), float(pb.detach().max()))
                #         print("[debug] pred_boxes shape:", tuple(outputs["pred_boxes"].shape))
                #     if isinstance(outputs, dict) and "pred_logits" in outputs:
                #         pl = outputs["pred_logits"].detach()
                #         print("[debug] pred_logits shape:", tuple(pl.shape), "range:", float(pl.min()), float(pl.max()))
                #     print("[debug] tgt_boxes  shape:", tuple(tb.shape))
                #     print("[debug] criterion box_fmt:", getattr(criterion, "box_fmt", "<no-box_fmt>"))

                # NaN/Inf 保护
                if isinstance(outputs, dict) and "pred_boxes" in outputs:
                    if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                        print(outputs["pred_boxes"])
                        state = model.state_dict()
                        new_state = {}
                        for key, value in state.items():
                            new_key = key.replace("module.", "")
                            new_state[new_key] = value
                        dist_utils.save_on_master({"model": new_state}, "./NaN.pth")

                # 兼容 criterion 是否需要 fea/depth_fea
                fea = outputs["fea"]
                try:
                    loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution,**metas)
                except TypeError:
                    # 老签名：不带 fea/depth_fea
                    try:
                        loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution, **metas)
                    except TypeError:
                        loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea, depth, gt_distribution,**metas)

                loss = sum(loss_dict.values())
            # 反传
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 非 AMP
            outputs, pred_depth, depth_fea , depth,gt_distribution = model(samples, targets=targets)
            
            fea = outputs["fea"]
            try:
                loss_dict = criterion(outputs, targets, pred_depth, fea, depth_fea,depth, gt_distribution, **metas)
            except TypeError:
                try:
                    loss_dict = criterion(outputs, targets, pred_depth,depth, **metas)
                except TypeError:
                    loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # EMA & warmup
        if ema is not None:
            ema.update(model)
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # 统计
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)})
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    is_visual=False,
    is_track=False,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    iou_types = coco_evaluator.iou_types

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    seen_img_ids = set()  # ✅ 每张图只评一次，避免 COCO 重复

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    if is_track:
        tracker = BYTEDTracker(args = argparse.Namespace(**{
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "mot20": False,
            "det_thresh": 0.2,
            }))

    all_online_targets = []

    # === 关键：评估时把 window_len 临时改为 1 ===
    with _temp_window1(data_loader):
        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            global_step = epoch * len(data_loader) + i

            if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
                save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

            samples = samples.to(device)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            outputs, depth_map , _ = model(samples)


            # 尺寸还原
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            
            results = postprocessor(outputs, orig_target_sizes)

            img_paths = [t["image_path"] for t in targets]
            scnes=[i.split('/')[-3] for i in img_paths]
            if is_visual:
                save_results_with_depth(results, img_paths, os.path.join('./visual_out', "detection_results"))


            if is_track:
                online_target = track(results, tracker, info_imgs=orig_target_sizes.cpu())
                # get_box_depth(targets, depth_map)
                if is_visual:
                    visualize_track_results(
                        img_paths=img_paths,
                        track_results=online_target,
                        output_dir="./visual_out",  # 跟踪可视化结果保存目录
                        class_names=kwargs.get("class_names", None),  # 类别名称列表（可选）
                        depth_maps=depth_map
                    )
                all_online_targets.append(online_target)

            
            # 组织成 {img_id: result}
            batch_res = {}
            for idx, (tgt, resi) in enumerate(zip(targets, results)):
                img_id = int(tgt["image_id"].item())
                if img_id in seen_img_ids:
                    continue
                seen_img_ids.add(img_id)
                batch_res[img_id] = resi

                # Validator 的 GT/Pred
                gt.append({
                    "boxes": scale_boxes(
                        tgt["boxes"],
                        (tgt["orig_size"][1], tgt["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": tgt["labels"],
                })
                if getattr(postprocessor, "remap_mscoco_category", False):
                    labs = torch.tensor(
                        [mscoco_category2label[int(x.item())] for x in resi["labels"].flatten()],
                        device=resi["labels"].device
                    ).reshape(resi["labels"].shape)
                else:
                    labs = resi["labels"]
                preds.append({"boxes": resi["boxes"], "labels": labs, "scores": resi["scores"]})

            if coco_evaluator is not None and len(batch_res) > 0:
                coco_evaluator.update(batch_res)

    if all_online_targets != []:
        visualize_depth_tracks(all_online_targets, save_path="./visual_out/depth_tracks.png")
        visualize_depth_video(all_online_targets, save_path="./visual_out/depth_scatter.mp4")

    # 计算统计
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        wandb.log({**{f"metrics/{k}": v for k, v in metrics.items()}, "epoch": epoch})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types and coco_evaluator.coco_eval.get("bbox") is not None:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types and coco_evaluator.coco_eval.get("segm") is not None:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator