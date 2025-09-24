# src/solver/det_solver.py
"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Modified from RT-DETR / DETR.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch

# === 新增：基于现有 val_dataloader 临时构建“单帧” eval loader（window=1） ===
from .utils_eval_loader import build_eval_loader_from


def _get_val_total_bs(cfg) -> int:
    """
    仅从 YAML 配置里读取 val 的 total_batch_size。
    注意：cfg.val_dataloader 是已构建的 DataLoader；真正配置在 cfg.yaml_cfg（dict）。
    """
    y = getattr(cfg, "yaml_cfg", None)
    if y is None:
        raise ValueError("cfg.yaml_cfg is None; cannot read val_dataloader.total_batch_size")

    vd = y.get("val_dataloader")
    if vd is None:
        raise ValueError("yaml_cfg['val_dataloader'] not found")

    if "total_batch_size" in vd:
        return int(vd["total_batch_size"])

    # 兜底：若未配置 total_batch_size，则用 batch_size；再不行就回退 1
    return int(vd.get("batch_size", 1))


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb
            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        top1 = 0.0
        best_stat = {"epoch": -1}

        # === 恢复后首次评估：改为使用临时 window=1 的 eval loader ===
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            eval_loader, restore = build_eval_loader_from(
                self.val_dataloader,
                epoch=getattr(self.val_dataloader, "epoch", 0),
                win_len=1,
                win_ivl=1,
                total_batch_size=_get_val_total_bs(self.cfg),
            )
            try:
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    eval_loader,
                    self.evaluator,
                    self.device,
                    self.last_epoch,
                    self.use_wandb,
                )
            finally:
                restore()

            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # 阶段切换时的 EMA/LR 等逻辑（保持你的原逻辑）
            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            self.last_epoch += 1

            # 训练早期保存中间 ckpt（保持你的原逻辑）
            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # === 每个 epoch 的评估：改为使用临时 window=1 的 eval loader ===
            module = self.ema.module if self.ema else self.model
            eval_loader, restore = build_eval_loader_from(
                self.val_dataloader,
                epoch=getattr(self.val_dataloader, "epoch", 0),
                win_len=1,
                win_ivl=1,
                total_batch_size=_get_val_total_bs(self.cfg),
            )
            try:
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    eval_loader,
                    self.evaluator,
                    self.device,
                    epoch,
                    self.use_wandb,
                    output_dir=self.output_dir,
                )
            finally:
                restore()

            # 统计与最优模型保存（保持你的原逻辑）
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}", v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {"epoch": -1}
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
            }

            if self.use_wandb:
                wandb_logs = {}
                if "coco_eval_bbox" in test_stats:  # 防止 key 不存在
                    for idx, metric_name in enumerate(metric_names):
                        wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                import wandb as _wandb  # 避免上面 shadow
                _wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model

        # === 验证阶段：强制 window=1 的临时 eval loader ===
        eval_loader, restore = build_eval_loader_from(
            self.val_dataloader,
            epoch=getattr(self.val_dataloader, "epoch", 0),
            win_len=1,
            win_ivl=1,
            total_batch_size=_get_val_total_bs(self.cfg),
        )
        try:
            
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                eval_loader,   # 单帧验证
                self.evaluator,
                self.device,
                epoch=-1,
                use_wandb=False,
                is_visual=self.cfg.visual,
                is_track=self.cfg.track
            )
        finally:
            restore()

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return