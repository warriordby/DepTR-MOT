# src/solver/utils_eval_loader.py
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

def _unwrap_base(ds):
    while hasattr(ds, "base"):
        ds = ds.base
    return ds

def build_eval_loader_from(orig_loader,
                           epoch: int = 0,
                           win_len: int = 1,
                           win_ivl: int = 1,
                           pick: str = "last",
                           total_batch_size: int | None = None):
    """
    从已有的 val_dataloader 构造一个“只取单帧”的 eval loader
    - total_batch_size: 从配置传入的全局批量大小（推荐），自动按 world_size 均分
    """
    ds = getattr(orig_loader, "dataset", None)
    if ds is None:
        raise ValueError("orig_loader has no dataset")
    base = _unwrap_base(ds)

    # 备份原设置
    prev = {
        "window_len": getattr(base, "window_len", None),
        "window_interval": getattr(base, "window_interval", None),
        "_eval_window1": getattr(base, "_eval_window1", False),
        "_eval_pick": getattr(base, "_eval_pick", "last"),
    }
    setattr(base, "_eval_window1", True)
    setattr(base, "_eval_pick", pick)

    if hasattr(base, "window_len"):
        base.window_len = int(win_len)
    if hasattr(base, "window_interval"):
        base.window_interval = int(win_ivl)
    if hasattr(base, "set_epoch"):
        base.set_epoch(epoch)

    # ====== 关键部分：按 total_batch_size // world_size 计算 ======
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    if total_batch_size is None:
        raise ValueError("必须传入 total_batch_size 才能正确计算 batch_size")

    if total_batch_size % world_size != 0:
        raise ValueError(f"total_batch_size={total_batch_size} 不能整除 world_size={world_size}")

    bs = total_batch_size // world_size
    # ===========================================================

    kwargs = {
        "dataset": ds,
        "batch_size": bs,
        "num_workers": getattr(orig_loader, "num_workers", 0),
        "collate_fn": getattr(orig_loader, "collate_fn", None),
        "pin_memory": getattr(orig_loader, "pin_memory", False),
        "drop_last": False,
        "shuffle": False,
    }

    if world_size > 1:
        kwargs["sampler"] = DistributedSampler(ds, shuffle=False, drop_last=False)

    new_loader = TorchDataLoader(**kwargs)

    print(f"[EvalLoader] force window_len={win_len}, interval={win_ivl}, pick={pick}")
    print(f"[EvalLoader] total_batch_size={total_batch_size}, world_size={world_size}, batch_size={bs}, num_workers={kwargs['num_workers']}")

    def restore():
        setattr(base, "_eval_window1", prev["_eval_window1"])
        setattr(base, "_eval_pick", prev["_eval_pick"])
        if prev["window_len"] is not None:
            base.window_len = prev["window_len"]
        if prev["window_interval"] is not None:
            base.window_interval = prev["window_interval"]
        if hasattr(base, "set_epoch"):
            base.set_epoch(epoch)

    return new_loader, restore