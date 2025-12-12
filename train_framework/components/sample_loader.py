# train_framework/components/sample_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader

from train_framework.components.loader.load_npz_dataset import (
    build_loaders_from_export,
)

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

Task = Literal["cls", "reg"]


# def _collate_force_last_dim_is_one():
#     """
#     Collate that:
#       - if X is (B, F)      -> (B, F, 1)
#       - if X is (B, L, F)   -> pass through
#       - if X is (B, F, 1)   -> pass through
#     """
#     def collate(batch):
#         xs, ys = zip(*batch)
#         X = torch.stack([torch.as_tensor(x) for x in xs], dim=0)
#         if X.dim() == 2:
#             B, F = X.shape
#             X = X.view(B, F, 1)  # force (B, F, 1)
#         elif X.dim() == 3:
#             pass  # already sequence-like
#         else:
#             raise ValueError(f"Unexpected input tensor shape {tuple(X.shape)}; expected 2D or 3D.")

#         y0 = ys[0]
#         if isinstance(y0, torch.Tensor):
#             Y = torch.stack([torch.as_tensor(y) for y in ys], dim=0)
#         else:
#             Y = torch.as_tensor(ys)
#         return X, Y
#     return collate

def _collate_force_last_dim_is_one(batch):
    """
    Collate that:
      - if X is (B, F)      -> (B, F, 1)
      - if X is (B, L, F)   -> pass through
      - if X is (B, F, 1)   -> pass through
    """
    xs, ys = zip(*batch)

    # Stack X
    X = torch.stack([torch.as_tensor(x) for x in xs], dim=0)
    if X.dim() == 2:
        B, F = X.shape
        X = X.view(B, F, 1)  # force (B, F, 1)
    elif X.dim() == 3:
        # (B, L, F) or (B, F, 1): pass through
        pass
    else:
        raise ValueError(f"Unexpected input tensor shape {tuple(X.shape)}; expected 2D or 3D.")

    # Stack Y
    y0 = ys[0]
    if isinstance(y0, torch.Tensor):
        Y = torch.stack([torch.as_tensor(y) for y in ys], dim=0)
    else:
        Y = torch.as_tensor(ys)

    return X, Y

''' create train and validate DataLoader used by TrainModel 
    it reads all npz files from base_dir/stem/key/task folder
    create train DataLoader from all train.part<dddd>.npz files
    create val DataLoader from all validate.part<dddd>.npz files
    pass both data loaders to TrainModel using ctx through pipeline
'''
@dataclass
class BuildLoadersFromExport(Component):
    """
    Build lazy train/val DataLoaders from exported NPZ parts and ensure:
      - (B, F) -> (B, F, 1)   (seq_len=F, input_size=1)
      - (B, L, F) untouched   (seq_len=L, input_size=F)
    """
    def run(self, ctx: Context) -> Context:
        base_dir = ctx.get("base_dir");  assert base_dir is not None, "ctx['base_dir'] required"
        stem     = ctx.get("stem");      assert stem     is not None, "ctx['stem'] required"
        key      = ctx.get("key");       assert key      is not None, "ctx['key'] required"
        task     = ctx.get("task");      assert task in ("cls", "reg"), "ctx['task'] must be 'cls' or 'reg'"

        batch_size  = int(ctx.get("batch_size", 64))
        num_workers = int(ctx.get("num_workers", 0))
        pin_memory  = bool(ctx.get("pin_memory", False))
        drop_last   = bool(ctx.get("drop_last", False))

        # Build base datasets/loaders
        train_loader, val_loader = build_loaders_from_export(
            base_dir=base_dir,
            stem=stem,
            key=key,
            task=task,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        # Always wrap with our forcing collate
        collate = _collate_force_last_dim_is_one#()
        train_loader = DataLoader(
            train_loader.dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_loader.dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            collate_fn=collate,
        )

        # Probe to record shapes
        xb, _ = next(iter(val_loader if len(val_loader) > 0 else train_loader))
        # Expect (B, F, 1) or (B, L, F)
        if xb.dim() == 3 and xb.shape[-1] == 1:
            _, F_final, _ = xb.shape
            ctx["seq_len"] = int(F_final)
            ctx["input_size"] = 1
            print(f"[BuildLoadersFromExport] enforced (B, F, 1): seq_len={ctx['seq_len']}, input_size=1")
        elif xb.dim() == 3:
            _, L_final, F_final = xb.shape
            ctx["seq_len"] = int(L_final)
            ctx["input_size"] = int(F_final)
            print(f"[BuildLoadersFromExport] detected (B, L, F): seq_len={ctx['seq_len']}, input_size={ctx['input_size']}")
        else:
            raise RuntimeError(f"Unexpected batch shape after collate: {tuple(xb.shape)}")

        ctx["train_loader"] = train_loader
        ctx["val_loader"]   = val_loader
        try:
            ctx["train_size"] = len(train_loader.dataset)  # type: ignore[attr-defined]
            ctx["val_size"]   = len(val_loader.dataset)    # type: ignore[attr-defined]
        except Exception:
            pass

        return ctx
