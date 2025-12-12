# train_framework/components/predict_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Iterable, List

import torch
from torch.utils.data import DataLoader

from train_framework.components.loader.load_npz_dataset import (
    build_loader_for_split,
)

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

Task  = Literal["cls", "reg"]
Split = Literal["train", "validate"]


def _collate_force_last_dim_is_one():
    """
    Collate that:
      - if X is (B, F)      -> (B, F, 1)
      - if X is (B, L, F)   -> pass through
      - if X is (B, F, 1)   -> pass through
    """
    def collate(batch):
        xs, ys = zip(*batch)
        X = torch.stack([torch.as_tensor(x) for x in xs], dim=0)
        if X.dim() == 2:
            B, F = X.shape
            X = X.view(B, F, 1)  # force (B, F, 1)
        elif X.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input tensor shape {tuple(X.shape)}; expected 2D or 3D.")
        y0 = ys[0]
        if isinstance(y0, torch.Tensor):
            Y = torch.stack([torch.as_tensor(y) for y in ys], dim=0)
        else:
            Y = torch.as_tensor(ys)
        return X, Y
    return collate


@dataclass
class BuildPredictLoader(Component):
    """
    Build a single predict loader from either 'train' or 'validate' split,
    optionally filtered by symbols (ctx['select_symbols']).

    Required in ctx:
      - base_dir: str | Path
      - stem: str
      - key: str
      - task: "cls" | "reg"

    Optional in ctx:
      - predict_split: "train" | "validate" (default "validate")
      - select_symbols: List[str] | Iterable[str]
      - batch_size: int = 64
      - num_workers: int = 0
      - pin_memory: bool = False
      - drop_last: bool = False

    Outputs in ctx:
      - 'predict_loader': DataLoader
      - 'predict_seq_len': int
      - 'predict_input_size': int
    """
    def run(self, ctx: Context) -> Context:
        base_dir = ctx.get("base_dir");  assert base_dir is not None, "ctx['base_dir'] required"
        stem     = ctx.get("stem");      assert stem     is not None, "ctx['stem'] required"
        key      = ctx.get("key");       assert key      is not None, "ctx['key'] required"
        task     = ctx.get("task");      assert task in ("cls", "reg"), "ctx['task'] must be 'cls' or 'reg'"

        split: Split = ctx.get("predict_split", "validate")
        if split not in ("train", "validate"):
            raise ValueError("ctx['predict_split'] must be 'train' or 'validate'")

        batch_size  = int(ctx.get("batch_size", 64))
        num_workers = int(ctx.get("num_workers", 0))
        pin_memory  = bool(ctx.get("pin_memory", False))
        drop_last   = bool(ctx.get("drop_last", False))
        select_symbols = ctx.get("select_symbols", None)

        # Build the base loader for the chosen split (with symbol filtering)
        loader = build_loader_for_split(
            base_dir=base_dir,
            stem=stem,
            key=key,
            task=task,
            split=split,
            select_symbols=select_symbols,
            batch_size=batch_size,
            shuffle=False,                     # prediction → no shuffle by default
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        # Wrap with collate to force (B, F) -> (B, F, 1) deterministically
        collate = _collate_force_last_dim_is_one()
        loader = DataLoader(
            loader.dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            collate_fn=collate,
        )

        # Probe to record shapes
        xb, _ = next(iter(loader))
        if xb.dim() == 3 and xb.shape[-1] == 1:
            _, F_final, _ = xb.shape
            ctx["predict_seq_len"] = int(F_final)
            ctx["predict_input_size"] = 1
        elif xb.dim() == 3:
            _, L_final, F_final = xb.shape
            ctx["predict_seq_len"] = int(L_final)
            ctx["predict_input_size"] = int(F_final)
        else:
            raise RuntimeError(f"Unexpected predict batch shape after collate: {tuple(xb.shape)}")

        ctx["predict_loader"] = loader
        return ctx
