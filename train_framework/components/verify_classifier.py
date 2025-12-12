# train_framework/components/verify_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

def _to_numpy_1d(x):
    try:
        import torch
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(x)
    if arr.ndim != 1: arr = arr.reshape(-1)
    return arr

@dataclass
class VerifyClassifier(Component):
    """
    Compares predicted labels vs ground truth and reports accuracy/error_rate.
    Expects:
      - ctx['pred_labels'] from PredictClassifier
      - ctx['dataset'].y (or ctx['true_labels']) for ground truth
    """
    def run(self, ctx: Context) -> Context:
        y_pred = ctx.get("pred_labels", None)
        if y_pred is None:
            raise KeyError("VerifyClassifier requires ctx['pred_labels'].")

        # true labels from dataset (preferred), fallback to ctx['true_labels']
        ds = ctx.get("dataset")
        y_true = getattr(ds, "y", None) if ds is not None else None
        if y_true is None:
            y_true = ctx.get("true_labels", None)
        if y_true is None:
            raise KeyError("VerifyClassifier needs dataset.y or ctx['true_labels'].")

        y_true = _to_numpy_1d(y_true)
        y_pred = _to_numpy_1d(y_pred)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"Length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}")

        n = y_true.shape[0]
        errors = int(np.sum(y_true != y_pred))
        accuracy = float(1.0 - errors / n) if n > 0 else 0.0
        error_rate = float(errors / n) if n > 0 else 0.0

        ctx["verify_metrics"] = {
            "n_samples": int(n),
            "n_errors": errors,
            "accuracy": accuracy,
            "error_rate": error_rate,
        }
        ctx["verify_report"] = f"VerifyClassifier: n={n}, errors={errors}, accuracy={accuracy:.4f}, error_rate={error_rate:.4f}"
        return ctx
