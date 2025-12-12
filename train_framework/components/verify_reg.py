# train_framework/components/verify_reg.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

@dataclass
class VerifyRegressor(Component):
    """Compute MAE, MSE, RMSE, and R^2 from predictions.
    Requires ctx['pred_values'] and 'true_values' (torch tensor or numpy array).
    Writes ctx['verify_metrics'] and 'verify_report'.
    """
    def run(self, ctx: Context) -> Context:
        yhat = ctx.get('pred_values')
        y    = ctx.get('true_values')
        if yhat is None or y is None:
            raise KeyError("VerifyRegressor requires ctx['pred_values'] and ctx['true_values'].")

        try:
            import torch
            if isinstance(yhat, torch.Tensor): yhat = yhat.detach().cpu().numpy()
            if isinstance(y, torch.Tensor):    y = y.detach().cpu().numpy()
        except Exception:
            pass

        yhat = np.asarray(yhat).reshape(-1)
        y    = np.asarray(y).reshape(-1)
        if yhat.shape[0] != y.shape[0]:
            raise ValueError(f'Length mismatch: pred={yhat.shape[0]} true={y.shape[0]}')

        n = y.shape[0]
        mse = float(np.mean((yhat - y)**2)) if n>0 else 0.0
        mae = float(np.mean(np.abs(yhat - y))) if n>0 else 0.0
        rmse = float(np.sqrt(mse))
        # R^2 (coefficient of determination)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) if n>0 else 0.0
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else 0.0

        ctx['verify_metrics'] = {
            'n_samples': int(n), 'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2
        }
        ctx['verify_report'] = (
            f"VerifyRegressor: n={n}, mse={mse:.6f}, mae={mae:.6f}, rmse={rmse:.6f}, r2={r2:.4f}"
        )
        return ctx
