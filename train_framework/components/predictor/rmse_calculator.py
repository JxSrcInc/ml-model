from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): ...
    class Component:
        def run(self, ctx: Context): return ctx


@dataclass
class RegressionEvaluator(Component):
    """
    Regression evaluator.

    Expects in ctx:
        - y:          true values, shape (N,)
        - pred_value: predicted values, shape (N,)

    Writes to ctx:
        - metric_mse:  float
        - metric_rmse: float
    """

    y_key: str = "y"
    pred_value_key: str = "pred_value"
    mse_key: str = "metric_mse"
    rmse_key: str = "metric_rmse"

    def run(self, ctx: Context) -> Context:
        if self.y_key not in ctx:
            raise KeyError(f"RegressionEvaluator requires ctx['{self.y_key}']")
        if self.pred_value_key not in ctx:
            raise KeyError(f"RegressionEvaluator requires ctx['{self.pred_value_key}']")

        y_true = np.asarray(ctx[self.y_key], dtype=float).reshape(-1)
        y_pred = np.asarray(ctx[self.pred_value_key], dtype=float).reshape(-1)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"RegressionEvaluator: length mismatch, y={y_true.shape[0]}, "
                f"pred={y_pred.shape[0]}"
            )

        mse = float(((y_true - y_pred) ** 2).mean())
        rmse = float(np.sqrt(mse))

        ctx[self.mse_key] = mse
        ctx[self.rmse_key] = rmse

        print(f"[RegressionEvaluator] RMSE = {rmse:.4f}, MSE = {mse:.6f}")
        return ctx
