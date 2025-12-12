# train_framework/components/predict_reg.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import os, torch
from torch import nn
from torch.utils.data import DataLoader
from util.model_report import model_report, save_report
from train_framework.helper import get_ckpt_dir

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

@dataclass
class PredictRegressor(Component):
    default_in: str = "model.pt"
    default_load_mode: Literal["weights","full"] = "weights"

    def _safe_load(self, path: str, *, map_location='cpu', weights_only: bool=False):
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def run(self, ctx: Context) -> Context:
        dl: DataLoader = ctx['predict_loader'] if 'predict_loader' in ctx else ctx['train_loader']
        model: nn.Module = ctx['model']

        in_path = get_ckpt_dir(ctx) + '/' + ctx.get('ckpt_file', self.default_in)
        load_mode = str(ctx.get('load_mode', self.default_load_mode)).lower()
        as_numpy = bool(ctx.get('as_numpy', False))

        device_str = str(ctx.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        device = torch.device(device_str)
        model.to(device)

        obj = self._safe_load(in_path, map_location=device, weights_only=(load_mode=='weights'))
        state = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
        model.load_state_dict(state, strict=False)

        preds = []
        targets = []
        model.eval()
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device).float().view(-1, 1)
                yhat = model(xb).float().view(-1, 1)
                preds.append(yhat.detach().cpu())
                targets.append(yb.detach().cpu())

        import torch as _T
        yhat_all = _T.cat(preds, dim=0)
        y_all    = _T.cat(targets, dim=0)

        # report
        pred_rep = model_report(model, name='reg_predict_model', extra_meta={
            'phase': 'predict',
            'task': 'reg',
            'model_type': ctx.get('model_type'),
            'input_size': ctx.get('input_size'),
        }, ckpt_path=in_path, ckpt_state=state)
        save_report(pred_rep, get_ckpt_dir(ctx) + '/reg_predict_model_report.json')

        if as_numpy:
            ctx['pred_values'] = yhat_all.numpy()
            ctx['true_values'] = y_all.numpy()
        else:
            ctx['pred_values'] = yhat_all
            ctx['true_values'] = y_all

        ctx['pred_device'] = str(device)
        ctx['pred_count'] = int(yhat_all.shape[0])
        return ctx
