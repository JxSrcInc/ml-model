from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, List
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from util.model_report import model_report, save_report, diff_reports
from train_framework.helper import get_ckpt_dir
try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx


@dataclass
class PredictClassifier(Component):
    """
    Uses ctx['train_loader'] as prediction loader (from LoadNPZDataset),
    ctx['model'] as nn.Module, and loads checkpoint like TrainClassifier.
    If the checkpoint head out_dim != current model head out_dim, it adapts
    the last Linear to match the checkpoint and loads with strict=False.

    Writes: 'pred_logits', 'pred_probs', 'pred_labels' (+ device, count).
    """
    default_in: str = "model.pt"
    default_load_mode: Literal["weights","full"] = "weights"

    # --------- exact same safe-load approach as TrainClassifier ----------
    def _safe_load(self, path: str, *, map_location="cpu", weights_only: bool = False):
        try:
            # PyTorch 1.12+ supports weights_only. Older versions might not.
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            # Fallback for older PyTorch versions where weights_only is not supported
            return torch.load(path, map_location=map_location)

    # --------- helpers to adapt model head to checkpoint ----------
    @staticmethod
    def _infer_out_dim_from_state(state_dict: dict) -> Optional[int]:
        # Try common classifier names first
        for k in (
            "fc.weight", "classifier.weight", "head.weight",
            "out.weight", "proj.weight", "linear.weight"
        ):
            v = state_dict.get(k, None)
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                return int(v.shape[0])
        # Fallback: last 2D weight tensor in the state dict order
        out_dim = None
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                out_dim = int(v.shape[0])
        return out_dim

    @staticmethod
    def _find_last_linear(model: nn.Module) -> Optional[nn.Linear]:
        last = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def _load_ckpt_like_trainer(self, mode: str, path: str, model: nn.Module) -> None:
        if not (path and os.path.isfile(path) and os.path.getsize(path) > 0):
            print(f"[PredictClassifier] No checkpoint found at: {path}")
            return

        print(f"[PredictClassifier] Loading checkpoint from: {path} (load_mode='{mode}')")

        try:
            # For mode='full', weights_only is False, loading the entire dict {model, opt, epoch...}
            # For mode='weights', weights_only is True, often loading just the state_dict
            obj = self._safe_load(path, map_location="cpu", weights_only=(mode == "weights"))
        except Exception as e:
            print(f"[PredictClassifier] Warn: failed to load checkpoint '{path}': {e}")
            return

        # Extract state dict (the model weights)
        if isinstance(obj, dict) and "model" in obj:
            # This handles models saved in 'full' mode and 'weights' mode (if saved with metadata)
            state = obj["model"]
            print(f"[PredictClassifier] Extracted model state_dict from 'model' key. "
                  f"(Full checkpoint or metadata-wrapped weights detected.)")
        else:
            state = obj  # This handles plain state_dict ('weights' mode)
            print(f"[PredictClassifier] Loaded model state_dict directly. (Plain weights-only checkpoint assumed.)")
        
        # Check head out_dim in the checkpoint
        ckpt_out_dim = self._infer_out_dim_from_state(state)
        last_linear = self._find_last_linear(model)

        if last_linear is not None and ckpt_out_dim is not None:
            cur_out = int(last_linear.out_features)
            if cur_out != ckpt_out_dim:
                # Rebuild last Linear to match checkpoint head
                in_f = int(last_linear.in_features)
                print(f"[PredictClassifier] Adapting classifier head: {cur_out} -> {ckpt_out_dim}")
                
                # Simpler approach: walk parents by attribute set
                def replace_last_linear(module):
                    replaced = False
                    for attr in dir(module):
                        try:
                            sub = getattr(module, attr)
                        except Exception:
                            continue
                        if sub is last_linear:
                            setattr(module, attr, nn.Linear(in_f, ckpt_out_dim))
                            return True
                    for child in module.children():
                        if replace_last_linear(child):
                            return True
                    return False

                if not replace_last_linear(model):
                    # Fallback: try to modify directly if it's top-level reference only
                    print("[PredictClassifier] Warning: could not locate last Linear parent; "
                          "attempting to load with strict=False without head swap.")

                # Load with strict=False (since new head has different shape)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[PredictClassifier] load_state_dict(strict=False) "
                          f"missing={list(missing)} unexpected={list(unexpected)}")
                return

        # If shapes match, load strictly (like trainer)
        model.load_state_dict(state, strict=True)

    # --------- main ----------
    def run(self, ctx: Context) -> Context:
        dl: DataLoader | None = ctx.get("predict_loader")  # reuse as predict loader
        model: nn.Module | None = ctx.get("model")
        if dl is None:
            raise KeyError("PredictClassifier requires ctx['predict_loader']") # Corrected key
        if model is None:
            raise KeyError("PredictClassifier requires ctx['model']")

        in_path = get_ckpt_dir(ctx)+'/'+ctx['ckpt_file']
        # in_path   = str(ctx.get("in", self.default_in))
        load_mode = str(ctx.get("load_mode", self.default_load_mode)).lower()
        as_numpy  = bool(ctx.get("as_numpy", False))
        device_cfg = ctx.get("device", None)

        if load_mode not in {"weights", "full"}:
            raise ValueError("load_mode must be 'weights' or 'full'")

        # Load checkpoint exactly like the trainer and adapt head if needed
        if in_path:
            self._load_ckpt_like_trainer(load_mode, in_path, model)

        device = torch.device(device_cfg) if device_cfg else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        ###################
        # Optional: log final layer out_features vs expected
        expected = ctx.get("num_classes")
        last_weight = None
        for name, p in model.named_parameters():
            if name.endswith(".weight") and p.ndim == 2:
                last_weight = p
        # last encountered 2D weight is often the classifier head
        if expected is not None and last_weight is not None:
            out_dim = last_weight.shape[0]
            if out_dim != expected:
                print(f"[PredictClassifier] WARNING: model head out_dim={out_dim} "
                    f"!= expected num_classes={expected}. Check pipeline order & ckpt.")
        #####################

        softmax = nn.Softmax(dim=1)
        logits_all, probs_all, preds_all = [], [], []
        true_all = []  # keep aligned ground-truth so Verify can't be derailed by shuffle

        with torch.no_grad():
            for xb, yb in dl:
                # Handle possible 1-D squeeze if input is (B, L, 1)
                xb = xb.to(device)
                logits = model(xb)                  # [B, C] -- now guaranteed to match ckpt head
                
                probs  = softmax(logits)
                preds  = logits.argmax(dim=1)
                
                logits_all.append(logits.detach().cpu())
                probs_all.append(probs.detach().cpu())
                preds_all.append(preds.detach().cpu())
                true_all.append(yb.detach().cpu())


        # Build a report that also summarizes the checkpoint state_dict we loaded
        # We need to load the full object again here, but ensure we extract the model's state for the report
        ckpt_obj = self._safe_load(in_path, map_location="cpu", weights_only=False) # Load the full dict for report, regardless of initial load_mode
        # Use the same logic to extract the state dict for reporting
        ckpt_state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj

        pred_rep = model_report(
            model, name="predict_model",
            extra_meta={
                "phase": "predict",
                "model_type": ctx.get("model_type"),
                "input_size": ctx.get("input_size"),
                "task": ctx.get("task"),
            },
            ckpt_path=in_path,
            ckpt_state=ckpt_state
        )
        report_path = get_ckpt_dir(ctx)+'/predict_model_report.json'
        save_report(pred_rep, ctx.get("predict_model_report", report_path))

        # Optional: compare to a training report if you provide it on disk
        train_report_path = ctx.get("train_model_report", None)
        if train_report_path and os.path.isfile(train_report_path):
            from util.model_report import load_report
            a = load_report(train_report_path)
            b = pred_rep
            diffs = diff_reports(a, b)
            if diffs:
                print("=== MODEL DIFF (train vs predict) ===")
                for d in diffs:
                    print(" -", d)
            else:
                print("=== MODEL DIFF: no differences detected ===")


        import torch as _torch
        logits_cat = _torch.cat(logits_all, 0) if logits_all else _torch.empty((0,0))
        probs_cat  = _torch.cat(probs_all,  0) if probs_all  else _torch.empty((0,0))
        preds_cat  = _torch.cat(preds_all,  0) if preds_all  else _torch.empty((0,))
        y_true_cat = _torch.cat(true_all,  0) if true_all  else _torch.empty((0,))

        if as_numpy:
            ctx["pred_logits"] = logits_cat.numpy()
            ctx["pred_probs"]  = probs_cat.numpy()
            ctx["pred_labels"] = preds_cat.numpy()
            ctx["true_labels"] = y_true_cat.numpy()
        else:
            ctx["pred_logits"] = logits_cat
            ctx["pred_probs"]  = probs_cat
            ctx["pred_labels"] = preds_cat
            ctx["true_labels"] = y_true_cat   # keep alignment exact

        ctx["pred_device"] = str(device)
        ctx["pred_count"] = int(preds_cat.shape[0])
        return ctx
