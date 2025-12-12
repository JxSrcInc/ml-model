from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
import numpy as np
import logging

log = logging.getLogger(__name__)

# Mock dependencies (replace with your actual imports)
class Context(dict): pass
class Component:
    def run(self, ctx: Context): return ctx
def get_ckpt_dir(ctx): return ctx.get('ckpt_dir', './checkpoints/default')


@dataclass
class PredictModel(Component):
    """
    Unified prediction component for both Classification and Regression.

    Expects in ctx:
      - "model": nn.Module (already constructed, same as training)
      - "training_model": "classification" or "regression"
      - "X": features (np.ndarray or torch.Tensor) with shape [N, ...]
      - "id" or "ids": list/array of length N (sample identifiers)

    Optional in ctx:
      - "ckpt_file": checkpoint file name (default "classifier.ckpt")
      - "ckpt_dir": directory containing checkpoint (get_ckpt_dir)
      - "device": "cuda" / "cpu" (default: auto "cuda" if available)
      - "batch_size": prediction batch size (default: 1024)

    Writes into ctx:
      - "pred_ids": np.ndarray of ids (aligned with X)
      - For classification:
          "pred_label": np.ndarray of predicted class indices (int64)
          "pred_proba": np.ndarray of probabilities [N, C]
      - For regression:
          "pred_value": np.ndarray of predicted values [N]
    """

    default_training_model: Literal["classification", "regression"] = "classification"
    default_batch_size: int = 1024

    # --- Checkpoint Helpers (simplified for inference only) ---
    def _safe_load(self, path: str, *, map_location="cpu"):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _load_ckpt(self, path: str, model: nn.Module):
        """Load model weights from checkpoint if available."""
        if not (path and os.path.isfile(path) and os.path.getsize(path) > 0):
            print(f"[PredictModel] No checkpoint found at '{path}'. Using current model weights.")
            return

        try:
            obj = self._safe_load(path, map_location="cpu")

            if isinstance(obj, dict) and "model" in obj:
                # New-style checkpoint with {"model": state_dict, ...}
                model.load_state_dict(obj["model"], strict=True)
                log.debug(f"Loaded 'model' state_dict from checkpoint.")
            else:
                # Old-style: bare state_dict
                model.load_state_dict(obj, strict=True)
                print(f"[PredictModel] Loaded bare state_dict from checkpoint.")
        except Exception as e:
            print(f"[PredictModel] Error loading checkpoint '{path}': {e}. Using current model weights.")

    # ------------------------------------
    # --------- Main Run Method ----------
    # ------------------------------------
    def run(self, ctx: Context) -> Context:
        # --- Basic configuration ---
        training_model = str(ctx.get("training_model", self.default_training_model)).lower()
        if training_model not in ("classification", "regression"):
            raise ValueError(
                f"Unknown training_model: {training_model}. "
                "Must be 'classification' or 'regression'."
            )

        model: nn.Module = ctx.get("model")
        if model is None:
            raise KeyError("PredictModel requires ctx['model']")

        # --- Get input data X and ids ---
        if "X" not in ctx:
            raise KeyError("PredictModel requires ctx['X'] (features array/tensor)")

        X_in = ctx["X"]
        ids_in = ctx.get("ids", ctx.get("id", None))
        if ids_in is None:
            raise KeyError("PredictModel requires ctx['id'] or ctx['ids']")

        # Convert ids to a 1D numpy array
        ids_arr = np.asarray(list(ids_in))
        # Convert X to a tensor
        if isinstance(X_in, np.ndarray):
            # Assume float features
            X_all = torch.from_numpy(X_in).float()
        elif isinstance(X_in, torch.Tensor):
            X_all = X_in
        else:
            raise TypeError("ctx['X'] must be a numpy.ndarray or torch.Tensor")

        N = X_all.shape[0]
        if ids_arr.shape[0] != N:
            raise ValueError(
                f"Length mismatch: X has {N} rows but id/ids has {ids_arr.shape[0]} entries"
            )

        # --- Device and checkpoint ---
        device = torch.device(ctx.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)

        ckpt_file = ctx.get("ckpt_file", "classifier.ckpt")
        ckpt_dir = get_ckpt_dir(ctx)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)

        # Load checkpoint weights if present
        self._load_ckpt(ckpt_path, model)

        # --- Inference loop (batched) ---
        batch_size = int(ctx.get("batch_size", self.default_batch_size))

        model.eval()
        all_labels = []   # for classification
        all_probs = []    # for classification
        all_values = []   # for regression

        with torch.no_grad():
            X_all = X_all.to(device)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb = X_all[start:end]

                logits = model(xb)

                if training_model == "classification":
                    # logits: [B, C]
                    probs = torch.softmax(logits, dim=1)
                    labels = logits.argmax(dim=1)

                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

                else:  # regression
                    # logits: [B] or [B, 1] or [B, D]
                    y = logits
                    if y.ndim == 2 and y.shape[1] == 1:
                        y = y.squeeze(1)
                    all_values.append(y.cpu().numpy())

        # --- Collate outputs and write back to ctx ---
        ctx["pred_ids"] = ids_arr  # [N]

        if training_model == "classification":
            pred_label = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
            pred_proba = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32)

            ctx["pred_label"] = pred_label  # [N]
            ctx["pred_proba"] = pred_proba  # [N, C]
            log.debug(f"Classification prediction done for {pred_label.shape[0]} samples.")

        else:  # regression
            pred_value = np.concatenate(all_values, axis=0) if all_values else np.empty((0,), dtype=np.float32)
            ctx["pred_value"] = pred_value  # [N]
            log.debug(f"Regression prediction done for {pred_value.shape[0]} samples.")

        return ctx
