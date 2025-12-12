from __future__ import annotations
import os
import time
import csv  # <-- NEW
from dataclasses import dataclass, field
from typing import Optional, Literal, Type
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
import numpy as np

# Mock dependencies (replace with your actual imports if necessary)
class Context(dict): pass
class Component:
    def run(self, ctx: Context): return ctx
def get_ckpt_dir(ctx): return ctx.get('ckpt_dir', './checkpoints/default')
def model_report(model, **kwargs): return {"meta": kwargs}
def save_report(report, path): pass

''' Main model train program '''
@dataclass
class TrainModel(Component):
    """
    A unified training component for both Classification (Acc) and 
    Regression (RMSE/MSE) tasks.
    
    Forking Points (controlled by ctx['training_model']):
    1. Data Loader Selection (now unified to 'train_loader'/'val_loader')
    2. Loss Function (CrossEntropyLoss vs. MSELoss)
    3. Metric Calculation (Accuracy vs. RMSE)
    4. Checkpoint/Early Stopping Criteria (Max Acc vs. Min Loss/RMSE)
    """
    # Default training configuration
    default_epochs: int = 10
    default_lr: float = 1e-3
    default_grad_clip: float = 1.0
    default_patience: int = 7
    default_training_model: Literal["classification", "regression"] = "classification" # UPDATED KEY
    default_save_mode: Literal["weights","full"] = "full"
    
    # --- Helper to calculate task-specific metrics for a batch ---
    def _get_batch_metrics(self, training_model: str, logits: torch.Tensor, labels: torch.Tensor, crit: nn.Module) -> dict: # UPDATED ARG
        """Calculates loss and primary metric (Acc or RMSE) for a batch."""
        
        # 1. Calculate Loss (Problem-type specific)
        loss_val = crit(logits, labels)
        
        # Store the tensor for backpropagation, and the float for logging/metrics
        metrics = {"loss_tensor": loss_val, "loss": loss_val.item()} 
        
        # 2. Calculate Primary Metric (Problem-type specific)
        if training_model == "classification":
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            metrics["acc"] = correct / total
            metrics["correct"] = correct
            metrics["total"] = total
            
        elif training_model == "regression":
            # RMSE is often used as the primary readable metric
            rmse = torch.sqrt(loss_val).item()
            metrics["rmse"] = rmse
            
        return metrics

    # --- Evaluation function (runs over entire loader) ---
    def _eval_metrics(self, model: nn.Module, dl: DataLoader, device: torch.device, training_model: str, crit: nn.Module) -> dict: # UPDATED ARG
        model.eval()
        total_loss, total_metric, total_count = 0.0, 0.0, 0
        
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                
                # Check shapes for regression (labels should be float)
                if training_model == "regression" and yb.ndim == 1:
                    yb = yb.unsqueeze(1) # Ensure target is [B, 1] if needed
                        
                batch_metrics = self._get_batch_metrics(training_model, logits, yb, crit)
                
                # Use the float value of the loss (batch_metrics["loss"]) for accumulation in total_loss
                total_loss += batch_metrics["loss"] * yb.size(0)
                total_count += yb.size(0)
                
                if training_model == "classification":
                    total_metric += batch_metrics["correct"]
                elif training_model == "regression":
                    # For regression, we track the loss sum, and calculate final RMSE outside
                    pass

        avg_loss = total_loss / max(1, total_count)
        
        if training_model == "classification":
            avg_metric = total_metric / max(1, total_count) # Final Accuracy
            return {"loss": avg_loss, "acc": avg_metric}
        
        elif training_model == "regression":
            final_rmse = np.sqrt(avg_loss) # RMSE is sqrt of avg MSE
            return {"loss": avg_loss, "rmse": final_rmse}


    # --- Checkpoint Helpers (Updated to include Scheduler) ---
    def _safe_load(self, path: str, *, map_location="cpu", weights_only: bool = False):
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _save_ckpt(self, mode: str, path: str, model: nn.Module, opt: torch.optim.Optimizer,
                   scheduler: Optional[object], epoch: int, best_metric: float, best_loss: float):
        if not path: return
        
        if mode == "full":
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric, # This could be ACC or LOSS depending on problem_type
                "best_loss": best_loss,
                "scheduler": scheduler.state_dict() if scheduler else None, 
                "env": {"cudnn_benchmark": torch.backends.cudnn.benchmark},
            }
            torch.save(ckpt, path)
        elif mode == "weights":
            # Save only model weights, but still wrapped in dict with metric info
            ckpt = {
                "model": model.state_dict(),
                "best_metric": best_metric,
            }
            torch.save(ckpt, path)
        print(f"[TrainModel] Checkpoint saved to: {path} (mode={mode})")

    def _load_ckpt(self, mode: str, path: str, model: nn.Module, opt: torch.optim.Optimizer,
                   scheduler: Optional[object]) -> tuple[int, float, float]:
        
        start_epoch, best_metric, best_loss = 0, float("-inf"), float("inf")
        if not (path and os.path.isfile(path) and os.path.getsize(path) > 0):
            return start_epoch, best_metric, best_loss
        
        try:
            obj = self._safe_load(path, map_location="cpu", weights_only=(mode == "weights"))
            
            if mode == "full" and isinstance(obj, dict) and "model" in obj and "opt" in obj:
                # Load FULL training state (Model, Opt, Scheduler, Meta)
                model.load_state_dict(obj["model"], strict=True)
                opt.load_state_dict(obj["opt"])
                
                if scheduler and obj.get("scheduler"):
                    try:
                        scheduler.load_state_dict(obj["scheduler"])
                        print(f"[TrainModel] Loaded Scheduler state.")
                    except Exception as e:
                        print(f"[TrainModel] Warn: could not load scheduler state: {e}")
                        
                start_epoch  = obj.get("epoch", 0) + 1
                best_metric  = obj.get("best_metric", best_metric)
                best_loss    = obj.get("best_loss", best_loss)

                print(f"[TrainModel] Full checkpoint loaded. Resuming from epoch {start_epoch-1}.")
                
            elif isinstance(obj, dict) and "model" in obj:
                # Load just weights, potentially with some metadata (weights mode)
                model.load_state_dict(obj["model"], strict=True)
                print(f"[TrainModel] Model weights loaded from checkpoint (Weights mode).")
                
            else:
                # Load bare state_dict (old weights mode)
                model.load_state_dict(obj, strict=True)
                print(f"[TrainModel] Bare state_dict loaded.")

        except Exception as e:
            print(f"[TrainModel] Error loading checkpoint '{path}': {e}. Starting from scratch.")
            
        return start_epoch, best_metric, best_loss

    # ------------------------------------
    # --------- Main Run Method ----------
    # ------------------------------------
    def run(self, ctx: Context) -> Context:
        run_time = time.perf_counter()
        # --- Configurable Parameters ---
        epochs    = int(ctx.get("epochs", self.default_epochs))
        lr        = float(ctx.get("lr", self.default_lr))
        grad_clip = float(ctx.get("grad_clip", self.default_grad_clip)) 
        patience  = int(ctx.get("patience", self.default_patience))
        
        # 🌟 FORKING POINT 1: Training Model Type and Data Loaders
        training_model = str(ctx.get("training_model", self.default_training_model)).lower()
        
        # NOTE: Simplified loader lookup per user's request to use unified keys 'train_loader'/'val_loader'
        train_loader: DataLoader | None = ctx.get("train_loader")
        val_loader: DataLoader | None = ctx.get("val_loader")

        if training_model not in ("classification", "regression"):
            raise ValueError(f"Unknown training_model: {training_model}. Must be 'classification' or 'regression'.")
            
        if train_loader is None or val_loader is None:
            raise KeyError(f"TrainModel requires 'train_loader' and 'val_loader' in the context, regardless of model type.")

        model: nn.Module = ctx.get("model")
        if model is None: raise KeyError("TrainModel requires ctx['model']")

        # --- Checkpoint & history paths ---
        ckpt_file = ctx.get('ckpt_file', 'classifier.ckpt')
        ckpt_dir  = get_ckpt_dir(ctx)
        os.makedirs(ckpt_dir, exist_ok=True)
        out_path  = os.path.join(ckpt_dir, ckpt_file)
        history_path = os.path.join(ckpt_dir, "train_history.csv")  # NEW: training history CSV path

        load_mode = str(ctx.get("load_mode", self.default_save_mode)).lower()
        save_mode = str(ctx.get("save_mode", self.default_save_mode)).lower()

        # --- Scheduler Config ---
        scheduler_type = str(ctx.get("scheduler_type", "plateau")).lower()
        scheduler_patience = int(ctx.get("scheduler_patience", 5))
        scheduler_factor = float(ctx.get("scheduler_factor", 0.5))

        # --- Training Setup ---
        device = torch.device(ctx.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        opt  = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 🌟 FORKING POINT 2: Loss Function Initialization
        if training_model == "classification":
            crit = nn.CrossEntropyLoss()
            best_metric_key = "acc"
            best_metric_init = float("-inf")
            plateau_mode = "max"
            print(f"[TrainModel] Starting {training_model} training. Metric: Accuracy.")
        else: # regression
            crit = nn.MSELoss()
            # For regression, we track min Loss (MSE) for early stopping/saving
            best_metric_key = "loss"
            best_metric_init = float("inf")
            plateau_mode = "min"
            print(f"[TrainModel] Starting {training_model} training. Metric: MSE/Loss.")
            
        best_val_loss_init = float("inf") # Always track min loss for scheduling/reporting

        # --- Scheduler Initialization (Re-added) ---
        scheduler = None
        if scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                opt, mode=plateau_mode, 
                factor=scheduler_factor, patience=scheduler_patience
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(opt, T_max=epochs)
        elif scheduler_type == "none":
            scheduler = None 
        
        # --- Load Checkpoint (including optimizer and scheduler state) ---
        start_epoch, best_metric, best_val_loss = self._load_ckpt(load_mode, out_path, model, opt, scheduler)
        
        # Adjust best_metric/best_loss if resuming with an unfavorable score (e.g. if checkpoint file was empty)
        if training_model == "classification":
             best_metric = max(best_metric, best_metric_init)
             best_val_loss = min(best_val_loss, best_val_loss_init)
        else: # regression
             best_metric = min(best_metric, best_metric_init)
             best_val_loss = min(best_val_loss, best_val_loss_init)

        # --- Initial Evaluation after Checkpoint Load ---
        print("\n[TrainModel] Performing initial validation after checkpoint load...")
        val_metrics = self._eval_metrics(model, val_loader, device, training_model, crit)
        current_val_loss = val_metrics["loss"]
        
        if training_model == "classification":
            current_val_metric = val_metrics["acc"]
            if current_val_metric > best_metric:
                best_metric = current_val_metric
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss            
            print(f"Initial Val Loss: {current_val_loss:.4f} | Initial Val Acc: {current_val_metric:.4f}")
            print(f"--- Resuming with Best {best_metric_key} (ACC): {best_metric:.4f} ---")
            
        else: # regression (using loss/RMSE)
            current_val_metric = val_metrics["rmse"]
            if current_val_loss < best_val_loss - 1e-6:
                best_val_loss = current_val_loss
                best_metric = current_val_metric
            elif best_val_loss == best_val_loss_init:
                best_val_loss = current_val_loss
                best_metric = current_val_metric
            
            print(f"Initial Val Loss: {current_val_loss:.4f} | Initial Val RMSE: {current_val_metric:.4f}")
            print(f"--- Resuming with Best {best_metric_key} (LOSS): {best_val_loss:.4f} ---")


        epochs_no_improve = 0

        # Prepare CSV header for history file (only if file does not exist)
        history_fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_metric",
            "val_loss",
            "val_metric",
            "best_loss",
            "best_metric",
            "scheduler_type",
            "train_record_number",
        ]
        history_exists = os.path.exists(history_path)
        if not history_exists:
            with open(history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=history_fieldnames)
                writer.writeheader()
        
        # --- Training Loop ---
        for epoch in range(start_epoch, epochs):
            model.train()
            current_lr = opt.param_groups[0]['lr']
            print(f"\n--- Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} ---")
            
            epoch_loss_sum, epoch_metric_sum, epoch_total = 0.0, 0.0, 0
            start_time = time.time()
            
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                # Check shapes for regression (labels should be float)
                if training_model == "regression" and yb.ndim == 1:
                    yb = yb.unsqueeze(1) 

                opt.zero_grad()
                logits = model(xb)
                
                batch_metrics = self._get_batch_metrics(training_model, logits, yb, crit)
                
                # Retrieve the original loss tensor for backpropagation
                loss_tensor = batch_metrics["loss_tensor"] 
                # Retrieve the float loss for accumulation and logging
                loss_float = batch_metrics["loss"]
                
                # Backprop
                loss_tensor.backward()
                
                # Gradient Clipping
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

                # Accumulate
                epoch_loss_sum += loss_float * yb.size(0)
                epoch_total += yb.size(0)
                
                if training_model == "classification":
                    epoch_metric_sum += batch_metrics["correct"]
                elif training_model == "regression":
                    pass
            
            # --- Epoch Summary ---
            train_loss = epoch_loss_sum / max(1, epoch_total)
            if training_model == "classification":
                train_metric = epoch_metric_sum / max(1, epoch_total)
                train_metric_str = f"Acc: {train_metric:.4f}"
            else:
                train_metric = np.sqrt(train_loss)
                train_metric_str = f"RMSE: {train_metric:.4f}"

            print(f"Train Loss: {train_loss:.4f} | {train_metric_str} | Time: {time.time()-start_time:.1f}s")

            # --- Validation Pass ---
            val_metrics = self._eval_metrics(model, val_loader, device, training_model, crit)
            val_loss = val_metrics["loss"]
            
            if training_model == "classification":
                val_metric = val_metrics["acc"]
                val_metric_str = f"Val Acc: {val_metric:.4f}"
            else:
                val_metric = val_metrics["rmse"]
                val_metric_str = f"Val RMSE: {val_metric:.4f}"

            print(f"Val Loss: {val_loss:.4f} | {val_metric_str} | Best {best_metric_key}: {best_metric:.4f}")
            
            # --- Step the Scheduler ---
            if scheduler:
                if scheduler_type == "plateau":
                    scheduler.step(val_loss) 
                else:
                    scheduler.step()       

            # --- Checkpoint and Early Stopping Logic ---
            improved = False
            
            if training_model == "classification":
                if val_metric > best_metric:
                    best_metric = val_metric
                    improved = True
                if val_loss < best_val_loss: 
                    best_val_loss = val_loss
                    
            elif training_model == "regression":
                if val_loss < best_val_loss: 
                    best_val_loss = val_loss
                    improved = True
                # TODO: may need metric update

            if improved:
                # best_metric = val_metric
                # best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_ckpt(save_mode, out_path, model, opt, scheduler, epoch, best_metric, best_val_loss)
            else:
                epochs_no_improve += 1

            # --- NEW: Append training history row for this epoch ---
            train_record_number = epoch_total  # how many training records were used this epoch
            with open(history_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=history_fieldnames)
                writer.writerow({
                    "epoch": epoch + 1,  # 1-based for readability
                    "learning_rate": current_lr,
                    "train_loss": float(train_loss),
                    "train_metric": float(train_metric),
                    "val_loss": float(val_loss),
                    "val_metric": float(val_metric),
                    "best_loss": float(best_val_loss),
                    "best_metric": float(best_metric),
                    "scheduler_type": scheduler_type,
                    "train_record_number": int(train_record_number),
                })

            # --- Early stop after logging ---
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No val improvement for {patience} epochs "
                      f"(Best {best_metric_key}={best_metric:.4f})")
                break

        # --- Final Context Update ---
        ctx["model"] = model
        ctx["training_model"] = training_model # UPDATED CONTEXT KEY
        ctx["final_train_loss"] = train_loss
        ctx["final_val_loss"] = val_loss
        ctx[f"final_val_{best_metric_key}"] = val_metric
        
        # Build final model report
        pred_rep = model_report(
            model, name="train_model",
            extra_meta={"phase": "train", "training_model": training_model}, # UPDATED REPORT KEY
        )
        report_path = os.path.join(ckpt_dir, "train_model_report.json")
        save_report(pred_rep, ctx.get("train_model_report", report_path))
        
        ctx['train_time'] = time.perf_counter() - run_time
        ctx['ckpt_path'] = out_path
        ctx['train_history_csv'] = history_path  # (optional) expose path in ctx
        return ctx
