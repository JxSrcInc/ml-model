# train_framework/pipelines/predict_verify_npz.py
from __future__ import annotations
import argparse

try:
    from framework.pipeline import Pipeline
    from framework.component import Context, Component
except Exception:
    # tiny fallbacks so this can run without the framework package
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx
    class Pipeline:
        def __init__(self, steps): self.steps = steps
        def run(self, ctx: Context):
            for s in self.steps: s.run(ctx)
            return ctx

# 1) dataset loader (from your upload)
from train_framework.components.loader.load_npz_dataset import LoadNPZDataset
# 2) model builder (from your upload)
from train_framework.components.build_model import BuildModel
# 3) predictors/verifier (drop-ins provided below)
from train_framework.components.predict_classifier import PredictClassifier
from train_framework.components.verify_classifier import VerifyClassifier
# from train_framework.components.loader.read_ckpt_meta import ReadCkptMeta

class InferModelDimsFromDataset(Component):
    """
    Reads ctx['dataset'] to set:
      - ctx['num_classes'] (for classifier head)
      - ctx['input_size']  (feature dim; 1 if (L,1), else last dimension)
    Also sets ctx['model_type'] and ctx['in'] (ckpt path) from args already placed in ctx.
    """
    def run(self, ctx: Context) -> Context:
        ds = ctx.get("dataset")
        if ds is None:
            raise KeyError("InferModelDimsFromDataset needs ctx['dataset'] (set by LoadNPZDataset).")
        # Derive input_size from one sample
        x0, y0 = ds[0]
        # x0 shape is (L,1) if add_channel=True, else (L,)
        ctx["input_size"] = int(x0.shape[-1]) if x0.ndim >= 2 else 1

#         # Derive num_classes from labels if task is classification
#         # LoadNPZDataset put full labels into ctx['dataset'].y; safe fallback:
#         import numpy as np
#         y_all = ds.y if hasattr(ds, "y") else None
#         if y_all is None:
#             raise KeyError("Dataset missing 'y' to infer num_classes.")
#         ctx["num_classes"] = int(np.unique(y_all).size)
        return ctx
    
# class InferModelDimsFromDataset(Component):
#     """
#     Sets input_size from a sample.
#     Sets num_classes from dataset ONLY if ckpt_num_classes is not provided.
#     """
#     def run(self, ctx: Context) -> Context:
#         ds = ctx.get("dataset")
#         if ds is None:
#             raise KeyError("InferModelDimsFromDataset needs ctx['dataset'].")

#         x0, _ = ds[0]
#         ctx["input_size"] = int(x0.shape[-1]) if getattr(x0, "ndim", 1) >= 2 else 1

#         import numpy as np
#         y_all = getattr(ds, "y", None)
#         if y_all is None:
#             raise KeyError("Dataset missing 'y' to infer num_classes.")

#         if "ckpt_num_classes" in ctx:
#             # Respect the checkpoint’s class count
#             ctx["num_classes"] = int(ctx["ckpt_num_classes"])
#         else:
#             ctx["num_classes"] = int(np.unique(y_all).size)

#         # Optional: warn if mismatch between dataset labels and ckpt classes
#         if "ckpt_num_classes" in ctx:
#             ds_classes = int(np.unique(y_all).size)
#             if ds_classes != ctx["num_classes"]:
#                 print(f"[InferModelDimsFromDataset] Warning: dataset has {ds_classes} classes "
#                       f"but checkpoint expects {ctx['num_classes']}. "
#                       f"Please ensure labels/NPZ match the training task.")
#         return ctx


def build_pipeline(args) -> Pipeline:
    return Pipeline([
        # Load NPZ -> ctx['dataset'], ctx['train_loader'] (we reuse it as predict loader)
        LoadNPZDataset(),  # uses ctx['train_data'], ctx['task'], ctx['batch_size'], etc.

        # Fill in num_classes & input_size from dataset shape/labels
        InferModelDimsFromDataset(),

        # Build model using your components/build_model.py factory
        BuildModel(),      # uses ctx['model_type'], 'num_classes', 'input_size', etc.

        # Predict with ckpt (PredictClassifier loads weights if ctx['in'] present)
        PredictClassifier(),

        # Verify predictions vs ground-truth labels from dataset
        VerifyClassifier(),
    ])


def main():
    ap = argparse.ArgumentParser("NPZ → BuildModel → Load .ckpt → Predict → Verify")
    # NPZ & task
    ap.add_argument("--npz", dest="train_data", default="/home/john/C:/Users/johnl/data-repos/ml-ts/cls/sprf_100_50/diff/validate.part0000.npz", help="Path to .npz with X and y_cls")
    # ap.add_argument("--npz", dest="train_data", default="/home/john/tmp/s7_100_20_diff_binar.npz", help="Path to .npz with X and y_cls")
    ap.add_argument("--task", default="diff", choices=["diff", "reg"], help="Classification=diff, Regression=reg")
    ap.add_argument("--add-channel", action="store_true", help="Reshape X to (L,1) for sequence models")
    ap.add_argument("--batch-size", type=int, default=512)

    # Model selection (must match your BuildModel factory: 'lstm' | 'rnn' | 'transformer')
    ap.add_argument("--model-type", default="transformer", choices=["lstm", "rnn", "transformer"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=256)

    # Checkpoint loading for prediction
    ap.add_argument("--ckpt", dest="in_path", default="part0000_trans_diff.ckpt", help="Path to saved checkpoint (.ckpt/.pt)")
    ap.add_argument("--load-mode", default="weights", choices=["weights", "full"])
    ap.add_argument("--device", default=None, help="'cuda', 'cuda:0', or 'cpu' (auto if omitted)")
    ap.add_argument("--as-numpy", action="store_true", help="Return predictions as NumPy arrays")

    args = ap.parse_args()

    # Build pipeline and seed ctx
    pipe = build_pipeline(args)
    # ctx = Context(run_id="predict_verify")
    ctx: Context = {"run_id":"predict_verify"}

    # Dataset settings expected by LoadNPZDataset
    ctx["train_data"] = args.train_data    # NPZ path
    ctx["task"] = args.task
    ctx["add_channel"] = bool(args.add_channel)
    ctx["batch_size"] = int(args.batch_size)

    # Model config expected by BuildModel
    ctx["model_type"] = args.model_type
    ctx["hidden"] = args.hidden
    ctx["num_layers"] = args.num_layers
    ctx["dropout"] = args.dropout
    ctx["nhead"] = args.nhead
    ctx["enc_layers"] = args.enc_layers
    ctx["dim_ff"] = args.dim_ff
    ctx["shuffle"] = False
    ctx["num_classes"] = 3

    # Prediction options expected by PredictClassifier
    ctx["in"] = args.in_path
    ctx["load_mode"] = args.load_mode
    ctx["device"] = args.device
    ctx["as_numpy"] = bool(args.as_numpy)
    ctx["add_channel"] = True
    ctx["train_model_report"] = "train_model_report.json"


    ctx = pipe.run(ctx)

    # Summary
    report = ctx.get("verify_report", "")
    metrics = ctx.get("verify_metrics", {})
    print("=== VERIFY SUMMARY ===")
    print(report)
    if metrics:
        print(f"n={metrics['n_samples']}, errors={metrics['n_errors']}, "
              f"accuracy={metrics['accuracy']:.4f}, error_rate={metrics['error_rate']:.4f}")


if __name__ == "__main__":
    main()
