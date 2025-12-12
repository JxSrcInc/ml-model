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
from train_framework.components.predict_loader import BuildPredictLoader
# 2) model builder (from your upload)
from train_framework.components.build_model import BuildModel
# 3) predictors/verifier (drop-ins provided below)
from train_framework.components.predict_classifier import PredictClassifier
from train_framework.components.verify_classifier import VerifyClassifier
    
def build_pipeline() -> Pipeline:
    return Pipeline([
        BuildPredictLoader(),
        # Build model using your components/build_model.py factory
        BuildModel(),      # uses ctx['model_type'], 'num_classes', 'input_size', etc.

        # Predict with ckpt (PredictClassifier loads weights if ctx['in'] present)
        PredictClassifier(),

        # Verify predictions vs ground-truth labels from dataset
        VerifyClassifier(),
    ])

def main():

    # Build pipeline and seed ctx
    pipe = build_pipeline()

    ctx = {
        "base_dir": "/home/john/git-2024/ml-stock/data/dayTrade",
        "stem": "sprf_50_20",
        "key": "diff",
        "task": "cls",
        "batch_size": 100,

        # Optional symbol filtering for prediction
        # "predict_split": "validate",            # or "train"
        "predict_split": "train",            # or "train"
        # "select_symbols": ["AAPL", "MSFT"],     # optional

        # model + train params...
        "model_type": "lstm",
        "num_classes": 3,
        "hidden": 128,        # LSTM hidden size
        "num_layers": 2,      # LSTM layers
        "ckpt_file": "lstm_diff.ckpt"

    }

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
