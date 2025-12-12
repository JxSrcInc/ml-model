from train_framework.pipelines.build_pipeline import build_pipeline
# from train_framework.pipelines.binar import build_binar_pipeline
from framework.component import Context  # or your own
import argparse

ap = argparse.ArgumentParser(description="data_framework runner")

ap.add_argument("--output_dir", default='C:/Users/johnl/data-repos/ml-ts', 
                # required=True,
                help="Directory to save plots (optional).")
args = ap.parse_args()
base_dir = args.output_dir

ctx = {
    "base_dir": base_dir,
    # "stem": "all_100_50",
    "stem": "all_100_80",
    "key": "diff",
    # "key": "count",
    "task": "cls",

    "batch_size": 256,   # a bit smaller than LSTM if needed for memory

    # model config (transformer)
    "model_type": "transformer",  # << IMPORTANT
    "num_classes": 3,

    # Transformer hyperparameters (match factory expectations)
    "hidden": 128,        # d_model
    "nhead": 8,           # must divide hidden
    "enc_layers": 4,
    "dim_ff": 512,        # or 4 * hidden
    "dropout": 0.1,

    # optional common:
    "grad_clip": 1.0,
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_type": "cosine",  # transformer often works nicely with cosine decay

    # training
    "epochs": 200,
    "lr": 3e-4,           # a bit smaller LR is often safer for transformers
    "ckpt_file": "trans_diff.ckpt",
    "patience": 30,
    "load_mode": "full",
    "save_mode": "full",

    "training_model": "classification", 
}

# pipe = build_binar_pipeline()
pipe = build_pipeline(ctx=ctx)
out = pipe.run(ctx)
print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")
