from train_framework.pipelines.build_pipeline import build_pipeline
from train_framework.tasks.config.lstm_binar_ctx import Config

base_dir ='C:/Users/johnl/data-repos/ml-ts'
# stem = "sprf_100_50"
stem = 'all_100_50'
# batch_size = 20
batch_size = 256
key='diff'
key='raw'


ctx = Config(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)
ctx = ctx.__dict__

pipe = build_pipeline(ctx=ctx)
out = pipe.run(ctx)
print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")

# from train_framework.pipelines.binar import build_binar_pipeline
# from framework.component import Context  # or your own
# import argparse

# ap = argparse.ArgumentParser(description="data_framework runner")

# ap.add_argument("--output_dir", default='C:/Users/johnl/data-repos/ml-ts', 
#                 # required=True,
#                 help="Directory to save plots (optional).")
# args = ap.parse_args()
# base_dir = args.output_dir

# ctx = {
#     "base_dir": base_dir,
#     "stem":"sprf_100_50",
#     # "stem":"all_100_50",
#     "key": "diff",
#     "task": "cls",
#     "batch_size": 20,

#     # model
#     "model_type": "lstm",
#     "num_classes": 3,
#     "hidden": 128,        # LSTM hidden size
#     "num_layers": 2,      # LSTM layers
#     "dropout": 0.3,       # optional; 0.0 disables

#     # training
#     "epochs": 200,
#     "lr": 1e-3,
#     "ckpt_file": "lstm_diff.ckpt",
#     "patience": 50,
#     "load_mode": "full",   # 'weights' or 'full'
#     "save_mode": "full",   # 'weights' or 'full'
#     # optional common:
#     "grad_clip": 1.0,           # Recommended starting value for gradient clipping
#     "scheduler_patience": 5,    # (Optional) Controls when LR is reduced
#     "scheduler_factor": 0.5,     # (Optional) Controls the reduction factor}
#     # no scheduler_type for tslm, always use ReduceLROrPlateau (plateau)

#     "training_model":"classification",

#  }

# pipe = build_pipeline(ctx=ctx)
# # pipe = build_binar_pipeline()
# out = pipe.run(ctx)
# print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")

