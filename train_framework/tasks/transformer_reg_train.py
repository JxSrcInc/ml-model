from train_framework.pipelines.reg import build_reg_train_pipeline
from framework.component import Context

ctx = {
    "base_dir": "/home/john/git-2024/ml-stock/data/dayTrade",
    "stem": "all_100_20",
    "key": "diff",
    "task": "reg",
    "batch_size": 1000,

    "model_type": "transformer",
    "hidden": 128,
    "nhead": 4,
    "enc_layers": 2,
    "dim_ff": 256,
    "dropout": 0.1,

    "epochs": 50,
    "lr": 1e-3,
    "ckpt_file": "trans_reg.pt",
    "patience": 20,
    "load_mode": "weights",
    "save_mode": "weights",
}

pipe = build_reg_train_pipeline()
out = pipe.run(ctx)
print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")
