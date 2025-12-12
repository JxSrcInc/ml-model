from train_framework.pipelines.reg import build_reg_predict_pipeline
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

    "ckpt_file": "trans_reg.pt",
    "load_mode": "weights",
    "as_numpy": True,
}

pipe = build_reg_predict_pipeline()
out = pipe.run(ctx)

m = out.get("verify_metrics", {})
if m:
    print(f"n={m['n_samples']}, mse={m['mse']:.6f}, mae={m['mae']:.6f}, rmse={m['rmse']:.6f}, r2={m['r2']:.4f}")
