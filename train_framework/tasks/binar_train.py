from train_framework.pipelines.build_pipeline import build_pipeline
from train_framework.tasks.config.lstm_binar_ctx import Config as LstmConfig
from train_framework.tasks.config.transformer_binar_ctx import Config as TransformerConfig


def main():
    base_dir = 'C:/Users/johnl/data-repos/ml-ts'
    predict_period = 5
    # stem = "sprf_100_50_5"
    # batch_size = 20
    # stem = 'all_100_90'
    stem = f'all_80_75_{predict_period}'
    batch_size = 256 * 2 * 3
    key = 'diff'

    # NOTE: last assignment wins; this selects the transformer by default
    model = 'lstm'
    model = 'transformer'

    if model == 'lstm':
        cfg = LstmConfig(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)
    else:
        cfg = TransformerConfig(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)

    ctx = cfg.__dict__
    ctx['predict_period'] = predict_period

    pipe = build_pipeline(ctx=ctx)
    out = pipe.run(ctx)
    print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # important on Windows when using num_workers > 0
    main()
