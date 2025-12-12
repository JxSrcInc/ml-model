from train_framework.pipelines.predict_pipeline import build_pipeline
from train_framework.tasks.config.lstm_binar_ctx import Config as LstmConfig
from train_framework.tasks.config.transformer_binar_ctx import Config as TransformerConfig

def predict():
    base_dir ='C:/Users/johnl/data-repos/ml-ts'
    # stem = "sprf_100_50_5"
    # batch_size = 20
    stem = 'all_100_90_5'
    batch_size = 256
    key='diff'
    # model = 'lstm'
    model = 'transformer'

    if model == 'lstm':
        ctx = LstmConfig(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)
    else:
        ctx = TransformerConfig(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)

    ctx = ctx.__dict__

    pipe = build_pipeline(ctx=ctx)
    return pipe.run(ctx)
    # print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")

out = predict()
print(out['pred_label'])

