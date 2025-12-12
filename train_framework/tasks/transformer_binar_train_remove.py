from train_framework.pipelines.build_pipeline import build_pipeline
from train_framework.tasks.config.transformer_binar_ctx import Config

base_dir ='C:/Users/johnl/data-repos/ml-ts'
stem = "sprf_100_50"
stem = 'all_100_50'
batch_size = 20
key = 'diff'

ctx = Config(base_dir=base_dir, stem=stem, batch_size=batch_size, key=key)
ctx = ctx.__dict__

pipe = build_pipeline(ctx=ctx)
out = pipe.run(ctx)
print(f"used_time: {out['train_time']:.2f}, ckpt_path: {out['ckpt_path']}")
