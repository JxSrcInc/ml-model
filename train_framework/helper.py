import os
def get_out_dir(ctx):
    base_dir = ctx["base_dir"]
    task = ctx["task"]
    stem     = ctx["stem"]
    key      = ctx["key"]

    out_dir  = f'{base_dir}/{task}/{stem}/{key}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def get_ckpt_dir(ctx):
    return get_out_dir(ctx) + '/model'
