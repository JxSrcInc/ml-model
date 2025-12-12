# components/build_model.py
from __future__ import annotations
from dataclasses import dataclass

try:
    from framework.component import Component, Context
except Exception:
    class Context(dict): pass
    class Component:
        def run(self, ctx: Context): return ctx

from ..models.sequence_models import build_sequence_model  # adjust import path

''' build train model and pass it to TrainModel using ctx through pipeline'''
@dataclass
class BuildModel(Component):
    """
    Build a sequence classification model and put it into ctx['model'].

    Expects (in ctx):
      - 'model_type': 'lstm' | 'rnn' | 'transformer'    (default 'lstm')
      - 'num_classes': int                              (required for classification)
      - Optional hyperparams depending on type:
          * common: 'input_size', 'hidden', 'num_layers', 'dropout'
          * transformer: 'nhead', 'enc_layers', 'dim_ff'

    Output:
      - ctx['model'] = nn.Module
    """
    default_model_type: str = "lstm"

    def run(self, ctx: Context) -> Context:
        mtype = str(ctx.get("model_type", self.default_model_type)).lower()
        if "num_classes" not in ctx:
            raise KeyError("BuildModel requires ctx['num_classes'] to construct the classifier head.")
        # pull a shallow config dict from ctx; unknown keys are ignored by factory
        cfg = {
            "num_classes": int(ctx["num_classes"]),
            "input_size": int(ctx.get("input_size", 1)),
            "hidden": int(ctx.get("hidden", 64)),
            "num_layers": int(ctx.get("num_layers", 1)),
            "dropout": float(ctx.get("dropout", 0.0)),
            "nhead": int(ctx.get("nhead", 4)),
            "enc_layers": int(ctx.get("enc_layers", 2)),
            "dim_ff": int(ctx.get("dim_ff", 256)),
        }
        ctx["model"] = build_sequence_model(mtype, **cfg)
        return ctx
