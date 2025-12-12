
from __future__ import annotations
from typing import Dict, Any
from framework.pipeline import Pipeline
import os
from train_framework.components.build_model import BuildModel
from train_framework.components.train_model import TrainModel
from train_framework.components.sample_loader import BuildLoadersFromExport

''' select classification and regression to train'''
# def build_pipeline(training_model:str, ctx: dict[str, Any]):
def build_pipeline(ctx: dict[str, Any]):
    training_model = ctx['training_model']
    # ctx['training_model'] = training_model
    model_type = ctx['model_type']
    ckpt_dir= f"{ctx['base_dir']}/{ctx['task']}/{ctx['stem']}/{ctx['key']}/{model_type}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ctx['ckpt_dir'] = ckpt_dir
    if training_model == 'classification':
        pipeline = [
        # create sample dataset for training
        BuildLoadersFromExport(),
        # create NN model for training
        BuildModel(),  
        # do training
        TrainModel(),  
        ]
        ctx['task'] = 'cls'
    elif training_model == 'regression':
        pass
    else:
        raise ValueError(f'Unknown training_model: {training_model}')

    return Pipeline(pipeline)
