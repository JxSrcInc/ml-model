
from __future__ import annotations
from typing import Dict, Any
from framework.pipeline import Pipeline
import os
from train_framework.components.build_model import BuildModel
from train_framework.components.loader.predict_npz_loader import NpzPredictLoader
from train_framework.components.model_predictor import PredictModel
from train_framework.components.prediction_verifier import PredictionVerifier
from train_framework.components.predictor.accuracu_calculater import AccuracyCalculator
from train_framework.components.predictor.rmse_calculator import RegressionEvaluator

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
    ctx['npz_paths'] = f"{ctx['base_dir']}/{ctx['task']}/{ctx['stem']}/{ctx['key']}/test.part0000.npz"
    # ctx['npz_paths'] = f"{ctx['base_dir']}/{ctx['task']}/{ctx['stem']}/{ctx['key']}/validate.part0000.npz"
    if training_model == 'classification':
        pipeline = [
        # create sample dataset for prediction
        NpzPredictLoader(),
        # create NN model for prediction
        BuildModel(),  
        # do prediction
        PredictModel(),  
        AccuracyCalculator(),
        ]
        ctx['task'] = 'cls'
    elif training_model == 'regression':
        pipeline = [
        # create sample dataset for prediction
        NpzPredictLoader(),
        # create NN model for prediction
        BuildModel(),  
        # do prediction
        PredictModel(),  
        RegressionEvaluator(),
        ]
        ctx['task'] = 'reg'

    return Pipeline(pipeline)
