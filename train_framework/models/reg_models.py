# models/reg_models.py
from __future__ import annotations
from typing import Any, Dict, Literal
import torch.nn as nn
from .sequence_models import build_sequence_model

ModelType = Literal["lstm", "transformer"]
 
def build_regression_model(model_type: ModelType, **cfg: Dict[str, Any]) -> nn.Module:
    """Build a sequence backbone with a 1-D regression head.
    This simply reuses the same backbones as classifiers but sets num_classes=1.
    """
    cfg = dict(cfg)
    cfg['num_classes'] = 1
    return build_sequence_model(model_type, **cfg)
