from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config:
    base_dir:str = 'C = /Users/johnl/data-repos/ml-ts'
    # stem:str = sprf_50_20
    stem:str = 'all_100_50'
    key:str = 'diff'
    task:str = 'cls'
    batch_size:int = 256

    # model
    model_type:str = 'lstm'
    num_classes:int = 3
    hidden:int = 128        # LSTM hidden size
    num_layers:int = 2      # LSTM layers
    dropout:float = 0.3       # optional; 0.0 disables

    # training
    epochs:int = 200
    lr:float = 1e-3
    ckpt_file:str = 'lstm_diff.ckpt'
    patience:int = 50
    load_mode:str = 'full'   # 'weights' or 'full'
    save_mode:str = 'full'   # 'weights' or 'full'
    # optional common = 
    grad_clip:float = 1.0           # Recommended starting value for gradient clipping
    scheduler_patience:int = 5    # (Optional) Controls when LR is reduced
    scheduler_factor:float = 0.5     # (Optional) Controls the reduction factor}
    # no scheduler_type for tslm always use ReduceLROrPlateau (plateau)

    training_model:str = 'classification'

