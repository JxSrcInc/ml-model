from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config:
    base_dir: str = 'C:/Users/johnl/data-repos/ml-ts'
    stem: str = 'all_100_50'
    key:str = 'diff'
    task: str = 'cls'

    batch_size: int = 256   # a bit smaller than LSTM if needed for memory

    # model config (transformer)
    model_type: str = 'transformer'
    num_classes: int = 3

    # Transformer hyperparameters (match factory expectations)
    hidden: int = 128       # d_model
    nhead: int = 8          # must divide hidden
    enc_layers: int = 4
    dim_ff: int = 512        # or 4 * hidden
    dropout: float = 0.1

    # optional common:
    grad_clip: float = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_type: str = 'cosine'  # transformer often works nicely with cosine decay

    # training
    epochs: int = 200
    lr: float = 3e-4           # a bit smaller LR is often safer for transformers
    ckpt_file: str = 'trans_diff.ckpt'
    patience: int = 30
    load_mode: str = 'full'
    save_mode: str = 'full'
    training_model: str = 'classification'

