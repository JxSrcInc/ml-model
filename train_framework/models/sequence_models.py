# models/sequence_models.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Any, Dict
import torch
from torch import nn

ModelType = Literal["lstm", "rnn", "transformer"]

# ------------------- Backbones -------------------

class SimpleRNNDiff(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=3, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.post_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # (B, L, 1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.post_dropout(last)
        return self.head(last)

class SimpleLSTMDiff(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=3, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 and dropout and dropout > 0 else 0.0,
            bidirectional=False,
        )
        self.post_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # (B, L, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.post_dropout(last)
        return self.head(last)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # NOTE: / d_model instead of / max_len

        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):  # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)
# change for push
class SimpleTransformerDiff(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        input_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project scalar/low-dim input into model dimension
        self.in_proj = nn.Linear(input_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Use proper positional encoding with dropout
        self.posenc = PositionalEncoding(d_model, dropout=dropout)

        # Global pooling over time
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final normalization and head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: (B, L, C)
        # Project to model dimension and scale
        h = self.in_proj(x) * math.sqrt(self.d_model)  # (B, L, D)

        # Add positional encoding
        h = self.posenc(h)  # (B, L, D)

        # Transformer encoder
        h = self.encoder(h)  # (B, L, D)

        # Global average pool over time dimension
        h = h.transpose(1, 2)                 # (B, D, L)
        h = self.pool(h).squeeze(-1)          # (B, D)

        # Final normalization + classification head
        h = self.norm(h)
        return self.head(h)                   # (B, num_classes)

# ------------------- Factory -------------------

def build_sequence_model(model_type: ModelType, **cfg) -> nn.Module:
    """
    Create a model by type without touching trainer code.
    Common keys:
      - num_classes (required)
      - input_size (default=1)
      - dropout, hidden/num_layers for RNN/LSTM
      - d_model, nhead, enc_layers, dim_ff, dropout for Transformer
    """
    num_classes = int(cfg.get("num_classes", 3))
    input_size  = int(cfg.get("input_size", 1))

    if model_type == "lstm":
        return SimpleLSTMDiff(
            input_size=input_size,
            hidden_size=int(cfg.get("hidden", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.0)),
        )
    elif model_type == "rnn":
        return SimpleRNNDiff(
            input_size=input_size,
            hidden_size=int(cfg.get("hidden", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.0)),
        )
    elif model_type == "transformer":
        hidden = int(cfg.get("hidden", 128))       # d_model
        nhead = int(cfg.get("nhead", 8))          # must divide hidden
        enc_layers = int(cfg.get("enc_layers", 4))
        dim_ff = int(cfg.get("dim_ff", 4 * hidden))  # standard FF dimension
        drop = float(cfg.get("dropout", 0.1))

        return SimpleTransformerDiff(
            d_model=hidden,
            nhead=nhead,
            num_layers=enc_layers,
            dim_feedforward=dim_ff,
            num_classes=num_classes,
            dropout=drop,
            input_size=input_size,
        )
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Use 'lstm' | 'rnn' | 'transformer'.")
