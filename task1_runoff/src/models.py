"""
Deep learning models for runoff forecast post-processing.

The final exam requires THREE architectures (LSTM, GRU, Transformer).
This file gives you one fully-working baseline (LSTM) and clean scaffolds
for the other two — you implement them, then compare all three.

All models share the same interface: they take a batch of shape
    (batch, lookback, n_features)
and return predictions of shape
    (batch,)

That way `train.py` stays model-agnostic — swap the model and run.

Design tip: keep the hidden size and number of layers the same across the
three models when you compare them, so any performance difference is
actually about the architecture and not about parameter count.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


# ===========================================================================
# 1. LSTM  ──────────── FULLY IMPLEMENTED (this is your working baseline) ───
# ===========================================================================

class LSTMPredictor(nn.Module):
    """An LSTM followed by a small MLP head.

    Forward pass:
      1. LSTM reads the whole lookback window and produces a hidden state
         sequence of shape (B, L, H).
      2. We take the final timestep's hidden state (B, H) as the summary.
      3. A 2-layer MLP maps H -> 1 scalar prediction.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, _ = self.lstm(x)            # (B, L, H)
        last = out[:, -1, :]             # (B, H)
        return self.head(last).squeeze(-1)  # (B,)


# ===========================================================================
# 2. GRU  ────────── TODO: IMPLEMENT THIS (Michael or Grady) ────────────────
# ===========================================================================

class GRUPredictor(nn.Module):
    """GRU counterpart to LSTMPredictor.

    TODO (should be straightforward once you've read the LSTM version):
      - Replace nn.LSTM with nn.GRU. The API is almost identical; GRU's forward
        returns (output, h_n) whereas LSTM returns (output, (h_n, c_n)).
      - Keep the MLP head the same so the comparison with LSTM is fair.
      - Match hidden_size / num_layers / dropout defaults to the LSTM so the
        only difference is the recurrent cell type.

    Why bother? GRU has 25%-ish fewer parameters than LSTM for the same hidden
    size (3 gates vs 4) and often trains faster with comparable accuracy.
    That's the kind of finding that makes a good comparison table in the
    report.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: build self.gru and self.head analogously to LSTMPredictor
        raise NotImplementedError("GRUPredictor — implement me!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===========================================================================
# 3. Transformer  ───── TODO: IMPLEMENT THIS (one of the team) ──────────────
# ===========================================================================

class TransformerPredictor(nn.Module):
    """A small time-series Transformer encoder.

    TODO:
      - Implement a learnable or sinusoidal positional encoding of length
        `lookback` added to the input projection.
      - Use nn.TransformerEncoder with a handful of layers (2-4 is plenty
        for this size of data).
      - Pool over the sequence axis (simplest: take the last token, like the
        LSTM version does; slightly fancier: mean-pool, or use a [CLS]-style
        learned query).
      - Feed the pooled vector into a 2-layer MLP head returning 1 scalar.

    Reasonable starting hyperparameters (tune later):
        d_model = 64, nhead = 4, num_layers = 2,
        dim_feedforward = 128, dropout = 0.1.

    Sinusoidal positional encoding implementation hint:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        lookback: int = 48,
    ):
        super().__init__()
        # TODO: implement
        raise NotImplementedError("TransformerPredictor — implement me!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===========================================================================
# Factory — keeps train.py simple
# ===========================================================================

MODEL_REGISTRY = {
    "lstm": LSTMPredictor,
    "gru":  GRUPredictor,
    "transformer": TransformerPredictor,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if name == "transformer":
        return cls(n_features=n_features, **kwargs)
    return cls(n_features=n_features, **kwargs)
