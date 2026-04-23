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
# 2. GRU  ────────────────── Implemented.  Near-twin of LSTMPredictor. ──────
# ===========================================================================

class GRUPredictor(nn.Module):
    """GRU counterpart to LSTMPredictor.

    A GRU (Gated Recurrent Unit) is an LSTM's simpler cousin. It has only
    two gates (reset + update) instead of the LSTM's three (input + forget +
    output), and no separate cell state. Fewer parameters, often faster
    training, usually within noise of LSTM on smallish time-series problems
    like ours.

    API-wise the only difference from nn.LSTM is that nn.GRU returns
    (output, h_n) instead of (output, (h_n, c_n)). Everything else —
    batch_first, dropout, stacked layers — is identical.

    Why bother comparing? The research question for the report is whether
    the extra gating of LSTM provides real benefit on NWM residuals, or
    whether GRU's simpler mechanism is sufficient. Training time and
    parameter count make good discussion points.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Keep the head identical to LSTMPredictor so the only architectural
        # difference between the two models is the recurrent cell type.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, _ = self.gru(x)              # (B, L, H)
        last = out[:, -1, :]              # (B, H) — final timestep summary
        return self.head(last).squeeze(-1)  # (B,)


# ===========================================================================
# 3. Transformer  ────────────────── Implemented. ──────────────────────────
# ===========================================================================

class _PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding from "Attention Is All You Need".

    A Transformer treats its input as an unordered set of tokens. For time
    series we care *a lot* about order (hour t is different from hour t-1),
    so we add a fixed pattern to each position that tells the model where in
    the sequence each token sits.

    The pattern uses sines and cosines at geometrically increasing wavelengths
    so that any two positions' encodings encode their *relative* distance in
    a way attention can learn to read.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer (moves with .to(device), but isn't a trainable param)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) — add the first L rows of the PE table
        return x + self.pe[:, : x.size(1), :]


class TransformerPredictor(nn.Module):
    """A small Transformer encoder for time-series residual prediction.

    Architecture (in order, reading the forward pass):
      1. Input projection: raw features (F dims) → model dim (d_model).
         Transformers work internally at a single fixed width regardless of
         how many input features you have.
      2. Positional encoding: add a fixed position signal to each timestep.
         Without this, self-attention treats the sequence as a bag of hours.
      3. Stack of TransformerEncoder layers. Each layer does multi-head
         self-attention (every hour looks at every other hour and decides
         how much to pay attention to each) followed by a feed-forward MLP.
      4. Pool over the sequence axis. We take the last timestep's
         representation, matching the LSTM and GRU for a fair comparison.
         Alternatives: mean-pool, or prepend a learnable [CLS] token.
      5. 2-layer MLP head → scalar prediction.

    Hyperparameter choices for this small-data regime:
      d_model=64, nhead=4, num_layers=2, dim_feedforward=128.
    We deliberately match d_model to the LSTM/GRU hidden_size so the three
    architectures are roughly comparable in capacity.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,         # used as d_model — keeps API consistent
        num_layers: int = 2,
        dropout: float = 0.1,          # transformers usually want lower dropout
        nhead: int = 4,
        dim_feedforward: int = 128,
        lookback: int = 48,
    ):
        super().__init__()
        d_model = hidden_size
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max(lookback + 1, 512))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,           # (B, L, d) convention, matches our data
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Same head shape as LSTM/GRU for fair comparison
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        h = self.input_proj(x)          # (B, L, d_model)
        h = self.pos_enc(h)              # add positional information
        h = self.encoder(h)              # (B, L, d_model) — self-attention + FFN
        last = h[:, -1, :]               # (B, d_model) — last timestep summary
        return self.head(last).squeeze(-1)  # (B,)


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

