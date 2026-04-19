"""
Proof-of-Concept Model — A Neural Network That Uses a Circuit Layer
=====================================================================
Two-stage proof architecture:

Stage A: "Does the circuit work inside a forward pass?"
    - Digit values are passed directly (no extraction learning)
    - Only output heads are learned (trivial linear classification)
    - Proves: circuit layer integration, gradient flow, training loop
    - If this fails → circuit integration is broken

Stage B: "Can a network learn to extract values for the circuit?"
    - Digit values are extracted from learned embeddings
    - Each digit token gets a 1D embedding that must converge to its integer value
    - Proves: a learned encoder can interface with an exact circuit layer
    - If this fails → the extraction/STE gradient path is broken

Stage C: "Does it work with a transformer encoder?"
    - Full transformer encoder processes token sequence
    - Value extractor maps encoded representation to digit values
    - Proves: the complete architecture (NLU → extraction → circuit → output) works
    - If this fails → transformer adds too much noise to the extraction

Each stage is a separate model class. Train them in order (A → B → C).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.compute_layer.arithmetic import round_ste
from src.compute_layer.circuit_layer import CircuitLayer


# ---------------------------------------------------------------------------
# Stage A: Minimal Integration Proof
# ---------------------------------------------------------------------------

class StageAModel(nn.Module):
    """Minimal proof: circuit layer in a training loop.

    No value extraction. Digit values are passed directly as floats.
    Only the output heads (sum classifier + carry classifier) are learned.
    This isolates the question: "can output heads learn to decode circuit results?"

    Params: 36 (Linear(2,10) + Linear(2,2))
    """

    def __init__(self):
        super().__init__()
        self.circuit = CircuitLayer()
        self.sum_head = nn.Linear(2, 10)
        self.carry_head = nn.Linear(2, 2)

    def forward(
        self, digit_a_ids: Tensor, digit_b_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Use token IDs directly as digit values (no extraction)
        a_val = digit_a_ids.float()
        b_val = digit_b_ids.float()

        circuit_in = torch.stack([a_val, b_val], dim=-1)
        circuit_out = self.circuit(circuit_in)  # (batch, 2): [sum_digit, carry]

        # Detach: output heads classify the circuit result as a fixed input.
        # No gradient flows through the circuit to the heads — the STE
        # approximation corrupts CE gradients. The circuit's job is to compute
        # the right answer; the heads' job is to classify that answer.
        sum_logits = self.sum_head(circuit_out.detach())
        carry_logits = self.carry_head(circuit_out.detach())

        # Return dummy raw_values (no extraction to supervise)
        raw_values = circuit_in
        return sum_logits, carry_logits, raw_values


# ---------------------------------------------------------------------------
# Stage B: Learned Value Extraction
# ---------------------------------------------------------------------------

class StageBModel(nn.Module):
    """Extraction proof: network learns to produce circuit-compatible values.

    Each digit gets a 1D embedding. The embedding for digit k must learn
    to represent the value k (so that the circuit receives correct inputs).
    Auxiliary MSE loss supervises the embeddings directly.

    Params: 20 (embeddings) + 36 (output heads) = 56
    """

    def __init__(self):
        super().__init__()
        # Each digit gets a scalar embedding that should converge to its integer value
        self.digit_embedding = nn.Embedding(10, 1)

        self.circuit = CircuitLayer()
        self.sum_head = nn.Linear(2, 10)
        self.carry_head = nn.Linear(2, 2)

        # Initialize embeddings near their target values
        # (makes the extraction problem easier — focus is on proving the concept)
        with torch.no_grad():
            for i in range(10):
                self.digit_embedding.weight[i, 0] = float(i) + 0.0

    def forward(
        self, digit_a_ids: Tensor, digit_b_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Extract digit values from embeddings
        a_val = self.digit_embedding(digit_a_ids).squeeze(-1)  # (batch,)
        b_val = self.digit_embedding(digit_b_ids).squeeze(-1)  # (batch,)

        raw_values = torch.stack([a_val, b_val], dim=-1)  # (batch, 2)

        # Round to integer (STE: forward=round, backward=identity)
        a_rounded = round_ste(a_val).clamp(0, 9)
        b_rounded = round_ste(b_val).clamp(0, 9)

        circuit_in = torch.stack([a_rounded, b_rounded], dim=-1)
        circuit_out = self.circuit(circuit_in)

        # Detach circuit output — heads learn from correct integers,
        # extractor learns from auxiliary MSE loss
        sum_logits = self.sum_head(circuit_out.detach())
        carry_logits = self.carry_head(circuit_out.detach())

        return sum_logits, carry_logits, raw_values


# ---------------------------------------------------------------------------
# Stage C: Transformer Encoder + Extraction
# ---------------------------------------------------------------------------

class StageCModel(nn.Module):
    """Full proof: transformer encoder → value extractor → circuit → output.

    Architecture:
        Token embeddings (20 tokens, d=64)
        → Positional encoding (2 positions)
        → TransformerEncoder (2 layers, 4 heads, d=64)
        → Value extractor (Linear → 2 values)
        → CircuitLayer (0 params)
        → Output heads (sum: 10 classes, carry: 2 classes)

    Params: ~30K
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()

        self.token_embedding = nn.Embedding(20, d_model)
        self.pos_embedding = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.circuit = CircuitLayer()
        self.sum_head = nn.Linear(2, 10)
        self.carry_head = nn.Linear(2, 2)

    def forward(
        self, digit_a_ids: Tensor, digit_b_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = digit_a_ids.shape[0]
        device = digit_a_ids.device

        token_ids = torch.stack([digit_a_ids, digit_b_ids + 10], dim=1)
        tok_emb = self.token_embedding(token_ids)
        positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb

        x = self.encoder(x)
        x_pooled = x.mean(dim=1)

        raw_values = self.value_head(x_pooled)
        values_rounded = round_ste(raw_values).clamp(0.0, 9.0)

        circuit_out = self.circuit(values_rounded)

        # Detach circuit output — heads learn from correct integers,
        # extractor learns from auxiliary MSE loss
        sum_logits = self.sum_head(circuit_out.detach())
        carry_logits = self.carry_head(circuit_out.detach())

        return sum_logits, carry_logits, raw_values


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(stage: str = "A", **kwargs) -> nn.Module:
    """Create a proof model for the specified stage.

    Args:
        stage: "A", "B", or "C"
    """
    if stage.upper() == "A":
        return StageAModel()
    elif stage.upper() == "B":
        return StageBModel()
    elif stage.upper() == "C":
        return StageCModel(**kwargs)
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'A', 'B', or 'C'.")
