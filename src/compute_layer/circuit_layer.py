"""
Circuit Layer — Drop-in Neural Network Layer for Exact Arithmetic
===================================================================
Wraps the decimal digit adder as a standard nn.Module that can be
inserted into any neural network architecture.

This layer has NO learnable parameters. Its computation is hard-coded
as a digital circuit (logic gates → adders → decimal arithmetic).
A forward pass through this layer IS circuit execution.

Usage:
    layer = CircuitLayer()
    x = torch.tensor([[7.0, 8.0]])  # two digits
    y = layer(x)                     # → tensor([[5.0, 1.0]])  (sum=5, carry=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.compute_layer.adders import DecimalDigitAdder


class CircuitLayer(nn.Module):
    """A neural network layer that performs exact single-digit addition.

    No learnable parameters. The 'weights' are the circuit topology
    (gate connections) encoded as algebraic operations in the
    underlying HalfAdder/FullAdder/RippleCarryAdder modules.

    Input:  (batch, 2) — two decimal digits as floats, each in [0, 9]
    Output: (batch, 2) — (sum_digit, carry) as floats
                         sum_digit in [0, 9], carry in {0, 1}
    """

    def __init__(self):
        super().__init__()
        self.digit_adder = DecimalDigitAdder()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, 2) tensor where x[:, 0] = digit_a, x[:, 1] = digit_b

        Returns:
            (batch, 2) tensor where out[:, 0] = sum_digit, out[:, 1] = carry
        """
        digit_a = x[..., 0]
        digit_b = x[..., 1]
        sum_digit, carry = self.digit_adder(digit_a, digit_b)
        return torch.stack([sum_digit, carry], dim=-1)

    def extra_repr(self) -> str:
        return "mode=decimal_digit_addition, learnable_params=0"
