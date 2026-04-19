"""
Comparison Programs — relational operators composed from ALU primitives
========================================================================
These address the Logical deficiency category.
Each function returns {0.0, 1.0} tensors — binary truth values.

Programs:
    Greater(a, b)   →  1 if a > b,  else 0
    Less(a, b)      →  1 if a < b,  else 0
    Equal(a, b)     →  1 if a == b, else 0
    GreaterEq(a, b) →  1 if a >= b, else 0
    LessEq(a, b)    →  1 if a <= b, else 0
"""

from __future__ import annotations

from torch import Tensor

from src.compute_layer.alu import Sub, Floor, Clamp, Abs


def Greater(a: Tensor, b: Tensor) -> Tensor:
    """1 if a > b, else 0.

    Wolfram: Boole[Greater[7, 3]] → 1

    For integer inputs: Clamp(a - b, 0, 1) gives 1 iff a > b,
    since all positive differences ≥ 1 clamp to 1, and ≤ 0 clamp to 0.

    For non-integer inputs: Floor(Clamp(diff, 0, 1)) ensures {0, 1} output.
    """
    diff = Sub(a, b)
    return Floor(Clamp(diff, 0.0, 1.0))


def Less(a: Tensor, b: Tensor) -> Tensor:
    """1 if a < b, else 0. Wolfram: Boole[Less[3, 7]] → 1."""
    return Greater(b, a)


def GreaterEq(a: Tensor, b: Tensor) -> Tensor:
    """1 if a >= b, else 0. Wolfram: Boole[GreaterEqual[7, 7]] → 1."""
    diff = Sub(a, b)
    # a >= b when diff >= 0. Clamp(diff + 1, 0, 1) = 1 for diff >= 0.
    return Floor(Clamp(Sub(diff, -1.0), 0.0, 1.0))


def LessEq(a: Tensor, b: Tensor) -> Tensor:
    """1 if a <= b, else 0. Wolfram: Boole[LessEqual[3, 7]] → 1."""
    return GreaterEq(b, a)


def Equal(a: Tensor, b: Tensor) -> Tensor:
    """1 if a == b, else 0. Wolfram: Boole[Equal[5, 5]] → 1.

    For integer inputs: |a - b| == 0 iff a == b.
    1 - Clamp(|a - b|, 0, 1) gives 1 iff diff == 0.
    """
    diff = Abs(Sub(a, b))
    return Sub(1.0, Floor(Clamp(diff, 0.0, 1.0)))
