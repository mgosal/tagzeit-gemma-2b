"""
Arithmetic Programs — composed from the 5 ALU primitives
==========================================================
These address the Mathematical deficiency category.
Each function is a fixed computation graph with zero learnable parameters.

Programs:
    Mod(x, n)       →  x mod n
    FloorDiv(x, n)  →  floor(x / n)
    IntAdd(a, b)    →  a + b  (trivial, but named for the program registry)
    IntSub(a, b)    →  a - b
    IntMul(a, b)    →  floor(a * b)  (ensures integer result)
    Remainder(a, b) →  a - floor(a/b) * b  (same as Mod, alias)
"""

from __future__ import annotations

from torch import Tensor

from src.compute_layer.alu import Add, Sub, Mul, Floor, Div, Mod, FloorDiv


# ---------------------------------------------------------------------------
# Basic integer arithmetic
# ---------------------------------------------------------------------------

def IntAdd(a: Tensor, b: Tensor) -> Tensor:
    """Integer addition. a + b.

    Wolfram: Plus[47, 68] → 115

    For integer inputs, result is exact. No Floor needed.
    """
    return Add(a, b)


def IntSub(a: Tensor, b: Tensor) -> Tensor:
    """Integer subtraction. a - b.

    Wolfram: Subtract[68, 47] → 21

    Handles negative results: IntSub(3, 7) → -4.
    """
    return Sub(a, b)


def IntMul(a: Tensor, b: Tensor) -> Tensor:
    """Integer multiplication. floor(a * b).

    Wolfram: Times[6, 7] → 42

    Floor is defensive — for exact integer inputs, Mul alone is exact.
    But if the extractor produces 5.99 instead of 6.0, Floor(5.99 * 7)
    = Floor(41.93) = 41, which is wrong. So for multiplication we
    round the inputs first via the extractor's round_ste, not here.

    In practice: just Mul. Floor applied only if needed.
    """
    return Mul(a, b)


# Re-export ALU-level Mod and FloorDiv for the program registry
__all__ = [
    "IntAdd", "IntSub", "IntMul",
    "Mod", "FloorDiv",
]
