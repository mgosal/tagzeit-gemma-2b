"""
ALU — The 5-Primitive Arithmetic Logic Unit
=============================================
Four free tensor operations + one STE-wrapped discretisation.
Every program in the compute layer composes from these.

Primitives:
    Add(a, b)   →  a + b           (free, no STE)
    Sub(a, b)   →  a - b           (free, no STE)
    Mul(a, b)   →  a * b           (free, no STE)
    Div(a, b)   →  a / b           (free, no STE)
    Floor(x)    →  floor_ste(x)    (STE: forward=floor, backward=identity)

Design:
    - All primitives operate on arbitrary float tensors
    - No unsigned-int limitation — negative numbers, large values, fractions all work
    - Floor is the ONLY discretisation point — the bridge between continuous and
      discrete math. Every integer operation LLMs get wrong (mod, carry, rollover)
      reduces to some combination of arithmetic + Floor.
    - Zero learnable parameters

See: ADR-002, Phase Zero (experiments/neural-compute/)
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.compute_layer.arithmetic import floor_ste


# ---------------------------------------------------------------------------
# The 5 Primitives
# ---------------------------------------------------------------------------

def Add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition. Gradient: ∂/∂a = 1, ∂/∂b = 1."""
    return a + b


def Sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction. Gradient: ∂/∂a = 1, ∂/∂b = -1."""
    return a - b


def Mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication. Gradient: ∂/∂a = b, ∂/∂b = a."""
    return a * b


def Div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division. Gradient: ∂/∂a = 1/b, ∂/∂b = -a/b².

    Guards against zero division: denominators in (-eps, eps) are clamped
    to ±eps. This is necessary because during early training, learned
    extractors produce near-zero values that become Mod/FloorDiv divisors.
    """
    eps = 1e-7
    b_safe = torch.where(b.abs() < eps, torch.full_like(b, eps), b)
    return a / b_safe


def Floor(x: Tensor) -> Tensor:
    """Floor with Straight-Through Estimator.

    Forward:  floor(x) — standard mathematical floor
    Backward: identity — gradient passes through unchanged

    This is the ONLY STE in the entire ALU. Every integer operation
    that LLMs fail at reduces to arithmetic + Floor.
    """
    return floor_ste(x)


# ---------------------------------------------------------------------------
# Composite operations (built from the 5 primitives)
# These are here because they're fundamental enough to be ALU-level,
# not program-level. Think of them as ALU micro-ops.
# ---------------------------------------------------------------------------

def Mod(x: Tensor, n: Tensor | float) -> Tensor:
    """Modular reduction: x mod n.

    Wolfram: Mod[125, 60] → 5

    Composition: x - floor(x/n) * n
    Ops: 1 Floor + 1 Div + 1 Mul + 1 Sub = 4 ops, 0 params
    """
    if not isinstance(n, Tensor):
        n = torch.tensor(n, dtype=x.dtype, device=x.device)
    return Sub(x, Mul(Floor(Div(x, n)), n))


def FloorDiv(x: Tensor, n: Tensor | float) -> Tensor:
    """Integer division: floor(x / n).

    Wolfram: Quotient[125, 60] → 2

    Composition: floor(x / n)
    Ops: 1 Floor + 1 Div = 2 ops, 0 params
    """
    if not isinstance(n, Tensor):
        n = torch.tensor(n, dtype=x.dtype, device=x.device)
    return Floor(Div(x, n))


def Abs(x: Tensor) -> Tensor:
    """Absolute value. Uses torch.abs (differentiable except at 0)."""
    return torch.abs(x)


def Clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    """Clamp to range [lo, hi]. Uses torch.clamp (differentiable in range)."""
    return torch.clamp(x, lo, hi)
