"""
Differentiable Arithmetic Primitives — Straight-Through Estimators
====================================================================
Floor division and modulo are non-differentiable (gradient = 0 a.e.).
These STE wrappers compute the exact result in the forward pass but
pass gradients straight through in the backward pass, enabling
gradient flow through the full circuit pipeline.

This is the standard technique from quantization-aware training
(Bengio et al., 2013) applied to circuit computation.

Forward: exact discrete operation (floor, fmod)
Backward: identity gradient (∂y/∂x ≈ 1)
"""

from __future__ import annotations

import torch
from torch import Tensor


class _FloorSTE(torch.autograd.Function):
    """Floor with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        # STE: pass gradient through as if floor were identity
        return grad_output


class _FmodSTE(torch.autograd.Function):
    """Fmod with straight-through gradient estimator.

    Note: standard torch.fmod already has gradient 1 w.r.t. input
    (since fmod(x, m) = x - m*floor(x/m) and d(floor)/dx = 0).
    This STE version ensures consistent gradient behavior when
    composed with other STE operations.
    """

    @staticmethod
    def forward(ctx, x: Tensor, modulus: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.modulus = modulus
        return torch.fmod(x, modulus)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output, None


class _RoundSTE(torch.autograd.Function):
    """Round with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def floor_ste(x: Tensor) -> Tensor:
    """Floor operation with STE gradient (forward: floor, backward: identity)."""
    return _FloorSTE.apply(x)


def fmod_ste(x: Tensor, modulus: float) -> Tensor:
    """Fmod operation with STE gradient (forward: fmod, backward: identity)."""
    return _FmodSTE.apply(x, modulus)


def round_ste(x: Tensor) -> Tensor:
    """Round operation with STE gradient (forward: round, backward: identity)."""
    return _RoundSTE.apply(x)
