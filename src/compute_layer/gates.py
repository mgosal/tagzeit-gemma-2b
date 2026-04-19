"""
Logic Gates — Algebraic Form over Binary Tensors
==================================================
Digital logic gates implemented as differentiable tensor operations.

For binary inputs (a, b ∈ {0, 1}), these algebraic identities produce
exact binary outputs AND have well-defined gradients:

    AND(a, b) = a * b                    ∂/∂a = b,  ∂/∂b = a
    OR(a, b)  = a + b - a * b            ∂/∂a = 1-b, ∂/∂b = 1-a
    XOR(a, b) = a + b - 2 * a * b        ∂/∂a = 1-2b, ∂/∂b = 1-2a
    NOT(a)    = 1 - a                    ∂/∂a = -1

No activation functions. No thresholds. No STE needed.
The algebra IS the logic.
"""

from __future__ import annotations

import torch
from torch import Tensor


def AND(a: Tensor, b: Tensor) -> Tensor:
    """Logical AND: returns 1 iff both inputs are 1.

    Algebraic form: a * b
    Truth table: (0,0)→0  (0,1)→0  (1,0)→0  (1,1)→1
    """
    return a * b


def OR(a: Tensor, b: Tensor) -> Tensor:
    """Logical OR: returns 1 if either input is 1.

    Algebraic form: a + b - a*b  (inclusion-exclusion)
    Truth table: (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→1
    """
    return a + b - a * b


def XOR(a: Tensor, b: Tensor) -> Tensor:
    """Logical XOR: returns 1 iff exactly one input is 1.

    Algebraic form: a + b - 2*a*b
    Truth table: (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→0
    """
    return a + b - 2.0 * a * b


def NOT(a: Tensor) -> Tensor:
    """Logical NOT: returns 1 if input is 0, 0 if input is 1.

    Algebraic form: 1 - a
    Truth table: 0→1  1→0
    """
    return 1.0 - a


def verify_truth_table(
    gate_fn,
    expected: dict[tuple, float],
    device: str = "cpu",
) -> bool:
    """Exhaustively verify a gate function against its truth table.

    Args:
        gate_fn: callable taking (a, b) or (a,) tensors
        expected: dict mapping input tuples to expected output float
        device: torch device

    Returns:
        True if all outputs match expected values exactly.

    Raises:
        AssertionError with details if any output is wrong.
    """
    for inputs, expected_val in expected.items():
        tensors = [torch.tensor(float(v), device=device) for v in inputs]
        result = gate_fn(*tensors)
        actual = result.item()
        assert actual == expected_val, (
            f"Gate {gate_fn.__name__}: inputs={inputs} → "
            f"got {actual}, expected {expected_val}"
        )
    return True
