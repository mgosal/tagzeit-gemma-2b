"""
Arithmetic Circuits — Adders Composed from Logic Gates
========================================================
Each adder is a non-learnable nn.Module that composes logic gates
from gates.py into arithmetic circuits.

Hierarchy:
    HalfAdder         — 2 bits → (sum, carry)
    FullAdder         — 2 bits + carry_in → (sum, carry_out)
    RippleCarryAdder  — N-bit + N-bit → (N+1)-bit result
    DecimalDigitAdder — digit(0-9) + digit(0-9) → (sum_digit, carry)

All modules have zero learnable parameters. The computation graph
IS the circuit. Activations ARE signal values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.compute_layer.arithmetic import floor_ste, fmod_ste

from src.compute_layer.gates import AND, OR, XOR


class HalfAdder(nn.Module):
    """Half-adder: adds two single bits.

    Circuit:
        sum  = XOR(a, b)
        carry = AND(a, b)

    Truth table:
        a  b │ sum  carry
        0  0 │  0    0
        0  1 │  1    0
        1  0 │  1    0
        1  1 │  0    1
    """

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        sum_bit = XOR(a, b)
        carry = AND(a, b)
        return sum_bit, carry


class FullAdder(nn.Module):
    """Full-adder: adds two bits plus a carry-in.

    Circuit:
        p    = XOR(a, b)
        sum  = XOR(p, cin)
        cout = OR(AND(a, b), AND(p, cin))

    Truth table:
        a  b  cin │ sum  cout
        0  0   0  │  0    0
        0  0   1  │  1    0
        0  1   0  │  1    0
        0  1   1  │  0    1
        1  0   0  │  1    0
        1  0   1  │  0    1
        1  1   0  │  0    1
        1  1   1  │  1    1
    """

    def forward(
        self, a: Tensor, b: Tensor, cin: Tensor
    ) -> tuple[Tensor, Tensor]:
        p = XOR(a, b)
        sum_bit = XOR(p, cin)
        g = AND(a, b)
        p_cin = AND(p, cin)
        cout = OR(g, p_cin)
        return sum_bit, cout


class RippleCarryAdder(nn.Module):
    """N-bit ripple-carry adder: chains full-adders for multi-bit addition.

    The carry output of each bit position feeds into the carry input
    of the next position (LSB to MSB). This mirrors physical hardware.

    Input:  a_bits, b_bits — (batch, n_bits) tensors, LSB at index 0
    Output: result_bits — (batch, n_bits + 1) tensor, LSB at index 0
            The extra bit is the final carry-out.

    For n_bits=4: adds two 4-bit numbers (0-15) → 5-bit result (0-30).
    """

    def __init__(self, n_bits: int = 4):
        super().__init__()
        self.n_bits = n_bits
        self.half_adder = HalfAdder()
        self.full_adder = FullAdder()

    def forward(
        self, a_bits: Tensor, b_bits: Tensor
    ) -> Tensor:
        """
        Args:
            a_bits: (batch, n_bits) binary tensor, LSB at index 0
            b_bits: (batch, n_bits) binary tensor, LSB at index 0

        Returns:
            (batch, n_bits + 1) binary tensor, LSB at index 0
        """
        result_bits = []

        # Bit 0: half-adder (no carry-in)
        s, carry = self.half_adder(a_bits[..., 0], b_bits[..., 0])
        result_bits.append(s)

        # Bits 1..n-1: full-adders with carry chain
        for i in range(1, self.n_bits):
            s, carry = self.full_adder(a_bits[..., i], b_bits[..., i], carry)
            result_bits.append(s)

        # Final carry-out
        result_bits.append(carry)

        return torch.stack(result_bits, dim=-1)


def decimal_to_binary(digits: Tensor, n_bits: int = 4) -> Tensor:
    """Convert decimal integer tensor to binary representation.

    Uses STE-wrapped floor/fmod so gradients flow through the conversion.

    Args:
        digits: tensor of integer values (as floats)
        n_bits: number of output bits

    Returns:
        (..., n_bits) tensor of binary values, LSB at index 0
    """
    bits = []
    remaining = digits
    for _ in range(n_bits):
        bit = fmod_ste(remaining, 2.0)
        bits.append(bit)
        remaining = floor_ste(remaining / 2.0)
    return torch.stack(bits, dim=-1)


def binary_to_decimal(bits: Tensor) -> Tensor:
    """Convert binary representation back to decimal integer.

    Args:
        bits: (..., n_bits) tensor of binary values, LSB at index 0

    Returns:
        tensor of decimal integer values (as floats)
    """
    n_bits = bits.shape[-1]
    powers = torch.pow(2.0, torch.arange(n_bits, dtype=bits.dtype, device=bits.device))
    return (bits * powers).sum(dim=-1)


class DecimalDigitAdder(nn.Module):
    """Adds two single decimal digits (0-9) using binary circuit logic.

    Pipeline:
        1. Convert each digit to 4-bit binary
        2. Add via RippleCarryAdder (4-bit → 5-bit result)
        3. Convert 5-bit result back to decimal
        4. Decompose into sum_digit (mod 10) and carry (div 10)

    Input:  digit_a, digit_b — tensors of values in [0, 9]
    Output: (sum_digit, carry) — sum_digit in [0, 9], carry in {0, 1}

    For all 100 input pairs (0+0 through 9+9), this produces exact results.
    """

    def __init__(self):
        super().__init__()
        self.adder = RippleCarryAdder(n_bits=4)

    def forward(
        self, digit_a: Tensor, digit_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            digit_a: tensor of digit values [0-9] (as floats)
            digit_b: tensor of digit values [0-9] (as floats)

        Returns:
            sum_digit: tensor of digit values [0-9]
            carry: tensor of carry values {0, 1}
        """
        a_bits = decimal_to_binary(digit_a, n_bits=4)
        b_bits = decimal_to_binary(digit_b, n_bits=4)

        result_bits = self.adder(a_bits, b_bits)

        result_decimal = binary_to_decimal(result_bits)

        # Decompose: result = carry * 10 + sum_digit
        # Uses STE floor so gradients flow through carry decomposition
        carry = floor_ste(result_decimal / 10.0)
        sum_digit = result_decimal - carry * 10.0

        return sum_digit, carry
