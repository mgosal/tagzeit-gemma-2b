"""
Temporal Programs — time arithmetic composed from ALU primitives
=================================================================
These address the Temporal deficiency category.
Each function is a fixed computation graph with zero learnable parameters.

Programs:
    TimeAdd(h1, m1, h2, m2)        →  (h, m, day_overflow)
    TimeSub(h1, m1, h2, m2)        →  (h, m, day_underflow)
    DurationBetween(h1, m1, h2, m2) → (hours, minutes, total_minutes)

These replicate the exact behaviour of core/computation/temporal_engine.js
using only the 5 ALU primitives.
"""

from __future__ import annotations

from torch import Tensor

from src.compute_layer.alu import Add, Sub, Mul, Mod, FloorDiv, Clamp


def TimeAdd(h1: Tensor, m1: Tensor, h2: Tensor, m2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Add two times: (h1:m1) + (h2:m2) → (result_h, result_m, day_overflow).

    Wolfram: DatePlus[TimeObject[{9, 58, 0}], Quantity[5, "Minutes"]]
             → TimeObject[{10, 3, 0}]

    Base-60 minutes, base-24 hours.

    Composition:
        total_m  = m1 + m2                    (Add)
        m_carry  = floor(total_m / 60)        (FloorDiv — 1 Floor)
        result_m = total_m mod 60             (Mod — 1 Floor)
        total_h  = h1 + h2 + m_carry          (Add × 2)
        h_carry  = floor(total_h / 24)        (FloorDiv — 1 Floor)
        result_h = total_h mod 24             (Mod — 1 Floor)

    Total: 4 Floor + 10 arithmetic = 14 ops. 0 params.
    """
    total_m = Add(m1, m2)
    m_carry = FloorDiv(total_m, 60.0)
    result_m = Mod(total_m, 60.0)

    total_h = Add(Add(h1, h2), m_carry)
    h_carry = FloorDiv(total_h, 24.0)
    result_h = Mod(total_h, 24.0)

    return result_h, result_m, h_carry


def TimeSub(h1: Tensor, m1: Tensor, h2: Tensor, m2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Subtract duration from time: (h1:m1) - (h2:m2) → (result_h, result_m, day_underflow).

    Wolfram: DatePlus[TimeObject[{2, 10, 0}], Quantity[-30, "Minutes"]]
             → TimeObject[{1, 40, 0}]

    Handles borrow: if minutes go negative, borrow from hours.
    If hours go negative, borrow from days (wrap around 24).

    Composition:
        total_m  = m1 + 60 - m2               (bring into positive range)
        result_m = total_m mod 60             (Mod — 1 Floor)
        m_borrow = 1 - floor(total_m / 60)    (FloorDiv — 1 Floor)
        total_h  = h1 + 24 - h2 - m_borrow   (bring into positive range)
        result_h = total_h mod 24             (Mod — 1 Floor)
        h_borrow = 1 - floor(total_h / 24)    (FloorDiv — 1 Floor)

    Total: 4 Floor + ~12 arithmetic = 16 ops. 0 params.
    """
    # Add 60 to ensure non-negative before mod, then check if we borrowed
    total_m = Add(Sub(m1, m2), 60.0)
    result_m = Mod(total_m, 60.0)
    # If total_m >= 60, no borrow needed (FloorDiv >= 1). If < 60, borrow = 1.
    m_borrow = Sub(1.0, Clamp(FloorDiv(total_m, 60.0), 0.0, 1.0))

    # Same pattern for hours: add 24, subtract, check borrow
    total_h = Sub(Sub(Add(h1, 24.0), h2), m_borrow)
    result_h = Mod(total_h, 24.0)
    h_borrow = Sub(1.0, Clamp(FloorDiv(total_h, 24.0), 0.0, 1.0))

    return result_h, result_m, h_borrow


def DurationBetween(h1: Tensor, m1: Tensor, h2: Tensor, m2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Duration from time1 to time2: (h2:m2) - (h1:m1) → (hours, minutes, total_minutes).

    Wolfram: DateDifference[TimeObject[{9, 0}], TimeObject[{17, 30}]]
             → Quantity[510, "Minutes"]

    If t2 < t1, assumes t2 is the next day (adds 24h).

    Composition:
        total1   = h1 * 60 + m1              (to total minutes)
        total2   = h2 * 60 + m2
        diff     = total2 - total1
        diff     = if diff < 0: diff + 1440  (add 24h in minutes)
        hours    = floor(diff / 60)
        minutes  = diff mod 60

    Total: 2 Floor + ~10 arithmetic = 12 ops. 0 params.
    """
    total1 = Add(Mul(h1, 60.0), m1)
    total2 = Add(Mul(h2, 60.0), m2)
    diff = Sub(total2, total1)

    # If negative, add 24 hours (1440 minutes)
    # diff_adjusted = ((diff + 1440) mod 1440) handles wrap-around
    diff_adjusted = Mod(Add(diff, 1440.0), 1440.0)

    hours = FloorDiv(diff_adjusted, 60.0)
    minutes = Mod(diff_adjusted, 60.0)

    return hours, minutes, diff_adjusted
