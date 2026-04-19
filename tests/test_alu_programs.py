"""
Exhaustive tests for the ALU and program library.
=====================================================
Tests cover:
    1. ALU primitives (5 operations)
    2. Arithmetic programs (Mod, FloorDiv, IntAdd, IntSub, IntMul)
    3. Temporal programs (TimeAdd, TimeSub, DurationBetween)
    4. Comparison programs (Greater, Less, Equal, GreaterEq, LessEq)
    5. Gradient flow through all programs
"""

import pytest
import torch
from torch import tensor

from src.compute_layer.alu import Add, Sub, Mul, Div, Floor, Mod, FloorDiv
from src.compute_layer.programs.arithmetic import IntAdd, IntSub, IntMul
from src.compute_layer.programs.temporal import TimeAdd, TimeSub, DurationBetween
from src.compute_layer.programs.comparison import Greater, Less, Equal, GreaterEq, LessEq


# ===================================================================
# ALU Primitives
# ===================================================================

class TestALU_Primitives:
    """Basic smoke tests for the 5 primitives."""

    def test_add(self):
        assert Add(tensor(3.0), tensor(4.0)).item() == 7.0

    def test_sub(self):
        assert Sub(tensor(10.0), tensor(3.0)).item() == 7.0

    def test_sub_negative(self):
        assert Sub(tensor(3.0), tensor(10.0)).item() == -7.0

    def test_mul(self):
        assert Mul(tensor(6.0), tensor(7.0)).item() == 42.0

    def test_div(self):
        assert Div(tensor(15.0), tensor(4.0)).item() == 3.75

    def test_floor(self):
        assert Floor(tensor(3.7)).item() == 3.0

    def test_floor_negative(self):
        assert Floor(tensor(-3.7)).item() == -4.0

    def test_floor_integer(self):
        assert Floor(tensor(5.0)).item() == 5.0

    def test_floor_gradient(self):
        """Floor uses STE — gradient should pass through."""
        x = tensor(3.7, requires_grad=True)
        y = Floor(x)
        y.backward()
        assert x.grad is not None
        assert x.grad.item() == 1.0  # STE: gradient = 1

    def test_batched(self):
        """All primitives work on batched tensors."""
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        assert torch.equal(Add(a, b), tensor([5.0, 7.0, 9.0]))
        assert torch.equal(Sub(a, b), tensor([-3.0, -3.0, -3.0]))
        assert torch.equal(Mul(a, b), tensor([4.0, 10.0, 18.0]))


# ===================================================================
# Mod and FloorDiv
# ===================================================================

class TestALU_Mod:
    """Exhaustive tests for Mod(x, n) = x - floor(x/n) * n."""

    @pytest.mark.parametrize("n", [7, 10, 12, 24, 60])
    def test_exhaustive_positive(self, n):
        """Mod(x, n) matches Python % for x in [0, 1000]."""
        for x in range(0, 1001):
            result = Mod(tensor(float(x)), float(n)).item()
            expected = x % n
            assert result == expected, f"Mod({x}, {n}): got {result}, expected {expected}"

    def test_mod_60_boundaries(self):
        """Minute boundaries."""
        assert Mod(tensor(0.0), 60.0).item() == 0.0
        assert Mod(tensor(59.0), 60.0).item() == 59.0
        assert Mod(tensor(60.0), 60.0).item() == 0.0
        assert Mod(tensor(61.0), 60.0).item() == 1.0
        assert Mod(tensor(119.0), 60.0).item() == 59.0
        assert Mod(tensor(120.0), 60.0).item() == 0.0

    def test_mod_24_boundaries(self):
        """Hour boundaries."""
        assert Mod(tensor(0.0), 24.0).item() == 0.0
        assert Mod(tensor(23.0), 24.0).item() == 23.0
        assert Mod(tensor(24.0), 24.0).item() == 0.0
        assert Mod(tensor(25.0), 24.0).item() == 1.0
        assert Mod(tensor(48.0), 24.0).item() == 0.0

    def test_mod_gradient_flow(self):
        x = tensor(125.0, requires_grad=True)
        y = Mod(x, 60.0)
        y.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad)


class TestALU_FloorDiv:
    """Exhaustive tests for FloorDiv(x, n) = floor(x / n)."""

    @pytest.mark.parametrize("n", [7, 10, 12, 24, 60])
    def test_exhaustive_positive(self, n):
        """FloorDiv(x, n) matches Python // for x in [0, 1000]."""
        for x in range(0, 1001):
            result = FloorDiv(tensor(float(x)), float(n)).item()
            expected = x // n
            assert result == expected, f"FloorDiv({x}, {n}): got {result}, expected {expected}"


# ===================================================================
# Arithmetic Programs
# ===================================================================

class TestArithmeticPrograms:

    def test_int_add_positive(self):
        assert IntAdd(tensor(47.0), tensor(68.0)).item() == 115.0

    def test_int_add_large(self):
        assert IntAdd(tensor(999.0), tensor(999.0)).item() == 1998.0

    def test_int_sub_negative(self):
        assert IntSub(tensor(3.0), tensor(10.0)).item() == -7.0

    def test_int_mul(self):
        assert IntMul(tensor(12.0), tensor(11.0)).item() == 132.0

    def test_int_add_batch(self):
        a = tensor([1.0, 47.0, 999.0])
        b = tensor([1.0, 68.0, 1.0])
        expected = tensor([2.0, 115.0, 1000.0])
        assert torch.equal(IntAdd(a, b), expected)


# ===================================================================
# Temporal Programs
# ===================================================================

class TestTimeAdd:
    """TimeAdd: (h1:m1) + (h2:m2) → (result_h, result_m, day_overflow)."""

    def test_simple(self):
        """9:58 + 0:05 = 10:03, no overflow."""
        h, m, overflow = TimeAdd(tensor(9.0), tensor(58.0), tensor(0.0), tensor(5.0))
        assert h.item() == 10.0
        assert m.item() == 3.0
        assert overflow.item() == 0.0

    def test_hour_carry(self):
        """9:30 + 1:00 = 10:30."""
        h, m, overflow = TimeAdd(tensor(9.0), tensor(30.0), tensor(1.0), tensor(0.0))
        assert h.item() == 10.0
        assert m.item() == 30.0
        assert overflow.item() == 0.0

    def test_minute_carry(self):
        """9:45 + 0:30 = 10:15."""
        h, m, overflow = TimeAdd(tensor(9.0), tensor(45.0), tensor(0.0), tensor(30.0))
        assert h.item() == 10.0
        assert m.item() == 15.0
        assert overflow.item() == 0.0

    def test_day_overflow(self):
        """23:50 + 0:15 = 00:05, day overflow."""
        h, m, overflow = TimeAdd(tensor(23.0), tensor(50.0), tensor(0.0), tensor(15.0))
        assert h.item() == 0.0
        assert m.item() == 5.0
        assert overflow.item() == 1.0

    def test_midnight(self):
        """23:59 + 0:01 = 00:00, day overflow."""
        h, m, overflow = TimeAdd(tensor(23.0), tensor(59.0), tensor(0.0), tensor(1.0))
        assert h.item() == 0.0
        assert m.item() == 0.0
        assert overflow.item() == 1.0

    def test_large_duration(self):
        """10:00 + 15:30 = 01:30, day overflow."""
        h, m, overflow = TimeAdd(tensor(10.0), tensor(0.0), tensor(15.0), tensor(30.0))
        assert h.item() == 1.0
        assert m.item() == 30.0
        assert overflow.item() == 1.0

    def test_zero_add(self):
        """14:20 + 0:00 = 14:20."""
        h, m, overflow = TimeAdd(tensor(14.0), tensor(20.0), tensor(0.0), tensor(0.0))
        assert h.item() == 14.0
        assert m.item() == 20.0
        assert overflow.item() == 0.0

    def test_batch(self):
        """Batched operation."""
        h1 = tensor([9.0, 23.0, 0.0])
        m1 = tensor([58.0, 50.0, 0.0])
        h2 = tensor([0.0, 0.0, 0.0])
        m2 = tensor([5.0, 15.0, 0.0])
        h, m, ov = TimeAdd(h1, m1, h2, m2)
        assert torch.equal(h, tensor([10.0, 0.0, 0.0]))
        assert torch.equal(m, tensor([3.0, 5.0, 0.0]))
        assert torch.equal(ov, tensor([0.0, 1.0, 0.0]))

    def test_exhaustive_sample(self):
        """1000 random time additions match Python arithmetic."""
        import random
        random.seed(42)
        for _ in range(1000):
            h1, m1 = random.randint(0, 23), random.randint(0, 59)
            h2, m2 = random.randint(0, 23), random.randint(0, 59)

            h, m, ov = TimeAdd(tensor(float(h1)), tensor(float(m1)),
                               tensor(float(h2)), tensor(float(m2)))

            total_m = m1 + m2
            exp_m = total_m % 60
            m_carry = total_m // 60
            total_h = h1 + h2 + m_carry
            exp_h = total_h % 24
            exp_ov = total_h // 24

            assert m.item() == exp_m, f"{h1}:{m1} + {h2}:{m2}: min got {m.item()}, expected {exp_m}"
            assert h.item() == exp_h, f"{h1}:{m1} + {h2}:{m2}: hr got {h.item()}, expected {exp_h}"
            assert ov.item() == exp_ov, f"{h1}:{m1} + {h2}:{m2}: ov got {ov.item()}, expected {exp_ov}"


class TestTimeSub:
    """TimeSub: (h1:m1) - (h2:m2) → (result_h, result_m, day_underflow)."""

    def test_simple(self):
        """10:30 - 0:15 = 10:15."""
        h, m, uf = TimeSub(tensor(10.0), tensor(30.0), tensor(0.0), tensor(15.0))
        assert h.item() == 10.0
        assert m.item() == 15.0
        assert uf.item() == 0.0

    def test_minute_borrow(self):
        """10:10 - 0:30 = 9:40."""
        h, m, uf = TimeSub(tensor(10.0), tensor(10.0), tensor(0.0), tensor(30.0))
        assert h.item() == 9.0
        assert m.item() == 40.0
        assert uf.item() == 0.0

    def test_day_underflow(self):
        """00:10 - 0:30 = 23:40, day underflow."""
        h, m, uf = TimeSub(tensor(0.0), tensor(10.0), tensor(0.0), tensor(30.0))
        assert h.item() == 23.0
        assert m.item() == 40.0
        assert uf.item() == 1.0

    def test_midnight_sub(self):
        """00:00 - 0:01 = 23:59, day underflow."""
        h, m, uf = TimeSub(tensor(0.0), tensor(0.0), tensor(0.0), tensor(1.0))
        assert h.item() == 23.0
        assert m.item() == 59.0
        assert uf.item() == 1.0

    def test_exact_zero(self):
        """10:30 - 10:30 = 00:00, no underflow."""
        h, m, uf = TimeSub(tensor(10.0), tensor(30.0), tensor(10.0), tensor(30.0))
        assert h.item() == 0.0
        assert m.item() == 0.0
        assert uf.item() == 0.0


class TestDurationBetween:
    """DurationBetween(h1, m1, h2, m2) → (hours, minutes, total_minutes)."""

    def test_simple(self):
        """9:00 → 17:30 = 8h 30m = 510m."""
        hrs, mins, total = DurationBetween(tensor(9.0), tensor(0.0), tensor(17.0), tensor(30.0))
        assert hrs.item() == 8.0
        assert mins.item() == 30.0
        assert total.item() == 510.0

    def test_same_time(self):
        """10:00 → 10:00 = 0h 0m = 0m."""
        hrs, mins, total = DurationBetween(tensor(10.0), tensor(0.0), tensor(10.0), tensor(0.0))
        assert hrs.item() == 0.0
        assert mins.item() == 0.0
        assert total.item() == 0.0

    def test_wrap_around(self):
        """22:00 → 06:00 (next day) = 8h 0m = 480m."""
        hrs, mins, total = DurationBetween(tensor(22.0), tensor(0.0), tensor(6.0), tensor(0.0))
        assert hrs.item() == 8.0
        assert mins.item() == 0.0
        assert total.item() == 480.0


# ===================================================================
# Comparison Programs
# ===================================================================

class TestComparison:

    def test_greater_true(self):
        assert Greater(tensor(7.0), tensor(3.0)).item() == 1.0

    def test_greater_false(self):
        assert Greater(tensor(3.0), tensor(7.0)).item() == 0.0

    def test_greater_equal_input(self):
        assert Greater(tensor(5.0), tensor(5.0)).item() == 0.0

    def test_less_true(self):
        assert Less(tensor(3.0), tensor(7.0)).item() == 1.0

    def test_less_false(self):
        assert Less(tensor(7.0), tensor(3.0)).item() == 0.0

    def test_equal_true(self):
        assert Equal(tensor(5.0), tensor(5.0)).item() == 1.0

    def test_equal_false(self):
        assert Equal(tensor(5.0), tensor(6.0)).item() == 0.0

    def test_greater_eq_true(self):
        assert GreaterEq(tensor(7.0), tensor(7.0)).item() == 1.0
        assert GreaterEq(tensor(8.0), tensor(7.0)).item() == 1.0

    def test_greater_eq_false(self):
        assert GreaterEq(tensor(6.0), tensor(7.0)).item() == 0.0

    def test_less_eq_true(self):
        assert LessEq(tensor(7.0), tensor(7.0)).item() == 1.0
        assert LessEq(tensor(6.0), tensor(7.0)).item() == 1.0

    def test_less_eq_false(self):
        assert LessEq(tensor(8.0), tensor(7.0)).item() == 0.0

    @pytest.mark.parametrize("a,b", [(i, j) for i in range(-10, 11) for j in range(-10, 11)])
    def test_exhaustive_greater(self, a, b):
        expected = 1.0 if a > b else 0.0
        assert Greater(tensor(float(a)), tensor(float(b))).item() == expected

    @pytest.mark.parametrize("a,b", [(i, j) for i in range(-10, 11) for j in range(-10, 11)])
    def test_exhaustive_equal(self, a, b):
        expected = 1.0 if a == b else 0.0
        assert Equal(tensor(float(a)), tensor(float(b))).item() == expected


# ===================================================================
# Gradient Flow
# ===================================================================

class TestGradientFlow:
    """Verify gradients flow through all programs."""

    def test_mod_gradient(self):
        x = tensor(125.0, requires_grad=True)
        y = Mod(x, 60.0)
        y.backward()
        assert x.grad is not None and not torch.isnan(x.grad)

    def test_time_add_gradient(self):
        h1 = tensor(9.0, requires_grad=True)
        m1 = tensor(58.0, requires_grad=True)
        h, m, _ = TimeAdd(h1, m1, tensor(0.0), tensor(5.0))
        loss = h + m
        loss.backward()
        assert h1.grad is not None and not torch.isnan(h1.grad)
        assert m1.grad is not None and not torch.isnan(m1.grad)

    def test_greater_gradient(self):
        a = tensor(7.0, requires_grad=True)
        y = Greater(a, tensor(3.0))
        y.backward()
        assert a.grad is not None and not torch.isnan(a.grad)
