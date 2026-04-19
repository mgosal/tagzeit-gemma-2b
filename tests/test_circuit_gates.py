"""
Test Suite — Exhaustive Verification of Circuit Primitives
============================================================
Tests every level of the circuit hierarchy with zero tolerance:
    Level 0: Logic gates (AND, OR, XOR, NOT)
    Level 1: Half-adder
    Level 2: Full-adder
    Level 3: 4-bit ripple-carry adder (256 combinations)
    Level 4: Decimal digit adder (100 combinations)
    + Gradient flow verification
    + Batch consistency

Total assertions: ~400+
Target: 100% pass.
"""

import unittest
import torch

from src.compute_layer.gates import AND, OR, XOR, NOT, verify_truth_table
from src.compute_layer.adders import (
    HalfAdder,
    FullAdder,
    RippleCarryAdder,
    DecimalDigitAdder,
    decimal_to_binary,
    binary_to_decimal,
)
from src.compute_layer.circuit_layer import CircuitLayer


class TestLevel0_LogicGates(unittest.TestCase):
    """Level 0: Verify each gate against its truth table."""

    def test_AND_truth_table(self):
        expected = {
            (0, 0): 0.0,
            (0, 1): 0.0,
            (1, 0): 0.0,
            (1, 1): 1.0,
        }
        self.assertTrue(verify_truth_table(AND, expected))

    def test_OR_truth_table(self):
        expected = {
            (0, 0): 0.0,
            (0, 1): 1.0,
            (1, 0): 1.0,
            (1, 1): 1.0,
        }
        self.assertTrue(verify_truth_table(OR, expected))

    def test_XOR_truth_table(self):
        expected = {
            (0, 0): 0.0,
            (0, 1): 1.0,
            (1, 0): 1.0,
            (1, 1): 0.0,
        }
        self.assertTrue(verify_truth_table(XOR, expected))

    def test_NOT_truth_table(self):
        expected = {
            (0,): 1.0,
            (1,): 0.0,
        }
        self.assertTrue(verify_truth_table(NOT, expected))

    def test_AND_gradient(self):
        """AND(a, b) = a*b → ∂/∂a = b, ∂/∂b = a"""
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        result = AND(a, b)
        result.backward()
        self.assertEqual(a.grad.item(), b.item())  # ∂/∂a = b
        self.assertEqual(b.grad.item(), a.item())  # ∂/∂b = a

    def test_OR_gradient(self):
        """OR(a, b) = a + b - a*b → ∂/∂a = 1-b, ∂/∂b = 1-a"""
        a = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        result = OR(a, b)
        result.backward()
        self.assertAlmostEqual(a.grad.item(), 1.0 - b.item())
        self.assertAlmostEqual(b.grad.item(), 1.0 - a.item())

    def test_XOR_gradient(self):
        """XOR(a, b) = a + b - 2ab → ∂/∂a = 1-2b, ∂/∂b = 1-2a"""
        a = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        result = XOR(a, b)
        result.backward()
        self.assertAlmostEqual(a.grad.item(), 1.0 - 2.0 * b.item())
        self.assertAlmostEqual(b.grad.item(), 1.0 - 2.0 * a.item())

    def test_NOT_gradient(self):
        """NOT(a) = 1-a → ∂/∂a = -1"""
        a = torch.tensor(1.0, requires_grad=True)
        result = NOT(a)
        result.backward()
        self.assertEqual(a.grad.item(), -1.0)


class TestLevel1_HalfAdder(unittest.TestCase):
    """Level 1: Verify half-adder for all 4 input combinations."""

    def setUp(self):
        self.ha = HalfAdder()

    def test_exhaustive(self):
        expected = {
            (0, 0): (0, 0),  # sum=0, carry=0
            (0, 1): (1, 0),  # sum=1, carry=0
            (1, 0): (1, 0),  # sum=1, carry=0
            (1, 1): (0, 1),  # sum=0, carry=1
        }
        for (a_val, b_val), (exp_sum, exp_carry) in expected.items():
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            s, c = self.ha(a, b)
            self.assertEqual(
                s.item(), exp_sum,
                f"HalfAdder({a_val},{b_val}): sum={s.item()}, expected {exp_sum}"
            )
            self.assertEqual(
                c.item(), exp_carry,
                f"HalfAdder({a_val},{b_val}): carry={c.item()}, expected {exp_carry}"
            )

    def test_gradient_flow(self):
        """Verify gradients flow through both outputs."""
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        s, c = self.ha(a, b)
        loss = s + c
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertFalse(torch.isnan(a.grad))
        self.assertFalse(torch.isnan(b.grad))


class TestLevel2_FullAdder(unittest.TestCase):
    """Level 2: Verify full-adder for all 8 input combinations."""

    def setUp(self):
        self.fa = FullAdder()

    def test_exhaustive(self):
        # All 2^3 = 8 combinations of (a, b, cin) → (sum, cout)
        expected = {
            (0, 0, 0): (0, 0),
            (0, 0, 1): (1, 0),
            (0, 1, 0): (1, 0),
            (0, 1, 1): (0, 1),
            (1, 0, 0): (1, 0),
            (1, 0, 1): (0, 1),
            (1, 1, 0): (0, 1),
            (1, 1, 1): (1, 1),
        }
        for (a_val, b_val, cin_val), (exp_sum, exp_cout) in expected.items():
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            cin = torch.tensor(float(cin_val))
            s, cout = self.fa(a, b, cin)
            self.assertEqual(
                s.item(), exp_sum,
                f"FullAdder({a_val},{b_val},{cin_val}): sum={s.item()}, expected {exp_sum}"
            )
            self.assertEqual(
                cout.item(), exp_cout,
                f"FullAdder({a_val},{b_val},{cin_val}): cout={cout.item()}, expected {exp_cout}"
            )

    def test_gradient_flow(self):
        """Verify gradients flow through all three inputs."""
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        cin = torch.tensor(0.0, requires_grad=True)
        s, cout = self.fa(a, b, cin)
        loss = s + cout
        loss.backward()
        for name, param in [("a", a), ("b", b), ("cin", cin)]:
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad), f"NaN gradient for {name}")


class TestLevel3_RippleCarryAdder(unittest.TestCase):
    """Level 3: Verify 4-bit adder for ALL 256 input combinations."""

    def setUp(self):
        self.rca = RippleCarryAdder(n_bits=4)

    def test_exhaustive_4bit(self):
        """Test all 16 × 16 = 256 additions of 4-bit numbers."""
        failures = []
        for a_val in range(16):
            for b_val in range(16):
                expected = a_val + b_val

                a_bits = decimal_to_binary(
                    torch.tensor(float(a_val)), n_bits=4
                ).unsqueeze(0)
                b_bits = decimal_to_binary(
                    torch.tensor(float(b_val)), n_bits=4
                ).unsqueeze(0)

                result_bits = self.rca(a_bits, b_bits)
                result = binary_to_decimal(result_bits).item()

                if result != expected:
                    failures.append(
                        f"  {a_val} + {b_val} = {int(result)} (expected {expected})"
                    )

        if failures:
            msg = f"{len(failures)} failures out of 256:\n" + "\n".join(failures[:20])
            self.fail(msg)

    def test_batch_consistency(self):
        """Same results regardless of batch size."""
        a_val, b_val = 13, 9
        expected = a_val + b_val

        # Single
        a1 = decimal_to_binary(torch.tensor(float(a_val)), 4).unsqueeze(0)
        b1 = decimal_to_binary(torch.tensor(float(b_val)), 4).unsqueeze(0)
        r1 = binary_to_decimal(self.rca(a1, b1)).item()

        # Batched
        a_batch = decimal_to_binary(
            torch.tensor([float(a_val)] * 8), 4
        )
        b_batch = decimal_to_binary(
            torch.tensor([float(b_val)] * 8), 4
        )
        r_batch = binary_to_decimal(self.rca(a_batch, b_batch))

        self.assertEqual(r1, expected)
        for i in range(8):
            self.assertEqual(r_batch[i].item(), expected)

    def test_gradient_flow_through_carry_chain(self):
        """Verify gradients propagate through the full 4-bit carry chain."""
        a_bits = decimal_to_binary(
            torch.tensor(15.0), n_bits=4
        ).unsqueeze(0).requires_grad_(True)
        b_bits = decimal_to_binary(
            torch.tensor(1.0), n_bits=4
        ).unsqueeze(0)
        # Need b_bits as leaf too
        b_bits = b_bits.clone().detach().requires_grad_(True)

        result_bits = self.rca(a_bits, b_bits)
        loss = result_bits.sum()
        loss.backward()

        self.assertIsNotNone(a_bits.grad)
        self.assertFalse(torch.any(torch.isnan(a_bits.grad)))
        self.assertIsNotNone(b_bits.grad)
        self.assertFalse(torch.any(torch.isnan(b_bits.grad)))


class TestLevel4_DecimalDigitAdder(unittest.TestCase):
    """Level 4: Verify decimal digit adder for ALL 100 input combinations."""

    def setUp(self):
        self.dda = DecimalDigitAdder()

    def test_exhaustive_single_digit(self):
        """Test all 10 × 10 = 100 single-digit additions."""
        failures = []
        for a in range(10):
            for b in range(10):
                expected_sum = (a + b) % 10
                expected_carry = (a + b) // 10

                digit_a = torch.tensor(float(a))
                digit_b = torch.tensor(float(b))
                sum_digit, carry = self.dda(digit_a, digit_b)

                s = sum_digit.item()
                c = carry.item()

                if s != expected_sum or c != expected_carry:
                    failures.append(
                        f"  {a} + {b}: got sum={int(s)},carry={int(c)} "
                        f"(expected sum={expected_sum},carry={expected_carry})"
                    )

        if failures:
            msg = f"{len(failures)} failures out of 100:\n" + "\n".join(failures[:20])
            self.fail(msg)

    def test_batched(self):
        """Verify batched execution produces same results."""
        a_vals = torch.arange(10, dtype=torch.float)
        b_vals = torch.full((10,), 7.0)

        sum_digits, carries = self.dda(a_vals, b_vals)

        for i in range(10):
            expected_sum = (i + 7) % 10
            expected_carry = (i + 7) // 10
            self.assertEqual(
                sum_digits[i].item(), expected_sum,
                f"Batched {i}+7: sum={sum_digits[i].item()}, expected {expected_sum}"
            )
            self.assertEqual(
                carries[i].item(), expected_carry,
                f"Batched {i}+7: carry={carries[i].item()}, expected {expected_carry}"
            )

    def test_edge_cases(self):
        """Specific edge cases for digit addition."""
        cases = [
            (0, 0, 0, 0),   # zero + zero
            (9, 9, 8, 1),   # max carry
            (5, 5, 0, 1),   # exact 10
            (9, 1, 0, 1),   # exact 10 from 9+1
            (1, 9, 0, 1),   # commutative check
            (0, 9, 9, 0),   # identity element
        ]
        for a, b, exp_sum, exp_carry in cases:
            s, c = self.dda(torch.tensor(float(a)), torch.tensor(float(b)))
            self.assertEqual(s.item(), exp_sum, f"{a}+{b} sum")
            self.assertEqual(c.item(), exp_carry, f"{a}+{b} carry")

    def test_gradient_flow(self):
        """Verify gradients flow end-to-end through the decimal circuit."""
        a = torch.tensor(7.0, requires_grad=True)
        b = torch.tensor(8.0, requires_grad=True)
        s, c = self.dda(a, b)
        loss = s + c
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        # Gradients may be zero for specific inputs (floor/mod have zero grad
        # in standard autograd), but they should not be NaN
        self.assertFalse(torch.isnan(a.grad), "NaN gradient for a")
        self.assertFalse(torch.isnan(b.grad), "NaN gradient for b")


class TestCircuitLayer(unittest.TestCase):
    """Test the CircuitLayer nn.Module wrapper."""

    def setUp(self):
        self.layer = CircuitLayer()

    def test_forward_shape(self):
        x = torch.tensor([[3.0, 4.0], [9.0, 9.0]])
        y = self.layer(x)
        self.assertEqual(y.shape, (2, 2))

    def test_forward_values(self):
        x = torch.tensor([[3.0, 4.0]])
        y = self.layer(x)
        self.assertEqual(y[0, 0].item(), 7.0)   # sum_digit
        self.assertEqual(y[0, 1].item(), 0.0)   # carry

        x = torch.tensor([[9.0, 9.0]])
        y = self.layer(x)
        self.assertEqual(y[0, 0].item(), 8.0)   # sum_digit (18 % 10)
        self.assertEqual(y[0, 1].item(), 1.0)   # carry (18 // 10)

    def test_no_learnable_params(self):
        """CircuitLayer must have zero learnable parameters."""
        n_params = sum(p.numel() for p in self.layer.parameters())
        self.assertEqual(n_params, 0)

    def test_exhaustive_via_layer(self):
        """All 100 digit additions through the CircuitLayer interface."""
        inputs = []
        for a in range(10):
            for b in range(10):
                inputs.append([float(a), float(b)])

        x = torch.tensor(inputs)
        y = self.layer(x)

        for idx, (a, b) in enumerate(inputs):
            a_int, b_int = int(a), int(b)
            expected_sum = (a_int + b_int) % 10
            expected_carry = (a_int + b_int) // 10
            self.assertEqual(
                y[idx, 0].item(), expected_sum,
                f"CircuitLayer: {a_int}+{b_int} sum"
            )
            self.assertEqual(
                y[idx, 1].item(), expected_carry,
                f"CircuitLayer: {a_int}+{b_int} carry"
            )


class TestConversions(unittest.TestCase):
    """Test decimal ↔ binary conversion helpers."""

    def test_decimal_to_binary(self):
        # 13 = 1101 in binary → bits LSB-first: [1, 0, 1, 1]
        bits = decimal_to_binary(torch.tensor(13.0), n_bits=4)
        self.assertEqual(bits.tolist(), [1.0, 0.0, 1.0, 1.0])

    def test_binary_to_decimal(self):
        bits = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 13 in LSB-first
        val = binary_to_decimal(bits)
        self.assertEqual(val.item(), 13.0)

    def test_roundtrip(self):
        """decimal → binary → decimal should be identity for [0, 15]."""
        for v in range(16):
            bits = decimal_to_binary(torch.tensor(float(v)), n_bits=4)
            result = binary_to_decimal(bits)
            self.assertEqual(
                result.item(), float(v),
                f"Roundtrip failed for {v}: got {result.item()}"
            )

    def test_zero(self):
        bits = decimal_to_binary(torch.tensor(0.0), n_bits=4)
        self.assertEqual(bits.tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_max(self):
        bits = decimal_to_binary(torch.tensor(15.0), n_bits=4)
        self.assertEqual(bits.tolist(), [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
