# Compute Layer — A RISC ALU for Neural Networks

Exact arithmetic embedded as a differentiable layer in the neural computation graph. 5 primitives. Zero learnable parameters. Everything else is a program.

**Status:** ALU + 3 program families verified (969/969 tests). See [ADR-002](../../brain/decisions/002-neural-compute-layer.md).

## The 5 Primitives

| # | Primitive | Implementation | STE? |
|---|-----------|---------------|------|
| 1 | `Add(a, b)` | `a + b` | No |
| 2 | `Sub(a, b)` | `a - b` | No |
| 3 | `Mul(a, b)` | `a * b` | No |
| 4 | `Div(a, b)` | `a / b` | No |
| 5 | `Floor(x)` | `floor_ste(x)` | **Yes** — the single discretisation point |

Four free tensor operations. One STE. That's the entire instruction set.

`Floor` is what makes integers possible. Every integer operation LLMs get wrong — modular arithmetic, carry propagation, time rollover — reduces to arithmetic + `Floor`.

## Programs

Programs are fixed compositions of the 5 primitives. Zero learnable parameters. Adding a new program = writing a Python function.

| Program | Wolfram Equivalent | Floors | Deficiency |
|---------|-------------------|--------|------------|
| `Mod(x, n)` | `Mod[125, 60] → 5` | 1 | Mathematical |
| `FloorDiv(x, n)` | `Quotient[125, 60] → 2` | 1 | Mathematical |
| `IntAdd(a, b)` | `Plus[47, 68] → 115` | 0 | Mathematical |
| `IntSub(a, b)` | `Subtract[68, 47] → 21` | 0 | Mathematical |
| `TimeAdd(h,m,dh,dm)` | `DatePlus[{9,58}, {0,5}] → {10,3}` | 4 | Temporal |
| `TimeSub(h,m,dh,dm)` | `DatePlus[{10,30}, {0,-15}] → {10,15}` | 4 | Temporal |
| `DurationBetween(...)` | `DateDifference[{9,0}, {17,30}] → 510` | 2 | Temporal |
| `Greater(a, b)` | `Boole[Greater[7, 3]] → 1` | 1 | Logical |
| `Less(a, b)` | `Boole[Less[3, 7]] → 1` | 1 | Logical |
| `Equal(a, b)` | `Boole[Equal[5, 5]] → 1` | 1 | Logical |

## Usage

```python
from src.compute_layer.alu import Add, Floor, Mod, FloorDiv
from src.compute_layer.programs.temporal import TimeAdd

# ALU primitives
Mod(tensor(125.0), 60.0)   # → tensor(5.0)

# Temporal program
h, m, overflow = TimeAdd(tensor(9.0), tensor(58.0), tensor(0.0), tensor(5.0))
# → h=10, m=3, overflow=0
```

## Module Map

```
src/compute_layer/
├── alu.py                   # 5 primitives + Mod/FloorDiv
├── programs/
│   ├── arithmetic.py        # IntAdd, IntSub, IntMul
│   ├── temporal.py          # TimeAdd, TimeSub, DurationBetween
│   └── comparison.py        # Greater, Less, Equal, GreaterEq, LessEq
│
├── gates.py                 # Phase Zero: binary logic gates (proof of tech)
├── arithmetic.py            # Phase Zero: STE wrappers (floor_ste, etc.)
├── adders.py                # Phase Zero: binary adders (proof of tech)
├── circuit_layer.py         # Phase Zero: binary CircuitLayer
└── proof_model.py           # Phase Zero: Stage A/B/C proof models
```

## Verification

```bash
# All tests (969 — 1.6 seconds)
python -m pytest tests/test_circuit_gates.py tests/test_alu_programs.py -v

# Phase Zero only (28 tests)
python -m pytest tests/test_circuit_gates.py -v

# ALU + programs only (941 tests)
python -m pytest tests/test_alu_programs.py -v
```

## Architecture Constraint: The Detach Pattern

Circuit/ALU output must be `.detach()`-ed before downstream classification heads. Discovered in Phase Zero — STE gradients through the computation pipeline corrupt CE loss. The learned interface (router, extractor, injector) trains via separate gradient paths:

1. **Output heads** ← CE loss ← `alu_output.detach()`
2. **Value extractor** ← MSE auxiliary loss ← ground-truth operands
