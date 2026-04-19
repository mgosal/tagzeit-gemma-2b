# Experiment 011: Neural Compute Layer

## Goal

Embed exact arithmetic into neural networks as differentiable computation layers. Eliminate the external Luxon engine dependency from Route-to-Luxon.

## Phases

### Phase Zero: Proof of Technology ✅

**Can a PyTorch `nn.Module` encode exact digital circuits?**

Built binary logic gates (`AND = a*b`, `XOR = a+b-2ab`) and composed them into adders → ripple-carry → decimal digit addition. A transformer learned to route inputs through the circuit and decode results.

| Stage | Params | 100% at | Time |
|-------|--------|---------|------|
| A (direct values → circuit) | 36 | epoch 4,000 | 4.5s |
| B (learned embeddings → circuit) | 46 | epoch 4,000 | 5.0s |
| C (transformer → circuit) | 72,678 | epoch 9,500 | 79.2s |

**Key discovery:** Circuit output must be `.detach()`-ed before output heads. STE gradient accumulation through the binary pipeline corrupts CE loss.

### RISC ALU: 5-Primitive Architecture ✅

Phase Zero proved binary circuits work but are fragile (60 ops, 6+ STEs per digit). The production architecture reduces to 5 primitives:

| Primitive | Implementation | STE? |
|-----------|---------------|------|
| `Add(a, b)` | `a + b` | No |
| `Sub(a, b)` | `a - b` | No |
| `Mul(a, b)` | `a * b` | No |
| `Div(a, b)` | `a / b` | No |
| `Floor(x)` | `floor_ste(x)` | **Yes** |

Everything else is a **program** — a fixed composition of these 5 with zero learnable parameters.

#### Programs Implemented

| Program | Wolfram | Floors | Deficiency |
|---------|---------|--------|------------|
| `Mod(x, n)` | `Mod[125, 60] → 5` | 1 | Mathematical |
| `FloorDiv(x, n)` | `Quotient[125, 60] → 2` | 1 | Mathematical |
| `IntAdd(a, b)` | `Plus[47, 68] → 115` | 0 | Mathematical |
| `TimeAdd(h,m,dh,dm)` | `DatePlus[{9,58},{0,5}] → {10,3}` | 4 | Temporal |
| `TimeSub(h,m,dh,dm)` | `DatePlus[{10,30},{0,-15}]` | 4 | Temporal |
| `DurationBetween(...)` | `DateDifference[...]` | 2 | Temporal |
| `Greater(a, b)` | `Boole[Greater[7,3]] → 1` | 1 | Logical |
| `Equal(a, b)` | `Boole[Equal[5,5]] → 1` | 1 | Logical |

#### Multi-Program Dispatch ✅

A 30K-parameter model routes between IntAdd, Mod, and Greater on 600 synthetic examples.

| Metric | Mean | Std | Description |
|--------|------|-----|-------------|
| **E2E accuracy** | **92.6%** | ±1.2% | Router selects program via argmax → extractor → ALU |
| Oracle accuracy | 99.4% | ±0.9% | Ground-truth program ID → extractor → ALU |
| Router accuracy | 93.2% | ±0.4% | Router argmax matches ground-truth label |

Results aggregated over 5 seeds (42, 123, 7, 2024, 31415). E2E accuracy is the primary metric.

**Per-program E2E accuracy (mean ± std):**

| Program | E2E | Notes |
|---------|-----|-------|
| IntAdd | 90.5% ± 2.1% | Confused with Greater by router (~10%) |
| Mod | 98.7% ± 2.9% | Fixed modulus (n=10) simplifies extraction |
| Greater | 88.7% ± 2.3% | Confused with IntAdd by router (~11%) |

#### Evaluation Protocol

Two accuracy metrics are reported to avoid conflating operand extraction with routing:

- **E2E accuracy** (`program_ids=None`): The router selects which program to execute via `argmax(router_logits)`. If it picks the wrong program, the ALU runs the wrong computation even if operand extraction is perfect. This is the honest end-to-end measure.

- **Oracle accuracy** (`program_ids=ground_truth`): Ground-truth program labels bypass the router. This isolates operand extraction quality — useful as a diagnostic but not a valid claim of system performance.

The gap between oracle (99.4%) and E2E (92.6%) quantifies the cost of imperfect routing: ~7% of examples are routed to the wrong program.

#### Ablation: Removing Operand Supervision

To test whether the system requires auxiliary operand MSE loss (credit-assignment question):

| Condition | E2E | Oracle | Router |
|-----------|-----|--------|--------|
| Full (router CE + operand MSE) | 92.6% ± 1.2% | 99.4% ± 0.9% | 93.2% ± 0.4% |
| **No aux (router CE only)** | **17.9% ± 4.8%** | **18.2% ± 4.7%** | **94.3% ± 1.0%** |

**Finding:** Removing operand supervision collapses E2E accuracy to chance (~18%) while the router still learns (~94%). The extractors receive zero gradient without aux MSE because the ALU output is `.detach()`-ed. This confirms:

1. The system is **not end-to-end differentiable** in any meaningful sense. Routing and extraction are trained by separate supervised losses, not by backpropagation through the ALU.
2. Auxiliary operand supervision is **load-bearing**, not optional.
3. The `.detach()` wall prevents credit assignment from the final output back through the symbolic executor to the operand extractors.

#### Calibration: Decomposing the E2E Gap

Of the ~7% E2E error, how much is pure routing failure vs accidental agreement (wrong program, right answer)?

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Router ✓ Answer ✓ | 557 | 92.8% | System works correctly |
| Router ✓ Answer ✗ | 0 | 0.0% | Extraction error despite correct routing |
| Router ✗ Answer ✓ | 1 | 0.2% | Accidental agreement (lucky misroute) |
| Router ✗ Answer ✗ | 42 | 7.0% | True routing failure |

**Finding:** Virtually all E2E errors (42/43 = 97.7%) are true routing failures — wrong program → wrong answer. Accidental agreement is negligible (1 case). And when the router is correct, extraction is perfect (0 category-B errors), confirming that the ~0.6% oracle failure in some seeds is marginal and seed-dependent, not a systematic extraction problem.

**Router confusion matrix** (seed 42):

| True \ Predicted | IntAdd | Mod | Greater |
|------------------|--------|-----|---------|
| IntAdd | 184 | 2 | 14 |
| Mod | 0 | 200 | 0 |
| Greater | 22 | 5 | 173 |

The dominant confusion is **Greater ↔ IntAdd** (22 + 14 = 36 cross-misroutes). Mod is perfectly routed. This makes sense: IntAdd and Greater both take two positive integers as operands, while Mod has a distinctive fixed second operand (always 10.0).

#### Ablation: Two-Stage Training and Aux Weight

Can we close the E2E gap by pre-training the router before joint training, or by rebalancing loss weights?

| Variant | E2E (mean ± std) | Oracle | Router |
|---------|-------------------|--------|--------|
| Baseline (joint from epoch 0) | 92.6% ± 1.2% | 99.4% | 93.2% |
| Two-stage: 5K router → 15K joint | 91.2% ± 6.5% | 97.6% | 93.5% |
| **Two-stage: 10K router → 10K joint** | **93.7% ± 1.3%** | **100.0%** | **93.6%** |
| Joint, aux_weight=0.5 | 92.3% ± 2.3% | 99.6% | 92.5% |
| Joint, aux_weight=2.0 | 92.9% ± 0.7% | 100.0% | 92.9% |

**Findings:**

1. **10K→10K two-stage is the best variant** at 93.7% ± 1.3% — a +1.1% improvement over baseline, with 100% oracle accuracy (zero extraction errors). Pre-training the router to 93.6% before introducing operand supervision gives extractors a stable routing landscape.

2. **5K→5K is unstable** — one seed collapsed to 79.7%, driving the high variance (±6.5%). The router wasn't fully converged at 5K epochs before joint training began.

3. **Aux weight has marginal effect.** Doubling or halving the operand MSE weight changes E2E by <1%. The router ceiling (~93%) is the binding constraint, not extraction quality.

4. **The bottleneck is routing, not extraction.** All variants with aux supervision achieve 99-100% oracle accuracy. The E2E ceiling is set by router accuracy, which plateaus around 93-95% regardless of training schedule.

## Test Results

```
tests/test_circuit_gates.py   28 passed    (Phase Zero circuits)
tests/test_alu_programs.py   941 passed    (ALU + programs)
─────────────────────────────────────────
Total:                       969 passed in 1.40s
```

## Key Findings

1. **Floor is the only special primitive.** The other 4 are standard PyTorch ops. Every integer computation LLMs fail at reduces to arithmetic + Floor.

2. **Div needs a zero guard.** During early training, extractors produce near-zero values → NaN in Mod. Fix: epsilon clamp on denominators.

3. **Mod is discontinuous in its modulus.** `Mod(125, 60)=5` vs `Mod(125, 59)=7`. The extractor must produce exact integers. Fixed-modulus Mod converges; variable moduli is a harder extraction problem.

4. **RISC beats CISC.** 5 primitives + composition vs dedicated binary circuits per operation. Fewer STEs, simpler gradients, same result accuracy.

5. **E2E accuracy ≠ oracle accuracy.** The original 100% result was measured with oracle routing. Under router-selected dispatch, accuracy drops to ~93% because ~7% of examples are mis-routed.

6. **Operand supervision is load-bearing.** Without aux MSE, E2E accuracy collapses to ~18% despite the router learning to ~94%. The `.detach()` wall means extractors have no gradient signal from the final output.

## Architecture

```
Hidden state (d_model)
         │
         ├─── Router ──────────► program_id (CE loss)
         │    Linear(d, N)
         │
         ├─── Extractor ───────► operand values (MSE aux loss)
         │    Per-program MLPs
         │
         ▼
    ┌──────────┐
    │ ALU      │  5 primitives, 0 params
    │ Program  │  Execute: Mod, TimeAdd, Greater, etc.
    └────┬─────┘
         │
         │ .detach()     ← gradient stops here
         │
         ▼
    Exact result (injected back or used directly)
```

**What is differentiable vs what is supervised:**
- Router → CE loss against ground-truth program labels
- Extractors → MSE loss against ground-truth operand values
- ALU → zero parameters, zero gradients, pure symbolic execution

The system is honest neuro-symbolic routing + supervised operand parsing + symbolic execution. The ALU provides *exact* arithmetic but requires *exact* operands and *correct* routing, both supplied by supervised learning.

## Reproducing

```bash
# All tests (< 2 seconds)
python -m pytest tests/test_circuit_gates.py tests/test_alu_programs.py -v

# Phase Zero training (~90s CPU)
python experiments/neural-compute/train_phase_zero.py --stage all

# Multi-program proof, 5 seeds (~4 min CPU)
python experiments/neural-compute/train_multi_program.py \
    --epochs 20000 --lr 0.001 --d_model 128 \
    --seeds 42,123,7,2024,31415

# Ablation: no operand supervision (~4 min CPU)
python experiments/neural-compute/train_multi_program_ablation.py \
    --epochs 20000 --lr 0.001 --d_model 128 \
    --seeds 42,123,7,2024,31415

# Two-stage and aux-weight ablation sweep (~20 min CPU)
python experiments/neural-compute/train_two_stage_ablation.py \
    --lr 0.001 --d_model 128 --seeds 42,123,7,2024,31415

# Failure diagnostics (single seed, ~1 min CPU)
python experiments/neural-compute/diagnose_failures.py \
    --seed 42 --epochs 20000 --lr 0.001 --d_model 128
```

## Files

```
src/compute_layer/
├── alu.py                    5 RISC primitives + Mod/FloorDiv
├── programs/
│   ├── arithmetic.py         IntAdd, IntSub, IntMul
│   ├── temporal.py           TimeAdd, TimeSub, DurationBetween
│   └── comparison.py         Greater, Less, Equal, GreaterEq, LessEq
├── interface/
│   └── __init__.py           (neural interface package)
├── layer.py                  ComputeLayer — router + extractors + dispatch
├── gates.py                  Phase Zero: binary logic gates
├── arithmetic.py             Phase Zero: STE wrappers
├── adders.py                 Phase Zero: binary adders
├── circuit_layer.py          Phase Zero: CircuitLayer
└── proof_model.py            Phase Zero: Stage A/B/C

tests/
├── test_circuit_gates.py     28 Phase Zero tests
└── test_alu_programs.py      941 ALU + program tests

experiments/neural-compute/
├── README.md                 This file
├── train_phase_zero.py       Phase Zero training
├── train_multi_program.py    Multi-program dispatch proof (5-seed, dual-metric)
├── train_multi_program_ablation.py  Ablation: no operand supervision
├── train_two_stage_ablation.py      Ablation: two-stage training + aux weight sweep
├── diagnose_failures.py             Calibration decomposition + confusion matrix
└── checkpoints/              Saved results

brain/decisions/
├── 002-neural-compute-layer.md   Phase Zero ADR
└── 003-risc-alu.md               RISC ALU ADR
```

## Next Steps

| Step | What | Status |
|------|------|--------|
| Router improvement | Dedicated router pre-training or curriculum to push E2E past 93% | Next |
| Variable-modulus extraction | Extractor produces exact integers from a discrete set | Research problem |
| Temporal engine parity | Cross-reference TimeAdd/TimeSub against Luxon output | Planned |
| Llama 3.2-1B integration | Inject ComputeLayer into the transformer | Phase 4 |
| Baselines | Compare against tool-call, CoT scratchpad, LoRA-on-arithmetic | Required for paper |
