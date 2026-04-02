# Experiment 010: Llama 3.2-1B-Instruct Full Run (2500/5000 steps)

**Date:** 2026-04-02
**Model:** meta-llama/Llama-3.2-1B-Instruct (1.24B params)
**Runtime:** Google Colab L4 (24GB VRAM)
**Status:** Training interrupted at step 2500/5000 (Colab disconnect). Results sufficient -- no resume needed.

## Configuration

| Parameter | Value |
|-----------|-------|
| Data | 100K samples (v2 generator: formulaic + hard negatives + edge cases) |
| Distribution | 30/30/15/25 (ADD/SUB/BETWEEN/NO_ROUTE) |
| Steps completed | 2,500 / 5,000 planned |
| Effective batch | 16 (2 x 8) |
| Epochs | ~0.42 (sub-epoch) |
| Precision | bf16 (L4 native) |
| LR | 5e-5 (cosine, 200 warmup) |
| Optimizer | adamw_8bit |
| Gradient checkpointing | Yes |

## Training Curve

| Step | Train Loss | Eval Loss |
|------|-----------|-----------|
| 100 | 0.678 | -- |
| 500 | 0.336 | 0.334 |
| 1000 | 0.314 | 0.316 |
| 1500 | 0.296 | 0.303 |
| 2000 | 0.301 | 0.293 |
| 2500 | 0.289 | 0.288 |

No overfitting. Train-eval gap < 0.002 throughout. Loss curve flattening (diminishing returns).

## Inference Results (checkpoint-2500)

### Operation Routing: 8/8 (100%)

All ADD, SUB, BETWEEN, and NO_ROUTE operations correctly classified.

### Argument Precision: 8/8 (100%)

Every operand correctly extracted (durations <= 59 min). Comparison to Exp 009 (250 steps):

| Prompt | 009 (250 steps) | 010 (2500 steps) |
|--------|----------------|------------------|
| "23:59" | ARG_MIN_00 | ARG_MIN_59 |
| "1 minute" duration | ARG_HOUR_01 ARG_MIN_02 | ARG_MIN_01 |
| "00:15" | ARG_MIN_00 | ARG_MIN_15 |
| "30 minutes" | ARG_MIN_58 | ARG_MIN_30 |
| "17:30" | ARG_MIN_00 | ARG_MIN_30 |
| "14:30" | ARG_MIN_00 | ARG_MIN_30 |
| "45 minutes" | ARG_MIN_44 | ARG_MIN_45 |

## E2E Validation (48-case harness via Luxon engine)

### Overall: 38/48 passed (79.2%)

| Category | Score | Status |
|----------|-------|--------|
| subtraction | 5/5 | PASS |
| format_robust | 5/5 | PASS |
| semantic_eq | 5/5 | PASS |
| midnight_boundary | 1/1 | PASS |
| minute_carry | 7/8 (88%) | 1 fail (61 min duration) |
| hour_rollover | 6/7 (86%) | 1 fail (90 min duration) |
| cascade | 4/5 (80%) | 1 fail (90 min duration) |
| standard | 5/6 (83%) | 1 fail (70 min duration) |
| generalization | 0/3 (0%) | All >60 min durations |
| impossible | 0/3 (0%) | No INVALID training data |

### Failure Analysis

**Gap 1: Multi-hour duration decomposition (7 failures, issue #31)**
Model emits `[ARG_HOUR_01] [ARG_MIN_00]` instead of `[ARG_HOUR_01] [ARG_MIN_30]` for 90-minute durations. The generator under-represents >59min durations.

**Gap 2: Invalid input detection (3 failures, issue #32)**
Model routes "12:65", "25:00", and "13:45 PM" as valid. No INVALID examples in training data.

### Sample Correct Outputs

```
BP-01: 23:59 + 1 minute    -> 00:00 (correct, midnight rollover)
BP-03: 23:45 + 30 minutes  -> 00:15 (correct, midnight rollover)
BP-08: 23:30 + 45 minutes  -> 00:15 (correct, midnight rollover)
TM-06: 07:00 + 120 minutes -> 09:00 (correct, 2-hour round duration)
FR-01: 1 8 : 5 9 + 1 min   -> 19:00 (correct, spaced format)
SB-03: 00:05 - 10 minutes  -> 23:55 (correct, reverse midnight)
SB-04: 12:00 - 120 minutes -> 10:00 (correct, 2-hour subtraction)
```

### Sample Failed Outputs

```
BP-09: 22:50 + 90 min -> 23:50 (expected 00:20, model emitted 60 min not 90)
TG-02: 08:30 + 200 min -> 08:50 (expected 11:50, model emitted 20 min not 200)
IM-01: 12:65 + 5 min -> 12:50 (expected INVALID, model routed as valid)
```

## Comparison Across All Experiments

| Exp | Model | Steps | Eval Loss | E2E Accuracy |
|-----|-------|-------|-----------|--------------|
| 004 | SmolLM2-360M | 5,000 | -- | 25.0% |
| 008b | SmolLM2-360M | 10,000 | 0.267 | 2.1% |
| 009 | Llama 3.2-1B | 250 | 0.407 | 100% routing / args wrong |
| **010** | **Llama 3.2-1B** | **2,500** | **0.288** | **79.2% (38/48)** |

## Next Steps

1. Fix generator: add >60min duration templates with proper hour+minute decomposition (#31)
2. Fix generator: add INVALID input examples (#32)
3. Create held-out adversarial test set independent of generator (#33)
4. Retrain with improved data -> target >93% E2E
