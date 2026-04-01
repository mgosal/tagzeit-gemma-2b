# Experiment 009: Llama 3.2-1B-Instruct Smoke Test (250 steps)

**Date:** 2026-04-01
**Model:** meta-llama/Llama-3.2-1B-Instruct (1.24B params)
**Runtime:** Google Colab L4 (24GB VRAM)

## Configuration

| Parameter | Value |
|-----------|-------|
| Data | 5K samples (v2 generator: formulaic + hard negatives + edge cases) |
| Distribution | 30/30/15/25 (ADD/SUB/BETWEEN/NO_ROUTE) |
| Steps | 250 |
| Effective batch | 16 (2 x 8) |
| Epochs | ~1.0 |
| Precision | bf16 (L4 native) |
| LR | 5e-5 |
| Optimizer | adamw_8bit |
| Gradient checkpointing | Yes |

## Training Results

| Step | Train Loss | Eval Loss |
|------|-----------|-----------|
| 125 | 0.516 | 0.477 |
| 250 | 0.421 | 0.407 |

**Final train loss:** 0.789 (epoch average)
**Final eval loss:** 0.407
**Eval perplexity:** 1.50
**Runtime:** 522s (8.7 min)

## Inference Results (6/6 = 100% operation routing)

| Test | Expected | Got | Status |
|------|----------|-----|--------|
| What time is it 1 minute after 23:59? | ADD | ROUTE_TIME_ADD | PASS |
| What time was it 30 minutes before 00:15? | SUB | ROUTE_TIME_SUB | PASS |
| How much time between 09:00 and 17:30? | BETWEEN | ROUTE_DURATION_BETWEEN | PASS |
| Meeting at 14:30, lasts 45 min. When end? | ADD | ROUTE_TIME_ADD | PASS |
| What is 42 + 18? | NO_ROUTE | NO_ROUTE (with reasoning) | PASS |
| When was The Secret Garden published? | NO_ROUTE | NO_ROUTE (with reasoning) | PASS |

## Observations

### What works
- **Operation routing: 100%** at 250 steps. Llama learns the grammar immediately.
- **NO_ROUTE discrimination: perfect.** Both hard negatives rejected with correct reasoning
  ("base-10 arithmetic, not temporal" and "factual knowledge lookup, not temporal arithmetic").
- **Grammar structure correct:** [ROUTE] [OP] [HEAD_*] [ARG_*] [/ROUTE]

### What needs more training (argument precision)
- Minute extraction off: "23:59" extracted as ARG_MIN_00 (should be 59)
- "14:30" extracted as ARG_MIN_00 (should be 30)
- Duration values imprecise: "1 minute" became ARG_HOUR_01 ARG_MIN_02
- Trailing token repetition after [/ROUTE] closure

### Comparison to SmolLM2-360M

| Model | Steps | Operation Routing |
|-------|-------|-------------------|
| SmolLM2-360M (Exp 004) | 500 | 12.5% |
| SmolLM2-360M (Exp 004) | 5,000 | 25.0% |
| SmolLM2-360M (Exp 008b) | 10,000 | 2.1% |
| **Llama 3.2-1B (Exp 009)** | **250** | **100%** |

The 3.4x parameter increase (362M to 1.24B) produced a qualitative jump in routing
capability. The model capacity hypothesis from the implementation plan is validated.

## Next Steps

Experiment 010: Full 5K step run with 100K data and 500-step checkpoints. Focus on
improving argument precision (minute extraction, duration values) and eliminating
trailing token repetition.
