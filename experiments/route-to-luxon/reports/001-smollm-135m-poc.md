# Experiment 001: SmolLM-135M — Route-to-Luxon PoC

**Date:** 2026-03-27
**Status:** Complete
**Model:** `HuggingFaceTB/SmolLM-135M` (135M parameters)
**Architecture:** Route-to-Luxon (ADR-001)

## Objective

Validate that the SFT training pipeline works end-to-end: can a small model learn
to emit structured `[ROUTE_*]` tokens instead of attempting temporal arithmetic?

## Hypothesis

Even a 135M model should learn the routing **syntax** (structural output format).
Argument **accuracy** is expected to be low at this scale — the goal is pipeline
validation, not production-quality routing.

## Setup

| Parameter | Value |
|:----------|:------|
| Model | `HuggingFaceTB/SmolLM-135M` |
| Training data | 5,656 samples (route format) |
| Eval data | 295 samples |
| Steps | 250 |
| Batch size | 2 (effective 8 with grad_accum=4) |
| Learning rate | 3e-4 (cosine schedule) |
| Warmup | 100 steps |
| Precision | float32 (MPS) |
| Optimizer | `adamw_torch` |
| LoRA | None (full fine-tune, tiny model) |
| Domain tokens | 104 (geometric sinusoidal init) |
| Hardware | Apple M4 Pro, 24GB RAM, MPS |

## Training Data

Generated via `core/synthetic_data/generators/temporal/generator.js --count 5000`:
- **Route examples**: NL question → `[ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_XX] [ARG_MIN_XX] [HEAD_DURATION] [ARG_MIN_XX] [/ROUTE]`
- **Shadow pairs**: base-10 arithmetic → `[NO_ROUTE]` (contrastive)
- **Negative examples**: Non-temporal questions → `[NO_ROUTE]` (~5%)
- **Subtraction**: ~20% of temporal examples use `[ROUTE_TIME_SUB]`

## Results

### Training Metrics

| Metric | Value |
|:-------|:------|
| Final train loss | 0.52 |
| Eval loss | 0.52 |
| Token accuracy | 83.3% |
| Training time | 7m 20s |
| Speed | 1.76s/step |

### Inference Test (5 prompts)

| Prompt | Route emitted? | Correct args? | Notes |
|:-------|:---------------|:--------------|:------|
| `5 min after 09:58` | ✗ (`[NO_ROUTE]`) | N/A | False negative — simple phrasing missed |
| `Meeting 14:00 + 45 min` | ✓ `[ROUTE_TIME_ADD]` | ✗ | Args: 15:50+27 (mode collapse) |
| `Train 23:45 + 30 min` | ✓ `[ROUTE_TIME_ADD]` | ✗ | Same collapsed args as above |
| `Baby 20:30 + 15 min` | ✓ `[ROUTE_TIME_ADD]` | ~ | Hour close (20), min wrong (50) |
| `10 min before 00:05` | ✓ `[ROUTE_TIME_ADD]` | ✗ | Should be `TIME_SUB` |

### Analysis

1. **Format learned**: 4/5 prompts produce syntactically valid `[ROUTE]...[/ROUTE]` blocks
2. **Arguments collapse**: The model converges to a few common value patterns (mode collapse on 15:50+27)
3. **Subtraction not learned**: Model always emits `TIME_ADD` — needs more training or capacity
4. **Simple phrasing missed**: "What time is it X after Y?" misclassified as non-temporal

### Comparison to Baselines

| Model | Architecture | Score |
|:------|:-------------|:------|
| SmolLM-135M (vanilla) | Direct answer | 0/48 (0%) |
| TinyLlama-1.1B (vanilla) | Direct answer | 0/48 (0%), 5 BASE_10_ERROR |
| **SmolLM-135M (SFT, this run)** | **Route-to-Luxon** | **4/5 route emission** (syntax only) |

## Conclusions

1. ✅ **Pipeline validated**: Data generation → SFT → route emission works end-to-end
2. ✅ **Domain tokens work**: 104 typed tokens registered and used in model output
3. ✅ **Geometric init works**: Model references real token IDs (not BPE fragments)
4. ⚠️ **135M too small for argument accuracy**: Need more parameters or training steps
5. ⚠️ **250 steps insufficient**: ~0.3 effective epochs; model hasn't seen enough examples

## Next Steps

- Run TinyLlama-1.1B (8x parameters) with same pipeline
- Run 1000+ steps if 135M is revisited
- Add route-mode evaluation to `validate.py` for 48-test harness scoring
