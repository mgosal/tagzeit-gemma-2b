# Experiment 002: TinyLlama-1.1B — Route-to-Luxon SFT

**Date:** 2026-03-27
**Status:** Complete
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)
**Architecture:** Route-to-Luxon (ADR-001)
**Predecessor:** Experiment 001 (SmolLM-135M PoC)

## Objective

Scale up from the SmolLM-135M proof-of-concept to a 1.1B parameter model. Test whether
increased capacity improves argument accuracy and route-type discrimination (ADD vs SUB),
which were failure modes in Experiment 001.

## Hypothesis

With 8× more parameters and LoRA fine-tuning, argument collapse should reduce and the
model should learn to discriminate between `TIME_ADD` and `TIME_SUB` operations.

## Setup

| Parameter | Value |
|:----------|:------|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Training data | 5,656 samples (route format) |
| Eval data | 295 samples |
| Steps | 500 |
| Batch size | 2 (effective 8 with grad_accum=4) |
| Learning rate | 5e-5 (cosine schedule) |
| Warmup | 100 steps |
| Precision | float32 (MPS) |
| Optimizer | `adamw_torch` |
| LoRA | r=64, alpha=128, dropout=0.05, all linear layers |
| Domain tokens | 104 (geometric sinusoidal init) |
| Hardware | Apple M4 Pro, 24GB RAM, MPS |
| Training time | 42m 01s |
| Speed | 5.04s/step |

## Training Data

Same dataset as Experiment 001, generated via
`core/synthetic_data/generators/temporal/generator.js --count 5000`:
- **Route examples**: NL question → `[ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_XX] ...`
- **Shadow pairs**: base-10 arithmetic → `[NO_ROUTE]` (contrastive)
- **Negative examples**: Non-temporal questions → `[NO_ROUTE]` (~5%)
- **Subtraction**: ~20% of temporal examples use `[ROUTE_TIME_SUB]`

## Results

### Training Curve

| Step | Train Loss | Token Accuracy | Notes |
|:-----|:-----------|:---------------|:------|
| 50 | 2.46 | — | Early learning |
| 100 | 1.32 | — | Rapid descent |
| 250 | 0.68 | 80.7% | Mid-point eval |
| 460 | 0.65 | 81.5% | Near-convergence |
| 480 | 0.63 | 82.0% | Peak train accuracy |
| **500 (final)** | **0.63** | **82.0%** | **Final train** |

### Final Evaluation

| Metric | Value |
|:-------|:------|
| Eval loss | 0.6511 |
| Eval token accuracy | 81.3% |
| Train/eval gap | 0.7% (no overfitting) |
| Eval runtime | 22.69s |

### Inference Test (5 free-form prompts)

Model produces natural language responses when given free-form prompts without the
training format. This is expected — the model learned to emit `[ROUTE]` blocks
specifically within the structured training format (NL → `[ROUTE]...[/ROUTE]`).

> [!NOTE]
> The inference test below uses free-form prompts, not the training format.
> These results show general language behavior, not route emission.

| Prompt | Response (natural language) |
|:-------|:---------------------------|
| `3 days after 2024-01-15?` | "It is 2024-02-01" ✗ |
| `Days between 2024-03-01 and 2024-03-15?` | "144 minutes" ✗ |
| `Day of week for 2024-07-04?` | "Saturday" ✗ |
| `Add 2 weeks to 2024-06-01` | "2024-07-01" ✗ |
| `Is 2024 a leap year?` | "Yes" ✓ |

### Comparison to All Experiments

| Model | Params | Architecture | Eval Loss | Token Acc | Notes |
|:------|:-------|:-------------|:----------|:----------|:------|
| SmolLM-135M (vanilla) | 135M | Direct answer | — | — | 0/48 baseline |
| TinyLlama-1.1B (vanilla) | 1.1B | Direct answer | — | — | 0/48 baseline, 5 BASE_10_ERROR |
| SmolLM-135M (SFT) | 135M | Route-to-Luxon | 0.52 | 83.3% | 4/5 route syntax, arg collapse |
| **TinyLlama-1.1B (SFT)** | **1.1B** | **Route-to-Luxon** | **0.65** | **81.3%** | **LoRA, needs route-mode eval** |

> [!IMPORTANT]
> SmolLM-135M shows higher token accuracy (83.3%) than TinyLlama-1.1B (81.3%) despite
> being 8× smaller. This is likely because SmolLM was full-fine-tuned while TinyLlama
> used LoRA with only ~2.5% trainable parameters. The larger model's advantage should
> emerge on out-of-distribution examples and route-mode E2E evaluation.

## Conclusions

1. ✅ **Training converged**: Loss dropped from 2.46 → 0.63, token accuracy reached 82%
2. ✅ **No overfitting**: Train/eval gap < 1% (0.63 vs 0.65)
3. ✅ **LoRA works**: Efficient fine-tuning with ~2.5% trainable parameters
4. ⚠️ **Token accuracy comparable to SmolLM**: LoRA may limit surface-level accuracy
5. ⚠️ **Route-mode eval needed**: The real test is whether the model emits correct
   `[ROUTE]` blocks that produce correct answers when piped through the Luxon engine

## Next Steps

- [ ] Add `--route_mode` to `validate.py` for end-to-end Luxon scoring
- [ ] Run 48-test harness in route mode against both SFT models
- [ ] Compare E2E accuracy: SFT route-mode vs vanilla direct-answer
- [ ] Consider full fine-tune of TinyLlama (if LoRA is the bottleneck)
- [ ] Scale to Gemma-2B with findings from this experiment
