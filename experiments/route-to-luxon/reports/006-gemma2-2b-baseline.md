# Experiment 006: Gemma-2B-it — Vanilla Baseline

**Date:** 2026-03-31
**Status:** Complete ✅
**Model:** `google/gemma-2-2b-it` (2.6B parameters)
**Architecture:** Direct evaluation (no Route-to-Luxon training)
**Method:** Zero-shot — no fine-tuning, no adapters
**Predecessor:** Experiment 005 (regression test confirming model-agnostic validator)

## Objective

Establish the **vanilla Gemma-2B-it baseline** for temporal reasoning. This is the first evaluation of a model significantly larger than SmolLM2-360M (2.6B vs 362M — a 7.2× parameter increase).

Key questions:
1. Does raw scale (7.2× more parameters) buy temporal reasoning ability for free?
2. Does Gemma exhibit the same failure modes as SmolLM2 (BASE_10_ERROR, TOKEN_COLLAPSE)?
3. What is the inference speed cost of the larger model on M4 Pro?

## Prompt Format Note

> **Gemma system-prompt merging:** Gemma-2B-it does not support a separate `system` role in its chat template. The model-agnostic `format_prompt()` function (added in Exp 005) automatically merges the system prompt into the user message. The resulting prompt looks like:
>
> ```
> <start_of_turn>user
> You are a precise time calculator. Given a starting time and a duration...
>
> What time is it 1 minute after 23:59?<end_of_turn>
> <start_of_turn>model
> ```
>
> This is functionally equivalent to ChatML's separate system role, but the model sees instructions and question as a single user turn.

## Setup

| Parameter | Value |
|:----------|:------|
| Model | `google/gemma-2-2b-it` (2.6B params) |
| Training | None (vanilla) |
| Eval harness | `tools/validate.py` (48 tests, model-agnostic) |
| Mode | Direct (zero-shot, no CoT) |
| Skins | Military only |
| Backend | auto (MLX → PyTorch fallback) |
| Hardware | Apple M4 Pro, 24GB RAM |

## SmolLM2 Comparison Baseline (from Exp 004/005)

| Metric | SmolLM2-360M (Vanilla) | SmolLM2-360M (5k SFT) |
|:-------|:----------------------|:----------------------|
| Parameters | 362M | 362M |
| E2E (Route) | N/A (untrained) | 25.0% (12/48) |
| Standard | — | 83% |
| Subtraction | — | 0% |

## Results

### Harness 1: `validate.py` (Direct Mode, 48 tests)

**Normalized Match: 25/48 (52.1%)**

| Category | Score | Notes |
|:---------|:------|:------|
| Standard | 5/6 (83%) | Matches SFT-trained SmolLM2 |
| Minute Carry | 5/8 (62%) | Far better than SmolLM2's 25% |
| Hour Rollover | 4/7 (57%) | SmolLM2 scored 0% even after SFT |
| Semantic Eq | 3/5 (60%) | Reasonable cross-format understanding |
| Cascade | 2/5 (40%) | SmolLM2 scored 0% |
| **Subtraction** | **5/5 (100%)** | **SmolLM2 scored 0% — even after 5k steps of SFT** |
| Midnight Boundary | 1/1 (100%) | |
| Generalization | 0/3 (0%) | Large-delta reasoning still weak |
| Format Robust | 0/5 (0%) | Spaced digits confuse Gemma completely |
| Impossible | 0/3 (0%) | No error detection ability |

**Failure Mode Distribution:**
| Mode | Count |
|:-----|:------|
| BASE_10_ERROR | 0 |
| TOKEN_COLLAPSE | 0 |
| FORMAT_HALLUCINATION | 0 |
| OTHER_ERROR | 23 |

**Performance:** 25.6 tok/s on M4 Pro (vs ~60 tok/s for SmolLM2-360M).

### Harness 2: `validate_route.py` (E2E Route-to-Luxon)

**E2E: 0/48 (0.0%)** — expected, as the model has never seen `[ROUTE_*]` tokens.

**Interesting finding:** Gemma spontaneously *attempted* to emit routing-style tokens (e.g., `[ROUTE_time_after]`, `[ROUTE_TIME_BEFORE]`, `[ROUTE_past_time]`). It hallucinated the concept of routing calls from the system prompt alone, but with incorrect casing, syntax, and argument structure. This suggests the 2B model has enough instruction-following capacity to "try" structured output formats — a promising signal for SFT.

## Head-to-Head: Gemma-2B-it vs SmolLM2-360M

| Category | SmolLM2-360M (Vanilla) | SmolLM2-360M (5k SFT) | **Gemma-2B-it (Direct)** | Gemma-2B-it (CoT) |
|:---------|:----------------------|:---------------------|:-------------------------|:------------------|
| Parameters | 362M | 362M | **2.6B (7.2×)** | 2.6B |
| **Overall** | ~0%* | 25.0% (route E2E) | **52.1%** | 35.4% |
| Standard | — | 83% | **83%** | 83% |
| Minute Carry | — | 25% | **62%** | 25% |
| Hour Rollover | — | 0% | **57%** | 43% |
| Subtraction | — | 0% | **100%** | 40% |
| Cascade | — | 0% | **40%** | 0% |
| Semantic Eq | — | — | 60% | 60% |
| Format Robust | — | 40% | 0% | 0% |
| Inference | ~60 tok/s | ~60 tok/s | 25.6 tok/s | 30.6 tok/s |

*SmolLM2 vanilla baseline was not formally recorded in direct mode.

### Harness 3: `validate.py` (CoT Mode, 48 tests)

**Normalized Match: 17/48 (35.4%) — ⚠️ WORSE than Direct (52.1%)**

CoT mode asks the model to "show your reasoning" before answering. Counter-intuitively, this **degrades Gemma's performance by 16.7 percentage points**.

| Category | Direct | CoT | Delta |
|:---------|:-------|:----|:------|
| Subtraction | **100%** | 40% | **−60%** |
| Minute Carry | 62% | 25% | −37% |
| Hour Rollover | 57% | 43% | −14% |
| Cascade | 40% | 0% | −40% |
| Standard | 83% | 83% | = |
| Semantic Eq | 60% | 60% | = |

**Failure Mode Distribution (CoT):**
| Mode | Direct | CoT |
|:-----|:-------|:----|
| BASE_10_ERROR | 0 | **4** |
| FORMAT_HALLUCINATION | 0 | **11** |
| OTHER_ERROR | 23 | 16 |

**Key observation:** CoT *introduces* failure modes that didn't exist in direct mode. Gemma in direct mode had **zero** BASE_10_ERROR and **zero** FORMAT_HALLUCINATION. In CoT mode, the model's reasoning chains lead it astray — it converts to 12-hour AM/PM mid-thought (causing FORMAT_HALLUCINATION), and attempts fractional arithmetic that breaks base-60 (causing BASE_10_ERROR).

Example (IM-02, CoT): The model attempted `25:00 + 10 minutes` by converting to fractional hours (`10/60 = 0.1667 hours`) — a valid approach for base-10 but wrong for clock arithmetic.

## Conclusions

1. **Scale buys temporal reasoning.** Vanilla Gemma-2B-it at 52.1% already outperforms SmolLM2-360M *after 5,000 steps of explicit SFT* (25.0%). The 7.2× parameter increase buys significant zero-shot arithmetic ability.

2. **Subtraction is solved at scale (in direct mode).** The most dramatic result: Gemma scores **100% on subtraction** with zero training. SmolLM2 scored 0% even after 5k steps — it consistently mapped "ago" to ADD. Gemma's larger model correctly understands directional semantics.

3. **CoT hurts Gemma on temporal math.** This is the most counter-intuitive finding. Chain-of-thought reasoning *degrades* performance from 52.1% → 35.4%. The model's reasoning chains introduce errors that don't exist in direct mode — particularly AM/PM format drift and base-10 vs base-60 confusion. This is strong evidence for **Route-to-Luxon**: even when the model "knows" the answer (direct mode), asking it to reason can corrupt the output. Offloading arithmetic to a deterministic engine avoids this entirely.

4. **Format robustness remains a weakness.** Spaced digits (`1 8 : 5 9`) completely break Gemma (0% in both modes). The model's tokenizer likely fragments these into nonsensical subwords.

5. **No BASE_10_ERROR in direct mode.** Gemma never produces `18:60` or `24:00` in direct mode. All failures are small arithmetic mistakes. CoT mode, however, introduces 4 BASE_10_ERROR failures.

6. **Route hallucination is a signal.** The model attempted routing-style output without training, suggesting SFT convergence should be faster than SmolLM2's.

7. **Inference cost is 2.4×.** At 25.6 tok/s vs ~60 tok/s, the larger model is noticeably slower but still practical for evaluation.

**Next step:** SFT Gemma-2B-it on Route-to-Luxon data and measure how quickly it converges compared to SmolLM2.
