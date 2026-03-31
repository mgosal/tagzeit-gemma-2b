# Experiment 007b: Checkpoint Progression Analysis

**Model:** Gemma-2B-it + LoRA (r=64, α=128, all-linear)  
**Training Data:** 5,640 samples, Gemma chat-template wrapped  
**Hardware:** Google Colab T4 (fp16, 8-bit optimizer)  
**Effective Batch:** 8 (batch=2 × grad_accum=4)  

## Hypothesis

At **250 steps (~0.35 epochs)**, the model learns the routing *grammar* but not the *numerical grounding*.
With more training, `[ARG_HOUR_*]` and `[ARG_MIN_*]` values should converge to match the input question.

---

## Checkpoint Results

### Step 250 (~0.35 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | **44/48 (91.7%)** |
| **E2E accuracy** | **0/48 (0%)** |
| **Grammar correct** | ✅ Consistent `[ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_*] ...` |
| **Operation correct** | ✅ All ADD cases use `[ROUTE_TIME_ADD]` |
| **Subtraction learned** | ❌ All SUB cases emit `[ROUTE_TIME_ADD]` instead |
| **Numerical grounding** | ❌ Wrong hour/minute values throughout |

**Failure modes at step 250:**
1. **Wrong ARG values** (44/48 cases): Grammar is perfect but numbers don't match input.
   - e.g. `08:30 + 45min` → `[ARG_HOUR_11] [ARG_MIN_28]` (should be `[ARG_HOUR_08] [ARG_MIN_30]`)
2. **`[ROUTE_future]` hallucination** (4/48 cases): Format-robust tests with spaced digits.
   - e.g. `1 8 : 5 9` → `[ROUTE_future 1 8 : 5 9 ...]` 
3. **No SUB operations**: Model defaults to ADD for all cases including explicit subtraction.

**Interpretation:** The model has learned the *syntax* of routing (token structure, ordering, closure)
but not the *semantics* (extracting correct values from natural language and choosing ADD vs SUB).
This is expected at <1 epoch — the model has seen each training example fewer than once.

### Step 500 (~0.71 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | **1/48 (2.1%)** ⚠️ regression |
| **E2E accuracy** | **0/48 (0%)** |
| **Grammar correct** | ❌ Abandoned `[ARG_*]` tokens for natural-language routing |
| **Operation correct** | ⚠️ Uses `[ROUTE_time_after]` instead of `[ROUTE_TIME_ADD]` |
| **Subtraction learned** | ⚠️ Starts using `[ROUTE_TIME_BEFORE]` / `[ROUTE_past_time]` |
| **Numerical grounding** | ✅ Now extracts correct raw values (e.g. `23:59 1`) |

**Failure modes at step 500:**
1. **Natural-language routing** (47/48 cases): Model invents its own compact notation:
   - e.g. `08:30 + 45min` → `[ROUTE_time_after] 08:30 45 minutes` (correct values, wrong format)
   - e.g. `01:15 - 30min` → `[ROUTE_TIME_BEFORE] 01:15 30 minutes` (correct operation!)
2. **No structured tokens**: Completely stopped using `[HEAD_TIME]`, `[ARG_HOUR_*]`, `[ARG_MIN_*]`
3. **Format-robust tests**: Handles spaced digits (`2 3:5 5`) by stripping spaces

**Interpretation:** The model is transitioning from memorized syntax to semantic understanding.
It has **learned what routing means** (extract time, extract duration, output a structured call)
but is expressing it in its own natural-language notation rather than our token grammar.
This is a classic intermediate overfitting pattern — the model finds a "shortcut" representation.

> **Critical observation:** The model at step 500 can correctly distinguish ADD from SUB
> (`[ROUTE_time_after]` vs `[ROUTE_TIME_BEFORE]`), which step 250 could not do.
> Semantic understanding is progressing even as format compliance regresses.

### Step 750 (~1.06 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 1000 (~1.42 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 1250 (~1.77 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 1500 (~2.13 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 1750 (~2.48 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 2000 (~2.84 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 2250 (~3.19 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

### Step 2500 (~3.55 epochs)

| Metric | Result |
|:-------|:-------|
| **Route emission rate** | TBD |
| **E2E accuracy** | TBD |

---

## Learning Curve Summary

| Steps | Epochs | Emission Rate | E2E Accuracy | Notes |
|:------|:-------|:-------------|:-------------|:------|
| 250 | 0.35 | 91.7% (44/48) | 0% | Grammar learned, numbers wrong |
| 500 | 0.71 | 2.1% (1/48) | 0% | ⚠️ Regression — natural-language routing, correct values |
| 750 | 1.06 | TBD | TBD | |
| 1000 | 1.42 | TBD | TBD | |
| 1250 | 1.77 | TBD | TBD | |
| 1500 | 2.13 | TBD | TBD | |
| 1750 | 2.48 | TBD | TBD | |
| 2000 | 2.84 | TBD | TBD | |
| 2250 | 3.19 | TBD | TBD | |
| 2500 | 3.55 | TBD | TBD | |

---

## Comparison: 007 (raw text) vs 007b (chat template)

| | Exp 007 (raw text) | Exp 007b (chat template) |
|:---|:---|:---|
| **Step 250 emission** | Hallucinated tokens (`[ROUTE_23]`) | Clean grammar (91.7%) |
| **Step 500 emission** | Hallucinated tokens | Natural-language routing (2.1%) ⚠️ |
| **Step 2500 emission** | Hallucinated tokens | TBD |
| **Best E2E** | 0% (all checkpoints) | TBD |
| **Root cause** | Training/eval format mismatch | — |

> **Key insight:** The chat template wrapping was the critical fix. With matching
> train/eval formats, the model learned perfect routing grammar in just 250 steps
> (~15 minutes of T4 compute). Numerical grounding is the remaining challenge.

---

## Exp 007 Loss Curve (reference — raw text training)

| Step | Train Loss | Val Loss |
|:-----|:-----------|:---------|
| 500 | 0.422 | **0.458** ← best |
| 1000 | 0.335 | 0.579 |
| 1500 | 0.289 | 0.669 |
| 2000 | 0.274 | 0.779 |
| 2500 | 0.263 | 0.803 |

Note: This loss curve is from the raw-text training run (007) and is not directly
comparable to 007b, since the input distribution changed. The 007b loss curve
will be captured from the Colab output when training completes.
