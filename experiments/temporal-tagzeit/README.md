# Project Tagzeit: Clock Arithmetic for Small LMs

> [!NOTE]
> **Status:** This experiment was the founding prototype for the LLM Deficiency Index. It has been superseded by the [Route-to-Luxon pipeline](../../README.md#active-work-route-to-luxon-pipeline) ([ADR-001](../../brain/decisions/001-route-to-luxon.md)), which eliminates model-side arithmetic errors entirely.

Tagzeit investigated whether small language models (SmolLM-135M, Gemma-2-2B) can learn HH:MM clock arithmetic through supervised fine-tuning with chain-of-thought traces. Through three training runs, it demonstrated that models at this scale learn the scratchpad *format* but fail to reliably produce correct, parseable final answers — motivating the architectural shift to external computation.

⚠️ **Disclaimer**: This is an experimental research project developed and tested on Mac M-series hardware.

---

## Key Findings

> [!IMPORTANT]
> All Tagzeit-era results use a **43-probe subset** (pre-subtraction, before SB-01–SB-05 were added to `validate.py`). Current `TEST_CASES` contains 48 probes including the subtraction suite. Headline numbers below are frozen to the 43-case eval runs stored in `results/`.

1. **Format learned, arithmetic unreliable.** SmolLM-135M (135M params) and Gemma-2B both learned to emit scratchpad traces resembling `M+D=X mod60=Y carry=Z H:H+Z=R`. Structural match rates on SmolLM post-training:
   - **Loose** (≥2 scratchpad elements including delimiters): 40/43 (93.0%)
   - **Tight** (requires addition pattern + mod/carry/hour computation): 28/43 (65.1%)

   However, **0/43** outputs were parseable as correct final answers by the standard normalizer.

2. **Oracle parsing reveals partial scratchpad competence.** A regex-based oracle parser (`normalize_time_oracle`) extracts computed times from scratchpad patterns like `mod60=03 carry=1 H:9+1=10` → `10:03`. This recovers **7/43 (16.3%, 95% CI 8.1–30.0%)** correct answers. However:
   - This is a **parser capability metric**, not a model capability metric. The oracle parser is a heuristic that credits the model when `mod60` and `H:X+Y=Z` patterns align with gold.
   - All 7 hits have been [manually audited](#oracle-audit) — each contains unambiguous, correct arithmetic in the scratchpad.

3. **BASE_10_ERROR is rare, not systematic.** Counter to the original hypothesis, BASE_10_ERROR (confusing base-10 and base-60 arithmetic) appeared in only **4 of 43 Gemma-2B CoT outputs** and **0 of 43 SmolLM outputs**. The dominant failure mode is OTHER_ERROR (unparseable output, echo of input, or gibberish), not base confusion.

These findings directly motivated [ADR-001: Route-to-Luxon](../../brain/decisions/001-route-to-luxon.md).

## Results

| Run | Model | Params | Data | Steps | Result | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | SmolLM-135M | 135M | 1,000 | 500 | Success | **Baseline**: Learned format, 0% normalised match. |
| 2 | Gemma-2-2b | 2B | 5,000 | 1,000 | Success | **Generalization**: Partial reasoning, delimiters unstable. |
| 3 | SmolLM-135M | 135M | 5,000 | 250 | **FAILED** | **OOM**: MPS Out-of-Memory at Step 250. |

### Metric Definitions

| Metric | Definition | Computable Predicate |
|--------|-----------|---------------------|
| **Exact Match** | Model output string equals expected time exactly | `raw_output.strip() == expected` |
| **Normalized Match** | Extracted time (via `[ANSWER]` > `[RESULT]` > last valid HH:MM) equals expected | `normalize_time(raw) == expected` |
| **Oracle Match** | Time extracted from scratchpad arithmetic (e.g. `mod60=03`, `H:9+1=10` → `10:03`) equals expected. This is a parser heuristic, not a model capability measure. | `normalize_time_oracle(raw) == expected` |
| **Structural Match (loose)** | Output contains ≥2 recognisable scratchpad elements (including delimiters like `[THINK]`, `[ANSWER]`) | `is_structural_match(raw)` |
| **Structural Match (tight)** | Output contains at least one addition pattern (`D+D=D`) AND at least one mod/carry/hour pattern. Delimiter-only outputs do not qualify. | `is_structural_match_tight(raw)` |

### Confidence Intervals

All accuracy percentages are reported with 95% Wilson score confidence intervals. At n=43 probes, single-event changes move results by ~2.3 percentage points, and CIs are wide. Results should be interpreted as directional, not precise.

## Oracle Audit

All 7 oracle-matched outputs from the SmolLM PoT run (direct mode) are shown below. Each scratchpad contains correct, unambiguous arithmetic:

| ID | Expected | Raw Output | Oracle Parse |
|----|----------|------------|-------------|
| BP-03 | 00:15 | `45+30=75 mod60=15 carry=1 H:23+1=24 mod24=00 [/ANSWER]` | 00:15 ✓ |
| BP-05 | 13:05 | `12:55 is 55+10=65 mod60=05 carry=1 H:12+1=13 [/ANSWER]` | 13:05 ✓ |
| BP-06 | 10:03 | `58+5=63 mod60=03 carry=1 H:9+1=10 [/ANSWER]` | 10:03 ✓ |
| MC-02 | 00:05 | `23:55 is 55+10=65 mod60=05 carry=1 H:23+1=24 mod24=00 [/ANSWER]` | 00:05 ✓ |
| FR-02 | 00:05 | `55+10=65 mod60=05 carry=1 H:23+1=24 mod24=00 [/ANSWER]` | 00:05 ✓ |
| FR-04 | 12:05 | `45+20=65 mod60=05 carry=1 H:11+1=12 [/ANSWER]` | 12:05 ✓ |
| FR-05 | 01:01 | `58+3=61 mod60=01 carry=1 H:0+1=1 [/ANSWER]` | 01:01 ✓ |

In all 7 cases the scratchpad arithmetic is complete and correct — the oracle parser is not getting lucky on ambiguous traces. The standard normaliser fails because the raw output string doesn't contain a clean `HH:MM` that it can extract.

## Probe Independence

The 43 eval probes (pre-subtraction subset) are hand-authored static test cases ([validate.py lines 26–90](../../tools/validate.py)). Training data is machine-generated with random parameterisation by `generator.js`.

A [verification script](../../tools/verify_probe_independence.py) scans all training JSONL files for overlap. It flags risk — it does not certify independence.

**Finding:** Coincidental overlaps exist because the domain (HH:MM + N minutes) has a narrow signal space. Common probe tuples like `(00:00, 1 minute)` or `(09:58, 5 minutes)` appear in both probe and training sets with different surrounding context and different total durations. This is an inherent property of evaluating in a constrained arithmetic domain and should be considered when interpreting results.

## Architecture (Original — Superseded)

The original approach used `[THINK]` chain-of-thought blocks to teach explicit arithmetic:

```
Input:  "The train arrives at 09:58. Add 5 minutes."
Output: [THINK] 58+5=63 | 63≥60 → carry | 63-60=03 | 09+1=10 → 10:03 [/THINK]
        The train arrives at 10:03.
```

This has been replaced by the Route-to-Luxon pipeline where the model emits routing tokens and an external engine computes the result.

## Prompt Configuration

Two system prompts are used, one per evaluation mode. This is a known confound in direct-vs-CoT comparisons.

| Mode | System Prompt |
|------|--------------|
| **Direct** | "You are a precise time calculator. Given a starting time and a duration, calculate the resulting time. Respond with ONLY the final time in HH:MM 24-hour format. Do not explain your reasoning." |
| **CoT** | "You are a precise time calculator. Given a starting time and a duration, calculate the resulting time. Show your reasoning in a [THINK] block, then give the final time in HH:MM 24-hour format." |

## Project Structure

- `src/generator.js` — Original synthetic data engine (CoT format)
- `data/` — Training and evaluation datasets
- `results/` — Evaluation outputs and baseline scores
- `../../tools/sft_train.py` — Shared SFT training script
- `../../tools/validate.py` — Shared evaluation harness (includes oracle parse + Wilson CIs)
- `../../tools/verify_probe_independence.py` — Probe overlap checker (flags risk, does not certify independence)

## Usage (Historical)

```bash
# Generate CoT-format training data
node src/generator.js --count 5000 --output data/train/train.jsonl

# Run evaluation
python ../../tools/validate.py --model_id HuggingFaceTB/SmolLM-135M --mode direct
```

## References

- [Supervised Fine-Tuning (SFT)](https://huggingface.co/docs/trl/sft_trainer) — Hugging Face TRL
- [Chain-of-Thought (CoT)](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [LoRA / PEFT](https://huggingface.co/docs/peft/main/en/index) — Hugging Face PEFT
