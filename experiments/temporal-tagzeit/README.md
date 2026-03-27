# Project Tagzeit: Temporal Reasoning for LLMs

> [!NOTE]
> **Status:** This experiment was the founding prototype for the LLM Deficiency Index. It has been superseded by the [Route-to-Luxon pipeline](../../README.md#active-work-route-to-luxon-pipeline) ([ADR-001](../../brain/decisions/001-route-to-luxon.md)), which eliminates model-side arithmetic errors entirely.

Tagzeit investigated how small language models handle temporal reasoning. Through three training runs, it demonstrated that small models can learn temporal *formatting* but consistently fail at the underlying *arithmetic* — motivating the architectural shift to external computation.

⚠️ **Disclaimer**: This is an experimental research project developed and tested on Mac M-series hardware.

---

## Key Findings

1. **Models learn format, not math.** SmolLM-135M and Gemma-2B both learned to output `[THINK]` traces, but arithmetic accuracy remained near 0%.
2. **BASE_10_ERROR is systematic.** Models confuse base-10 and base-60 arithmetic: `58 + 5 = 63` instead of performing the 60-carry to get `1:03`.
3. **Chain-of-thought is insufficient for 2B models.** Even structured CoT traces don't help when the model can't perform the underlying computation.

These findings directly motivated [ADR-001: Route-to-Luxon](../../brain/decisions/001-route-to-luxon.md).

## Results

| Run | Model | Data | Steps | Result | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | SmolLM-135M | 1,000 | 500 | Success | **Baseline**: Learned format, but 0% math. |
| 2 | Gemma-2-2b | 5,000 | 1,000 | Success | **Generalization**: Partial reasoning, delims unstable. |
| 3 | SmolLM-135M | 5,000 | 250 | **FAILED** | **OOM**: MPS Out-of-Memory at Step 250. |

## Architecture (Original — Superseded)

The original approach used `[THINK]` chain-of-thought blocks to teach explicit arithmetic:

```
Input:  "The train arrives at 09:58. Add 5 minutes."
Output: [THINK] 58+5=63 | 63≥60 → carry | 63-60=03 | 09+1=10 → 10:03 [/THINK]
        The train arrives at 10:03.
```

This has been replaced by the Route-to-Luxon pipeline where the model emits routing tokens and an external engine computes the result.

## Project Structure

- `src/generator.js` — Original synthetic data engine (CoT format)
- `data/` — Training and evaluation datasets
- `results/` — Evaluation outputs and baseline scores
- `../../tools/sft_train.py` — Shared SFT training script
- `../../tools/validate.py` — Shared evaluation harness

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
