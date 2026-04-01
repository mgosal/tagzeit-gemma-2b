# Route-to-Luxon SFT Experiments

Experiments for the Route-to-Luxon pipeline (ADR-001). Models learn to emit `[ROUTE_*]`
tokens — delegating all arithmetic to a deterministic Luxon engine.

## Architecture

```
NL Input → [Stage 1: Detector] → Typed Tokens → [Stage 2: LLM] → [ROUTE_*] → [Stage 3: Luxon] → Result
```

The model only learns **NLU + routing**. It never does math.

## Experiments

| Run | Model | Data | Steps | E2E Accuracy | Key Finding |
|:----|:------|:-----|:------|:-------------|:------------|
| 001 | SmolLM-135M | 5.6K | 250 | 4/5 syntax ✓ | PoC: grammar learnable |
| 003 | SmolLM2-360M | 5.6K | 500 | 12.5% | Grammar acquired, direction biased |
| 004 | SmolLM2-360M | 5.6K | 5,000 | 25.0% | ADD/SUB confused, grammar ossified |
| 006 | Gemma-2B-it | — (vanilla) | — | 52.1% Direct | Scale buys reasoning for free |
| 007 | Gemma-2B-it | 5.6K (LoRA) | 5,000 | 0% | LoRA failed — no grammar learning |
| 008b | SmolLM2-360M | 95K balanced | 10,000 | 2.1% | Training-test distribution mismatch |
| **009** | **Llama 3.2-1B** | **95K diverse** | **5,000** | **Pending** | Formulaic + hard negatives + edge cases |

### Active: Experiment 009

First model scaling experiment. Moves from SmolLM2-360M to Llama 3.2-1B-Instruct (3.4× more params).

**Data improvements (v2 generator):**
- 28 formulaic templates matching validation harness patterns
- 991 external hard negatives (ComplexTempQA) + 12 synthetic temporal-sounding non-computables
- 10% boundary-biased edge-case time sampling (midnight rollover, noon, minute carry)
- Distribution: 30% ADD / 30% SUB / 15% BETWEEN / 25% NO_ROUTE
- 6 sparse domains expanded with additional templates

**Pipeline improvements:**
- `validate_route.py` — system prompt parameterised via `--system_prompt` CLI arg
- `sft_train.py` — `--bf16` and `--gradient_checkpointing` flags for larger models
- `extract_hard_negatives.py` — extracts temporal-sounding but non-computable questions from ComplexTempQA

**Model progression roadmap:** SmolLM2-360M → **Llama 3.2-1B** → Qwen2.5-1.5B → Gemma-2B

### Colab Notebook

Open [`009_llama1b_fullft_colab.ipynb`](./009_llama1b_fullft_colab.ipynb) in Google Colab:

1. Set runtime to **L4 GPU** (24GB VRAM required)
2. Add HF token as Colab Secret `HF_TOKEN` (Llama is gated)
3. Run all cells — data generation, formatting, training are fully self-contained

## Shared Infrastructure

- `../../tools/sft_train.py` — SFT training with domain token registration + geometric init
- `../../tools/validate.py` — 48-test diagnostic harness (model-agnostic via `apply_chat_template`)
- `../../tools/validate_route.py` — E2E evaluator: routes → Luxon engine → correctness check
- `../../tools/format_for_model.py` — Wraps raw Q/A in model-specific chat template
- `../../tools/extract_hard_negatives.py` — ComplexTempQA hard-negative extractor
- `../../experiments/temporal-tagzeit/src/generator_route.js` — Route-format data generator (v2)
- `../../core/computation/temporal_engine.js` — Luxon computation engine
