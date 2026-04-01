# Route-to-Luxon SFT Experiments

Experiments for the Route-to-Luxon pipeline (ADR-001). Models learn to emit `[ROUTE_*]`
tokens — delegating all arithmetic to a deterministic Luxon engine.

## Architecture

```
NL Input -> [Stage 1: Detector] -> Typed Tokens -> [Stage 2: LLM] -> [ROUTE_*] -> [Stage 3: Luxon] -> Result
```

The model only learns **NLU + routing**. It never does math.

## Experiments

| Run | Model | Data | Steps | Eval Loss | E2E Accuracy | Key Finding |
|:----|:------|:-----|:------|:----------|:-------------|:------------|
| 001 | SmolLM-135M | 5.6K | 250 | — | 4/5 syntax | PoC: grammar learnable |
| 003 | SmolLM2-360M | 5.6K | 500 | — | 12.5% | Grammar acquired, direction biased |
| 004 | SmolLM2-360M | 5.6K | 5,000 | — | 25.0% | ADD/SUB confused, grammar ossified |
| 006 | Gemma-2B-it | — | — | — | 52.1% Direct | Scale buys reasoning for free |
| 007 | Gemma-2B-it | 5.6K LoRA | 5,000 | — | 0% | LoRA failed for grammar |
| 008b | SmolLM2-360M | 95K bal | 10,000 | 0.267 | 2.1% | Distribution mismatch |
| 009a | SmolLM2-360M | 14K v2 | 1,000 | NaN | 0% | fp16 on T4 = weight corruption |
| **009** | **Llama 3.2-1B** | **5K v2** | **250** | **0.407** | **100% routing** | **6/6 ops at 250 steps** |
| **010** | **Llama 3.2-1B** | **100K v2** | **5,000** | — | **Pending** | Full run + checkpoints |

### Active: Experiment 010

Full 5K step run on Llama 3.2-1B-Instruct with 100K v2 data. Building on 009's
100% operation routing at just 250 steps.

**What 009 proved:** Operation routing (ADD/SUB/BETWEEN/NO_ROUTE) is solved at 250 steps.
**What 010 targets:** Argument precision (minute extraction, duration values) and
eliminating trailing token repetition.

### Colab Notebook

Open [`010_llama1b_fullrun_colab.ipynb`](./010_llama1b_fullrun_colab.ipynb) in Google Colab:

1. Set runtime to **L4 GPU** (24GB VRAM, bf16 required)
2. Add HF token as Colab Secret `HF_TOKEN` (Llama is gated)
3. Mount Google Drive for checkpoint persistence
4. Run all cells — ~2-3 hours

Checkpoints saved every 500 steps to Google Drive. Resume supported if disconnected.

## Key Learnings

- **Model capacity matters dramatically.** Llama 3.2-1B achieved in 250 steps what SmolLM2-360M could not in 10K steps.
- **Never use fp16 on T4** for full fine-tuning. Use bf16 on L4/A100, or fp32 on T4.
- **Full FT required** for novel token grammar. LoRA is insufficient.
- **Data distribution must match test distribution.** V2 generator with formulaic templates fixes this.

## Shared Infrastructure

- `../../tools/sft_train.py` — SFT training with domain token registration + geometric init
- `../../tools/validate.py` — 48-test diagnostic harness
- `../../tools/validate_route.py` — E2E evaluator: routes -> Luxon engine
- `../../tools/format_for_model.py` — Model-specific chat template wrapper
- `../../tools/extract_hard_negatives.py` — ComplexTempQA hard-negative extractor
- `../../experiments/temporal-tagzeit/src/generator_route.js` — Route-format data generator (v2)
- `../../core/computation/temporal_engine.js` — Luxon computation engine
