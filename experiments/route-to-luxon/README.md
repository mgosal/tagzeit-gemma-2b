# Route-to-Luxon SFT Experiments

Experiments for the Route-to-Luxon pipeline (ADR-001). Unlike the original Tagzeit
experiments which used `[THINK]` CoT blocks, these train models to emit `[ROUTE_*]`
tokens — delegating all arithmetic to a deterministic Luxon engine.

## Architecture

```
NL Input → [Stage 1: Detector] → Typed Tokens → [Stage 2: LLM] → [ROUTE_*] → [Stage 3: Luxon] → Result
```

The model only learns **NLU + routing**. It never does math.

## Experiments

| Run | Model | Data | Steps | Token Accuracy | Route Emission | Card |
|:----|:------|:-----|:------|:---------------|:---------------|:-----|
| 001 | SmolLM-135M | 5,656 | 250 | 83.3% | 4/5 (syntax ✓, args ✗) | [001](reports/001-smollm-135m-poc.md) |
| 003 | SmolLM2-360M | 5,656 | 500 | 89.4% | (Initial: 12.5% Success) | [003](reports/003-smollm2-360m.md) |
| 004 | SmolLM2-360M | 5,656 | 5,000 | 90.2% | **25.0% E2E Success** | [004](reports/004-smollm2-360m-5k.md) |
| 005 | SmolLM2-360M | — | — (eval only) | — | **25.0% E2E** (regression ✅) | [005](reports/005-regression-model-agnostic.md) |
| 006 | **Gemma-2B-it** | — | — (vanilla) | — | **52.1% Direct / 35.4% CoT / 0% Route** | [006](reports/006-gemma2-2b-baseline.md) |
| 007 | **Gemma-2B-it** | 5,640 | 5,000 (LoRA) | TBD | **Pending** (Colab) | [007](reports/007-gemma2-2b-lora.md) |

### Key Findings (Exp 006)

- **Scale buys reasoning for free.** Vanilla Gemma-2B-it (52.1%) outperforms SmolLM2-360M after 5k steps of SFT (25.0%). No training required.
- **CoT hurts temporal math.** Chain-of-thought *degrades* Gemma from 52.1% → 35.4%, introducing BASE_10_ERROR and FORMAT_HALLUCINATION that don't exist in direct mode. This validates the Route-to-Luxon thesis: reasoning chains corrupt arithmetic.
- **Subtraction solved at scale.** Gemma scores 100% on subtraction (direct mode). SmolLM2 scored 0% even after SFT.
- **Route hallucination.** Vanilla Gemma spontaneously attempted `[ROUTE_*]` style output — promising for fast SFT convergence.

## Shared Infrastructure

- `../../tools/sft_train.py` — SFT training with domain token registration + geometric init (supports `--resume_from_checkpoint`)
- `../../tools/validate.py` — 48-test diagnostic harness (model-agnostic via `apply_chat_template`)
- `../../tools/validate_route.py` — E2E evaluator which extracts routes and pipes them through the Luxon engine
- `../../core/synthetic_data/generators/temporal/generator.js` — Route-format data generator
- `../../core/computation/temporal_engine.js` — Luxon computation engine

## Quickstart (Training)

The training script `02_run_training_5k.sh` is now **"Resume-Aware"**. If a training run crashes or is interrupted (e.g., via Ctrl+C), simply running the script again will:
1.  Scan for existing `checkpoint-*` directories in the output folder.
2.  Automatically resume from the latest state.
3.  Load the optimizer, scheduler, and model weights to ensure training continuity.
