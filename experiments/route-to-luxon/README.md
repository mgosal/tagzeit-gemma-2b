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

## Shared Infrastructure

- `../../tools/sft_train.py` — SFT training with domain token registration + geometric init
- `../../tools/validate.py` — 48-test evaluation harness
- `../../core/synthetic_data/generators/temporal/generator.js` — Route-format data generator
- `../../core/computation/temporal_engine.js` — Luxon computation engine
