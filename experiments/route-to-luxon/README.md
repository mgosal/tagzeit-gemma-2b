# Route-to-Luxon SFT Experiments

Experiments for the Route-to-Luxon pipeline (ADR-001). Models learn to emit `[ROUTE_*]`
tokens -- delegating all arithmetic to a deterministic Luxon engine.

## Architecture

```
NL Input -> [Stage 1: Detector] -> Typed Tokens -> [Stage 2: LLM] -> [ROUTE_*] -> [Stage 3: Luxon] -> Result
```

The model only learns **NLU + routing**. It never does math.

## Experiments

| Run | Model | Data | Steps | Eval Loss | E2E | Key Finding |
|:----|:------|:-----|:------|:----------|:----|:------------|
| 001 | SmolLM-135M | 5.6K | 250 | -- | 4/5 syntax | PoC: grammar learnable |
| 003 | SmolLM2-360M | 5.6K | 500 | -- | 12.5% | Grammar acquired, direction biased |
| 004 | SmolLM2-360M | 5.6K | 5,000 | -- | 25.0% | ADD/SUB confused, grammar ossified |
| 006 | Gemma-2B-it | -- | -- | -- | 52.1% Direct | Scale buys reasoning for free |
| 007 | Gemma-2B-it | 5.6K LoRA | 5,000 | -- | 0% | LoRA failed for grammar |
| 008b | SmolLM2-360M | 95K bal | 10,000 | 0.267 | 2.1% | Distribution mismatch |
| 009a | SmolLM2-360M | 14K v2 | 1,000 | NaN | 0% | fp16 on T4 = weight corruption |
| 009 | Llama 3.2-1B | 5K v2 | 250 | 0.407 | 100% routing | Args imprecise |
| **010** | **Llama 3.2-1B** | **100K v2** | **2,500** | **0.288** | **79.2% (38/48)** | **See below** |

## Exp 010 Results

**38/48 E2E accuracy** on the full 48-case harness through Luxon engine.

### What works (100%)
- Subtraction (5/5)
- Format robustness -- spaced times like "1 8 : 5 9" (5/5)
- Semantic equivalence -- varied phrasings (5/5)
- Midnight boundary crossings (1/1)
- All durations <= 59 minutes

### What fails
- **Multi-hour durations >59min (7 failures, #31):** "90 minutes" decomposed as [ARG_HOUR_01][ARG_MIN_00] instead of [ARG_HOUR_01][ARG_MIN_30]. Generator under-represents these.
- **Invalid inputs (3 failures, #32):** "12:65", "25:00", "13:45 PM" routed as valid. No INVALID training examples.

### Fix path
Both are generator data gaps, not model architecture issues. Fix the generator, retrain -> target >93%.

## Key Learnings

- **Model capacity is decisive.** Llama 3.2-1B achieved in 250 steps what SmolLM2-360M could not in 10K.
- **Never use fp16 on T4.** Use bf16 on L4/A100, or fp32 on T4.
- **Full FT required** for novel token grammar. LoRA is insufficient.
- **Data distribution must match test distribution.** V2 generator with formulaic templates fixes this.
- **Sub-epoch training works.** 010 used only 0.42 epochs with no overfitting.
- **The test harness must be independent of the generator.** Circular validation risks overstating generalization (#33).

## Shared Infrastructure

- `../../tools/sft_train.py` -- SFT training with domain token registration + geometric init
- `../../tools/validate.py` -- 48-test diagnostic harness
- `../../tools/validate_route.py` -- E2E evaluator: routes -> Luxon engine
- `../../tools/format_for_model.py` -- Model-specific chat template wrapper
- `../../tools/extract_hard_negatives.py` -- ComplexTempQA hard-negative extractor
- `../../experiments/temporal-tagzeit/src/generator_route.js` -- Route-format data generator (v2)
- `../../core/computation/temporal_engine.js` -- Luxon computation engine
