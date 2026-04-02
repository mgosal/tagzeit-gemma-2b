# Route-to-Luxon SFT Experiments

Experiments for the Route-to-Luxon pipeline (ADR-001). Models learn to emit `[ROUTE_*]`
tokens -- delegating all arithmetic to a deterministic Luxon engine.

## Architecture

```
NL Input -> [Stage 1: Detector] -> Typed Tokens -> [Stage 2: LLM] -> [ROUTE_*] -> [Stage 3: Luxon] -> Result
```

The model only learns **NLU + routing**. It never does math.

## Experiments

| Run | Model | Data | Steps | Eval Loss | Result | Key Finding |
|:----|:------|:-----|:------|:----------|:-------|:------------|
| 001 | SmolLM-135M | 5.6K | 250 | -- | 4/5 syntax | PoC: grammar learnable |
| 003 | SmolLM2-360M | 5.6K | 500 | -- | 12.5% E2E | Grammar acquired, direction biased |
| 004 | SmolLM2-360M | 5.6K | 5,000 | -- | 25.0% E2E | ADD/SUB confused, grammar ossified |
| 006 | Gemma-2B-it | -- | -- | -- | 52.1% Direct | Scale buys reasoning for free |
| 007 | Gemma-2B-it | 5.6K LoRA | 5,000 | -- | 0% | LoRA failed for grammar |
| 008b | SmolLM2-360M | 95K bal | 10,000 | 0.267 | 2.1% E2E | Distribution mismatch |
| 009a | SmolLM2-360M | 14K v2 | 1,000 | NaN | 0% | fp16 on T4 = weight corruption |
| 009 | Llama 3.2-1B | 5K v2 | 250 | 0.407 | 100% routing | 6/6 ops, args imprecise |
| **010** | **Llama 3.2-1B** | **100K v2** | **2,500** | **0.288** | **100% routing + args** | **Perfect extraction** |

### Current: E2E Validation

Exp 010 achieved 100% operation routing AND 100% argument precision at checkpoint-2500.
Sample output (truncated at first [/ROUTE]):

```
Q: What time is it 1 minute after 23:59?
A: [ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_23] [ARG_MIN_59] [HEAD_DURATION] [ARG_MIN_01] [/ROUTE]

Q: The meeting starts at 14:30 and lasts 45 minutes. When does it end?
A: [ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_30] [HEAD_DURATION] [ARG_MIN_45] [/ROUTE]

Q: What is 42 + 18?
A: [NO_ROUTE] This is base-10 arithmetic, not temporal.
```

Next step: pipe these ROUTE expressions through the Luxon engine to validate computed answers.

**Known issue:** Trailing token repetition after [/ROUTE]. Fixable via post-processing truncation.

## Key Learnings

- **Model capacity is decisive.** Llama 3.2-1B achieved in 250 steps what SmolLM2-360M could not in 10K steps.
- **Never use fp16 on T4** for full fine-tuning. Use bf16 on L4/A100, or fp32 on T4.
- **Full FT required** for novel token grammar. LoRA is insufficient.
- **Data distribution must match test distribution.** V2 generator with formulaic templates fixes this.
- **Sub-epoch training works.** 010 used only 0.42 epochs with no overfitting.

## Shared Infrastructure

- `../../tools/sft_train.py` -- SFT training with domain token registration + geometric init
- `../../tools/validate.py` -- 48-test diagnostic harness
- `../../tools/validate_route.py` -- E2E evaluator: routes -> Luxon engine
- `../../tools/format_for_model.py` -- Model-specific chat template wrapper
- `../../tools/extract_hard_negatives.py` -- ComplexTempQA hard-negative extractor
- `../../experiments/temporal-tagzeit/src/generator_route.js` -- Route-format data generator (v2)
- `../../core/computation/temporal_engine.js` -- Luxon computation engine
