# Experiment 009a: SmolLM2-360M V2 Generator Pipeline Test (FAILED)

**Date:** 2026-04-01
**Model:** HuggingFaceTB/SmolLM2-360M-Instruct
**Runtime:** Google Colab T4 (16GB VRAM)

## Configuration

| Parameter | Value |
|-----------|-------|
| Data | 14K samples (v2 generator: formulaic + hard negatives + edge cases) |
| Distribution | 30/30/15/25 (ADD/SUB/BETWEEN/NO_ROUTE) |
| Steps | 1,000 |
| Effective batch | 16 (4 x 4) |
| Epochs | ~1.2 |
| Precision | fp16 mixed precision (T4) |
| LR | 1e-4 |

## Results

| Step | Train Loss | Eval Loss |
|------|-----------|-----------|
| 250 | 3.334 | NaN |
| 500 | 0.000 | NaN |
| 750 | 0.000 | NaN |
| 1000 | 0.000 | NaN |

**Final train loss (reported):** 1.269
**Final eval loss:** NaN
**Routing accuracy:** 0/10 (0%)
**Inference output:** All responses were endoftext token spam. Model completely dead.

## Root Cause

**fp16 mixed precision on T4 caused gradient overflow, corrupting model weights.**

The model was loaded in float32 but trained with fp16=True. This caused numerical
instability — the eval loss went NaN from the first evaluation at step 250, and the
model weights became degenerate. The training loss reporting 0.000 is likely a display
artifact of NaN values in the HF Trainer.

Previous successful experiments (003, 004) used the sft_train.py script which defaults
to fp32 when bf16 is not supported. Exp 008 used bf16=True on L4 (which has native
bf16 support). The 009a notebook incorrectly used fp16=True on T4.

## Learning

- **Never use fp16 for full fine-tuning on these models.** Use bf16 on L4/A100, or fp32 on T4.
- Add a runtime assertion: `assert torch.cuda.is_bf16_supported()` to fail fast if on wrong GPU.
- This failure is unrelated to the v2 generator data quality.
