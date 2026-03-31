# Experiment 008a: SmolLM2-360M — Full Fine-Tune on 100K Routed Data (FAILED)

**Date:** 2026-03-31
**Status:** Failed ❌ (VM reset, checkpoints lost)
**Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (362M parameters)
**Method:** Full fine-tune
**Hardware:** Google Colab L4 GPU

## What Happened

Training started in float32 (0.61 it/s), switched to bf16 at step 1000 (1.25 it/s). Colab VM reset at ~step 1996. Checkpoints were saved to local VM storage, not Google Drive — all lost.

## What We Learned

| Finding | Detail |
|:--------|:-------|
| **bf16 is stable** | No loss instability after float32→bf16 switch at step 1000 |
| **bf16 is 2× faster** | 0.61 → 1.25 it/s on L4 |
| **Loss looks excellent** | 0.272 at step 1500 (vs 0.326 in Exp 003 at step 500) |
| **Save to Google Drive** | Colab VMs are ephemeral — always save to Drive |
| **float32 unnecessary** | SmolLM2-360M is stable in bf16 on L4 |
| **`processing_class`** | transformers v4.46+ renamed `tokenizer` param in Trainer |

## Partial Results (Before Crash)

| Step | Train Loss | Eval Loss | Precision |
|:-----|:-----------|:----------|:----------|
| 500  | 0.283      | 0.282     | float32   |
| 1500 | 0.272      | 0.273     | bf16      |

## Fixes Applied for 008b

1. Checkpoints save to Google Drive (`/content/drive/MyDrive/tagzeit-008/`)
2. bf16 enabled from start
3. Fixed `tokenizer` → `processing_class` for Trainer
4. Fixed `total_mem` → `total_memory` for VRAM check
5. L4 GPU recommended in notebook header
