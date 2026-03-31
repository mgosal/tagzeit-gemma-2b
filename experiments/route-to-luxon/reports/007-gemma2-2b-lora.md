# Experiment 007: Gemma-2B-it — LoRA SFT on Route-to-Luxon

**Date:** 2026-03-31
**Status:** Pending 🔲 (awaiting Colab run)
**Model:** `google/gemma-2-2b-it` (2.6B parameters)
**Method:** LoRA (r=64, alpha=128, all-linear, 3.1% trainable ~83M params)
**Hardware:** Google Colab T4 (16GB VRAM)
**Predecessor:** Experiment 006 (vanilla baseline: 52.1% Direct, 35.4% CoT, 0% Route)

## Objective

Fine-tune Gemma-2B-it via LoRA to emit `[ROUTE_*]` tokens, teaching Gemma the routing grammar that offloads temporal arithmetic to the deterministic Luxon engine.

Key questions:
1. How quickly does a 2.6B model with 52.1% vanilla accuracy converge on routing syntax?
2. Does LoRA SFT improve E2E Route accuracy from 0% to a competitive level?
3. How does LoRA Gemma compare to full-SFT SmolLM2 (25.0% E2E)?

## Why Colab?

Local M4 Pro smoke test measured **~128s/step** (MPS, float32). At 5,000 steps, that's ~7.4 days.

A Colab T4 (CUDA, float16, 8-bit optimizer) is expected to be ~10-20× faster, completing in **2-3 hours**.

## Setup

| Parameter | Value |
|:----------|:------|
| Model | `google/gemma-2-2b-it` (2.6B params) |
| Method | LoRA (r=64, alpha=128, all-linear) |
| Trainable | 83M / 2.7B (3.1%) |
| Data | 5,640 train / 303 eval samples |
| Steps | 5,000 |
| Batch | 4 × 2 grad_accum = 8 effective |
| LR | 5e-5 (cosine, 100 warmup) |
| Optimizer | adamw_8bit |
| Precision | float16 |
| Hardware | Google Colab T4 (16GB VRAM) |

## Comparison Baselines

| Model | Method | Direct | Route E2E |
|:------|:-------|:-------|:----------|
| SmolLM2-360M | Full SFT 5k steps | — | 25.0% |
| Gemma-2B-it | Vanilla (Exp 006) | 52.1% | 0% |
| **Gemma-2B-it** | **LoRA SFT (this)** | **TBD** | **TBD** |

## Results

### Training Metrics

*Pending Colab run...*

### Harness 1: `validate.py` (Direct Mode)

*Pending...*

### Harness 2: `validate_route.py` (E2E Route-to-Luxon)

*Pending...*

## Conclusions

*To be filled after training and evaluation.*

## Notebook

The training notebook is at: [`007_gemma_lora_colab.ipynb`](007_gemma_lora_colab.ipynb)
