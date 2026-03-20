# Experiment Report: Temporal Baseline & Architectural Validation (Run 0-1)

## Objective
Establish a baseline for temporal reasoning in small language models (SmolLM-135M) and validate the "Continued Pre-Training" (CPT) architectural approach using structured `[THINK]` blocks.

## Run 0: Zero-Shot Baseline
- **Model**: `HuggingFaceTB/SmolLM-135M`
- **Method**: Direct Zero-Shot evaluation across 43 diagnostic probes.
- **Results**:
    - Exact Match: 0.0%
    - Normalized Match: 2.3% (Success on 1 logic check)
- **Finding**: Model lacks inherent base-60 arithmetic primitives. Total hallucination of results is the primary failure mode.

## Phase 0: Data Hardening
- **Corpus**: 56,938 synthetic records.
- **Diversity**: 12 humanity domains (Domestic, Tech, Logistics, etc.) balanced at ~7% each.
- **Features**: 
    - 11.5% Format Jitter (Spaced digits) for tokenization hardening.
    - 6.3% Subtraction logic (Backwards temporal reasoning).
    - Shadow Pair adjacency for contrastive math learning.

## Run 1: Proof of Training (PoT)
- **Model**: `SmolLM-135M` (QLoRA, 1,000 iterations, 5k sample).
- **Metric**: Structural vs. Arithmetic Success.
- **Findings**:
    - **Structural Match (~85%)**: Model successfully adopted the `[THINK] ... [RESULT]` output format.
    - **Arithmetic Stability (Low)**: Model struggles to execute the carry logic reliably at this scale.
- **Conclusion**: The architectural "Thinking" pattern is learnable. Scaling to the 56k record PoC dataset is required for arithmetic stabilization.

## Next Steps
1. Scale training to full 56,938 PoC records.
2. Monitor arithmetic stability on "Cascade" (triple-carry) scenarios.
3. Validate subtraction logic on the 6.3% backwards-reasoning subset.
