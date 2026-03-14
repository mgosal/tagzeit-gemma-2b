# Project Tagzeit: Temporal Reasoning for LLMs

Project Tagzeit is a research initiative focused on hardening Large Language Models' (LLMs) ability to perform precise temporal reasoning. By moving beyond simple text prediction to a deterministic base-60 state machine logic, Tagzeit enables models to handle complex time math, boundary conditions, and multi-unit carries with high reliability.

---

## Models & Environment

Tagzeit is designed for a tiered training and evaluation hierarchy:

| Tier | Model ID | Purpose | Backend |
| :--- | :--- | :--- | :--- |
| **PoT (Proof of Training)** | `HuggingFaceTB/SmolLM-135M` | Initial 50-probe format & block-output validation | PyTorch / MLX |
| **PoC (Proof of Concept)**  | `HuggingFaceTB/SmolLM-135M` | 56k-record arithmetic scaling & stability | PyTorch / MLX |

**Hardware Requirement**: 
- **Local Research**: Runs on standard Mac CPU/M-series.
- **PoC**: Runs on standard Mac CPU/M-series.

---

## Phase 0: Foundation & Baseline

The core of Tagzeit revolves around a script that generates formatted synthetic data to train and evaluate LLM temporal arithmetic.

### Synthetic Corpus Generation (`generator.js`)
The generator produces a diverse dataset across **12 everyday domains** (Domestic, Logistics, Professional, Wellness, Social, Entertainment, Parenting, Financial, Maintenance, History, Tech, Procrastination).

Key Technical Features:
- **Temporal Logic Levels**: Targets minute carries (e.g., 09:58 + 5m), hour rollovers (23:59 + 1m), and day rollovers.
- **Subtraction Support**: Includes backwards temporal reasoning (e.g., "What time was it 20m before 01:10?") with internal base-60 borrow logic. Currently accounts for ~6% of the Phase 0 corpus.
- **Tokenization Hardening (Format Jitter)**: Employs "Format Jitter" (applied to ~11% of records) to prevent model over-reliance on standard patterns.
- **Shadow Pairs**: Injects pure base-10 arithmetic problems (e.g., `What is 45 + 20?`) immediately adjacent to temporal problems using the same numbers. This forces the model to learn the contrastive difference between standard math and base-60 temporal math within the same attention window.
- **Human-Fuzzy Time**: Support for "Temporal Context Anchors" (e.g., "half past six") with an internal translation step to formal time.

### Zero-Shot Baseline & Diagnostics (`validate.py`)
Before training, models are measured using a diagnostic probe to establish a performance baseline. This aligns with formal frameworks like BIG-bench "Date Understanding".

| Category | Description | Examples |
| :--- | :--- | :--- |
| **Cascade** | Multi-unit carries (triple carry logic). | `23:59 + 2 min -> 00:01` |
| **Day/Rollover** | Crossing the midnight boundary. | `23:45 + 30 min -> 00:15` |
| **Format Robustness**| Jittered input formats (spaced/clumped). | `1 2 : 5 9 + 1 min` |
| **Hour/Minute Carry**| Simple mathematical carries. | `09:58 + 5 min -> 10:03` |
| **Impossible/Error** | Detecting invalid input timestamps. | `12:65 + 5 min -> INVALID` |
| **Semantic Eq.** | Matching fuzzy anchors to formal time. | `"noon" + 30 min -> 12:30` |

- **Failure Categorization**: Tracks `BASE_10_ERROR`, `TOKEN_COLLAPSE`, `FORMAT_HALLUCINATION`, and `OTHER_ERROR`.
- **Scoring Tiers**: Exact Match (100% string identity) vs. Normalized extraction.

### Run 0: Baseline Evaluation
Evaluation of **SmolLM-135M** (Base Model, Zero-Shot) across 43 diagnostic probes. 

| Metric | Result (Direct) | Observation |
| :--- | :--- | :--- |
| **Exact Match** | 0.0% | Model did not adhere to 24h output constraints. |
| **Normalized Match** | 2.3% | Model succeeded on a single "Impossible" logic check. |
| **Primary Failure** | `OTHER_ERROR` | High frequency of total logic or hallucination collapses. |
| **Format Robustness**| 0.0% | Jittered formats treated as out-of-vocabulary tokens. |

**Technical Conclusion**: The zero-shot evaluation resulted in 0.0% Exact Match and a 2.3% Normalized Match (one successful logic check). This establishes the model's baseline level of temporal validation prior to Continuous Pre-Training (CPT). The diagnostic suite is slated for expansion to **100+ unique probe cases** in future iterations to ensure comprehensive coverage across edge conditions.

### Run 1: Proof of Training (PoT) Results
Post-training evaluation after 1,000 iterations on a 5k sample set.

| Metric | Result (CoT) | Observation |
| :--- | :--- | :--- |
| **Structural Match**| ~85% | Model successfully learned to emit `[THINK]` and `[RESULT]` blocks. |
| **Normalized Match**| 2.3% | Model follows the "Thinking" pattern but fails the arithmetic carry. |
| **Training Loss** | 0.164 | Significant drop from 2.97 starting loss. |
| **Status** | **Experimental**| Model successfully learned the `[THINK]` block formatting, but failed the arithmetic. Scaling to larger PoC dataset to continue testing. |

### Run 2: Generative Baseline Evaluation
Zero-shot evaluation of **SmolLM-135M** against 500 dynamically generated temporal reasoning records (using `generator.js`) across 12 domains. Test cases included format jitter, boundary rollovers, and multi-unit cascades.

| Metric | Result (Direct) | Observation |
| :--- | :--- | :--- |
| **Exact Match** | 0.0% | Model failed to produce a valid 24h format for any of the 469 valid temporal constraints. |
| **Normalized Match**| 0.0% | Model exhibited zero underlying temporal calculation capacity across all domains. |
| **Token Collapse** | 51 cases | Frequent hallucination of completely unrelated text. |
| **Conclusion** | **Verified** | Dynamic synthesis confirms the static probe results: the base model lacks all inherent temporal reasoning capacity. |

---

## Phase 1: Tiered Training

Tagzeit uses a progressive training approach to move from rapid local proof-of-concepts to production-scale models.

- **Phase 1a: PoC (TINY Mode)**: Training `SmolLM-135M` on CPU or local Mac hardware. 
    - **Status**: **Phase 0 Validated**. 56,938 records generated with balanced domain distribution (~7% per domain) and validated subtraction coverage.
    - **Objective**: Stabilize the mechanical arithmetic of the [THINK] block.
- **The [THINK] Block Methodology**: Models are trained to generate a chain-of-thought (CoT) trace that follows a deterministic state machine:
    1. Unit Isolation
    2. Overflow Check
    3. Carry Primitive
    4. Zero-Padding/Rollover Logic

---

## Phase 2: Validation & Diagnostics

Post-training, models are re-evaluated using the diagnostic suite to measure the training delta.

- **Hardware Benchmarking**: Recording tokens per second (TPS) across MLX and PyTorch backends.
- **Delta Analysis**: Comparing post-CPT (Continued Pre-Training) scores against the Phase 0 baseline saved in `baseline_smollm.json`.
- **Deployment**: Resulting LoRA adapters are exported for research use with `mlx-lm` or `transformers`.

---

## Roadmap & Generalization

Tagzeit is expanding beyond simple addition to cover more complex arithmetic and linguistic scenarios:

1.  **Temporal Subtraction**: Hardening reasoning for "What time was it X minutes ago?" which requires backwards borrow logic (base-60 borrow).
2.  **Multimodal Math**: Extending the 12 domains to general arithmetic (e.g., financial calculations like "market costs") while maintaining distinct reasoning chains for number vs. time problems.
3.  **12/24h Translation**: Direct training on converting between colloquial 12h formats and formal 24h targets.

---

## Project Structure

- `generator.js`: The synthetic data engine (Node.js).
- `validate.py`: The diagnostic probe and hardware benchmarker (Python).
- `cpt_train.py`: Tiered training script (supports Unsloth and standard PEFT).
- `train_poc.jsonl`: Generated PoC dataset.
- `baseline_smollm.json`: Reference scores for the 135M base model.

---

## Usage Guide

### 1. Installation
```bash
# Node dependencies (Data Gen)
npm install

# Python dependencies (Training/Eval)
pip install -r requirements.txt
```

### 2. Generate Data (Phase 0)
```bash
# General purpose training set
node generator.js --count 5000 --output train.jsonl

# Specific scripts from package.json
npm run generate:poc   # 50k records
```

### 3. Run Baseline Evaluation (Phase 0)
```bash
python validate.py --model_id HuggingFaceTB/SmolLM-135M --mode direct --output results.json
```

### 4. Continuous Pre-Training (Phase 1)
```bash
# Local PoC
python cpt_train.py --tiny --train_file train_poc.jsonl --eval_file eval_poc.jsonl
```
