# Project Tagzeit: Temporal Reasoning for LLMs

Tagzeit is an experiment in teaching Large Language Models (LLMs) precise temporal reasoning through deterministic chain-of-thought (CoT) formatting. By training models to follow a specific base-60 logic trace, Tagzeit enables reliable arithmetic for time math, boundary conditions, and multi-unit carries.

---

## Models & Environment

| Tier | Model ID | Purpose | Backend |
| :--- | :--- | :--- | :--- |
| **PoT (Proof of Training)** | `HuggingFaceTB/SmolLM-135M` | Core logic & block-output validation | PyTorch / MLX |
| **PoC (Proof of Concept)**  | `google/gemma-2-2b` | Scale testing for complex reasoning | PyTorch / MLX |

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

### Engineering Log & Run History

| Run | Model | Data | Steps | Result | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | SmolLM-135M | 1,000 (v1) | 500 | Success | **Baseline**: Learned format, but 0% math. |
| 2 | Gemma-2-2b | 5,000 (v2) | 1,000 | Success | **Generalization**: Partial reasoning, delims unstable. |
| 3 | SmolLM-135M | 5,000 (v3) | 250 | **FAILED** | **OOM**: MPS Out-of-Memory at Step 250 (PoT). |
| 3.1 | SmolLM-135M | 5,000 (v3) | 250 | *Planned* | **Reduced Weight**: Dialed down to 250 steps. |

---

## Phase 1: Tiered Training

Tagzeit uses a progressive training approach to move from rapid local proof-of-concepts to production-scale models.

- **Phase 1a: PoC (TINY Mode)**: Training `SmolLM-135M` on CPU or local Mac hardware. 
    - **Status**: **Phase 0 Validated**. 56,938 records generated with balanced domain distribution and validated subtraction coverage.
    - **Objective**: Stabilize the mechanical arithmetic of the [THINK] block.
- **The [THINK] Block Methodology**: Models are trained to generate a chain-of-thought (CoT) trace that follows a deterministic state machine:
    1. **Unit Isolation**: Identifying the starting hours and minutes.
    2. **Overflow Check**: Determining if the addition/subtraction triggers a rollover.
    3. **Carry/Borrow Primitives**: Executing the base-60 arithmetic logic.
    4. **Resolution**: Emitting the final 24h formatted result.

---

## Phase 2: Validation & Diagnostics

Post-training, models are re-evaluated using the diagnostic suite to measure the training delta.

- **Hardware Benchmarking**: Recording tokens per second (TPS) across MLX and PyTorch backends.
- **Delta Analysis**: Comparing post-CPT (Continued Pre-Training) scores against the Phase 0 baseline saved in `baseline_smollm.json`.
- **Deployment**: Resulting LoRA adapters are exported for research use with `mlx-lm` or `transformers`.

---

## Data Generation (v3)
The Tagzeit generator recently underwent a major refactor to v3 (following a high-fidelity Opus audit) to ensure arithmetic grounding and logic variety.

### Generator Comparison (v2 vs v3)

| Feature | v2 (Baseline) | v3 (Hardened) | Training Impact |
| :--- | :--- | :--- | :--- |
| **Math Traces** | Implicit ("Carry 1") | **Explicit** (`1 * 60 = 60`) | Eliminates "magic" carries. |
| **Delimiters** | Inconsistent | **Strict** `[THINK] ... [/THINK]` | Enforces structural adherence. |
| **12h Support** | 0% (Fuzzy only) | **35%** (AM/PM jitter) | Real-world format resilience. |
| **Logic Variety**| Basic Domains | **12 Human Areas** | Generalizes beyond simple +/-. |
| **Humanized Logic**| None | Medicine, TZ, Calendar | Logic-state machine diversity. |
| **Stability** | No EOS | `<|endoftext|>` | Predictable sequence termination. |

### Domain Variety (v3 Categories)
- **Medicine**: Frequency reasoning ("Every 8 hours", "3 times a day").
- **Travel/Tech**: Time Zone offsets (UTC±X arithmetic).
- **Calendar/History**: Day-boundary logic ("tomorrow", "crossing days").
- **Procrastination/Social**: Corrected semantic direction (Fixing v2 inversions).

---

## Roadmap & Generalization

Tagzeit is expanding beyond simple addition to cover more complex arithmetic and linguistic scenarios:

1.  **Temporal Subtraction**: Hardening reasoning for "What time was it X minutes ago?" which requires backwards borrow logic (base-60 borrow).
2.  **12h/24h Translation**: Direct training on converting between colloquial 12h formats (AM/PM) and formal 24h targets.
3.  **Arithmetic Hardening (Run 3)**: Ensuring the model doesn't just learn formatting, but actually executes the internal carries correctly.

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
