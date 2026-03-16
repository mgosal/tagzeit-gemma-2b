# Project Tagzeit: Temporal Reasoning for LLMs

> [!NOTE]
> **Repository Scope**: While the repository is named `tagzeit-gemma-2b` to reflect the ultimate target model, current developmental work and the Proof of Technology (PoT) use `SmolLM-135M` as a baseline.

Tagzeit is an experiment in teaching Large Language Models (LLMs) precise temporal reasoning through deterministic chain-of-thought (CoT) formatting. By training models to follow a specific base-60 logic trace, Tagzeit enables reliable arithmetic for time math, boundary conditions, and multi-unit carries.

⚠️ **Disclaimer**: This is an experimental research project. It is developed and tested on a specific local environment (Mac M-series). We make no guarantees that it will work on your system, hardware, or setup. Use at your own risk.

---

## Key Concepts & Terminology

We link to authoritative sources rather than providing our own definitions, as we are learning alongside the community:

- [Supervised Fine-Tuning (SFT)](https://huggingface.co/docs/trl/sft_trainer) — Hugging Face TRL documentation
- [Continued Pre-Training (CPT)](https://huggingface.co/blog/continued-pretraining) — Hugging Face Blog
- [Chain-of-Thought (CoT)](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [LoRA / PEFT](https://huggingface.co/docs/peft/main/en/index) — Hugging Face PEFT documentation
- [BIG-bench](https://github.com/google/BIG-bench) — Google Research Date Understanding

---

## Roadmap: Phased Training

1. **Current Phase: SFT (Supervised Fine-Tuning)**
   - Focus: Teaching the model *how to show its work* using explicit [THINK] traces.
   - Script: `src/sft_train.py`
2. **Future Phase: CPT (Continued Pre-Training)**
   - Goal: Teaching the model *temporal intuition* by immersing it in unlabeled temporal text before instruction tuning.
   - Script: `src/cpt_train.py` (Planned)

---

## Results & Observations

| Run | Model | Data | Steps | Result | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | SmolLM-135M | 1,000 | 500 | Success | **Baseline**: Learned format, but 0% math. |
| 2 | Gemma-2-2b | 5,000 | 1,000 | Success | **Generalization**: Partial reasoning, delims unstable. |
| 3 | SmolLM-135M | 5,000 | 250 | **FAILED** | **OOM**: MPS Out-of-Memory at Step 250 (PoT). |

Detailed results and baseline scores are stored in the [`results/`](./results/) directory.

---

## Project Structure

- `src/`
  - `generator.js`: The synthetic data engine (Node.js).
  - `sft_train.py`: Supervised Fine-Tuning script (trl/peft).
  - `validate.py`: Diagnostic probe and hardware benchmarker.
- `data/`
  - `train/`: Generated training corpora (JSONL).
  - `eval/`: Evaluation datasets (JSONL).
- `results/`: Evaluation outputs and baseline scores.
- `configs/`: LoRA and model configurations.

---

## Usage Guide

### 1. Installation

```bash
# Node dependencies (Data Gen)
npm install

# Python dependencies (Training/Eval)
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# General purpose training set
node src/generator.js --count 5000 --output data/train/train.jsonl

# Specific scripts from package.json
npm run generate:poc   # 50k records
```

### 3. Run Evaluation

```bash
# Zero-shot baseline or post-training check
python src/validate.py --model_id HuggingFaceTB/SmolLM-135M --mode direct --output results/baseline.json
```

### 4. Supervised Fine-Tuning (SFT)

```bash
# Local PoC (TINY mode)
python src/sft_train.py --tiny --train_file data/train/train_poc.jsonl --eval_file data/eval/eval_poc.jsonl
```
