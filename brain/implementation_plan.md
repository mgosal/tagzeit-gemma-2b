# Run 3: Proof of Training (PoT) Execution

The goal of this task is to execute **Run 3**: proving that the `SmolLM-135M` model can actually learn the temporal reasoning arithmetic (via the `[THINK]` block) by training it on a freshly generated 5k sample set.

## Proposed Changes

### 1. Data Generation
We will use `generator.js` to create a 5,000-record dataset specifically for the Initial Training Run.
- **Train File**: `train_initial_training_run.jsonl`
- **Eval File**: `eval_initial_training_run.jsonl`
- **Flag**: `--compact` (since `SmolLM-135M` is a small model, the compact base-60 trace is better suited for its attention window).

### 2. Deep Script Audit (OpenRouter/Opus Bridge)
Before modifying the training script, we will use the `openrouter_ask_model` tool to consult **Claude 3 Opus**.
- We will provide it with `cpt_train.py` and a sample of `train_initial_training_run.jsonl`.
- Goal: Get a high-fidelity audit of the `formatting_func` and `SFTTrainer` configuration to ensure compatibility with our specific JSONL `text` field and the tiny `135M` model architecture.

### 3. Updating `cpt_train.py`
Based on the Opus audit, we will modify `cpt_train.py`:
- Modify the `formatting_func` to correctly pass through the `text` field.
- Ensure `max_seq_length` and batch sizes are optimized for the `SmolLM-135M` context window and local CPU/MPS training.

### 4. Training Execution
Run the modified `cpt_train.py` in `--tiny` mode to train `SmolLM-135M` on CPU/local hardware.
- Monitor the training loss to ensure it is dropping.
- Save the resulting LoRA adapter.

### 4. Documentation
Document the Run 3 results directly in the `README.md` engineering log, noting the final training loss, and evaluating the adapter using `validate.py` in `--mode cot` to see if it learned the arithmetic.

## Verification Plan
### Automated Tests
Run `validate.py` passing the trained `--adapter_path` to verify the delta against the 0.0% baseline.
