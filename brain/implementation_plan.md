# Run 3: Proof of Training (PoT) Execution

The goal of this task is to execute **Run 3**: proving that the `SmolLM-135M` model can actually learn the temporal reasoning arithmetic (via the `[THINK]` block) by training it on a freshly generated 5k sample set.

## Proposed Changes

### 1. Data Generation
We will use `generator.js` to create a 5,000-record dataset specifically for the PoT.
- **Train File**: `train_pot.jsonl`
- **Eval File**: `eval_pot.jsonl`
- **Flag**: `--compact` (since `SmolLM-135M` is a small model, the compact base-60 trace is better suited for its attention window).

### 2. Updating `cpt_train.py`
The `cpt_train.py` script currently expects a JSONL with `instruction` and `response` fields. Our generator outputs a single `text` field containing the prompt and the expected CoT response.
#### [MODIFY] `cpt_train.py`
- Modify the `formatting_func` to just return the `text` field directly from the dataset, or map the `text` field correctly so `SFTTrainer` can consume it.

### 3. Training Execution
Run the modified `cpt_train.py` in `--tiny` mode to train `SmolLM-135M` on CPU/local hardware.
- Monitor the training loss to ensure it is dropping.
- Save the resulting LoRA adapter.

### 4. Documentation
Document the Run 3 results directly in the `README.md` engineering log, noting the final training loss, and evaluating the adapter using `validate.py` in `--mode cot` to see if it learned the arithmetic.

## Verification Plan
### Automated Tests
Run `validate.py` passing the trained `--adapter_path` to verify the delta against the 0.0% baseline.
