#!/bin/bash
set -e

# Change into the project root
cd "$(dirname "$0")/../.."

# Ensure datasets exist, otherwise prompt
if [ ! -f "data/train/train_routed.jsonl" ]; then
    echo "ERROR: Training data not found. Please run experiments/route-to-luxon/00_generate_data.sh first."
    exit 1
fi

echo "Starting SmolLM2-360M Route-to-Luxon Fine-Tuning..."
echo "Using Configuration: experiments/route-to-luxon/configs/003-smollm2-360m.yaml"
echo "We are strictly mapping the output artifacts to the experiment namespace."

# Kick off the training program with explicit file paths and parameters
# We set HF_HOME locally to bypass global Cache permission errors
HF_HOME="$(pwd)/.hf_cache" .venv/bin/python tools/sft_train.py \
    --model_id "HuggingFaceTB/SmolLM2-360M-Instruct" \
    --train_file "data/train/train_routed.jsonl" \
    --eval_file "data/eval/eval_routed.jsonl" \
    --output_dir "experiments/route-to-luxon/weights/003-smollm2-360m-instruct" \
    --no_lora

echo "================================================================"
echo "Training successful!"
echo "Model safely saved in: experiments/route-to-luxon/weights/003-smollm2-360m-instruct"
echo "Next step: You should run pipeline testing on this checkpoint:"
echo "python tools/validate_route.py --model_id experiments/route-to-luxon/weights/003-smollm2-360m-instruct"
