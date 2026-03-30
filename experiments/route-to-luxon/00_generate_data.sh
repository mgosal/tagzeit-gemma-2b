#!/bin/bash
set -e

# Move to project root to run node script
cd "$(dirname "$0")/../.."

# Create necessary output directories
mkdir -p data/train data/eval

echo "Generating Route-to-Luxon Temporal Dataset..."
# Generate 5,000 data points configured for the model's structure
node core/synthetic_data/generators/temporal/generator.js \
    --count 5000 \
    --output data/train/train_routed.jsonl \
    --eval data/eval/eval_routed.jsonl \
    --negatives 0.05

echo "Generation complete."
echo "Train JSONL output to: data/train/train_routed.jsonl"
echo "Eval JSONL output to: data/eval/eval_routed.jsonl"
