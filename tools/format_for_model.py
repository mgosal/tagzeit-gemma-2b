#!/usr/bin/env python3
"""
Training Data Format Adapter
==============================
Converts model-agnostic raw training data (question\\nanswer) into
chat-template-wrapped format for a specific target model.

The generator stays model-agnostic. This adapter sits between the
generator output and the training pipeline:

    Generator (raw Q/A) → format_for_model.py → Model-specific JSONL

Supports any HuggingFace model with a chat template.

Usage:
    python tools/format_for_model.py \\
        --model_id google/gemma-2-2b-it \\
        --input data/train/train_routed.jsonl \\
        --output data/train/train_routed_gemma.jsonl

    python tools/format_for_model.py \\
        --model_id HuggingFaceTB/SmolLM2-360M-Instruct \\
        --input data/train/train_routed.jsonl \\
        --output data/train/train_routed_smollm2.jsonl
"""

import argparse
import json
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_roles(model_id, tokenizer):
    """Detect whether the model uses 'assistant' or 'model' as the response role.

    Gemma uses 'model', most others use 'assistant'.
    """
    # Try 'assistant' first (most common)
    test_msgs = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "reply"},
    ]
    try:
        tokenizer.apply_chat_template(test_msgs, tokenize=False)
        return "assistant"
    except Exception:
        pass

    # Try 'model' (Gemma)
    test_msgs[1]["role"] = "model"
    try:
        tokenizer.apply_chat_template(test_msgs, tokenize=False)
        return "model"
    except Exception:
        pass

    print(f"WARNING: Could not detect response role for {model_id}, defaulting to 'assistant'")
    return "assistant"


def has_system_role(tokenizer):
    """Check if the model's chat template supports a system role."""
    test_msgs = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "test"},
    ]
    try:
        tokenizer.apply_chat_template(test_msgs, tokenize=False)
        return True
    except Exception:
        return False


def format_sample(tokenizer, question, answer, response_role, system_supported, system_prompt=None):
    """Wrap a raw Q/A pair in the model's chat template.

    Returns the fully formatted text string ready for SFT (with the
    model's response included, so the trainer learns to produce it).
    """
    messages = []

    # Add system prompt if provided and supported
    if system_prompt and system_supported:
        messages.append({"role": "system", "content": system_prompt})
    elif system_prompt and not system_supported:
        # Merge into user message (Gemma-style)
        question = f"{system_prompt}\n\n{question}"

    messages.append({"role": "user", "content": question})
    messages.append({"role": response_role, "content": answer})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw training data to model-specific chat format"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID (e.g. google/gemma-2-2b-it)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file (raw format: question\\nanswer)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file (chat-template wrapped)")
    parser.add_argument("--eval_input", type=str, default=None,
                        help="Optional eval JSONL input")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="Optional eval JSONL output")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Optional system prompt to prepend")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print 3 samples and exit")
    args = parser.parse_args()

    # Load tokenizer
    hf_home = os.environ.get("HF_HOME", None)
    from transformers import AutoTokenizer
    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Detect model capabilities
    response_role = detect_roles(args.model_id, tokenizer)
    system_supported = has_system_role(tokenizer)
    print(f"  Response role: {response_role}")
    print(f"  System role supported: {system_supported}")

    def process_file(input_path, output_path):
        """Process a single JSONL file."""
        count = 0
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                record = json.loads(line.strip())
                raw_text = record["text"]

                # Split on first newline: question\nanswer
                parts = raw_text.split("\n", 1)
                if len(parts) != 2:
                    print(f"  WARNING: Skipping malformed record (no newline): {raw_text[:80]}")
                    continue

                question, answer = parts

                formatted = format_sample(
                    tokenizer, question, answer,
                    response_role, system_supported,
                    args.system_prompt
                )

                fout.write(json.dumps({"text": formatted}) + "\n")
                count += 1

                if args.dry_run and count >= 3:
                    print(f"\n--- Sample {count} ---")
                    print(formatted)
                    print()

        return count

    if args.dry_run:
        print(f"\n=== DRY RUN: Showing first 3 samples ===\n")
        process_file(args.input, "/dev/null")
        return

    # Process train file
    print(f"\nProcessing {args.input} → {args.output}")
    train_count = process_file(args.input, args.output)
    print(f"  Wrote {train_count} samples")

    # Process eval file if provided
    if args.eval_input and args.eval_output:
        print(f"Processing {args.eval_input} → {args.eval_output}")
        eval_count = process_file(args.eval_input, args.eval_output)
        print(f"  Wrote {eval_count} samples")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
