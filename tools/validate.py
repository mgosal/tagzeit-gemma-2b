"""
Tagzeit Diagnostic Probe — validate.py
Zero-shot baseline measurement and post-CPT delta testing.
Supports: SmolLM2-135M (via MLX or PyTorch), Gemma-2-2b (via standard pipeline).

Features:
  1. Multi-Format Input Controller (Military / Standard / Spoken)
  2. Format Robustness (Spaced Digits)
  3. Boundary Probe Suite (hour / day / month rollovers)
  4. Two-Tier Scoring (Exact Match + Normalized Match)
  5. Zero-Shot Prompt Standardization (no CoT in baseline)
  6. Failure Mode Categorization (BASE_10_ERROR, TOKEN_COLLAPSE, FORMAT_HALLUCINATION)
  7. Hardware Benchmarking (tokens/sec)
  8. Two-Stage Post-Training ("No-Think Zone": Direct vs CoT)
"""

import re
import time
import json
import argparse

# ---------------------------------------------------------------------------
# Test Cases: Each has an "id", the core question parts, and the expected answer.
# The harness wraps each into 3 linguistic skins automatically.
# ---------------------------------------------------------------------------
TEST_CASES = [
    # === Boundary Probe Suite (Rollover-Only) ===
    {"id": "BP-01", "start": "23:59", "delta": "1 minute",  "expected": "00:00", "category": "hour_rollover"},
    {"id": "BP-02", "start": "11:59", "delta": "1 minute",  "expected": "12:00", "category": "hour_rollover"},
    {"id": "BP-03", "start": "23:45", "delta": "30 minutes","expected": "00:15", "category": "hour_rollover"},
    {"id": "BP-04", "start": "18:59", "delta": "1 minute",  "expected": "19:00", "category": "hour_rollover"},
    {"id": "BP-05", "start": "12:55", "delta": "10 minutes","expected": "13:05", "category": "hour_rollover"},
    {"id": "BP-06", "start": "09:58", "delta": "5 minutes", "expected": "10:03", "category": "minute_carry"},
    {"id": "BP-07", "start": "00:00", "delta": "1 minute",  "expected": "00:01", "category": "midnight_boundary"},
    {"id": "BP-08", "start": "23:30", "delta": "45 minutes","expected": "00:15", "category": "hour_rollover"},
    {"id": "BP-09", "start": "22:50", "delta": "90 minutes","expected": "00:20", "category": "hour_rollover"},
    {"id": "BP-10", "start": "17:45", "delta": "20 minutes","expected": "18:05", "category": "minute_carry"},

    # === Standard Time Math ===
    {"id": "TM-01", "start": "08:30", "delta": "45 minutes","expected": "09:15", "category": "standard"},
    {"id": "TM-02", "start": "14:00", "delta": "90 minutes","expected": "15:30", "category": "standard"},
    {"id": "TM-03", "start": "06:15", "delta": "30 minutes","expected": "06:45", "category": "standard"},
    {"id": "TM-04", "start": "10:10", "delta": "50 minutes","expected": "11:00", "category": "minute_carry"},
    {"id": "TM-05", "start": "15:40", "delta": "25 minutes","expected": "16:05", "category": "minute_carry"},
    {"id": "TM-06", "start": "07:00", "delta": "120 minutes","expected": "09:00", "category": "standard"},
    {"id": "TM-07", "start": "20:30", "delta": "15 minutes","expected": "20:45", "category": "standard"},
    {"id": "TM-08", "start": "11:11", "delta": "11 minutes","expected": "11:22", "category": "standard"},
    {"id": "TM-09", "start": "03:50", "delta": "70 minutes","expected": "05:00", "category": "minute_carry"},
    {"id": "TM-10", "start": "19:25", "delta": "35 minutes","expected": "20:00", "category": "minute_carry"},

    # === Temporal Generalization (Unseen Domains) ===
    {"id": "TG-01", "start": "14:45", "delta": "85 minutes","expected": "16:10", "category": "generalization"},
    {"id": "TG-02", "start": "08:30", "delta": "200 minutes","expected": "11:50", "category": "generalization"},
    {"id": "TG-03", "start": "23:15", "delta": "130 minutes","expected": "01:25", "category": "generalization"},
    {"id": "TG-04", "start": "05:55", "delta": "7 minutes", "expected": "06:02", "category": "minute_carry"},
    {"id": "TG-05", "start": "16:48", "delta": "12 minutes","expected": "17:00", "category": "minute_carry"},

    # === Multi-Unit Cascade (Triple Carry) ===
    {"id": "MC-01", "start": "23:59", "delta": "2 minutes", "expected": "00:01", "category": "cascade"},
    {"id": "MC-02", "start": "23:55", "delta": "10 minutes","expected": "00:05", "category": "cascade"},
    {"id": "MC-03", "start": "23:01", "delta": "59 minutes","expected": "00:00", "category": "cascade"},
    {"id": "MC-04", "start": "23:30", "delta": "31 minutes","expected": "00:01", "category": "cascade"},
    {"id": "MC-05", "start": "22:59", "delta": "61 minutes","expected": "00:00", "category": "cascade"},

    # === Impossible / Error Detection ===
    {"id": "IM-01", "start": "12:65", "delta": "5 minutes", "expected": "INVALID", "category": "impossible"},
    {"id": "IM-02", "start": "25:00", "delta": "10 minutes","expected": "INVALID", "category": "impossible"},
    {"id": "IM-03", "start": "13:45 PM", "delta": "5 minutes","expected": "INVALID", "category": "impossible"},

    # === Semantic Equivalence Anchors ===
    {"id": "SE-01", "start": "13:15", "delta": "10 minutes","expected": "13:25", "category": "semantic_eq"},
    {"id": "SE-02", "start": "06:30", "delta": "5 minutes", "expected": "06:35", "category": "semantic_eq"},
    {"id": "SE-03", "start": "09:45", "delta": "15 minutes","expected": "10:00", "category": "semantic_eq"},
    {"id": "SE-04", "start": "12:00", "delta": "30 minutes","expected": "12:30", "category": "semantic_eq"},
    {"id": "SE-05", "start": "18:00", "delta": "45 minutes","expected": "18:45", "category": "semantic_eq"},

    # === Spaced Digit Format Robustness ===
    {"id": "FR-01", "start": "1 8 : 5 9", "delta": "1 minute","expected": "19:00", "category": "format_robust"},
    {"id": "FR-02", "start": "2 3 : 5 5", "delta": "10 minutes","expected": "00:05", "category": "format_robust"},
    {"id": "FR-03", "start": "0 9 : 5 0", "delta": "15 minutes","expected": "10:05", "category": "format_robust"},
    {"id": "FR-04", "start": "1 1 : 4 5", "delta": "20 minutes","expected": "12:05", "category": "format_robust"},
    {"id": "FR-05", "start": "0 0 : 5 8", "delta": "3 minutes","expected": "01:01", "category": "format_robust"},

    # === Subtraction Suite (Backwards Borrow) ===
    {"id": "SB-01", "start": "10:00", "delta": "15 minutes ago", "expected": "09:45", "category": "subtraction"},
    {"id": "SB-02", "start": "01:15", "delta": "30 minutes ago", "expected": "00:45", "category": "subtraction"},
    {"id": "SB-03", "start": "00:05", "delta": "10 minutes ago", "expected": "23:55", "category": "subtraction"},
    {"id": "SB-04", "start": "12:00", "delta": "120 minutes ago","expected": "10:00", "category": "subtraction"},
    {"id": "SB-05", "start": "06:30", "delta": "45 minutes ago", "expected": "05:45", "category": "subtraction"},
]

# ---------------------------------------------------------------------------
# Linguistic Skin Mappings (for Semantic Equivalence)
# ---------------------------------------------------------------------------
SPOKEN_MAP = {
    "13:15": "quarter past one in the afternoon",
    "06:30": "half past six in the morning",
    "09:45": "quarter to ten in the morning",
    "12:00": "noon",
    "18:00": "six o'clock in the evening",
}

STANDARD_MAP = {
    "13:15": "1:15 PM",
    "06:30": "6:30 AM",
    "09:45": "9:45 AM",
    "12:00": "12:00 PM",
    "18:00": "6:00 PM",
}

def load_test_cases_from_jsonl(filepath):
    cases = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            text = data.get("text", "")
            if "[ANSWER]" in text:
                parts = text.split("\n[THINK]")
                if len(parts) >= 2:
                    question = parts[0].strip()
                    ans_match = re.search(r'\[ANSWER\] (.*?) \[\/ANSWER\]', text)
                    if ans_match:
                        expected = ans_match.group(1).strip()
                        cat = "shadow_pair" if "What is" in question else "generated_temporal"
                        cases.append({
                            "id": f"GEN-{i:03d}",
                            "start": "",
                            "delta": "",
                            "expected": expected,
                            "category": cat,
                            "raw_question": question
                        })
    return cases

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a precise time calculator. Given a starting time and a duration, calculate the resulting time. Respond with ONLY the final time in HH:MM 24-hour format. Do not explain your reasoning."

SYSTEM_PROMPT_COT = "You are a precise time calculator. Given a starting time and a duration, calculate the resulting time. Show your reasoning in a [THINK] block, then give the final time in HH:MM 24-hour format."
SYSTEM_PROMPT_ROUTE = "You are a precise temporal reasoning engine. Extract the underlying temporal logic from the user prompt and express it using [ROUTE_...] tokens. Do not provide a verbal explanation, only the routing call."


def build_prompt(test_case, skin="military", mode="direct"):
    """Build a prompt for a given test case, skin, and mode."""
    start = test_case["start"]
    delta = test_case["delta"]

    if skin == "spoken" and start in SPOKEN_MAP:
        start = SPOKEN_MAP[start]
    elif skin == "standard" and start in STANDARD_MAP:
        start = STANDARD_MAP[start]

    if "ago" in delta:
        question = f"What time was it {delta} before {start}?"
    else:
        question = f"What time is it {delta} after {start}?"

    if test_case["category"] == "impossible":
        question = f"What time is it {delta} after {start}? If the input time is invalid, respond with INVALID."

    if mode == "route":
        sys = SYSTEM_PROMPT_ROUTE
    else:
        sys = SYSTEM_PROMPT if mode == "direct" else SYSTEM_PROMPT_COT
    return sys, question


# ---------------------------------------------------------------------------
# Time Normalizer (Two-Tier Scoring)
# ---------------------------------------------------------------------------
def normalize_time(raw_output):
    """Extract and normalize a time from model output.
    Priority: [ANSWER] > [RESULT] > last valid HH:MM > INVALID > raw.
    """
    # Priority 1: [ANSWER] block
    answer_match = re.search(r'\[ANSWER\]\s*(\d{1,2})\s*:\s*(\d{2})', raw_output)
    if answer_match:
        h, m = int(answer_match.group(1)), int(answer_match.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"

    # Priority 2: [RESULT] block
    result_match = re.search(r'\[RESULT\]\s*(\d{1,2})\s*:\s*(\d{2})', raw_output)
    if result_match:
        h, m = int(result_match.group(1)), int(result_match.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"

    # Priority 3: Last valid HH:MM in output (not first — avoids grabbing start time)
    all_times = re.findall(r'(\d{1,2})\s*:\s*(\d{2})', raw_output)
    for h_str, m_str in reversed(all_times):
        h, m = int(h_str), int(m_str)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"

    # Check for INVALID
    if "INVALID" in raw_output.upper() or "invalid" in raw_output.lower():
        return "INVALID"

    return raw_output.strip()


# ---------------------------------------------------------------------------
# Failure Mode Classifier
# ---------------------------------------------------------------------------
def classify_failure(expected, actual, raw_output):
    """Classify the type of failure."""
    if expected == actual:
        return "PASS"

    # BASE_10_ERROR: outputs like 18:60, 23:60, 24:00
    if re.search(r'\d{1,2}:6[0-9]', raw_output) or re.search(r'\d{1,2}:9[0-9]', raw_output):
        return "BASE_10_ERROR"
    if re.search(r'24:\d{2}', raw_output):
        return "BASE_10_ERROR"

    # TOKEN_COLLAPSE: treats timestamp as a year or number
    if re.search(r'(century|year|age|era|18\d\d\s)', raw_output, re.IGNORECASE):
        return "TOKEN_COLLAPSE"

    # FORMAT_HALLUCINATION: changed format unexpectedly
    if re.search(r'(half past|quarter|o\'clock|AM|PM)', raw_output, re.IGNORECASE):
        return "FORMAT_HALLUCINATION"

    return "OTHER_ERROR"


# ---------------------------------------------------------------------------
# Model Loading (Abstracted for MLX / PyTorch)
# ---------------------------------------------------------------------------
def load_model(model_id, adapter_path=None, backend="auto"):
    """Load a model. Tries MLX first on macOS, falls back to PyTorch."""
    model = None
    tokenizer = None
    engine = None

    if backend in ("mlx", "auto"):
        try:
            from mlx_lm import load, generate as mlx_generate
            print(f"Loading {model_id} via MLX...")
            model, tokenizer = load(model_id, adapter_path=adapter_path)
            engine = "mlx"
            print(f"  ✓ MLX loaded successfully.")
            return model, tokenizer, engine
        except (ImportError, Exception) as e:
            if backend == "mlx":
                raise
            print(f"  MLX unavailable ({e}), trying PyTorch...")

    if backend in ("torch", "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os, json
        print(f"Loading {model_id} via PyTorch...")

        # Detect if model_id is a PEFT adapter directory
        adapter_config_path = os.path.join(model_id, "adapter_config.json")
        is_peft_dir = os.path.isfile(adapter_config_path)

        if is_peft_dir:
            # ── LoRA adapter directory ────────────────────────────────
            # 1. Read the base model name from adapter_config.json
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", model_id)
            print(f"  Detected PEFT adapter. Base model: {base_model_id}")

            # 2. Load tokenizer from the adapter dir (includes domain tokens)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 3. Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.float16, device_map="auto"
            )

            # 4. Resize embeddings if tokenizer has extra tokens (domain tokens)
            if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
                print(f"  Resizing embeddings: {base_model.get_input_embeddings().weight.shape[0]} → {len(tokenizer)}")
                base_model.resize_token_embeddings(len(tokenizer))

            # 5. Apply the LoRA adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, model_id)
            print(f"  ✓ LoRA adapter applied.")
        else:
            # ── Standard model or full fine-tune ──────────────────────
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)

        engine = "torch"
        print(f"  ✓ PyTorch loaded successfully.")

    return model, tokenizer, engine


def format_prompt(tokenizer, system_prompt, question):
    """Build a prompt string using the tokenizer's built-in chat template.

    Handles model-specific formatting automatically:
      - SmolLM2 / ChatML models → <|im_start|>system ... <|im_start|>assistant
      - Gemma-2B-it             → <start_of_turn>user ... <start_of_turn>model
      - Others                  → Whatever template the tokenizer defines

    Falls back to raw ChatML for base models with no chat template.

    NOTE (Gemma): Gemma instruct models do not support a "system" role.
    When this is detected, the system prompt is merged into the user message.
    This is functionally equivalent but should be noted in experiment logs.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    # --- Try the tokenizer's native chat template first ---
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some models (e.g. Gemma) don't support a "system" role.
            # Merge system instructions into the user message and retry.
            merged = [
                {"role": "user", "content": f"{system_prompt}\n\n{question}"},
            ]
            return tokenizer.apply_chat_template(
                merged,
                tokenize=False,
                add_generation_prompt=True,
            )

    # --- Fallback: raw ChatML (for base models / older checkpoints) ---
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_response(model, tokenizer, engine, system_prompt, question, max_tokens=128):
    """Generate a response from the model."""
    start_time = time.time()
    total_tokens = 0
    prompt = format_prompt(tokenizer, system_prompt, question)

    if engine == "mlx":
        from mlx_lm import generate as mlx_generate
        response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        total_tokens = len(tokenizer.encode(response))
    else:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        total_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    elapsed = time.time() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0

    return response, tps


# ---------------------------------------------------------------------------
# Main Harness
# ---------------------------------------------------------------------------
def run_validation(model, tokenizer, engine, test_cases, mode="direct", skins=None):
    """Run the full diagnostic probe."""
    if skins is None:
        skins = ["military"]

    results = []
    total_exact = 0
    total_normalized = 0
    total_tests = 0
    failure_counts = {"BASE_10_ERROR": 0, "TOKEN_COLLAPSE": 0, "FORMAT_HALLUCINATION": 0, "OTHER_ERROR": 0}
    tps_samples = []
    category_scores = {}

    for tc in test_cases:
        for skin in skins:
            if "raw_question" in tc:
                if skin != skins[0]: continue
                question = tc["raw_question"]
                sys_prompt = SYSTEM_PROMPT if mode == "direct" else SYSTEM_PROMPT_COT
            else:
                # Skip non-military skins for non-semantic-eq tests
                if skin != "military" and tc["category"] != "semantic_eq":
                    continue
                # Skip spoken/standard if no mapping exists
                if skin == "spoken" and tc["start"] not in SPOKEN_MAP:
                    continue
                if skin == "standard" and tc["start"] not in STANDARD_MAP:
                    continue

                sys_prompt, question = build_prompt(tc, skin=skin, mode=mode)
                
            raw_response, tps = generate_response(model, tokenizer, engine, sys_prompt, question)
            tps_samples.append(tps)

            normalized = normalize_time(raw_response)
            expected = tc["expected"]

            exact_match = (raw_response.strip() == expected)
            norm_match = (normalized == expected)
            failure_tag = classify_failure(expected, normalized, raw_response)

            total_tests += 1
            if exact_match:
                total_exact += 1
            if norm_match:
                total_normalized += 1
            if failure_tag != "PASS" and failure_tag in failure_counts:
                failure_counts[failure_tag] += 1

            cat = tc["category"]
            if cat not in category_scores:
                category_scores[cat] = {"total": 0, "correct": 0}
            category_scores[cat]["total"] += 1
            if norm_match:
                category_scores[cat]["correct"] += 1

            result = {
                "id": tc["id"],
                "skin": skin,
                "mode": mode,
                "question": question,
                "expected": expected,
                "raw_response": raw_response.strip()[:200],
                "normalized": normalized,
                "exact_match": exact_match,
                "norm_match": norm_match,
                "failure_tag": failure_tag,
                "tokens_per_second": round(tps, 1),
            }
            results.append(result)

            status = "✓" if norm_match else f"✗ [{failure_tag}]"
            print(f"  {tc['id']:6s} [{skin:8s}] {status:25s} expected={expected:6s} got={normalized:6s}")

    # Summary
    avg_tps = sum(tps_samples) / len(tps_samples) if tps_samples else 0
    print(f"\n{'='*60}")
    print(f"RESULTS ({mode.upper()} mode)")
    print(f"{'='*60}")
    print(f"  Total Tests:      {total_tests}")
    print(f"  Exact Match:      {total_exact}/{total_tests} ({100*total_exact/total_tests:.1f}%)")
    print(f"  Normalized Match: {total_normalized}/{total_tests} ({100*total_normalized/total_tests:.1f}%)")
    print(f"  Avg tok/s:        {avg_tps:.1f}")
    print(f"\n  Failure Modes:")
    for tag, count in failure_counts.items():
        print(f"    {tag}: {count}")
    print(f"\n  Category Breakdown:")
    for cat, scores in sorted(category_scores.items()):
        pct = 100 * scores["correct"] / scores["total"] if scores["total"] > 0 else 0
        print(f"    {cat:20s}: {scores['correct']}/{scores['total']} ({pct:.0f}%)")

    return {
        "mode": mode,
        "total_tests": total_tests,
        "exact_match": total_exact,
        "normalized_match": total_normalized,
        "exact_pct": round(100 * total_exact / total_tests, 1) if total_tests else 0,
        "norm_pct": round(100 * total_normalized / total_tests, 1) if total_tests else 0,
        "avg_tps": round(avg_tps, 1),
        "failure_counts": failure_counts,
        "category_scores": {k: {**v, "pct": round(100*v["correct"]/v["total"], 1) if v["total"] else 0} for k,v in category_scores.items()},
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Tagzeit Diagnostic Probe")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID (HuggingFace or local path)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (post-CPT)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "mlx", "torch"])
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "cot", "both"])
    parser.add_argument("--skins", type=str, default="military", help="Comma-separated: military,standard,spoken")
    parser.add_argument("--output", type=str, default="results/evaluation.json", help="Save results to JSON file")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a generated JSONL file to use instead of static TEST_CASES")
    args = parser.parse_args()

    if args.input_file:
        test_cases = load_test_cases_from_jsonl(args.input_file)
        skins = ["generated"]
    else:
        test_cases = TEST_CASES
        skins = [s.strip() for s in args.skins.split(",")]
    model, tokenizer, engine = load_model(args.model_id, args.adapter_path, args.backend)

    if model is None:
        print("ERROR: Could not load model.")
        return

    all_results = {}

    modes = ["direct", "cot"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} mode baseline...")
        print(f"{'='*60}")
        result = run_validation(model, tokenizer, engine, test_cases, mode=mode, skins=skins)
        all_results[mode] = result

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
