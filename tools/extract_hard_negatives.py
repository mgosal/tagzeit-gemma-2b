#!/usr/bin/env python3
"""
Extract Hard-Negative NO_ROUTE Samples from ComplexTempQA
==========================================================
Downloads the ComplexTempQA_small.json dataset (~90K records) and extracts
temporal-sounding questions that are NOT computable by our clock arithmetic
pipeline. These serve as hard negatives for NO_ROUTE discrimination training.

Output format (JSONL):
  {"question": "...", "reason": "..."}

Usage:
  python tools/extract_hard_negatives.py \
    --output data/hard_negatives/complex_tempqa_noroute.jsonl \
    --count 7000
"""

import json
import os
import random
import argparse
import urllib.request

SOURCE_URL = (
    "https://raw.githubusercontent.com/DataScienceUIBK/ComplexTempQA"
    "/main/ComplexTempQA_small.json"
)
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "hard_negatives"
)
CACHE_FILE = os.path.join(CACHE_DIR, "ComplexTempQA_small.json")

# Pool of varied NO_ROUTE reason strings
REASON_POOL = [
    "This requires factual knowledge lookup, not temporal arithmetic.",
    "This is a trivia question that cannot be computed from a clock.",
    "No temporal computation is possible here — this needs external knowledge.",
    "This question asks for a historical fact, not a time calculation.",
    "This cannot be answered with clock arithmetic — it requires a knowledge base.",
    "This is a factual recall question, not a temporal reasoning problem.",
    "Answering this requires looking up real-world data, not computing time.",
    "This question is about facts, not about adding or subtracting time.",
]

# Patterns that indicate temporal-sounding but non-computable questions
TEMPORAL_HARD_NEG_PATTERNS = [
    "when was the publication date",
    "when was the date of birth",
    "when was the date of death",
    "when was the inception date",
    "how long was the duration of",
    "when was the founding date",
    "when did",
    "what year",
    "how old",
]

# Patterns for general factual questions (softer negatives)
FACTUAL_PATTERNS = [
    "who was the",
    "what was the country",
    "what was the genre",
    "where was the",
    "what was the highest",
    "who was the founder",
    "who was the director",
    "who was the composer",
    "who was the screenwriter",
    "who was the producer",
    "how long was the duration",  # movie durations — looks temporal, isn't clock math
]


def download_dataset():
    """Download ComplexTempQA_small.json if not cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        print(f"Using cached dataset: {CACHE_FILE}")
        return CACHE_FILE

    print(f"Downloading ComplexTempQA_small.json...")
    urllib.request.urlretrieve(SOURCE_URL, CACHE_FILE)
    size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB → {CACHE_FILE}")
    return CACHE_FILE


def load_questions(path):
    """Load all questions from the JSON file.

    ComplexTempQA_small.json has a mixed format: an initial JSON array
    followed by individual JSON objects. We use raw_decode to parse
    all top-level values.
    """
    print(f"Loading questions from {path}...")
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    records = []
    idx = 0
    while idx < len(content):
        # Skip whitespace
        while idx < len(content) and content[idx] in " \t\n\r,":
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict) and "question" in obj:
                records.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1

    print(f"Loaded {len(records):,} records")
    return records


def is_hard_negative(question_text):
    """Check if a question is temporal-sounding but non-computable."""
    q = question_text.lower()
    for pattern in TEMPORAL_HARD_NEG_PATTERNS:
        if pattern in q:
            return True, "temporal"
    for pattern in FACTUAL_PATTERNS:
        if pattern in q:
            return True, "factual"
    return False, None


def extract(records, count, seed=42):
    """Extract a balanced sample of hard negatives."""
    random.seed(seed)

    temporal_hits = []
    factual_hits = []

    for rec in records:
        q = rec.get("question", "")
        if not q or len(q) < 10:
            continue

        is_neg, neg_type = is_hard_negative(q)
        if is_neg:
            if neg_type == "temporal":
                temporal_hits.append(q)
            else:
                factual_hits.append(q)

    print(f"Found {len(temporal_hits):,} temporal hard negatives")
    print(f"Found {len(factual_hits):,} factual hard negatives")

    # Prefer temporal (harder) negatives: 60/40 split
    temporal_target = min(int(count * 0.6), len(temporal_hits))
    factual_target = min(count - temporal_target, len(factual_hits))

    # If we don't have enough temporal, fill from factual
    if temporal_target + factual_target < count:
        factual_target = min(count - temporal_target, len(factual_hits))

    random.shuffle(temporal_hits)
    random.shuffle(factual_hits)

    selected = temporal_hits[:temporal_target] + factual_hits[:factual_target]
    random.shuffle(selected)

    actual = len(selected)
    if actual < count:
        print(f"WARNING: Only {actual} hard negatives available (requested {count})")

    print(f"Selected {actual} samples ({temporal_target} temporal + {factual_target} factual)")
    return selected


def write_output(questions, output_path, seed=42):
    """Write hard negatives as JSONL with varied reason strings."""
    random.seed(seed + 1)  # Different seed for reason assignment
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for q in questions:
            reason = random.choice(REASON_POOL)
            record = {"question": q, "reason": reason}
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(questions)} records → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hard-negative NO_ROUTE samples from ComplexTempQA"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/hard_negatives/complex_tempqa_noroute.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--count", type=int, default=7000,
        help="Number of hard negatives to extract"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    dataset_path = download_dataset()
    records = load_questions(dataset_path)
    selected = extract(records, args.count, args.seed)
    write_output(selected, args.output, args.seed)

    # Show a few samples
    print("\n--- Sample hard negatives ---")
    with open(args.output) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            print(f"  Q: {rec['question'][:80]}")
            print(f"  R: {rec['reason']}")
            print()


if __name__ == "__main__":
    main()
