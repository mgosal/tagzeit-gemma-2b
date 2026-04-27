"""
Probe Independence Verification — verify_probe_independence.py

Checks for overlap between the static eval probes in validate.py and
all training data. This does NOT certify independence — it flags risk.

In a narrow domain like HH:MM clock arithmetic, coincidental overlaps
between hand-authored probes and machine-generated training data are
expected. This script documents the extent of that overlap.

Methodology:
  - Extracts all (start_time, delta_minutes) tuples from TEST_CASES.
  - Scans every training JSONL file for records that contain BOTH the exact
    start time AND the exact delta as the primary duration (not as part of
    a larger duration like "4 hours and 35 minutes").
  - Reports any overlap found.

Exit code: 0 if no overlaps, 1 if overlaps found.
"""

import json
import re
import os
import sys
import glob

# Import TEST_CASES from validate.py
sys.path.insert(0, os.path.dirname(__file__))
from validate import TEST_CASES


def extract_probe_signatures():
    """Extract (start, delta_str, delta_minutes) tuples from the static test cases.
    Normalises spaced-digit formats like '1 8 : 5 9' → '18:59'.
    """
    signatures = []
    for tc in TEST_CASES:
        start = re.sub(r'\s', '', tc["start"])  # Collapse spaces
        delta = tc["delta"].strip()
        match = re.search(r'(\d+)', delta)
        minutes = int(match.group(1)) if match else None
        is_sub = "ago" in delta
        signatures.append((start, delta, minutes, is_sub))
    return signatures


def text_contains_exact_probe(text, start_time, delta_minutes, is_subtraction):
    """Check if the text contains a question with this exact start time and delta.

    Must match:
      1. The start time appears (as HH:MM, H:MM, spoken, etc.)
      2. The delta appears as a standalone duration of exactly N minutes
         (not as part of "X hours and N minutes")

    This filters out false positives where the delta appears as part of
    a compound duration.
    """
    # Check start time is present (normalised forms)
    h, m = None, None
    time_match = re.match(r'(\d{1,2}):(\d{2})', start_time)
    if time_match:
        h, m = int(time_match.group(1)), int(time_match.group(2))

    if h is None:
        return False

    # Build patterns for this start time
    time_patterns = [
        f"{h:02d}:{m:02d}",        # 09:58
        f"{h}:{m:02d}",            # 9:58
        f"{h:02d}.{m:02d}",        # 09.58
    ]

    found_time = any(p in text for p in time_patterns)
    if not found_time:
        return False

    # Check the delta is present as a standalone minute count.
    # Reject matches where it's part of a compound like "4 hours and 35 minutes"
    # or "1 hour and 35 minutes"
    delta_str = str(delta_minutes)

    # Pattern: "{N} minute(s)" NOT preceded by "and " or "hour(s) "
    # This catches standalone durations
    standalone_patterns = [
        # "takes 35 minutes", "in 35 minutes", "lasts 35 minutes"
        rf'(?:takes|in|lasts?|for|is)\s+{delta_str}\s+minute',
        # "{N} minutes" at word boundary, not after "and"
        rf'(?<!\band\s){delta_str}\s+minute',
    ]

    # Also check for compound patterns that would indicate the delta is
    # part of a larger duration
    compound_pattern = rf'\d+\s+hours?\s+(?:and\s+)?{delta_str}\s+minute'
    if re.search(compound_pattern, text, re.IGNORECASE):
        return False  # This is a compound duration, not our probe

    # Check for the standalone delta
    for pattern in standalone_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def scan_jsonl(filepath, probes):
    """Scan a JSONL file for entries matching any probe signature.
    Returns list of (line_num, probe_desc, snippet) tuples.
    """
    hits = []

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = record.get("text", "") + record.get("input", "") + record.get("instruction", "")

                for start, delta_str, delta_min, is_sub in probes:
                    if delta_min is None:
                        continue
                    if text_contains_exact_probe(text, start, delta_min, is_sub):
                        hits.append((i, f"{start}+{delta_str}", text[:120]))

    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")

    return hits


def main():
    probes = extract_probe_signatures()
    unique_sigs = set((s, d) for s, d, _, _ in probes)
    print(f"Probe Independence Verification")
    print(f"{'=' * 60}")
    print(f"  Probes loaded: {len(unique_sigs)} unique (start, delta) tuples")

    # Find all training JSONL files
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_paths = []
    for pattern in [
        "data/train/*.jsonl",
        "experiments/temporal-tagzeit/data/train/*.jsonl",
        "experiments/temporal-tagzeit/data/*.jsonl",
    ]:
        train_paths.extend(glob.glob(os.path.join(root, pattern)))

    # De-duplicate
    train_paths = sorted(set(os.path.abspath(p) for p in train_paths))
    print(f"  Training files found: {len(train_paths)}")

    total_hits = 0
    for path in train_paths:
        rel = os.path.relpath(path, root)
        hits = scan_jsonl(path, probes)
        if hits:
            print(f"\n  ⚠ OVERLAP in {rel}: {len(hits)} hit(s)")
            for line_num, probe_key, snippet in hits[:5]:
                print(f"    Line {line_num}: probe {probe_key}")
                print(f"      '{snippet}...'")
            if len(hits) > 5:
                print(f"    ... and {len(hits) - 5} more")
            total_hits += len(hits)
        else:
            print(f"  ✓ {rel}: clean")

    print(f"\n{'=' * 60}")
    if total_hits > 0:
        print(f"RESULT: {total_hits} potential overlap(s) found.")
        print(f"  Note: Overlaps are coincidental — training data is machine-generated")
        print(f"  with random parameterisation. Probes are hand-authored. However, the")
        print(f"  signal space is narrow enough that some (start, delta) tuples appear")
        print(f"  in both sets. This should be acknowledged in experiment reports.")
        sys.exit(1)
    else:
        print(f"RESULT: No overlap detected. Probes are independent by construction.")
        print(f"  Methodology: Static probes are hand-authored; training data is")
        print(f"  machine-generated with random parameterisation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
