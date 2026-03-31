#!/usr/bin/env python3
"""
Route-to-Luxon Evaluation Harness
=================================
Evaluates a model trained for Route-to-Luxon temporal math.
Passes the 48 baseline tests from validate.py.
Extracts the [ROUTE] output from the model, executes it in batch
via the Luxon engine (using a /tmp workaround), and calculates correctness.
"""

import sys
import os
import json
import argparse
import subprocess
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.validate import (
    TEST_CASES,
    load_model,
    generate_response,
    build_prompt,
    normalize_time,
    SYSTEM_PROMPT
)

def extract_routing_calls(raw_output):
    """
    Extracts purely the [ROUTE...] tokens and subsequent tokens.
    Regex looks for a routing operation token ([ROUTE_TIME_ADD], etc.)
    followed by structured argument tokens.
    Skips [ROUTE_START] which some checkpoints hallucinate as a prefix.
    """
    # Strip [ROUTE_START] if present — it's not part of the grammar
    cleaned = re.sub(r'\[ROUTE_START\]\s*', '', raw_output)
    match = re.search(r'\[ROUTE_[A-Z_]+\](?:\s*\[[A-Z0-9_]+\])+', cleaned)
    if match:
        return re.sub(r'\s+', ' ', match.group(0)).strip()
    return None

def execute_routes_in_batch(routing_calls):
    """
    Execute a batch of string routing calls using temporal_engine.js in /tmp
    to workaround npm permission issues locally.
    """
    import shutil
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engine_path = os.path.join(project_root, "core", "computation", "temporal_engine.js")
    
    tmp_dir = "/tmp/tagzeit-engine-test"
    os.makedirs(tmp_dir, exist_ok=True)
    shutil.copy2(engine_path, os.path.join(tmp_dir, "temporal_engine.js"))
    
    # Ensure Luxon is available in /tmp
    if not os.path.exists(os.path.join(tmp_dir, "node_modules", "luxon")):
        subprocess.run(["npm", "init", "-y"], cwd=tmp_dir, capture_output=True)
        # Using a custom cache path bypasses the user's broken global cache
        subprocess.run(["npm", "install", "luxon", "--cache", "/tmp/npm-cache"], cwd=tmp_dir, capture_output=True)
        
    js_script = f"""
const {{ parseRoutingCall, execute }} = require('./temporal_engine.js');
const fs = require('fs');

const calls = JSON.parse(fs.readFileSync('calls.json', 'utf8'));
const results = [];

for (const call of calls) {{
    if (!call) {{
        results.push({{ resultString: null, error: "No routing call extracted" }});
        continue;
    }}
    try {{
        const parsed = parseRoutingCall(call);
        const result = execute(parsed);
        results.push({{ resultString: result.resultString, error: null }});
    }} catch (e) {{
        results.push({{ resultString: null, error: e.message }});
    }}
}}
process.stdout.write(JSON.stringify(results));
"""
    script_path = os.path.join(tmp_dir, "runner.js")
    calls_path = os.path.join(tmp_dir, "calls.json")
    
    with open(script_path, "w") as f:
        f.write(js_script)
    with open(calls_path, "w") as f:
        json.dump(routing_calls, f)
        
    result = subprocess.run(
        ["node", "runner.js"],
        capture_output=True, text=True,
        cwd=tmp_dir,
    )
    
    if result.returncode != 0:
        print(f"  ✗ Engine failed: {result.stderr}")
        sys.exit(1)
        
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  ✗ JSON Parse Error. Engine stdout: {result.stdout}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Route-to-Luxon Diagnostic Probe")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID (HuggingFace or local path)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "mlx", "torch"])
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Initializing Route-to-Luxon Evaluator for: {args.model_id}")
    print("=" * 70)

    model, tokenizer, engine = load_model(args.model_id, args.adapter_path, args.backend)
    if model is None:
        print("ERROR: Could not load model.")
        return

    validation_cases = []
    
    print("\n--- Generating Route Predictions (SFT Model) ---\n")
    
    passed = 0
    failed = 0
    skipped = 0
    
    routing_calls = []
    
    for tc in TEST_CASES:
        skin = "military"
        if tc["category"] == "semantic_eq" and skin != "military":
            continue
            
        sys_prompt, question = build_prompt(tc, skin=skin, mode="route")
        
        raw_response, tps = generate_response(model, tokenizer, engine, sys_prompt, question)
        extracted_route = extract_routing_calls(raw_response)
        
        if extracted_route:
            print(f"  ✓ {tc['id']:6s} Route emitted: {extracted_route[:60]}...")
        else:
            print(f"  ✗ {tc['id']:6s} No route found. Raw: {raw_response[:60].strip()}")
            
        routing_calls.append(extracted_route)
        validation_cases.append(tc)
        
    print("\n--- Computing Extracted Routes via Luxon Engine ---\n")
    
    engine_results = execute_routes_in_batch(routing_calls)
    
    category_stats = {}
    
    for i, (tc, engine_result) in enumerate(zip(validation_cases, engine_results)):
        cat = tc["category"]
        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "failed": 0}
            
        expected = tc["expected"]
        
        if engine_result["error"]:
            print(f"  ✗ {tc['id']:6s} Engine error: {engine_result['error']}")
            failed += 1
            category_stats[cat]["failed"] += 1
            continue
            
        actual = normalize_time(engine_result["resultString"])
        
        if expected == "INVALID":
            # If actual is "INVALID" or engine failed handled parsing gracefully
            if actual == "INVALID" or engine_result["resultString"] is None:
                print(f"  ✓ {tc['id']:6s} INVALID correctly handled")
                passed += 1
                category_stats[cat]["passed"] += 1
                continue
                
        if actual == expected:
            print(f"  ✓ {tc['id']:6s} {tc['start']} + {tc['delta']:20s} → {actual} (correct)")
            passed += 1
            category_stats[cat]["passed"] += 1
        else:
            print(f"  ✗ {tc['id']:6s} {tc['start']} + {tc['delta']:20s} → {actual} (expected {expected})")
            failed += 1
            category_stats[cat]["failed"] += 1
            
    total = passed + failed
    print(f"\n{'=' * 70}")
    print(f"E2E RESULTS: {passed}/{total} passed ({failed} failed, {passed/total*100:.1f}%)")
    print(f"{'=' * 70}")

    print(f"\nPer-category breakdown:")
    for cat, stats in sorted(category_stats.items()):
        total_cat = stats["passed"] + stats["failed"]
        pct = (stats["passed"] / total_cat * 100) if total_cat > 0 else 0
        status = "✅" if stats["failed"] == 0 else "❌"
        print(f"  {status} {cat:20s}: {stats['passed']}/{total_cat} ({pct:.0f}%)")

if __name__ == "__main__":
    main()
