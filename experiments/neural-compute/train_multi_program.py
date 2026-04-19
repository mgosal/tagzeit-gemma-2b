#!/usr/bin/env python3
"""
Multi-Program Proof: Can a network learn to route between different ALU programs?
==================================================================================
Step 3 proof from the RISC ALU plan.

The model receives pairs of values + a program label, and must:
    1. Route to the correct program (router matches the label)
    2. Extract the correct operand values (extractor produces the right numbers)
    3. The ALU then produces the exact result (no output head needed — the
       detached ALU output IS the answer)

Dataset: mixed expressions from 3 program families:
    - IntAdd(a, b) → a + b          (arithmetic)
    - Mod(a, n) → a mod n            (modular)
    - Greater(a, b) → 1 if a > b     (comparison)

Evaluation protocol
-------------------
Two accuracy metrics are reported at every eval checkpoint:

    oracle_accuracy:  ALU result correctness when ground-truth program IDs are
                      provided (tests operand extraction in isolation).
    e2e_accuracy:     ALU result correctness when the router selects the program
                      via argmax (tests routing + extraction + execution jointly).

Only e2e_accuracy is a valid end-to-end measure. oracle_accuracy is a diagnostic.

Multi-seed support
------------------
Pass --seeds 42,123,7 to run multiple seeds and aggregate mean ± std.

Lesson from Phase Zero: the output head is trained on detached circuit output.
Here we go one step further — the ALU output IS the answer, so we don't need
an output head at all. The only things that need to learn are:
    1. Router: which program? (CE loss)
    2. Extractors: what operand values? (MSE aux loss)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.compute_layer.alu import Mod
from src.compute_layer.programs.arithmetic import IntAdd
from src.compute_layer.programs.comparison import Greater
from src.compute_layer.layer import ComputeLayer, ProgramSpec


# ---------------------------------------------------------------------------
# Reduced program set for the proof
# ---------------------------------------------------------------------------

PROOF_PROGRAMS = [
    ProgramSpec("IntAdd",  IntAdd,  n_operands=2, n_results=1),
    ProgramSpec("Mod",     lambda a, b: Mod(a, b), n_operands=2, n_results=1),
    ProgramSpec("Greater", Greater, n_operands=2, n_results=1),
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def generate_dataset(n_per_program: int = 200, seed: int = 42) -> list[dict]:
    """Generate mixed-program expressions."""
    rng = random.Random(seed)
    data = []

    # IntAdd: a + b, a,b in [0, 30]
    for _ in range(n_per_program):
        a, b = rng.randint(0, 30), rng.randint(0, 30)
        data.append({
            "program_id": 0, "program_name": "IntAdd",
            "operands": [float(a), float(b)],
            "result": float(a + b),
        })

    # Mod: a mod 10, a in [0, 99], fixed modulus
    # Fixed modulus so the proof tests routing, not discontinuous-extraction.
    # Variable moduli require exact extraction of a discrete set — a separate
    # problem from multi-program dispatch.
    MOD_N = 10.0
    for _ in range(n_per_program):
        a = rng.randint(0, 99)
        data.append({
            "program_id": 1, "program_name": "Mod",
            "operands": [float(a), MOD_N],
            "result": float(a % int(MOD_N)),
        })

    # Greater: 1 if a > b else 0
    for _ in range(n_per_program):
        a, b = rng.randint(-20, 20), rng.randint(-20, 20)
        data.append({
            "program_id": 2, "program_name": "Greater",
            "operands": [float(a), float(b)],
            "result": 1.0 if a > b else 0.0,
        })

    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, programs):
    """Run dual-metric evaluation: oracle accuracy + end-to-end accuracy.

    Returns dict with all metrics.
    """
    with torch.no_grad():
        hidden = encoder(operands_t)

        # --- Oracle evaluation (ground-truth program IDs) ---
        ev_oracle = compute(hidden, program_ids=program_ids_t)
        oracle_results = ev_oracle["result"][:, 0]
        oracle_correct = (oracle_results == results_t).float()
        oracle_acc = oracle_correct.mean().item()

        # --- E2E evaluation (router selects program) ---
        ev_e2e = compute(hidden, program_ids=None)
        e2e_results = ev_e2e["result"][:, 0]
        e2e_correct = (e2e_results == results_t).float()
        e2e_acc = e2e_correct.mean().item()

        # Router accuracy (same for both evals — computed from logits)
        router_preds = ev_oracle["router_logits"].argmax(dim=-1)
        router_acc = (router_preds == program_ids_t).float().mean().item()

        # Per-program breakdown (E2E — the metric that matters)
        per_prog_e2e = {}
        per_prog_oracle = {}
        for prog_idx, spec in enumerate(programs):
            mask = (program_ids_t == prog_idx)
            if mask.any():
                per_prog_e2e[spec.name] = (e2e_results[mask] == results_t[mask]).float().mean().item()
                per_prog_oracle[spec.name] = (oracle_results[mask] == results_t[mask]).float().mean().item()

        # E2E failure details
        e2e_failures = []
        wrong_mask = (e2e_results != results_t)
        n_wrong = wrong_mask.sum().item()
        if 0 < n_wrong <= 20:
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
            for idx in wrong_indices[:10]:
                d = dataset[idx.item()]
                e2e_failures.append({
                    "program": d["program_name"],
                    "operands": d["operands"],
                    "expected": d["result"],
                    "got": e2e_results[idx].item(),
                    "router_selected": ev_e2e["program_ids"][idx].item(),
                })

    return {
        "e2e_accuracy": e2e_acc,
        "oracle_accuracy": oracle_acc,
        "router_accuracy": router_acc,
        "per_program_e2e": per_prog_e2e,
        "per_program_oracle": per_prog_oracle,
        "e2e_failures": e2e_failures,
        "n_e2e_correct": int(e2e_correct.sum().item()),
        "n_oracle_correct": int(oracle_correct.sum().item()),
    }


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def train_single(
    seed: int,
    epochs: int = 10000,
    lr: float = 0.005,
    d_model: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train on a single seed and return metrics dict."""
    # Seed everything
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = generate_dataset(n_per_program=200, seed=seed)
    n = len(dataset)

    # Pre-compute tensors
    operands_t = torch.tensor([d["operands"] for d in dataset], dtype=torch.float32, device=device)
    program_ids_t = torch.tensor([d["program_id"] for d in dataset], dtype=torch.long, device=device)
    results_t = torch.tensor([d["result"] for d in dataset], dtype=torch.float32, device=device)

    # Build model: encoder + compute layer (no output head needed)
    encoder = nn.Sequential(
        nn.Linear(2, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    ).to(device)

    compute = ComputeLayer(d_model=d_model, programs=PROOF_PROGRAMS).to(device)

    all_params = list(encoder.parameters()) + list(compute.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)  # C3: plain Adam
    router_criterion = nn.CrossEntropyLoss()
    operand_criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in all_params if p.requires_grad)

    if verbose:
        print(f"\n  Seed {seed}")
        print(f"  {'-' * 56}")

    start = time.time()
    converged_at = None

    for epoch in range(1, epochs + 1):
        # --- Forward (oracle routing during training is correct — router
        # has its own CE loss, extraction needs known-correct program) ---
        hidden = encoder(operands_t)
        out = compute(hidden, program_ids=program_ids_t)

        # Loss 1: Router (which program?)
        loss_router = router_criterion(out["router_logits"], program_ids_t)

        # Loss 2: Operand extraction (C2 — direct supervision)
        loss_operand = torch.tensor(0.0, device=device)
        for prog_idx, spec in enumerate(PROOF_PROGRAMS):
            mask = (program_ids_t == prog_idx)
            if not mask.any():
                continue
            raw = out["raw_operands"][spec.name][mask]
            target = operands_t[mask, :spec.n_operands]
            loss_operand = loss_operand + operand_criterion(raw, target)

        loss = loss_router + loss_operand

        # Guard against NaN propagating into optimizer
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # --- Evaluate ---
        if epoch % 500 == 0 or epoch == 1 or epoch == epochs:
            metrics = evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, PROOF_PROGRAMS)

            if verbose:
                print(
                    f"    Epoch {epoch:5d} │ loss={loss.item():.4f} │ "
                    f"router={metrics['router_accuracy']*100:.1f}% │ "
                    f"e2e={metrics['e2e_accuracy']*100:.1f}% │ "
                    f"oracle={metrics['oracle_accuracy']*100:.1f}% │ "
                    + " │ ".join(f"{k}={v*100:.0f}%" for k, v in metrics["per_program_e2e"].items())
                )

            if metrics["e2e_accuracy"] == 1.0 and metrics["router_accuracy"] == 1.0:
                converged_at = epoch
                if verbose:
                    print(f"    ✓ 100% E2E accuracy at epoch {epoch}")
                break

    elapsed = time.time() - start

    # Final evaluation
    final = evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, PROOF_PROGRAMS)

    if verbose:
        e2e = final["e2e_accuracy"]
        oracle = final["oracle_accuracy"]
        router = final["router_accuracy"]
        status = (
            "✓ PASS" if e2e == 1.0 and router == 1.0
            else "~ PARTIAL" if e2e > 0.9
            else "✗ FAIL"
        )
        print(f"    {status} │ e2e={e2e*100:.1f}% │ oracle={oracle*100:.1f}% │ router={router*100:.1f}%")

        # Show E2E failures
        if final["e2e_failures"]:
            print(f"    E2E failures ({n - final['n_e2e_correct']}):")
            for f in final["e2e_failures"][:10]:
                print(f"      {f['program']}({f['operands']}) = {f['expected']}, "
                      f"got {f['got']} (router→{f['router_selected']})")

    return {
        "seed": seed,
        "e2e_accuracy": final["e2e_accuracy"],
        "oracle_accuracy": final["oracle_accuracy"],
        "router_accuracy": final["router_accuracy"],
        "per_program_e2e": final["per_program_e2e"],
        "per_program_oracle": final["per_program_oracle"],
        "n_e2e_correct": final["n_e2e_correct"],
        "n_oracle_correct": final["n_oracle_correct"],
        "n_total": n,
        "converged_at": converged_at,
        "epochs_run": epoch,
        "elapsed": round(elapsed, 2),
        "n_params": n_params,
        "e2e_failures": final["e2e_failures"],
    }


# ---------------------------------------------------------------------------
# Multi-seed harness
# ---------------------------------------------------------------------------

def train_multi_seed(
    seeds: list[int],
    epochs: int = 10000,
    lr: float = 0.005,
    d_model: int = 64,
    device: str = "cpu",
):
    """Run training across multiple seeds and aggregate results."""
    print(f"Multi-Program Proof Training")
    print(f"{'=' * 60}")
    print(f"  Programs:   {', '.join(p.name for p in PROOF_PROGRAMS)}")
    print(f"  Seeds:      {seeds}")
    print(f"  LR:         {lr}")
    print(f"  Epochs:     {epochs}")
    print(f"  d_model:    {d_model}")

    per_seed_results = []

    for seed in seeds:
        result = train_single(
            seed=seed,
            epochs=epochs,
            lr=lr,
            d_model=d_model,
            device=device,
            verbose=True,
        )
        per_seed_results.append(result)

    # --- Aggregate ---
    e2e_accs = [r["e2e_accuracy"] for r in per_seed_results]
    oracle_accs = [r["oracle_accuracy"] for r in per_seed_results]
    router_accs = [r["router_accuracy"] for r in per_seed_results]

    def fmt_stat(vals):
        mu = statistics.mean(vals)
        if len(vals) > 1:
            std = statistics.stdev(vals)
            return f"{mu*100:.1f}% ± {std*100:.1f}%"
        return f"{mu*100:.1f}%"

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE ({len(seeds)} seeds)")
    print(f"  {'─' * 54}")
    print(f"  E2E accuracy:     {fmt_stat(e2e_accs)}")
    print(f"  Oracle accuracy:  {fmt_stat(oracle_accs)}")
    print(f"  Router accuracy:  {fmt_stat(router_accs)}")

    # Per-program E2E aggregate
    all_prog_names = per_seed_results[0]["per_program_e2e"].keys()
    for prog_name in all_prog_names:
        vals = [r["per_program_e2e"][prog_name] for r in per_seed_results]
        print(f"  {prog_name} (E2E):  {fmt_stat(vals)}")

    print(f"  {'─' * 54}")

    # --- Save results ---
    save_dir = project_root / "experiments" / "neural-compute" / "checkpoints" / "multi_program"
    os.makedirs(save_dir, exist_ok=True)

    aggregate = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "e2e_accuracy_mean": statistics.mean(e2e_accs),
        "e2e_accuracy_std": statistics.stdev(e2e_accs) if len(e2e_accs) > 1 else 0.0,
        "oracle_accuracy_mean": statistics.mean(oracle_accs),
        "oracle_accuracy_std": statistics.stdev(oracle_accs) if len(oracle_accs) > 1 else 0.0,
        "router_accuracy_mean": statistics.mean(router_accs),
        "router_accuracy_std": statistics.stdev(router_accs) if len(router_accs) > 1 else 0.0,
        "per_program_e2e_mean": {
            prog: statistics.mean([r["per_program_e2e"][prog] for r in per_seed_results])
            for prog in all_prog_names
        },
        "per_program_e2e_std": {
            prog: (statistics.stdev([r["per_program_e2e"][prog] for r in per_seed_results])
                   if len(seeds) > 1 else 0.0)
            for prog in all_prog_names
        },
        "n_params": per_seed_results[0]["n_params"],
        "n_total": per_seed_results[0]["n_total"],
        "epochs_max": epochs,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    with open(save_dir / "results_per_seed.json", "w") as f:
        # Strip non-serializable bits
        clean = []
        for r in per_seed_results:
            clean.append({k: v for k, v in r.items()})
        json.dump(clean, f, indent=2)

    print(f"\n  Results saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,123,7,2024,31415",
                        help="Comma-separated seeds for multi-seed runs")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    train_multi_seed(
        seeds=seeds,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        device=args.device,
    )
