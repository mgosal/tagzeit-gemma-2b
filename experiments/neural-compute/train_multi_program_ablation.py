#!/usr/bin/env python3
"""
Ablation: What happens without auxiliary operand supervision?
=============================================================
Identical to train_multi_program.py but with operand MSE loss removed.

This answers the reviewer's question: "What breaks if we remove auxiliary
operand supervision?" If the extractors have no direct training signal for
operand values, the only learning path is:
    - Router learns from CE loss (which program)
    - Extractors receive NO gradient (no aux MSE, and ALU output is detached)

Expected outcome: extractors should collapse because they have zero gradient
signal. The router might still learn (it has CE), but the ALU will produce
wrong results because operands are random.

This directly addresses the credit-assignment critique: the detach means the
system *requires* auxiliary supervision to function. It is not "end-to-end
differentiable" in any meaningful sense.
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

# Import shared components from the main script
from train_multi_program import (
    PROOF_PROGRAMS,
    generate_dataset,
    evaluate,
)


# ---------------------------------------------------------------------------
# Single-seed training (NO operand supervision)
# ---------------------------------------------------------------------------

def train_single_no_aux(
    seed: int,
    epochs: int = 10000,
    lr: float = 0.005,
    d_model: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train with router CE loss ONLY — no operand MSE."""
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = generate_dataset(n_per_program=200, seed=seed)
    n = len(dataset)

    operands_t = torch.tensor([d["operands"] for d in dataset], dtype=torch.float32, device=device)
    program_ids_t = torch.tensor([d["program_id"] for d in dataset], dtype=torch.long, device=device)
    results_t = torch.tensor([d["result"] for d in dataset], dtype=torch.float32, device=device)

    encoder = nn.Sequential(
        nn.Linear(2, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    ).to(device)

    compute = ComputeLayer(d_model=d_model, programs=PROOF_PROGRAMS).to(device)

    all_params = list(encoder.parameters()) + list(compute.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    router_criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in all_params if p.requires_grad)

    if verbose:
        print(f"\n  Seed {seed} (NO aux operand supervision)")
        print(f"  {'-' * 56}")

    start = time.time()

    for epoch in range(1, epochs + 1):
        hidden = encoder(operands_t)
        out = compute(hidden, program_ids=program_ids_t)

        # ONLY router loss — no operand MSE
        loss = router_criterion(out["router_logits"], program_ids_t)

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1 or epoch == epochs:
            metrics = evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, PROOF_PROGRAMS)

            if verbose:
                print(
                    f"    Epoch {epoch:5d} │ loss={loss.item():.4f} │ "
                    f"router={metrics['router_accuracy']*100:.1f}% │ "
                    f"e2e={metrics['e2e_accuracy']*100:.1f}% │ "
                    f"oracle={metrics['oracle_accuracy']*100:.1f}%"
                )

    elapsed = time.time() - start
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

    return {
        "seed": seed,
        "ablation": "no_aux_operand_loss",
        "e2e_accuracy": final["e2e_accuracy"],
        "oracle_accuracy": final["oracle_accuracy"],
        "router_accuracy": final["router_accuracy"],
        "per_program_e2e": final["per_program_e2e"],
        "n_e2e_correct": final["n_e2e_correct"],
        "n_total": n,
        "epochs_run": epoch,
        "elapsed": round(elapsed, 2),
        "n_params": n_params,
    }


def run_ablation(seeds, epochs, lr, d_model, device):
    print(f"ABLATION: No Auxiliary Operand Supervision")
    print(f"{'=' * 60}")
    print(f"  Programs:   {', '.join(p.name for p in PROOF_PROGRAMS)}")
    print(f"  Seeds:      {seeds}")
    print(f"  Loss:       Router CE only (NO operand MSE)")

    per_seed = []
    for seed in seeds:
        result = train_single_no_aux(seed=seed, epochs=epochs, lr=lr, d_model=d_model, device=device)
        per_seed.append(result)

    e2e_accs = [r["e2e_accuracy"] for r in per_seed]
    oracle_accs = [r["oracle_accuracy"] for r in per_seed]
    router_accs = [r["router_accuracy"] for r in per_seed]

    def fmt(vals):
        mu = statistics.mean(vals)
        if len(vals) > 1:
            std = statistics.stdev(vals)
            return f"{mu*100:.1f}% ± {std*100:.1f}%"
        return f"{mu*100:.1f}%"

    print(f"\n{'=' * 60}")
    print(f"  ABLATION AGGREGATE ({len(seeds)} seeds)")
    print(f"  {'─' * 54}")
    print(f"  E2E accuracy:     {fmt(e2e_accs)}")
    print(f"  Oracle accuracy:  {fmt(oracle_accs)}")
    print(f"  Router accuracy:  {fmt(router_accs)}")

    save_dir = project_root / "experiments" / "neural-compute" / "checkpoints" / "ablation_no_aux"
    os.makedirs(save_dir, exist_ok=True)

    aggregate = {
        "ablation": "no_aux_operand_loss",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "e2e_accuracy_mean": statistics.mean(e2e_accs),
        "e2e_accuracy_std": statistics.stdev(e2e_accs) if len(e2e_accs) > 1 else 0.0,
        "oracle_accuracy_mean": statistics.mean(oracle_accs),
        "oracle_accuracy_std": statistics.stdev(oracle_accs) if len(oracle_accs) > 1 else 0.0,
        "router_accuracy_mean": statistics.mean(router_accs),
        "router_accuracy_std": statistics.stdev(router_accs) if len(router_accs) > 1 else 0.0,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    with open(save_dir / "results_per_seed.json", "w") as f:
        json.dump(per_seed, f, indent=2)

    print(f"\n  Results saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,123,7,2024,31415")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    run_ablation(seeds, args.epochs, args.lr, args.d_model, args.device)
