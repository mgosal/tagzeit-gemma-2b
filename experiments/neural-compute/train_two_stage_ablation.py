#!/usr/bin/env python3
"""
Ablation: Two-Stage Training (Router-first, then Joint)
========================================================
Can we close the E2E gap by pre-training the router before joint training?

Hypothesis: If the router converges first, extractors see stable program
assignments from the start and can specialize more effectively.

Stage 1: Train router CE only (extractors update too but with no aux signal)
          for N epochs. This is essentially "teach routing first."
Stage 2: Switch to joint training (router CE + operand MSE) for M epochs.
          Now extractors get aux supervision with a stable router.

Compare with baseline (joint from epoch 0) and with varying aux weights.
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
from train_multi_program import (
    PROOF_PROGRAMS,
    generate_dataset,
    evaluate,
)


def train_two_stage(
    seed: int,
    router_epochs: int = 5000,
    joint_epochs: int = 15000,
    lr: float = 0.001,
    d_model: int = 128,
    aux_weight: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Two-stage training: router-first then joint."""
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
    operand_criterion = nn.MSELoss()

    total_epochs = router_epochs + joint_epochs

    if verbose:
        print(f"\n  Seed {seed} | Stage 1: {router_epochs} router-only → Stage 2: {joint_epochs} joint (aux_weight={aux_weight})")
        print(f"  {'-' * 70}")

    start = time.time()
    stage1_router_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        in_stage1 = epoch <= router_epochs

        hidden = encoder(operands_t)
        out = compute(hidden, program_ids=program_ids_t)

        loss_router = router_criterion(out["router_logits"], program_ids_t)

        if in_stage1:
            # Stage 1: router CE only
            loss = loss_router
        else:
            # Stage 2: router CE + weighted operand MSE
            loss_operand = torch.tensor(0.0, device=device)
            for prog_idx, spec in enumerate(PROOF_PROGRAMS):
                mask = (program_ids_t == prog_idx)
                if not mask.any():
                    continue
                raw = out["raw_operands"][spec.name][mask]
                target = operands_t[mask, :spec.n_operands]
                loss_operand = loss_operand + operand_criterion(raw, target)
            loss = loss_router + aux_weight * loss_operand

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Log at stage boundary and periodic checkpoints
        if epoch == router_epochs or epoch % 2500 == 0 or epoch == 1 or epoch == total_epochs:
            metrics = evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, PROOF_PROGRAMS)
            stage_label = "S1" if in_stage1 else "S2"

            if epoch == router_epochs:
                stage1_router_acc = metrics["router_accuracy"]

            if verbose:
                print(
                    f"    [{stage_label}] Epoch {epoch:5d} │ loss={loss.item():.4f} │ "
                    f"router={metrics['router_accuracy']*100:.1f}% │ "
                    f"e2e={metrics['e2e_accuracy']*100:.1f}% │ "
                    f"oracle={metrics['oracle_accuracy']*100:.1f}%"
                )

            if metrics["e2e_accuracy"] == 1.0 and metrics["router_accuracy"] == 1.0:
                if verbose:
                    print(f"    ✓ 100% E2E at epoch {epoch}")
                break

    elapsed = time.time() - start
    final = evaluate(encoder, compute, operands_t, program_ids_t, results_t, dataset, PROOF_PROGRAMS)

    if verbose:
        status = "✓ PASS" if final["e2e_accuracy"] == 1.0 else "~ PARTIAL" if final["e2e_accuracy"] > 0.9 else "✗ FAIL"
        print(f"    {status} │ e2e={final['e2e_accuracy']*100:.1f}% │ oracle={final['oracle_accuracy']*100:.1f}% │ router={final['router_accuracy']*100:.1f}%")

    return {
        "seed": seed,
        "variant": f"two_stage_r{router_epochs}_j{joint_epochs}_w{aux_weight}",
        "router_epochs": router_epochs,
        "joint_epochs": joint_epochs,
        "aux_weight": aux_weight,
        "stage1_router_accuracy": stage1_router_acc,
        "e2e_accuracy": final["e2e_accuracy"],
        "oracle_accuracy": final["oracle_accuracy"],
        "router_accuracy": final["router_accuracy"],
        "per_program_e2e": final["per_program_e2e"],
        "n_total": n,
        "elapsed": round(elapsed, 2),
    }


def run_sweep(seeds, lr, d_model, device):
    """Run multiple training variants and compare."""
    variants = [
        # (router_epochs, joint_epochs, aux_weight, label)
        (0,     20000, 1.0, "baseline (joint from epoch 0)"),
        (5000,  15000, 1.0, "two-stage: 5K router → 15K joint"),
        (10000, 10000, 1.0, "two-stage: 10K router → 10K joint"),
        (0,     20000, 0.5, "joint, aux_weight=0.5"),
        (0,     20000, 2.0, "joint, aux_weight=2.0"),
    ]

    print(f"Two-Stage Training Ablation Sweep")
    print(f"{'=' * 70}")
    print(f"  Seeds: {seeds}")
    print(f"  Variants: {len(variants)}")

    all_results = []

    for router_ep, joint_ep, aux_w, label in variants:
        print(f"\n{'─' * 70}")
        print(f"  VARIANT: {label}")

        per_seed = []
        for seed in seeds:
            r = train_two_stage(
                seed=seed,
                router_epochs=router_ep,
                joint_epochs=joint_ep,
                lr=lr,
                d_model=d_model,
                aux_weight=aux_w,
                device=device,
                verbose=True,
            )
            per_seed.append(r)

        e2e_vals = [r["e2e_accuracy"] for r in per_seed]
        oracle_vals = [r["oracle_accuracy"] for r in per_seed]
        router_vals = [r["router_accuracy"] for r in per_seed]

        def fmt(v):
            return f"{statistics.mean(v)*100:.1f}% ± {statistics.stdev(v)*100:.1f}%" if len(v) > 1 else f"{v[0]*100:.1f}%"

        summary = {
            "label": label,
            "router_epochs": router_ep,
            "joint_epochs": joint_ep,
            "aux_weight": aux_w,
            "e2e_mean": statistics.mean(e2e_vals),
            "e2e_std": statistics.stdev(e2e_vals) if len(e2e_vals) > 1 else 0.0,
            "oracle_mean": statistics.mean(oracle_vals),
            "router_mean": statistics.mean(router_vals),
            "per_seed": per_seed,
        }
        all_results.append(summary)

        print(f"\n  {label}:")
        print(f"    E2E:    {fmt(e2e_vals)}")
        print(f"    Oracle: {fmt(oracle_vals)}")
        print(f"    Router: {fmt(router_vals)}")

    # --- Summary comparison ---
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON TABLE")
    print(f"  {'─' * 64}")
    print(f"  {'Variant':<40s} {'E2E':>12s} {'Oracle':>12s} {'Router':>12s}")
    print(f"  {'─' * 64}")
    for r in all_results:
        e2e_s = f"{r['e2e_mean']*100:.1f}% ± {r['e2e_std']*100:.1f}%"
        oracle_s = f"{r['oracle_mean']*100:.1f}%"
        router_s = f"{r['router_mean']*100:.1f}%"
        print(f"  {r['label']:<40s} {e2e_s:>12s} {oracle_s:>12s} {router_s:>12s}")

    # Save
    save_dir = project_root / "experiments" / "neural-compute" / "checkpoints" / "two_stage_ablation"
    os.makedirs(save_dir, exist_ok=True)

    # Strip per_seed for the summary (keep it in detail file)
    summary_out = [{k: v for k, v in r.items() if k != "per_seed"} for r in all_results]
    with open(save_dir / "results_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    detail_out = [{k: v for k, v in r.items()} for r in all_results]
    with open(save_dir / "results_detail.json", "w") as f:
        json.dump(detail_out, f, indent=2)

    print(f"\n  Results saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,123,7,2024,31415")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    run_sweep(seeds, args.lr, args.d_model, args.device)
