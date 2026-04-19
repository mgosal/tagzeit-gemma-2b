#!/usr/bin/env python3
"""
Diagnostic: Decompose E2E vs Oracle gap and characterize failure modes.
========================================================================
Answers three reviewer questions:

1. Calibration: Of the ~7% where router picks the wrong program, how many
   produce the wrong answer (true errors) vs accidentally produce the right
   answer (lucky misroutes)?

2. Oracle failures: What causes the ~0.6% failures even with correct program?
   STE rounding? Edge operand values? Numerical noise?

3. Per-program confusion matrix: Which programs does the router confuse?
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from train_multi_program import (
    PROOF_PROGRAMS,
    generate_dataset,
    evaluate,
)
from src.compute_layer.layer import ComputeLayer


def run_diagnostics(seed: int, epochs: int = 20000, lr: float = 0.001, d_model: int = 128, device: str = "cpu"):
    """Train a model and then perform detailed failure analysis."""
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

    print(f"Training seed {seed} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        hidden = encoder(operands_t)
        out = compute(hidden, program_ids=program_ids_t)
        loss_router = router_criterion(out["router_logits"], program_ids_t)
        loss_operand = torch.tensor(0.0, device=device)
        for prog_idx, spec in enumerate(PROOF_PROGRAMS):
            mask = (program_ids_t == prog_idx)
            if not mask.any():
                continue
            raw = out["raw_operands"][spec.name][mask]
            target = operands_t[mask, :spec.n_operands]
            loss_operand = loss_operand + operand_criterion(raw, target)
        loss = loss_router + loss_operand
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

    print(f"Training done. Running diagnostics...\n")

    # --- Detailed analysis ---
    with torch.no_grad():
        hidden = encoder(operands_t)

        # Oracle pass
        ev_oracle = compute(hidden, program_ids=program_ids_t)
        oracle_results = ev_oracle["result"][:, 0]

        # E2E pass
        ev_e2e = compute(hidden, program_ids=None)
        e2e_results = ev_e2e["result"][:, 0]
        router_preds = ev_e2e["program_ids"]

        # Router logits for oracle path (same hidden)
        router_logits_pred = ev_oracle["router_logits"].argmax(dim=-1)

    prog_names = {i: spec.name for i, spec in enumerate(PROOF_PROGRAMS)}

    # ======================================================================
    # Q1: Calibration — decompose the E2E gap
    # ======================================================================
    print("=" * 70)
    print("Q1: CALIBRATION — Decompose E2E vs Oracle gap")
    print("=" * 70)

    router_correct = (router_preds == program_ids_t)
    e2e_correct = (e2e_results == results_t)
    oracle_correct = (oracle_results == results_t)

    # Four categories:
    # A: router right + answer right
    # B: router right + answer wrong (extraction error with correct program)
    # C: router wrong + answer right (lucky misroute / accidental agreement)
    # D: router wrong + answer wrong (true routing failure)

    cat_A = (router_correct & e2e_correct).sum().item()
    cat_B = (router_correct & ~e2e_correct).sum().item()
    cat_C = (~router_correct & e2e_correct).sum().item()
    cat_D = (~router_correct & ~e2e_correct).sum().item()

    print(f"\n  Router correct + Answer correct (A): {cat_A:3d} ({cat_A/n*100:.1f}%)")
    print(f"  Router correct + Answer wrong   (B): {cat_B:3d} ({cat_B/n*100:.1f}%)  ← extraction error despite correct routing")
    print(f"  Router wrong   + Answer correct (C): {cat_C:3d} ({cat_C/n*100:.1f}%)  ← accidental agreement / lucky misroute")
    print(f"  Router wrong   + Answer wrong   (D): {cat_D:3d} ({cat_D/n*100:.1f}%)  ← true routing failure")
    print(f"  Total router errors: {(~router_correct).sum().item()} = C({cat_C}) + D({cat_D})")
    print(f"  Total answer errors: {(~e2e_correct).sum().item()} = B({cat_B}) + D({cat_D})")

    if cat_C + cat_D > 0:
        print(f"\n  Of {cat_C + cat_D} router errors:")
        print(f"    {cat_C} ({cat_C/(cat_C+cat_D)*100:.1f}%) produced the right answer anyway")
        print(f"    {cat_D} ({cat_D/(cat_C+cat_D)*100:.1f}%) produced the wrong answer")

    # ======================================================================
    # Q2: Oracle failures — characterize the ~0.6%
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("Q2: ORACLE FAILURES — What causes errors even with correct program?")
    print("=" * 70)

    oracle_failures = (~oracle_correct).nonzero(as_tuple=True)[0]
    n_oracle_fail = len(oracle_failures)
    print(f"\n  Oracle failures: {n_oracle_fail}/{n} ({n_oracle_fail/n*100:.2f}%)")

    if n_oracle_fail > 0:
        print(f"\n  Failure details:")
        for idx in oracle_failures[:30]:
            i = idx.item()
            d = dataset[i]
            # Get the raw extracted operands for this example's program
            raw_ops = ev_oracle["raw_operands"][d["program_name"]][program_ids_t == d["program_id"]]
            # Find which sub-index this example is within its program mask
            prog_mask = (program_ids_t == d["program_id"])
            prog_indices = prog_mask.nonzero(as_tuple=True)[0]
            local_idx = (prog_indices == idx).nonzero(as_tuple=True)[0].item()
            raw = raw_ops[local_idx]
            rounded = torch.round(raw)

            print(f"    [{d['program_name']}] operands={d['operands']} expected={d['result']}")
            print(f"      raw_extracted: [{', '.join(f'{v:.4f}' for v in raw.tolist())}]")
            print(f"      rounded:      [{', '.join(f'{v:.0f}' for v in rounded.tolist())}]")
            print(f"      oracle_result: {oracle_results[i].item()}")
            extraction_error = max(abs(raw[j].item() - d['operands'][j]) for j in range(len(d['operands'])))
            print(f"      max_extraction_error: {extraction_error:.4f}")
            print()
    else:
        print("  No oracle failures on this seed.")

    # ======================================================================
    # Q3: Router confusion matrix
    # ======================================================================
    print(f"{'=' * 70}")
    print("Q3: ROUTER CONFUSION MATRIX")
    print("=" * 70)

    n_progs = len(PROOF_PROGRAMS)
    confusion = torch.zeros(n_progs, n_progs, dtype=torch.long)
    for true_id, pred_id in zip(program_ids_t, router_preds):
        confusion[true_id.item()][pred_id.item()] += 1

    # Print confusion matrix
    header = "True \\ Pred".ljust(15) + "  ".join(f"{prog_names[j]:>10}" for j in range(n_progs))
    print(f"\n  {header}")
    print(f"  {'─' * len(header)}")
    for i in range(n_progs):
        row = prog_names[i].ljust(15)
        for j in range(n_progs):
            val = confusion[i][j].item()
            marker = " " if i == j else "*" if val > 0 else " "
            row += f"  {val:>9d}{marker}"
        print(f"  {row}")

    # Show specific misroute examples (Category D)
    print(f"\n  Sample misrouted examples (Category D: wrong program → wrong answer):")
    shown = 0
    for idx in range(n):
        if not router_correct[idx] and not e2e_correct[idx] and shown < 10:
            d = dataset[idx]
            true_prog = prog_names[program_ids_t[idx].item()]
            pred_prog = prog_names[router_preds[idx].item()]
            print(f"    {true_prog}({d['operands']})={d['result']}  →  routed to {pred_prog}, got {e2e_results[idx].item()}")
            shown += 1

    # ======================================================================
    # Summary JSON
    # ======================================================================
    save_dir = project_root / "experiments" / "neural-compute" / "checkpoints" / "diagnostics"
    import os
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        "seed": seed,
        "n_total": n,
        "calibration": {
            "router_correct_answer_correct": cat_A,
            "router_correct_answer_wrong": cat_B,
            "router_wrong_answer_correct": cat_C,
            "router_wrong_answer_wrong": cat_D,
            "accidental_agreement_rate": cat_C / (cat_C + cat_D) if (cat_C + cat_D) > 0 else 0,
        },
        "oracle_failures": {
            "count": n_oracle_fail,
            "rate": n_oracle_fail / n,
        },
        "confusion_matrix": {
            f"{prog_names[i]}_true": {
                f"{prog_names[j]}_pred": confusion[i][j].item()
                for j in range(n_progs)
            }
            for i in range(n_progs)
        },
    }

    with open(save_dir / f"diagnostics_seed_{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Diagnostics saved to {save_dir}/diagnostics_seed_{seed}.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_diagnostics(seed=args.seed, epochs=args.epochs, lr=args.lr, d_model=args.d_model, device=args.device)
