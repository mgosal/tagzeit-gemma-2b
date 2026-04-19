#!/usr/bin/env python3
"""
Phase Zero Training Script — Proof-of-Concept Circuit Model
==============================================================
Trains three progressively harder proof models to verify that
neural networks can learn to interface with hard-coded circuit layers.

    Stage A: No extraction — digit values passed directly. Proves circuit
             layer works in a training loop and output heads can decode results.

    Stage B: Learned embeddings — each digit has a 1D embedding that must
             converge to its integer value. Proves extraction → circuit works.

    Stage C: Transformer encoder — full architecture with NLU-style processing.
             Proves the complete detection → extraction → circuit → output pipeline.

Key training insights (discovered during Phase Zero):
    1. Circuit outputs must be DETACHED before output heads. STE gradients
       through multi-layer circuit pipelines corrupt the CE loss landscape.
    2. The extraction (value extractor) learns via auxiliary MSE loss, NOT
       through backpropagation through the circuit.
    3. No weight decay, no LR scheduling — plain Adam converges reliably
       for this problem scale.

Usage:
    python experiments/neural-compute/train_phase_zero.py                  # Run all stages
    python experiments/neural-compute/train_phase_zero.py --stage A        # Run stage A only
    python experiments/neural-compute/train_phase_zero.py --eval_only --stage B --checkpoint best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.compute_layer.proof_model import create_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DigitAdditionDataset(Dataset):
    """All 100 single-digit addition pairs: (a, b) → (sum_digit, carry)."""

    def __init__(self):
        self.pairs = []
        for a in range(10):
            for b in range(10):
                sum_digit = (a + b) % 10
                carry = (a + b) // 10
                self.pairs.append((a, b, sum_digit, carry))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, sum_digit, carry = self.pairs[idx]
        return (
            torch.tensor(a, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(sum_digit, dtype=torch.long),
            torch.tensor(carry, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Default hyperparameters per stage
# ---------------------------------------------------------------------------

STAGE_DEFAULTS = {
    "A": {"lr": 0.01, "epochs": 10000, "needs_aux": False},
    "B": {"lr": 0.01, "epochs": 10000, "needs_aux": True},
    "C": {"lr": 0.003, "epochs": 15000, "needs_aux": True},
}


# ---------------------------------------------------------------------------
# Training for a single stage
# ---------------------------------------------------------------------------

def train_stage(
    stage: str,
    epochs: int | None = None,
    lr: float | None = None,
    aux_weight: float = 1.0,
    device: str = "cpu",
    save_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train one stage and return results."""
    defaults = STAGE_DEFAULTS[stage.upper()]
    if epochs is None:
        epochs = defaults["epochs"]
    if lr is None:
        lr = defaults["lr"]
    needs_aux = defaults["needs_aux"]

    if save_dir is None:
        save_dir = str(
            project_root / "experiments" / "neural-compute" / "checkpoints" / f"stage_{stage.lower()}"
        )
    os.makedirs(save_dir, exist_ok=True)

    dataset = DigitAdditionDataset()

    model = create_model(stage).to(device)

    # Key insight: plain Adam, no weight decay, no scheduler.
    # Weight decay + cosine annealing prevented convergence for the
    # narrow class boundaries in the 10-class sum digit classifier.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sum_criterion = nn.CrossEntropyLoss()
    carry_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_circuit_params = sum(p.numel() for p in model.circuit.parameters())

    # Pre-compute full batch tensors
    a_ids = torch.tensor([p[0] for p in dataset.pairs], dtype=torch.long, device=device)
    b_ids = torch.tensor([p[1] for p in dataset.pairs], dtype=torch.long, device=device)
    sum_targets = torch.tensor([p[2] for p in dataset.pairs], dtype=torch.long, device=device)
    carry_targets = torch.tensor([p[3] for p in dataset.pairs], dtype=torch.long, device=device)

    if verbose:
        print(f"\n  Stage {stage.upper()}: {model.__class__.__name__}")
        print(f"  {'─' * 50}")
        print(f"  Learnable params: {n_params:,}")
        print(f"  Circuit params:   {n_circuit_params}")
        print(f"  Aux loss:         {'yes (weight={})'.format(aux_weight) if needs_aux else 'no (direct values)'}")
        print(f"  LR: {lr}, Epochs: {epochs}")
        print()

    best_accuracy = 0.0
    best_epoch = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        sum_logits, carry_logits, raw_values = model(a_ids, b_ids)

        loss_sum = sum_criterion(sum_logits, sum_targets)
        loss_carry = carry_criterion(carry_logits, carry_targets)
        loss = loss_sum + loss_carry

        if needs_aux:
            value_targets = torch.stack(
                [a_ids.float(), b_ids.float()], dim=-1
            )
            loss_aux = value_criterion(raw_values, value_targets)
            loss = loss + aux_weight * loss_aux

        loss.backward()
        optimizer.step()

        # Evaluate periodically
        log_interval = 500 if epoch > 100 else 100
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            accuracy, sum_acc, carry_acc = evaluate(model, a_ids, b_ids, sum_targets, carry_targets)

            record = {
                "epoch": epoch,
                "loss": round(loss.item(), 4),
                "accuracy": round(accuracy, 4),
                "sum_accuracy": round(sum_acc, 4),
                "carry_accuracy": round(carry_acc, 4),
            }
            history.append(record)

            if verbose:
                print(
                    f"    Epoch {epoch:5d} │ loss={loss.item():.4f} │ "
                    f"acc={accuracy*100:5.1f}% │ "
                    f"sum={sum_acc*100:5.1f}% │ carry={carry_acc*100:5.1f}%"
                )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

            if accuracy == 1.0:
                if verbose:
                    print(f"\n    ✓ 100% accuracy at epoch {epoch}.")
                break

    elapsed = time.time() - start_time
    final_accuracy, _, _ = evaluate(model, a_ids, b_ids, sum_targets, carry_targets)
    detailed = detailed_evaluation(model, dataset, device)

    torch.save(model.state_dict(), os.path.join(save_dir, "final.pt"))

    results = {
        "stage": stage.upper(),
        "model_class": model.__class__.__name__,
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "elapsed_seconds": round(elapsed, 2),
        "n_params": n_params,
        "lr": lr,
        "history": history,
        "detailed_results": detailed,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        n_correct = sum(1 for r in detailed if r["correct"])
        status = "✓ PASS" if final_accuracy == 1.0 else "✗ FAIL"
        print(f"\n    {status} — {n_correct}/100 correct in {elapsed:.1f}s")

        if final_accuracy < 1.0:
            failures = [r for r in detailed if not r["correct"]]
            for r in failures[:10]:
                print(f"      {r['a']}+{r['b']}: expected {r['expected_sum']} got {r['pred_sum']}")
        else:
            print(f"    Circuit layer integration verified.")

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, a_ids, b_ids, sum_targets, carry_targets):
    model.eval()
    with torch.no_grad():
        sum_logits, carry_logits, _ = model(a_ids, b_ids)
        sum_preds = sum_logits.argmax(dim=-1)
        carry_preds = carry_logits.argmax(dim=-1)

        sum_correct = (sum_preds == sum_targets).sum().item()
        carry_correct = (carry_preds == carry_targets).sum().item()
        correct = ((sum_preds == sum_targets) & (carry_preds == carry_targets)).sum().item()
        total = len(sum_targets)

    return correct / total, sum_correct / total, carry_correct / total


def detailed_evaluation(model, dataset, device):
    model.eval()
    results = []
    with torch.no_grad():
        a_ids = torch.tensor([p[0] for p in dataset.pairs], dtype=torch.long, device=device)
        b_ids = torch.tensor([p[1] for p in dataset.pairs], dtype=torch.long, device=device)
        sum_logits, carry_logits, raw_values = model(a_ids, b_ids)
        sum_preds = sum_logits.argmax(dim=-1)
        carry_preds = carry_logits.argmax(dim=-1)

        for i, (a, b, exp_sum, exp_carry) in enumerate(dataset.pairs):
            results.append({
                "a": a, "b": b,
                "expected_sum": exp_sum, "expected_carry": exp_carry,
                "pred_sum": sum_preds[i].item(),
                "pred_carry": carry_preds[i].item(),
                "correct": (sum_preds[i].item() == exp_sum and carry_preds[i].item() == exp_carry),
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase Zero: Circuit proof-of-concept")
    parser.add_argument("--stage", type=str, default="all",
                        help="'A', 'B', 'C', or 'all'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epochs for stage")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override default learning rate")
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="best")
    args = parser.parse_args()

    if args.eval_only:
        stage = args.stage.upper()
        save_dir = str(
            project_root / "experiments" / "neural-compute" / "checkpoints" / f"stage_{stage.lower()}"
        )
        ckpt_path = os.path.join(save_dir, f"{args.checkpoint}.pt")
        model = create_model(stage).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device, weights_only=True))
        dataset = DigitAdditionDataset()
        a_ids = torch.tensor([p[0] for p in dataset.pairs], dtype=torch.long, device=args.device)
        b_ids = torch.tensor([p[1] for p in dataset.pairs], dtype=torch.long, device=args.device)
        s_tgt = torch.tensor([p[2] for p in dataset.pairs], dtype=torch.long, device=args.device)
        c_tgt = torch.tensor([p[3] for p in dataset.pairs], dtype=torch.long, device=args.device)
        acc, s_acc, c_acc = evaluate(model, a_ids, b_ids, s_tgt, c_tgt)
        print(f"Stage {stage}: acc={acc*100:.1f}% sum={s_acc*100:.1f}% carry={c_acc*100:.1f}%")
        return

    stages = ["A", "B", "C"] if args.stage.lower() == "all" else [args.stage.upper()]

    print(f"Phase Zero: Circuit Proof-of-Concept Training")
    print(f"{'=' * 55}")
    print(f"  Running stages: {', '.join(stages)}")

    all_results = {}
    for stage in stages:
        results = train_stage(
            stage=stage,
            epochs=args.epochs,
            lr=args.lr,
            aux_weight=args.aux_weight,
            device=args.device,
        )
        all_results[stage] = results

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  SUMMARY")
    print(f"  {'─' * 50}")
    for stage, r in all_results.items():
        status = "✓" if r["final_accuracy"] == 1.0 else "✗"
        print(
            f"  {status} Stage {stage}: {r['final_accuracy']*100:.1f}% "
            f"({r['model_class']}, {r['n_params']:,} params, "
            f"epoch {r['best_epoch']}, {r['elapsed_seconds']:.1f}s)"
        )


if __name__ == "__main__":
    main()
