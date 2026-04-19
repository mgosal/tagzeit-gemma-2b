"""
Compute Layer — Drop-in nn.Module wrapping the ALU + programs + interface
==========================================================================
This is the "evaluator" — the Wolfram Kernel equivalent.

It dispatches expressions (operator_id, operands) to programs and returns
exact results. The neural interface (router, extractor) is learned; the
programs are exact.

Architecture:
    Hidden state (batch, d_model)
         │
         ├─── Router ──────────► program_id
         ├─── Extractors ─────► operand values (per-program heads)
         │
         ▼
    ┌──────────┐
    │   ALU    │  Program executes → exact result
    │ (0 params)│
    └────┬─────┘
         │ .detach()          ← C1: gradient stops here
         ▼
    ┌──────────┐
    │ Output   │  Classify result (hour: 24 classes, etc.)
    │  Heads   │  OR inject back into hidden state
    └──────────┘

Gradient Paths:
    Path 1: CE loss → output heads ← detached ALU output
    Path 2: Aux MSE → extractors ← ground-truth operands
    Path 3: CE loss → router ← ground-truth program labels
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from src.compute_layer.alu import Add, Sub, Mul, Div, Floor, Mod, FloorDiv
from src.compute_layer.programs.arithmetic import IntAdd, IntSub, IntMul
from src.compute_layer.programs.temporal import TimeAdd, TimeSub, DurationBetween
from src.compute_layer.programs.comparison import Greater, Less, Equal


# ---------------------------------------------------------------------------
# Program Registry
# ---------------------------------------------------------------------------

class ProgramSpec:
    """Specification for a registered program."""

    def __init__(self, name: str, fn: Callable, n_operands: int, n_results: int):
        self.name = name
        self.fn = fn
        self.n_operands = n_operands
        self.n_results = n_results


# Default V1 program registry
DEFAULT_PROGRAMS: list[ProgramSpec] = [
    ProgramSpec("IntAdd",           IntAdd,           n_operands=2, n_results=1),
    ProgramSpec("IntSub",           IntSub,           n_operands=2, n_results=1),
    ProgramSpec("IntMul",           IntMul,           n_operands=2, n_results=1),
    ProgramSpec("Mod",              lambda a, b: Mod(a, b), n_operands=2, n_results=1),
    ProgramSpec("FloorDiv",         lambda a, b: FloorDiv(a, b), n_operands=2, n_results=1),
    ProgramSpec("TimeAdd",          TimeAdd,          n_operands=4, n_results=3),
    ProgramSpec("TimeSub",          TimeSub,          n_operands=4, n_results=3),
    ProgramSpec("DurationBetween",  DurationBetween,  n_operands=4, n_results=3),
    ProgramSpec("Greater",          Greater,          n_operands=2, n_results=1),
    ProgramSpec("Equal",            Equal,            n_operands=2, n_results=1),
]


# ---------------------------------------------------------------------------
# Neural Interface Components
# ---------------------------------------------------------------------------

class ProgramRouter(nn.Module):
    """Classifies which program to invoke from the hidden state.

    Determines the "Head" of the Wolfram expression.

    Input:  (batch, d_model)
    Output: (batch, n_programs) — logits
    """

    def __init__(self, d_model: int, n_programs: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_programs)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.classifier(hidden)


class PerProgramExtractor(nn.Module):
    """Per-program operand extraction heads.

    Each program gets its own small MLP that extracts the right number
    of operands from the hidden state. The router selects which
    extractor's output to use.

    Trained via auxiliary MSE loss (constraint C2).
    """

    def __init__(self, d_model: int, programs: list[ProgramSpec]):
        super().__init__()
        self.programs = programs
        self.heads = nn.ModuleDict()
        for spec in programs:
            self.heads[spec.name] = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, spec.n_operands),
            )

    def forward(self, hidden: Tensor, program_name: str) -> Tensor:
        """Extract operands for a specific program.

        Returns: (batch, n_operands) — raw extracted values
        """
        return self.heads[program_name](hidden)

    def forward_all(self, hidden: Tensor) -> dict[str, Tensor]:
        """Extract operands for all programs (for loss computation during training)."""
        return {name: head(hidden) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# ComputeLayer — The Evaluator
# ---------------------------------------------------------------------------

class ComputeLayer(nn.Module):
    """Programmable ALU as a neural network layer.

    Usage during training (with ground-truth program labels):
        layer = ComputeLayer(d_model=64, programs=DEFAULT_PROGRAMS)
        result = layer(hidden_state, program_ids=labels)

    Usage during inference (router selects program):
        result = layer(hidden_state)

    Returns dict with:
        - 'result': (batch, max_results) — ALU output, DETACHED
        - 'router_logits': (batch, n_programs) — for router CE loss
        - 'raw_operands': dict[str, (batch, n_operands)] — for aux MSE loss
        - 'program_ids': (batch,) — selected program indices
    """

    def __init__(
        self,
        d_model: int,
        programs: list[ProgramSpec] | None = None,
    ):
        super().__init__()
        self.programs = programs or DEFAULT_PROGRAMS
        self.n_programs = len(self.programs)
        self.max_operands = max(p.n_operands for p in self.programs)
        self.max_results = max(p.n_results for p in self.programs)

        self.router = ProgramRouter(d_model, self.n_programs)
        self.extractor = PerProgramExtractor(d_model, self.programs)

    def forward(
        self,
        hidden: Tensor,
        program_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size = hidden.shape[0]
        device = hidden.device

        # 1. Route: which program?
        router_logits = self.router(hidden)

        if program_ids is None:
            program_ids = router_logits.argmax(dim=-1)

        # 2. Extract operands for each program
        raw_operands = self.extractor.forward_all(hidden)

        # 3. Execute programs and collect results
        results = torch.zeros(batch_size, self.max_results, device=device)

        for prog_idx, spec in enumerate(self.programs):
            mask = (program_ids == prog_idx)
            if not mask.any():
                continue

            # Get operands for this program's examples
            operands = raw_operands[spec.name][mask]  # (n_matched, n_operands)

            # Round operands to integers via STE (extraction constraint)
            from src.compute_layer.arithmetic import round_ste
            operands_rounded = round_ste(operands)

            # Execute program
            operand_list = [operands_rounded[:, i] for i in range(spec.n_operands)]
            program_result = spec.fn(*operand_list)

            # Pack result into output tensor
            if isinstance(program_result, tuple):
                for r_idx, r_val in enumerate(program_result):
                    results[mask, r_idx] = r_val
            else:
                results[mask, 0] = program_result

        return {
            "result": results.detach(),  # C1: detach before output heads
            "router_logits": router_logits,
            "raw_operands": raw_operands,
            "program_ids": program_ids,
        }
