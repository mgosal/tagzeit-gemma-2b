# ADR-003: RISC ALU — 5-Primitive Architecture

**Status:** Accepted (multi-program proof validated)
**Date:** 2026-04-19
**Supercedes:** Extends ADR-002

## Context

ADR-002 established that digital circuits can be embedded as differentiable neural network layers. Phase Zero used a CISC approach — full binary gate circuits (AND, OR, XOR, NOT → half-adder → full-adder → ripple-carry → decimal digit adder). This worked but required ~60 operations per digit addition and stacked 6+ STE layers, making gradients fragile.

We needed to determine the minimal instruction set for a general-purpose compute layer that could address multiple deficiency categories (temporal, mathematical, logical) without the complexity and fragility of binary circuits.

## Decision

Reduce the instruction set to 5 primitives:

1. `Add(a, b) = a + b` — no STE
2. `Sub(a, b) = a - b` — no STE
3. `Mul(a, b) = a * b` — no STE
4. `Div(a, b) = a / b` — no STE (with zero-guard)
5. `Floor(x) = floor_ste(x)` — single STE

All higher-level operations (Mod, TimeAdd, Greater, etc.) are **programs** — fixed compositions of the 5 primitives with zero learnable parameters. Adding a new computation means writing a Python function, not training a model.

The neural interface consists of:
- **Router:** classifies which program to invoke (learned, CE loss)
- **Per-program extractors:** extract operand values from hidden state (learned, MSE aux loss)
- **ALU:** executes the program (zero learnable parameters, detached output)

## Alternatives Considered

1. **CISC (ADR-002 status quo):** Dedicated binary circuits per operation. Rejected — 60+ ops per digit, deep STE chain, fragile gradient path.

2. **Learned arithmetic:** Let the network learn Mod/Floor through gradient descent. Rejected — discrete operations are a known failure mode for gradient-based learning (this is the whole problem the project exists to solve).

3. **Shared extractor across programs:** One MLP extracts operands for all programs. Rejected — each program needs different operand counts and value ranges. Per-program heads are cleaner and the parameter budget (0.18% of a transformer layer) easily accommodates them.

## Consequences

- The ALU uses at most 2-4 `Floor` operations per program, vs 6+ STEs in the binary pipeline. Gradient quality is significantly better.
- Programs are exact and exhaustively verifiable — 969 tests covering 380+ input combinations passed.
- Multi-program dispatch works: a 30K-parameter model routes between IntAdd, Mod, and Greater with 100% result accuracy.
- Mod with variable moduli remains an open problem — the extractor must produce exact integer divisors, which is a harder extraction task. Fixed-modulus Mod works.
- C3 refinement: extractor training needs lower LR (0.001) than Phase Zero's heads (0.01) due to Mod's sensitivity to extraction precision.
