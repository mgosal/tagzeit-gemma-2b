# ADR-001: Route-to-Luxon Architecture (Option B)

**Date:** 2026-03-27
**Status:** Accepted
**Decision Makers:** @mandipgosal

## Context

Tagzeit originally used a chain-of-thought (`[THINK]`) approach where the LLM was trained to show and execute temporal arithmetic step-by-step. This had two fundamental problems:

1. **BASE_10_ERROR**: Small models (2B parameters) confuse base-10 and base-60 arithmetic. `09:58 + 5min` → they compute `58 + 5 = 63` then face a carry decision they frequently get wrong.
2. **Training complexity**: Teaching a 2B model to perform reliable arithmetic requires enormous amounts of training data covering every edge case (carries, borrows, rollovers, leap years, DST).

## Decision

We adopted **Option B: Route-to-Luxon** — a three-stage delegation architecture where the LLM never performs temporal arithmetic.

```
NL Input → [Stage 1: Detector] → Typed Tokens → [Stage 2: LLM] → [ROUTE_*] → [Stage 3: Luxon] → Result
```

- **Stage 1 (Pre-LLM):** A deterministic `SymbolicExpressionDetector` canonicalises natural language temporal expressions into Wolfram-style typed tokens (`[HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_20]`).
- **Stage 2 (LLM):** The model performs NLU and intent routing only — it emits structured `[ROUTE_TIME_ADD]` calls instead of computing answers.
- **Stage 3 (Post-LLM):** A Luxon-powered computation engine receives the routing call and performs the arithmetic deterministically.

## Alternatives Considered

### Option A: Enhanced Chain-of-Thought (`[THINK]` blocks)
Train the model to show its arithmetic work in a structured trace, then validate the trace.
- **Rejected because:** Even with structured traces, a 2B model still makes arithmetic errors ~15-30% of the time. The model must learn both the format AND the math.

### Option C: External API Calls
Have the model emit function calls to a datetime API (like GPT-4's tool use).
- **Rejected because:** This requires a more capable model that understands tool-call protocols. A 2B model struggles with structured output generation. Option B simplifies the model's job to routing only.

## Consequences

### Positive
- **100% arithmetic accuracy** — Luxon handles all computation, eliminating BASE_10_ERROR entirely.
- **Simpler training** — Model only learns NLU + routing (one skill), not NLU + routing + arithmetic (three skills).
- **Reusable infrastructure** — The detector/compiler plugin architecture can be extended to other domains (currency, distance, weight).

### Negative
- **Two-language stack** — Python (tokenizer, model) + JavaScript (Luxon engine). Adds operational complexity.
- **Inference latency** — Requires a cross-process call from Python to Node.js during inference.
- **Reduced model autonomy** — The model cannot answer temporal questions without the engine being available.

### Risks
- A 2B model may struggle to emit syntactically valid routing calls (structured token grammar).
- The circadian rhythm defaults for AM/PM resolution may not generalise to all domains.

## Related
- [Issue #7](../../issues/7): Relative temporal references
- [Issue #13](../../issues/13): Domain-typed tokenizer
- [Issue #17](../../issues/17): Temporal compiler
- Opus Peer Review: 21 issues identified and resolved
