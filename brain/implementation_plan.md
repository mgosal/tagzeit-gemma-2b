# Route-to-Luxon Pipeline — Implementation Plan

**Architecture:** Option B — LLM routes, Luxon computes. The model never does math.
**ADR:** [ADR-001: Route-to-Luxon](./decisions/001-route-to-luxon.md)

## Pipeline Overview

```
NL Input → [Stage 1: Detector] → Typed Tokens → [Stage 2: LLM] → [ROUTE_*] → [Stage 3: Luxon] → Result
```

### Stage 1: Symbolic Expression Detector (Pre-LLM)
- **`src/tokenizer/domain_tokenizer.py`** — Plugin architecture with confidence-gated BPE fallback.
- **`src/tokenizer/compilers/temporal_compiler.py`** — Circadian-aware temporal detection (24h, 12h, bare hours, o'clock, fuzzy compound, durations).
- Context cues ("in the morning") boost confidence; ambiguous fuzzy times fall to BPE.

### Stage 2: LLM Routing
- Model is trained to emit `[ROUTE_*]` tokens only (no answer in training data).
- Token vocabulary: 104 special tokens (HEAD, ARG_HOUR, ARG_MIN, ROUTE, REF).
- **`src/utils/resize_embeddings.py`** — Geometric sinusoidal initialization with orthogonal subspaces.

### Stage 3: Luxon Computation Engine (Post-LLM)
- **`core/computation/temporal_engine.js`** — Deterministic arithmetic via Luxon.
- v1 operations: `TIME_ADD`, `TIME_SUB`, `DURATION_BETWEEN`.
- Calendar/timezone operations deferred to v2.

### Training Data
- **`core/synthetic_data/generators/temporal/generator.js`** — Route-format output, 12 domains, shadow pairs, negative examples.

## v1 Scope
| Feature | Status |
|---------|--------|
| Time arithmetic (add, sub, between) | ✅ |
| Circadian AM/PM defaults | ✅ |
| Bare hours, o'clock, fuzzy compound | ✅ |
| Geometric embedding init | ✅ |
| Calendar shifts, timezone conversion | 🔲 v2 |
| Stage 4 (result → NL wrapping) | 🔲 v2 |

## Verification
- Python tests: 25/25 (detector + compiler)
- JS tests: 26/26 (Luxon engine)
- Opus peer review: 21/21 issues addressed
