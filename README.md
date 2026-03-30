# LLM Deficiency Index

A systematic, experiment-driven catalogue of large-language-model failure modes—and the fine-tuning strategies that address them.

---

## Why This Exists

Large language models fail in predictable, documentable ways. They hallucinate dates, collapse under multi-step arithmetic, lose track of goals in agentic pipelines, and mishandle dozens of other tasks that appear trivial to human reasoners. These failures are widely observed but rarely studied in a unified, reproducible framework.

The **LLM Deficiency Index** is that framework. Each deficiency area gets its own isolated experiment—with dedicated data, training scripts, and evaluation harnesses—while sharing a common methodology and tooling layer across the project.

## Taxonomy

Deficiency areas are organised into high-level categories derived from [Issue #9](../../issues/9). This taxonomy is a living document; categories will be refined as experiments mature.

| Category | Description | Status |
|---|---|---|
| **Temporal** | Date arithmetic, relative time reasoning, calendar-system awareness | 🟢 Active — [Tagzeit](./experiments/temporal-tagzeit/) |
| **Mathematical** | Base-N arithmetic, symbolic manipulation, numerical precision | 🔲 Planned |
| **Planning** | Multi-step goal decomposition, loop handling, constraint satisfaction | 🔲 Planned |
| **Agentic** | Tool use, state tracking across turns, recovery from failed actions | 🔲 Planned |
| **Linguistic** | Pragmatics, idiomatic interpretation, cross-lingual transfer gaps | 🔲 Planned |
| **Logical** | Deductive and abductive reasoning, syllogistic consistency | 🔲 Planned |
| **Spatial** | Relative positioning, map/graph traversal, geometric reasoning | 🔲 Planned |
| **Ethical / Safety** | Value alignment edge cases, refusal calibration, bias propagation | 🔲 Planned |

> **Note:** The full, granular list of 30+ deficiency areas lives in [Issue #9](../../issues/9). The table above reflects the top-level groupings.

## Repository Structure

```text
llm-deficiency-index/
├── core/                                   # Shared infrastructure
│   ├── computation/
│   │   └── temporal_engine.js              # Luxon-powered deterministic engine
│   └── synthetic_data/
│       └── generators/temporal/
│           └── generator.js                # Route-format training data generator
├── src/                                    # Model-facing code
│   ├── tokenizer/
│   │   ├── domain_tokenizer.py             # Symbolic Expression Detector (plugin arch)
│   │   └── compilers/
│   │       └── temporal_compiler.py        # Circadian-aware temporal compiler
│   └── utils/
│       └── resize_embeddings.py            # Geometric sinusoidal embedding init
├── experiments/                            # Individual deficiency studies
│   └── temporal-tagzeit/                   # 🟢 Original prototype (SFT/CoT approach)
├── tests/                                  # Test suites
│   └── test_domain_tokenizer.py            # 25 tests (detector + compiler)
├── brain/                                  # Project documentation
│   ├── decisions/                          # Architectural Decision Records
│   │   └── 001-route-to-luxon.md           # Option B architecture decision
│   ├── implementation_plan.md              # Current pipeline plan
│   └── task.md                             # Task tracking
├── common-docs/                            # Contribution guides
├── tools/                                  # Shared CLI scripts (training, evaluation)
└── README.md                               # ← You are here
```

### Design Principles

1. **Sub-project autonomy.** Each experiment owns its README, configuration, and local tests. You can clone the repo and work inside a single experiment without touching anything else.
2. **Shared infrastructure.** Common computation engines, data generators, and training harnesses live in `core/` and `tools/`. Experiments import from these rather than reinventing them.
3. **Reproducibility first.** Every experiment ships with pinned dependencies, seed-controlled data generation, and versioned model checkpoints where feasible.
4. **Architectural decisions recorded.** Major choices are documented in `brain/decisions/` using ADR format.

## Active Work: Route-to-Luxon Pipeline

The current focus is the **Route-to-Luxon** pipeline ([ADR-001](./brain/decisions/001-route-to-luxon.md)), a three-stage architecture where the LLM never performs temporal arithmetic:

```
NL Input → [Stage 1: Detector] → Typed Tokens → [Stage 2: LLM] → [ROUTE_*] → [Stage 3: Luxon] → Result
```

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 (Pre-LLM) | Symbolic Detector | Canonicalises NL time into `[HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_20]` tokens |
| 2 (LLM) | Routing Layer | Model emits `[ROUTE_TIME_ADD]` calls instead of computing answers |
| 3 (Post-LLM) | Luxon Engine | Deterministic arithmetic via Luxon — 100% accuracy, zero BASE_10_ERROR |

**Key features:**
- Circadian rhythm defaults for AM/PM resolution (waking-hours bias)
- Context cues ("in the morning") for high-confidence compilation
- Geometric sinusoidal embedding init with orthogonal subspaces
- Shadow pair training data to teach base-10 vs base-60 distinction

### Quick start

```bash
# Install dependencies
npm install

# Run the Luxon engine tests
node core/computation/temporal_engine.test.js  # 26 tests

# Run the tokenizer/compiler tests
python3 tests/test_domain_tokenizer.py         # 25 tests

# Generate training data (Route-format)
node core/synthetic_data/generators/temporal/generator.js --count 5000

# Dry-run embedding resize (no ML stack needed)
python3 -m src.utils.resize_embeddings --dry_run
```

## Original Experiment: Tagzeit

The [Tagzeit experiment](./experiments/temporal-tagzeit/) was the founding prototype, using a `[THINK]` chain-of-thought approach with SmolLM-135M. It demonstrated that small models can learn temporal formatting but struggle with the underlying arithmetic — motivating the architectural shift to Route-to-Luxon.

See the [Tagzeit README](./experiments/temporal-tagzeit/README.md) for the original experiment documentation.

## Contributing

We welcome contributions at every level:

- **New deficiency areas.** Open a discussion or comment on [Issue #9](../../issues/9) to propose a new category or specific failure mode.
- **Experiment development.** Pick an experiment marked 🔲 Planned and open a PR with an initial scaffold.
- **Core infrastructure.** Improvements to shared evaluators, data loaders, or CLI tools benefit every experiment.

Please read [`common-docs/CONTRIBUTING.md`](./common-docs/CONTRIBUTING.md) before submitting a pull request.

## Roadmap

| Milestone | Status |
|---|---|
| Tagzeit prototype stable on `main` | ✅ Complete |
| Route-to-Luxon pipeline implemented | ✅ Complete (v1) |
| Opus peer review (21 issues) | ✅ Resolved |
| Baseline measurement (vanilla Gemma-2B) | 🔲 Next |
| End-to-end pipeline validation | ✅ Complete (Initial: 12.5% format success) |
| Second experiment scaffolded | 🔲 Future |
| Shared evaluation dashboard | 🔲 Future |

## License

This project is licensed under [LICENSE](./LICENSE). Individual experiments may include third-party datasets or model weights subject to their own licenses; these are documented in each experiment's README.

---

**Maintained by mandipgosal.** Questions, ideas, and critique are welcome—open an issue or start a discussion.
