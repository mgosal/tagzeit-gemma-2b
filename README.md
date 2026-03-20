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
├── core/                   # Shared utilities: evaluators, data loaders, base-model wrappers
├── experiments/            # Individual deficiency studies (one per directory)
│   ├── temporal-tagzeit/   # 🟢 Temporal reasoning — first prototype
│   ├── arithmetic-base-n/  # 🔲 Planned
│   └── planning-loops/     # 🔲 Planned
├── common-docs/            # Taxonomy deep-dives, methodology notes, contribution guides
├── tools/                  # Shared CLI scripts for training, evaluation, and reporting
├── LICENSE
└── README.md               # ← You are here
```

### Design Principles

1. **Sub-project autonomy.** Each experiment owns its README, configuration, and local tests. You can clone the repo and work inside a single experiment without touching anything else.
2. **Shared infrastructure.** Common evaluation metrics, data-loading patterns, and training harnesses live in `core/` and `tools/`. Experiments import from these rather than reinventing them.
3. **Reproducibility first.** Every experiment ships with pinned dependencies, seed-controlled data generation, and versioned model checkpoints where feasible.

## Active Experiments

### [Tagzeit](./experiments/temporal-tagzeit/) — Temporal Reasoning

The founding experiment. Tagzeit investigates how small language models (starting with Gemma 2B) handle date arithmetic, relative-time expressions, and calendar-system edge cases. It includes a synthetic data generator, a LoRA fine-tuning pipeline, and a structured evaluation suite.

**Quick start:**

```bash
cd experiments/temporal-tagzeit
pip install -r requirements.txt
python src/generate_data.py
python ../../tools/sft_train.py --config config/default.yaml
```

See the [Tagzeit README](./experiments/temporal-tagzeit/README.md) for full documentation.

## Contributing

We welcome contributions at every level:

- **New deficiency areas.** Open a discussion or comment on [Issue #9](../../issues/9) to propose a new category or specific failure mode.
- **Experiment development.** Pick an experiment marked 🔲 Planned and open a PR with an initial scaffold following the structure in `experiments/temporal-tagzeit/`.
- **Core infrastructure.** Improvements to shared evaluators, data loaders, or CLI tools benefit every experiment.

Please read [`common-docs/CONTRIBUTING.md`](./common-docs/CONTRIBUTING.md) before submitting a pull request.

## Roadmap

| Milestone | Target |
|---|---|
| Tagzeit prototype stable on `main` | Current |
| Monorepo structure finalised | `refactor/structure-migration` branch |
| Second experiment scaffolded (Arithmetic or Planning) | Next |
| Shared evaluation dashboard | Future |
| Published taxonomy paper / report | Future |

## License

This project is licensed under [LICENSE](./LICENSE). Individual experiments may include third-party datasets or model weights subject to their own licenses; these are documented in each experiment's README.

---

**Maintained by mandipgosal.** Questions, ideas, and critique are welcome—open an issue or start a discussion.
