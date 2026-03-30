# ADR 002: MLOps Artifact Tracking & Model Weights Retention

## Context
In early experiments (up to Experiment 003), Model weights (e.g. `*.safetensors`), training datasets (`*.jsonl`), and evaluation datasets were generated dynamically through node scripts and Python training loops. Because `.gitignore` correctly prevents gigabyte-heavy models from bloating the Git repository, these generated artefacts were frequently discarded, lost during machine migrations, or deleted to save space. Consequently, the experiment reports were present, but the physical artefacts (the neural network "brain" and the data it learned from) were not reproducible without manually remembering CLI commands.

## Decision
We will decouple **Code**, **Data**, and **Weights**, but strictly link their provenance using formal automated scripts and configuration invariants.

1. **Data Provenance via Scripting**: We will never run data generation scripts anonymously. All `.jsonl` files must be outputted by an explicit, version-controlled execution script (e.g., `00_generate_data.sh`) that explicitly guarantees the data format, path, and seed align with the trailing experiments.
2. **Explicit Artifact Storage**: Training Configurations (e.g., `003-smollm2-360m.yaml`) must explicitly define their `output_dir` (e.g., `experiments/route-to-luxon/weights/...`). We will no longer rely on tool defaults. 
3. **Model Registry & Archiving**: While `.gitignore` will continue to ignore `*.safetensors`, the "good practice" will be to push successful runs directly to a Model Registry (like HuggingFace Hub) or ensure explicit local symlinking/backing up of the resulting weights directories.

## Consequences
- **Positive:** Full reproducibility. Any developer can check out the repository, run `00_generate_data.sh`, then `01_run_training.sh`, and exactly reproduce the experiment results locally without manual CLI arguments.
- **Negative:** Slightly more boilerplate required per-experiment (shell scripts mapping pipelines).
