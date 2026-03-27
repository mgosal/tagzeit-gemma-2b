# Pipeline Task Status

Architecture: **Option B** — LLM routes, Luxon computes. The model never does math.

- [x] Symbolic Expression Detector (`src/tokenizer/domain_tokenizer.py`)
- [x] Temporal Compiler Plugin (`src/tokenizer/compilers/temporal_compiler.py`)
- [x] Luxon Computation Engine (`core/computation/temporal_engine.js`)
- [x] Synthetic Data Generator (`core/synthetic_data/generators/temporal/generator.js`)
- [x] Geometric Embedding Initialization (`src/utils/resize_embeddings.py`)
- [x] Opus Peer Review — 21 issues resolved
- [ ] Baseline Measurement (vanilla Gemma-2B)
- [ ] End-to-End Pipeline Validation
