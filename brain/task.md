# Task Checklist: Refining README and Project Documentation

- [x] Strip out embellishments and grandiose language from `README.md`
- [x] Explicitly state this is an experimental starting point, not the definitive core.
- [x] Keep the baseline conclusion objective (0% Exact Match, single logic check success).
- [x] Adjust PoT observation (Phase 1) to reflect the current known state (Format vs Math learning).
- [x] Remove "Humanity 12" branding for a simpler "12 everyday domains".

## Phase 2: Generative Evaluation
## Phase 2: Generative Evaluation
- [x] Modify `validate.py` to optionally consume a `.jsonl` input file.
- [x] Generate 500 records using `generator.js`.
- [x] Document the Run 2 results in the `README.md` engineering log.

## Phase 3: Run 3 Proof of Training (PoT)
- [ ] Generate 5k records for PoT (`--compact` mode).
- [ ] Modify `cpt_train.py` to directly consume the JSONL `text` field.
- [ ] Execute `cpt_train.py --tiny` to generate the LoRA adapter.
- [ ] Evaluate the adapter using `validate.py --mode cot`.
- [ ] Document final Run 3 results in `README.md`.
