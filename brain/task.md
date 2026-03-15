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

## Phase 3: Generator & Pipeline Hardening
- [x] Refactor `generator.js`: Standardize delimiters and linearize math traces.
- [x] Expand `generator.js`: Implement 12h AM/PM with case jitter.
- [x] Expand `generator.js`: Add "Humanized" categories (Medicine, Travel, Calendars).
- [x] Expand `generator.js`: Implement Time Zone logic (UTC offsets).
- [x] Expand `generator.js`: Add Noon/Midnight special token reasoning.
- [x] Patch `cpt_train.py`: Fix field-mapping, streaming/packing compatibility, and learning rate.
- [x] Generate fresh 5k records and perform a final Opus Audit check.
- [ ] Execute Run 3 Training (Deferred).
- [ ] Evaluate results and document in `README.md`.
- [x] Publish internal artifacts to GitHub in `brain/` directory.
