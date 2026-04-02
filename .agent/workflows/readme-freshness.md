---
description: Ensure READMEs are updated with every commit to reflect current project state
---

# README Freshness Rule

Every commit to this repository must include README updates if the commit changes anything that READMEs describe. This is not optional.

## What to check before committing

1. **Root `README.md`**: experiment history table, roadmap status, key findings, tool listings
2. **`experiments/route-to-luxon/README.md`**: experiment table, current/active experiment description, key learnings, sample outputs
3. **Any sub-project README** that describes changed functionality

## When to update

- New experiment results → update experiment tables in both READMEs
- New tool added → update tool listings
- Roadmap milestone completed → update roadmap status
- Architecture change → update architecture description
- Even a single word change counts if it makes the README more accurate

## How to update

- Keep changes minimal and precise. Don't rewrite sections unnecessarily.
- If an experiment moves from "Active" to "Complete", update the status.
- If results are recorded, add the actual numbers (eval loss, accuracy).
- If a new file is created, check if any README references need updating.

## Enforcement

This is a standing rule for all commits in this project. The agent must self-check README accuracy before every `git commit` command.
