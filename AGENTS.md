# Repository rules

This is the clean rebuild of the adaptive semantic ROI controller for causal classical rPPG. The project will include its own implementation, tests, training, and evaluation pipeline after the current migration and design work is approved.

## Dataset custody and independence

- **Absolute MMPD rule:** MMPD is evaluation-only. Never use MMPD data, labels, metadata, subjects, conditions, or evaluation outcomes to train, fine-tune, calibrate, select checkpoints or hyperparameters, design rewards or features, build curricula, or early-stop.
- MCD is the only dataset for training and development decisions. MMPD must remain a frozen, held-out transfer evaluation.
- Do not feed MMPD findings back into any MCD decision. Document a result as transfer evidence, not as tuning guidance.
- Never delete, move, rename, archive, compress, chmod, relocate, or copy local MMPD material without explicit approval naming the exact source, destination, and operation.
- The legacy repository is read-only by default. Copy knowledge, not code. Never make it a dependency or alter it.

## Evidence and claims

- Every pivotal number must be recomputed from row-level CSVs, with the aggregation rule, cohort, split, and artifact path recorded.
- Claims based only on remote Polaris output remain unverified until checked directly; state that limitation clearly.
- Separate `FACT`, `HYPOTHESIS`, `LIMITATION`, `SUPERSEDED`, `NEGATIVE RESULT`, and `INCOMPLETE` evidence. Preserve failed experiments and do not revive superseded evidence silently.
- Use simple language and do not overclaim. Do not import the legacy repository’s giant scoreboard or stale historical rules.

## Current build boundary

- During the current migration phase, do not add `src/` or implementation work until the system design and interfaces are explicitly approved. Update this rule when implementation begins.
- Keep the repository small. Later production code belongs under `src/`; launchers belong under `slurm/` or `scripts/`.
- Put temporary work outside the committed project or under an explicitly ignored `temp/` directory. Do not commit generated result trees by default.
- Do not access secrets, credentials, remote systems, datasets, checkpoints, or legacy project material unless the current task explicitly authorizes the exact need.
- Do not delete or move project material without explicit exact approval. Preserve unrelated edits.

## Before numerics or training

Before running numerical code or training, record exact provenance and run:

1. A synthetic known-answer smoke test, including short, empty, and NaN inputs where relevant.
2. Dataset split and leakage checks, including subject-level separation.
3. Subject-level uncertainty reporting and the row-level artifacts needed to reproduce it.

Any later implementation must include tests appropriate to its interfaces and must document the evaluation ruler, fixed inputs, and failure limits. No experiment is complete until its source, configuration, command, outputs, and limitations are recorded.
