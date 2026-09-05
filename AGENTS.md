# Repository rules

This is the clean rebuild of the adaptive semantic ROI controller for causal classical rPPG. The system design and interfaces are approved. Build the implementation in the gated order defined in `docs/system_design.md`.

## Communication between agents

- Write all agent handoffs, questions, progress updates, and final reports in clear, concise English. Use complete sentences, correct grammar and punctuation, and proper spaces between words and numbers. Preserve exact code identifiers, paths, commands, and error messages.
- Start each handoff with the task and expected result. State the relevant evidence, scope, file ownership, constraints, and acceptance criteria without repeating unrelated conversation history.
- Report findings with concrete file references and verification results. Distinguish observed facts from assumptions, and state blockers and the next action plainly. Avoid compressed shorthand, unexplained abbreviations, and unnecessary jargon.
- Follow the active four-role workflow. The root passes relevant findings between agents and resolves conflicting reports. Agents may ask one another focused questions when useful, but must report decisions and blockers to the root and must not expand their scope or spawn other agents.
- Remind each implementation agent that it shares the workspace: preserve others' edits and modify only its assigned files.

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
- Keep historical results, retired methods, old jobs, and legacy paths in `docs/legacy_project_map.md` and `docs/evidence_registry.md`, not in this file.

## Research discipline and PI stance

- Act as a critical research collaborator. Challenge an idea when its metric, preprocessing, split, control, comparison, or expected information gain is weak.
- Separate verified facts, implementation details, assumptions, inferences, unknowns, and publication-safe claims. Never present an inference as a fact.
- Check preprocessing, data integrity, split leakage, metric definition, ruler compatibility, fair baselines, and negative controls before blaming or changing the model.
- Prefer experiments that change a research decision. Do not launch work merely because it might improve one number.
- After every result review, state: what we learned; what remains ambiguous; what could make the result misleading; and whether to continue, fix, or stop.
- Be direct when evidence is not publication-safe. Preserve negative results and narrow the claim instead of hiding them.

## Experiment protocol

Before running or endorsing a diagnostic, record:

1. `Question`
2. `Required arms`
3. `Negative control`
4. `Interpretation limits`
5. `Stop/next rule`

Do not run the diagnostic until the comparison is fair and the result that would change direction is explicit.

## Implementation and engineering discipline

- During debugging and experiment work, start with the observed behavior and trace the actual data/code path to a fix. Treat project labels, run markers, hashes, and protective framework code as background metadata; do not introduce, foreground, or discuss them unless they directly help solve the current technical problem.
- Do not use gates, SHA checks, lifecycle markers, or custom save guards as the main way to reason about or report work. Use them only when the user explicitly asks, or when a concrete failure requires them. Prefer direct inspection of code, data, visuals, logs, scheduler status, and actual outputs.
- Follow the implementation order and acceptance gates in `docs/system_design.md`. No training is allowed until gates 1-6 pass.
- Keep the repository small. Production code belongs under `src/`; launchers belong under `scripts/` or `slurm/`.
- Prefer the smallest clear implementation that satisfies the current gate. Do not add speculative frameworks, duplicate validators, compatibility layers, configuration systems, or production-scale abstractions without a demonstrated current need.
- Keep tests focused and consolidated. Test research-critical boundaries and observed failures; avoid large collections of repetitive unit tests or tests for impossible internal states.
- Treat a large line-count increase as a review trigger, not as evidence of completeness. Before adding a new module or helper, check whether the existing code can express the behavior clearly; remove dead or superseded implementation once its evidence is recorded. Never trade away correctness, dataset boundaries, or reproducibility merely to reduce line count.
- Put temporary work outside the committed project or under an explicitly ignored `temp/` directory. Do not commit generated result trees by default.
- Keep one-off probes and diagnostics out of production modules. Name temporary scripts for the question they answer.
- Any implementation derived from a paper must include an inline citation immediately above the relevant code: author, year, title, venue, and section, equation, or algorithm. This includes signal extraction, filters, spectral estimation, window lengths, and paper-derived constants.
- Discuss implications, alternatives, and verification before changing rewards, training loops, preprocessing, evaluation logic, or anything that changes reproducibility.
- Do not access secrets, credentials, remote systems, datasets, checkpoints, or legacy project material unless the current task explicitly authorizes the exact need.
- Do not delete or move project material without explicit exact approval. Preserve unrelated edits.
- Before a new gate changes accepted earlier-gate code, contracts, configuration, or evidence, explicitly flag the compatibility impact, explain the reason and affected evidence/tests, obtain user approval, and rerun the affected acceptance checks. Prefer a consumer-side adapter when the prior contract is sufficient.

## Before numerics or training

Before running numerical code or training, record exact provenance and run:

1. A synthetic known-answer smoke test, including short, empty, and NaN inputs where relevant.
2. Dataset split and leakage checks, including subject-level separation.
3. Subject-level uncertainty reporting and the row-level artifacts needed to reproduce it.

Any later implementation must include tests appropriate to its interfaces and must document the evaluation ruler, fixed inputs, and failure limits. No experiment is complete until its source, configuration, command, outputs, and limitations are recorded.

## Compute safety

- Run heavy work through Slurm. Never run long CPU or GPU workloads on the Polaris login node.
- Before any parallel job, test the actual parallel path on Linux/Polaris. Confirm worker speedup or safe failure, use an explicit multiprocessing context where needed, and set thread limits before Python imports numerical libraries.
- Any recurrent training-path change must pass a tiny Polaris GPU smoke that exercises prediction followed by backward propagation before a full job is submitted.
- Preserve the job ID, exact command, configuration, code commit, input hashes, output paths, logs, and output hashes for every remote run.
- Never use `rsync --delete` without explicit authorization for the exact deletion scope.
- Do not treat a local SSH or DNS failure as proof that Polaris or a remote job failed. Verify scheduler state and logs directly.

## Advisor communication

- Student: Atharva Walawalkar. Advisor: Professor Kazunori Okada at SFSU.
- Use plain language in advisor-facing communication.
- Do not use em dashes or the word `bug` in advisor-facing writing. Use `issue`, `refinement`, or `update based on prior findings`, as appropriate.
