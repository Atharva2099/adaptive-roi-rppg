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
- There will be no verifier module, no lifecycle markers, no `COMPLETE.json`, and no test-run scaffolding in the research path. Keep one direct runner, one fixed configuration, and only a few focused unit tests, then run the full probe once. Use Slurm job state and exit status, logs, direct output checks, and essential checks inside the scientific runner to establish execution and result completeness. Historical marker requirements in design documents do not override this rule or block a new run. Preserve existing historical artifacts.
- Keep scientific checks proportional and direct: verify the intended cohort, ground-truth separation, relevant baseline reproduction, complete output rows, and uncertainty calculations. A successful Slurm exit establishes execution success; inspect the results before making scientific claims. Record the command, inputs, configuration, and seeds without adding a custom completion protocol.
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

Before running numerical code or training, record exact provenance and confirm:

1. Dataset split and leakage checks, including subject-level separation.
2. Subject-level uncertainty reporting and the row-level artifacts needed to reproduce it.

Any later implementation must include tests appropriate to its interfaces and must document the evaluation ruler, fixed inputs, and failure limits. No experiment is complete until its source, configuration, command, outputs, and limitations are recorded.

## Compute safety

- Run heavy work through Slurm. Never run long CPU or GPU workloads on the Polaris login node.
- Parallel jobs must use an explicit multiprocessing context where needed, set thread limits before Python imports numerical libraries, propagate worker failures, and record actual-job timing.
- Recurrent training-path changes require focused unit coverage of prediction and backward propagation before submission; do not add a separate smoke job.
- Preserve the job ID, exact command, configuration, code commit, input hashes, output paths, logs, and output hashes for every remote run.
- Never use `rsync --delete` without explicit authorization for the exact deletion scope.
- Do not treat a local SSH or DNS failure as proof that Polaris or a remote job failed. Verify scheduler state and logs directly.

## Polaris access and job operations

### Remote files are protected

- Never delete, move, rename, archive, compress in place, relocate, or recursively change permissions on anything on Polaris without explicit user approval naming the exact paths and operation. This includes source, datasets, checkpoints, logs, failed runs, temporary files, caches, and old outputs. Do not perform automatic remote cleanup.
- Never use `rm`, `mv`, `rsync --delete`, destructive Git commands, broad globs, or shell redirection to replace existing remote material as part of routine job setup. Do not overwrite an existing checkout or output directory. Stage authorized code in a fresh, uniquely named directory and write each run to a new output path. Copy only the files needed for the authorized task; leave remote datasets and checkpoints in place.
- Before any approved destructive operation, inspect the exact targets read-only. Do not expand approval for one path into approval for its parent or neighboring runs. Preserve unrelated jobs and files.

### Connect and inspect from the Codex terminal

- Use terminal commands directly for an authorized Polaris experiment; do not ask the user to manually perform ordinary SSH, submission, or status checks. Start with `ssh -o ConnectTimeout=10 -o BatchMode=yes polaris 'hostname'`. The configured alias uses the user's existing authentication; never print credentials, read private keys, or disable host-key verification. An authentication prompt or changed host key needs user attention.
- Use `ssh polaris 'REMOTE_COMMAND'` for bounded commands. Keep remote commands in single quotes so remote variables are expanded remotely. For complex work, stage an inspectable script and run it; avoid nested `sbatch --wrap` quoting and long inline Python programs.
- If the terminal sandbox blocks DNS/network access, retry the same required SSH command through the tool's escalation mechanism. A local connection failure does not establish cluster or job failure.
- Inspect actual remote paths and Python availability before staging or submission. Use explicit paths for code, interpreter, inputs, checkpoints, logs, and outputs. Do not assume the login-node environment or current directory carries into a batch job.
- For authorized staging, create a fresh remote directory, then use `scp` for explicitly selected source/configuration files. Never synchronize the entire local workspace or modify the legacy checkout as a convenience. Creating new run files is allowed within an authorized experiment; existing files remain protected.

### Partitions, wall times, and QoS

The following values were read directly from Polaris on **2026-09-08** using `sinfo`, `scontrol`, and `sacctmgr`. Recheck them before submission; they are configuration observations, not permanent guarantees.

| Partition | Configured maximum wall time | Intended resource choice |
|---|---|---|
| `cpucluster` (default) | `18:00:00` | CPU work |
| `cputest` | `01:00:00` | Short CPU checks |
| `gpucluster` | `08:00:00` | GPU work |
| `gpuquick` | `02:00:00` | Short GPU work |
| `highmem` | Unlimited in partition configuration | Work with a demonstrated large-memory need |

- The current user association is account `researchers`, allowed QoS `simple-qos`. This QoS has **MaxJobsPU=2** (running jobs per user) and **MaxSubmitPU=5** (submitted jobs, including pending and running). Its MaxWall field and the association's default QoS were blank. Specify `--qos=simple-qos` explicitly; do not assume other listed QoS names are available to this user.
- Request a finite `--time` based on measured runtime with reasonable margin, within the applicable partition/QoS/association limits. Examples: `00:15:00` for a short analysis, `04:00:00` for a measured multi-hour evaluation, or `16:00:00` for a long CPU run. These are request examples, not extra QoS tiers. Do not choose `highmem` merely to evade a time limit.
- Recheck limits and current occupancy with:

```bash
ssh polaris 'sinfo -o "%P %a %l %D %c %m"'
ssh polaris 'scontrol show partition'
ssh polaris 'sacctmgr -n -P show assoc where user=$USER format=Cluster,Account,User,Partition,QOS,DefaultQOS'
ssh polaris 'sacctmgr -n -P show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU,Flags'
ssh polaris 'squeue -u "$USER" -o "%.18i %.12P %.24j %.10T %.10M %.10l %.24R"'
```

- Arrays consume job limits through their tasks. `--array=0-15%2` limits concurrent array tasks but does not guarantee admission under a five-submitted-job limit. Account for existing jobs; use appropriately sized submission batches or a tested worker pool inside one allocation. Preserve the full scientific cohort when batching.
- `QOSMaxJobsPerUserLimit` means a running-job limit is preventing scheduling; `QOSMaxSubmitJobPerUserLimit` means submitted-job capacity is exhausted. Inspect the queue and wait or reduce submission batch size. Do not cancel other work or change QoS to evade limits. `TIMEOUT`, `OUT_OF_MEMORY`, and nonzero application exits require inspecting logs and the requested resources before retrying.

### Submit, monitor, and retrieve results

- Run heavy CPU/GPU computation only inside Slurm allocations. On the login node, limit work to bounded inspection, staging, syntax checks, and submission. Set CPU/thread limits before importing numerical libraries; request CPUs, memory, and GPUs explicitly for the actual workload. Discover available GPU resources before choosing a GPU request.
- Syntax-check the actual launcher with `bash -n /ABSOLUTE/STAGED/slurm/run.slurm`; verify its interpreter, inputs, working directory, and output parent exist. Use an explicit script, `--chdir`, and persistent log paths. Replace every placeholder below with a verified path before execution:

```bash
ssh polaris 'sbatch --parsable --account=researchers --partition=cpucluster --qos=simple-qos --time=04:00:00 --chdir=/ABSOLUTE/STAGED --output=/ABSOLUTE/NEW_RUN/slurm-%j.out --error=/ABSOLUTE/NEW_RUN/slurm-%j.err /ABSOLUTE/STAGED/slurm/run.slurm'
```

- Record the returned job ID immediately. After an ambiguous SSH response, inspect `squeue` and `sacct` for the submitted job before resubmitting, to avoid duplicate runs.
- Monitor a specific job using the following commands, replacing `JOB_ID` and log paths with actual values:

```bash
ssh polaris 'squeue -j JOB_ID -o "%.18i %.10T %.10M %.10l %.30R"'
ssh polaris 'scontrol show job JOB_ID'
ssh polaris 'sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,AllocCPUS,MaxRSS -P'
ssh polaris 'tail -n 80 /ABSOLUTE/NEW_RUN/slurm-JOB_ID.out /ABSOLUTE/NEW_RUN/slurm-JOB_ID.err'
```

- Disappearance from `squeue` is not evidence of success. Use `sacct` to inspect allocation and batch/step exit states; blank accounting fields mean unavailable. The launcher must propagate application failures (`set -euo pipefail`, or explicit handling of worker exit codes), so Slurm reflects the actual program outcome.
- Successful completion uses Slurm state/exit code and logs, followed by direct CSV/output checks and row-level scientific analysis. Do not add custom completion-marker files or publication frameworks. Retrieve authorized result files with `scp polaris:/EXACT/RESULT /NEW/LOCAL/DESTINATION` while leaving remote originals intact.
- Cancel, requeue, or change resources for an existing job only when the user authorizes that operation on the specific job. Use `scancel JOB_ID` only for that exact approved ID; never use broad user-wide cancellation. Record failures and reruns, retain their logs, and give every retry a fresh output path.

## Advisor communication

- Before thesis writing or editing, read `thesis/rules.md`; use `thesis/evidence.md` for sources and version decisions and `thesis/plan.md` for progress. Use `thesis/thesis.docx` for the working Word document once created from the school template. Keep thesis guidance in `thesis/rules.md`, with this root file as the only AGENTS.md for thesis work.

- Student: Atharva Walawalkar. Advisor: Professor Kazunori Okada at SFSU.
- Use plain language in advisor-facing communication.
- Do not use em dashes or the word `bug` in advisor-facing writing. Use `issue`, `refinement`, or `update based on prior findings`, as appropriate.
