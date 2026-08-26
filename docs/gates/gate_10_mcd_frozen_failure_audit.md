# Gate 10: frozen MCD failure audit protocol

Status: `IMPLEMENTATION-ONLY`. No MCD audit result has been run or accepted.

## Question and boundary

For the existing nine frozen MCD controllers, are difficult decisions better
explained by invalid measurements, a valid immediately better ROI that the
policy did not execute, or no valid immediate ROI? This is a one-hop,
post-training audit. It does not train, select a checkpoint, alter a reward,
or make a sequential-regret claim. MMPD is not an input to this gate.

## Immutable plan and retry design

`audit_plan.json` is created once under a user-supplied stable `RUN_ID`. It
binds the MCD split/cohort, source and environment hashes, nine checkpoint
identities, canonical 12-ROI order, expected hops, fixed random-control rule,
and logical ownership. Slurm job IDs and attempt IDs are deliberately outside
that identity.

The launcher code hash is a deterministic NUL-delimited hash of the current
working-tree bytes for Git-tracked files only, ordered by UTF-8 path bytes.
Untracked scheduler smoke logs do not change it; tracked source changes do.

Subject assignment is deterministic longest-processing-time balancing: sort
subjects by expected authoritative hop work descending, subject ID ascending,
then assign each to the least-loaded logical shard with shard-index tie break.
Logical shard count is independent from Slurm worker count. A completed,
validated shard is reused; a failed attempt is retained and only the missing
logical shard is resubmitted.

For a requested worker count and global shard batch, the plan also freezes the
physical-rank to ordered-logical-shard map. The launcher gives each rank only
that explicit local list. Its time check uses the sum of that rank's policy
work; a worker never redistributes its own list again.

This follows the **clean** Gate 8 CPU, subject-disjoint shard then merge shape,
not legacy code: [Gate 8 frozen MCD evaluation](gate_08_mcd_frozen_models.md).
The ROI ordering follows the fixed-sequence crossover contract:
[Gate 6.1](gate_06_1_fixed_sequence_crossover.md).

## Exact held-out contract

The runner rejects an evaluation plan unless it has 540 source clips, the
seven named Gate 8 exclusions, 533 retained clips, 89 subjects, no train/eval
subject overlap, nine checkpoint arms, and exactly 91,227 hops per checkpoint.
The merged compact table therefore has exactly `91,227 × 9 = 821,043` rows.

Each row is one checkpoint-hop and stores its 12 alternatives as canonical,
validated compact JSON. It avoids twelve repeated metadata rows while retaining
the selected decision, proposed/executed/override state, immediate validity,
best valid action, immediate regret, and a deterministic one-hop random
control. The alternatives do not change the actual frozen rollout state.

## Compute precondition

CPU is the only supported profile: one task and one numerical-library thread
per worker. A train-only smoke must finish and merge successfully before it can
write a benchmark record. The record derives worker and merge rates from the
observed outputs, then applies a 1.5 safety factor plus a 300-second fixed
reserve. Held-out shards and merge both reject before replay/labels if a
matching benchmark is absent or `AUDIT_WALLTIME_SECONDS` is insufficient.

GPU use is out of scope. It requires a separate deterministic resource profile
and a measured CPU case showing that the CPU profile is inadequate.

## Verified Polaris dispatch smoke (2026-08-26)

`FACT`: the scheduler-only smoke completed as Slurm job `48080` on Polaris
`cputest`: one node, 16 tasks, one CPU per task, five-minute allocation, exit
code `0:0`, and elapsed time `00:00:01`. All ranks `0` through `15` emitted
one `NO_DATA_ACCESS` proof with `worker_count=16`.

`FACT`: this was one `sbatch` allocation containing 16 `srun` tasks. It was
accepted under the observed `simple-qos` limits; it was not treated as 16
submitted jobs.

`FACT`: Polaris did not export `SLURM_TIMELIMIT` to the batch script. The
launcher was verified to obtain the allocated limit from `scontrol show job -o
$SLURM_JOB_ID` instead. The smoke used the explicit supported interpreter
`/Users/924254653/miniforge3/bin/python` (Python 3.13).

`LIMITATION`: this validates dispatch, rank identity, QoS acceptance, and the
Polaris runtime setup only. It did not open an MCD manifest, state vector,
label, checkpoint, or MMPD path; it establishes no physiological or model
result. `MaxRSS` was not reported for this short job.

Artifacts on Polaris:

- `/Users/924254653/adaptive_roi_mcd_failure/smoke/cputest_dispatch_64368f5/slurm-48080.out`
- `/Users/924254653/adaptive_roi_mcd_failure/smoke/cputest_dispatch_64368f5/slurm-48080.err`

Next: a one-worker, train-only `cpucluster` smoke must measure actual replay,
POS construction, output size, and merge time before choosing 4, 8, or 16
workers or submitting the held-out 533-clip audit.
