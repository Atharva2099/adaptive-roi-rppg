# MCD simple-selector diagnostic

## Contract

**Question:** Does a parameter-free current-signal selector outperform the
frozen Advantage PPO controllers, and does switching alone help?

**Required arms:** Fixed full-face POS; `max_ppr`; and Advantage PPO
seeds 0, 1, and 2. The deterministic `random_legal` null control is not
re-run here. A historical causal streaming evaluation on the same 533 clips
and 91,227 hops produced 13.3657 BPM for uniform random ROI requests passed
through the minimum-hold controller, versus 12.3865 BPM for its matched
full-face arm. That result is supporting historical context, not a jointly
bootstrapped arm in the current pipeline. Oracle B (3.7328175200960665 BPM)
is accepted descriptive headroom context, not a jointly bootstrapped arm.

**Negative control:** The newly replayed full-face arm must reproduce the
accepted Gate 8 rows within the explicitly recorded tolerance. Any mismatch,
missing row, nonfinite value, or subject overlap stops the run.

**Interpretation limits:** MCD evaluation only, 533 clips and 89 subjects.
Labels are joined after every policy trajectory is complete. The selector is
not tuned, and MMPD is never used. The three PPO seeds are independent runs;
their arithmetic clip MAE mean is not an ensemble.

**Bootstrap estimand:** 10,000 draws. Each draw resamples the 89 subjects
with replacement (a subject drawn twice carries its clips twice) and
computes each method's equal-clip MAE over the resampled subjects. The
paired comparison is `max_ppr - advantage_mean`, where
`advantage_mean` is the per-draw mean of the three Advantage PPO seeds.
Percentiles (2.5/97.5) are taken on the per-draw differences to form the 95%
CI.

The same paired draws also compute `max_ppr - full_face`. This interval tests
whether the observed difference between those policies is consistent across
subjects.

**Outputs:** the run shards clips subject-disjointly across `--shard-count`
producers. Each shard writes `per_hop.csv`, `observations_advantage_ppo_seed1.csv`
(only rows for the seed-1 Advantage PPO checkpoint; empty if the shard's
clips produce none), and `shard_accum.json` (the shard's per-clip error/hop
totals and behavioral counters). A separate `--merge` invocation reads every
shard's `shard_accum.json`, checks full coverage with no duplicated
(method, subject, clip) keys, and writes the merged `report.json`. With the
default `--shard-count 1` (no `--merge`), the runner still produces
`per_hop.csv`, the observations CSV, and `report.json` in one pass.

**Stop/next rule:** If max-ppr is worse than full-face, interpret this
as evidence that the current PPR feature is not a reliable selection
score and may be over-trusted by `R=r0/max(confidence,1e-6)`, not as evidence
that adaptive ROI selection is impossible. If PPO is better, that supports an
RL-specific advantage over this rule; if max-ppr is better, reconsider
the controller; if the interval crosses zero, the comparison is inconclusive.

The max-ppr rule uses valid measurements only. Exact ties keep the
current executed ROI; otherwise the lowest ROI index wins deliberately, so an
unresolved tie defaults to ROI 0 (`full_face`). All-invalid hops stay on the
previous action, or use ROI 0 on the first hop. Both selector arms use the
canonical two-hop minimum-hold rule.

**Score change (2026-09-08):** the arm originally ranked by
`measurement.confidence`. It now ranks by `measurement.peak_power_ratio`. The
measurements behind that change are recorded under "Score change and its
provenance" below.

The runner also stores each exact pre-action 101-D observation and 12
same-state alternative transitions. Alternatives never chain and ground truth
can change only scored error columns.

Run date: 2026-09-08. Cluster: Polaris (`n1.hpc.at.sfsu.edu`), partition
`cpucluster`, account `researchers`, QoS `simple-qos`. Job submission was
performed by the user.

This note records measurements only. Interpretation is not included.

## Cohort

MCD evaluation split. 533 clips, 89 subjects, 91,227 hops per arm. The cohort
is the Gate 8 binding produced by `_bindings` in
`scripts/verify_gate8_mcd_frozen_models.py`, which applies the seven E-031
exclusions to the 540-clip eval inventory. No MMPD data were accessed.

## Arms

| method_id | family | Definition |
|---|---|---|
| `full_face` | `fixed_full_face` | Requests canonical ROI 0 at every hop |
| `max_ppr` | `simple_selector` | Requests the valid measurement with maximum `peak_power_ratio`; exact ties keep the current executed ROI, otherwise the lowest ROI index; all-invalid hops keep the previous action, or ROI 0 at hop 0 |
| `advantage_ppo_seed0/1/2` | `advantage_ppo` | Frozen RecurrentPPO checkpoints from the Gate 8 checkpoint package |

`random_legal` was implemented and tested but not executed in these runs.

All arms pass their requested action through `control_step`, so the canonical
two-hop minimum-hold rule applies to every arm. Ground-truth labels are joined
only in `score_simple_rollout`, after each trajectory is complete.

## Equal-clip MAE, full cohort (job 49143)

Aggregation weights clips equally. Intervals are 10,000-draw subject-block
bootstrap, seed 8101, subjects resampled with replacement.

| method_id | Equal-clip MAE (BPM) | 95% CI (BPM) | Headroom captured |
|---|---:|---|---:|
| `advantage_ppo_seed1` | 7.5007 | 6.761 to 8.292 | 57.7% |
| `advantage_ppo_seed0` | 7.6375 | 6.903 to 8.397 | 56.1% |
| `advantage_ppo_seed2` | 7.7315 | 6.967 to 8.534 | 55.1% |
| `full_face` | 12.6317 | 11.363 to 13.924 | 0.0% |
| `max_ppr` | 12.9582 | 11.487 to 14.470 | −3.7% |

Headroom captured is `(12.631680651997897 − MAE) / (12.631680651997897 −
3.7328175200960665) × 100`, using the E-031 full-face value and the E-033
Oracle B value as fixed constants.

Paired difference, computed per bootstrap draw before percentiles are taken:

| Estimand | Point (BPM) | 95% CI (BPM) |
|---|---:|---|
| `max_ppr` − mean of the three Advantage PPO seeds | +5.3350 | +4.337 to +6.351 |
| `max_ppr` − `full_face` | +0.3265 | −0.153 to +0.802 |

The `max_ppr` − `full_face` interval was computed after job completion from
the existing 32 `shard_accum.json` files. It uses the same 10,000 subject-block
draws and seed 8101; no selector trajectory was rerun.

## Negative control

The replayed `full_face` arm returned equal-clip MAE
`12.631680651997897`, matching the accepted E-031 value within the runner's
`1e-9` tolerance. The run's assertion on this value was armed (it is skipped
only when `--limit-clips` or `--clip-offset` is in effect) and did not fire.

The three Advantage PPO values match E-031 (`7.637492779923465`,
`7.5006634485642305`, `7.731478074893967`) to the four decimal places reported
above.

## Behavioral statistics, full cohort

Switch rate counts hop-to-hop changes in executed action within a clip, so each
clip's first hop is excluded from the denominator. Override rate is the
fraction of hops where `proposed_action != executed_action`. Invalid-selected
rate is the fraction of hops whose selected measurement was invalid.

| method_id | Switch rate | Min-hold override rate | Invalid-selected rate |
|---|---:|---:|---:|
| `full_face` | 0.0000 | 0.0000 | 0.0459 |
| `max_ppr` | 0.3744 | 0.2089 | 0.0459 |
| `advantage_ppo_seed0` | 0.2925 | 0.0450 | 0.0464 |
| `advantage_ppo_seed1` | 0.3490 | 0.0956 | 0.0472 |
| `advantage_ppo_seed2` | 0.3591 | 0.0301 | 0.0483 |

Executed-action histograms, ROI 0 through ROI 11, 91,227 hops per arm:

| method_id | Histogram |
|---|---|
| `full_face` | 91227, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |
| `max_ppr` | 20816, 6976, 6864, 5669, 5466, 6959, 7750, 7400, 5933, 6105, 5629, 5660 |
| `advantage_ppo_seed0` | 45519, 2246, 11684, 5247, 1280, 2084, 1488, 2425, 5367, 277, 8581, 5029 |
| `advantage_ppo_seed1` | 37510, 1180, 3262, 86, 16299, 13771, 3796, 875, 2015, 6524, 3443, 2466 |
| `advantage_ppo_seed2` | 39186, 823, 1119, 598, 17012, 4255, 1269, 5290, 7124, 2009, 5978, 6564 |

## Preceding smoke runs

Smoke runs used `--limit-clips`, which disables both the E-031 assertion and
the 533-clip coverage check. Their outputs are partial-cohort measurements.

| Job | Date | Clips | Offset | Shards | Selector score | Elapsed | State |
|---|---|---:|---:|---:|---|---|---|
| 49095 | 2026-09-08 | 12 | 0 | 2 | `confidence` | 00:02:07 | COMPLETED 0:0 |
| 49096 | 2026-09-08 | 12 | 0 | 2 | `confidence` | 00:01:45 | COMPLETED 0:0 |
| 49105 | 2026-09-08 | 12 | 200 | 2 | `peak_power_ratio` | 00:01:54 | COMPLETED 0:0 |

Jobs 49095 and 49096 are duplicate submissions of the same configuration.

In job 49095 the selector scored by `measurement.confidence`, defined in
`src/adaptive_roi_rppg/signal/pos.py` as `ppr * coverage`. Its executed-action
histogram was `[2063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` over 2,063 hops, and
its equal-clip MAE was `17.4599`, equal to the `full_face` arm in the same run.
Per-ROI means measured from that run's stored observations (1,032 rows,
shard-0): `peak_power_ratio` ranged 0.1776 to 0.1883 across the 12 ROIs, and
`coverage` was 0.9316 for ROI 0 against 0.0405 to 0.1327 for ROI 1 through 11.
ROI 0 held the maximum confidence on 100.0% of those hops.

### Score change and its provenance

The pre-registered rule in `configs/evaluation/mcd_simple_selectors_v1.json`
originally scored by `measurement.confidence`. After job 49095 the score was
changed to `measurement.peak_power_ratio`, dropping the `coverage` factor. The
change was made after observing job 49095's outputs, not before any run. The
recorded basis is the measurement above: `coverage` varies by a factor of 7 to
23 between ROI 0 and ROI 1-11 while `peak_power_ratio` varies by under 6%
across all 12, so the argmax of their product selected ROI 0 on 100.0% of the
hops measured, and the arm's equal-clip MAE equalled the `full_face` arm's to
the four decimals reported.

Job 49105 used a clip range disjoint from jobs 49095 and 49096. Job 49143 used
the full 533-clip cohort, which includes the clips used in all three smokes.
No threshold, hysteresis, or weighting parameter was introduced by the change;
both scores are parameter-free. No other scoring formula was executed on this
cohort.

Job 49105's partial-cohort equal-clip MAE values were
`max_ppr` 7.1716, `advantage_ppo_seed1` 7.4968, `advantage_ppo_seed2` 8.5563,
`full_face` 9.9310, `advantage_ppo_seed0` 10.4674, over 2,057 hops from 12
clips and 3 subjects.

## Full run job record

| Field | Value |
|---|---|
| Job ID | 49143 |
| State / exit code | COMPLETED `0:0` |
| Elapsed | 00:05:13 |
| Submitted | 2026-09-08 |
| Partition / QoS / account | `cpucluster` / `simple-qos` / `researchers` |
| Allocation | `--nodes=1 --ntasks=32 --cpus-per-task=2 --time=04:00:00` |
| Shards | 32, subject-disjoint via `canonical_subject_shard` |

## Source files

Staged at `/Users/924254653/adaptive_roi_simple_selectors_20260908_v2` on
Polaris. Local SHA-256 at the time of the run:

| File | SHA-256 |
|---|---|
| `src/adaptive_roi_rppg/evaluation/simple_selectors.py` | `3ebd644e4f0295d9967b1769155b90eb2c2ad7d738ad9b0ac863c5ebbfde53da` |
| `scripts/evaluate_mcd_simple_selectors.py` | `25346dd25abc8ee4c15025e8be89a09f782727aad45a0cda33bab060370b83e7` |
| `scripts/verify_gate8_mcd_frozen_models.py` | `1d5bd14ad95fda55c50b10bd2cac075818be01ef6be1d8763226e935c9590bc1` |
| `slurm/mcd_simple_selectors_full.slurm` | `2fa0184f83a7cc6a1f85a1a9068ba21eb50a851e0058bcab786a78abe86df85c` |
| `configs/evaluation/mcd_simple_selectors_v1.json` | `d8b85bce8c26107d04a2ec9e6357172d88bb16ffd952f66e6a592cc6c57cc476` |
| `configs/evaluation/mcd_frozen_models_v1.json` | `757b2bf104aafaf0e9fb9f70e1571c9f3db55b630ffde627e574c44c7258c0f1` |

Note: `scripts/verify_gate8_mcd_frozen_models.py` was not modified for this
diagnostic; it supplies `_bindings` and `_runtime_checkpoints` only.

## Inputs

| Input | Path |
|---|---|
| Manifest tree | `/Users/924254653/adaptive_roi_gate2_reviewed_clean_20260811.fdGR97/mcd_manifest_structured_v2` |
| State root | `/Users/924254653/mcd_stage/precomputed_rgb/semantic_state_vectors_12roi_full` |
| GT root | `/Users/924254653/mcd_stage/precomputed_rgb/ground_truths_full` |
| Checkpoint root | `/Users/924254653/adaptive_roi_gate8_checkpoints_20260814` |

## Outputs

Run root `/Users/924254653/adaptive_roi_simple_selector_runs_20260908/full/full-49143`.

| Artifact | Location | Content |
|---|---|---|
| `report.json` | `merged/` | Aggregates, bootstrap, behavioral statistics. 2,815 bytes. SHA-256 `28dd27a098f6587f473610f94d771a31bfb563a76dbe8f310309589c72bab2e6` |
| `per_hop.csv` | `shards/shard-*/` | 456,135 data rows across 32 shards (91,227 hops x 5 arms) |
| `observations_advantage_ppo_seed1.csv` | `shards/shard-*/` | 91,227 data rows: 101 float32 observation columns, 12 alternative absolute-error columns, executed action, GT |
| `shard_accum.json` | `shards/shard-*/` | Per-shard clip totals and behavioral counters consumed by the merge |

Shard directory total: 200 MB. Verified from the stored rows: 533 distinct
clip IDs, 89 distinct subject IDs.

## Runtime environment

Python `/Users/924254653/miniforge3/bin/python3`; numpy 2.2.0;
stable-baselines3 2.8.0; torch 2.13.0+cu130. Thread limits `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` set to 1
before Python start. Compute nodes provide 128 CPUs and approximately 580 GB.

## Scope

MCD evaluation split only. No training, tuning, checkpoint selection, or MMPD
access occurred. The Advantage PPO checkpoints were replayed frozen. Oracle B
and full-face constants are quoted from E-033 and E-031 and were not recomputed
here, except for the `full_face` arm, which was replayed and matched.
