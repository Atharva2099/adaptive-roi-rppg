# Gate 7: MCD Oracle B/C teachers

Gate 7 is a frozen, oracle-only MCD training-split computation. The accepted
MCD manifest inventory has 3,060 train clips from 510 subjects and 540 evaluation
clips from 90 subjects, with no train/evaluation subject overlap. After the
three declared all-invalid clips are excluded, the eligible teacher cohort is
3,057 train clips from the same 510 subjects. It does not access MMPD, train a
model, run PPO, select a model or checkpoint, or tune a training decision. The
evaluation split is used only to prove the split boundary and is not scored.

## Frozen protocol

The verifier builds the plan from the authenticated MCD manifests and requires
`split=train`, 3,060 train clips, 510 train subjects, 540 evaluation clips, 90
evaluation subjects, and no subject overlap. Each clip uses the causal POS measurement contract
(`POS_CONFIG_ID`): trailing 8-second windows, 1-second hops, and the declared
MCD GT rule (`GT_RULE_ID`). The control contract is frozen as
`CONTROL_CONFIG_ID`, with `q=0.1`, `r0=200`, initial HR 70, a free first action,
and minimum hold 2.

The current source is the authenticated canonical MCD state CSV plus the
authenticated GT CSV. `read_mcd_canonical_frames` re-reads canonical frame
records; `build_pos_measurements` reconstructs each ROI's POS measurement. GT
is not passed to POS or to the control observation. It is loaded separately
for teacher selection and for scoring. Missing required RGB makes an ROI
invalid; there is no fill or clipping in the POS contract. Invalid GT, missing
or duplicate hop identity, and frame/label identity mismatches are rejected.

The only declared teacher exclusions are `7412_FullHDwebcam_after`,
`7412_IriunWebcam_after`, and `7412_USBVideo_after`. Each has exactly 171
invalid labels, all with `invalid_reason=degenerate_variation`. Any other
invalid label, or any mismatch in these three exclusion records, remains
fail-closed and rejects the plan.

### What B and C mean

- **Oracle B (`greedy_b`)** is a one-step GT-informed teacher. At each hop it
  tries each legal action, advances the current belief with that measurement,
  and chooses the action with the smallest immediate post-update
  `abs(post_belief_hr_bpm - gt_hr_bpm)`. Equal scores use the lowest canonical
  action index. B uses the current hop's GT, but does not inspect later camera
  hops or later GT values when choosing the current action.
- **Oracle C (`beam_c`)** is a sequence-aware GT-informed teacher. It expands
  legal action sequences with beam width 8, carries each candidate's complete
  control and belief state, and ranks by cumulative post-update absolute error,
  then action sequence. C can use full-clip future GT and future measurements
  while choosing an earlier action.

Both are offline teachers. B is not a deployable causal policy because it needs
current GT. C is additionally noncausal because it uses future full-clip
information. Neither result is deployment evidence.

### Belief and scoring

For every candidate action, `control_step` selects the proposed/executed ROI
measurement and updates the shared post-belief. The belief is a constant-
velocity one-dimensional Kalman state: prediction, then a measurement update
whose noise uses the selected measurement confidence. Invalid selected ROI
measurements produce the declared predict-only behavior. The stored score is
the post-update absolute HR error against the current GT label. Oracle C's
chosen sequence is replayed from a fresh initial state and every derived row
field must match.

The GT never enters POS input. POS receives only the canonical frame RGB
summaries and camera FPS; GT enters after POS, in B/C action scoring and in the
reported error. This separation is enforced by the interfaces and by the
replay checks in `src/adaptive_roi_rppg/evaluation/oracles.py`.

## Source-authoritative merge and acceptance

Shard rows are not authoritative. The merge first validates every shard's
plan, subject assignment, hashes, marker set, row identity, legal hold chain,
and summary. It then re-reads the exact authenticated canonical frames and GT
for every bound train clip, rebuilds POS measurements, and recomputes B and C
with bounded workers. Any row or summary mismatch against the shard is
rejected. The final report and aggregation are made from this fresh
recomputation, then the report is written, hashed, re-read, and validated
again. The aggregate is equal-clip MAE with subject-level and view/condition
summaries; `dCB = B_MAE - C_MAE` is derived from those clip rows.

Each shard and the merged report uses the marker protocol below:

1. Create a fresh output directory and write `STARTED.json` exclusively.
2. Write and validate the substantive artifact and its hash sidecar.
3. Write `COMPLETE.json` exclusively **last**, after all fallible checks.
4. If a run fails before completion, retain `STARTED.json` and write
   `FAILED.json` best-effort. A directory with `STARTED.json` plus
   `FAILED.json`, partial output, or any mixed marker state is rejected. A
   `COMPLETE.json` is accepted only when its report hash and provenance match.

All 16 shard completions, the merged report sidecar, the reread report, and
the final merged `COMPLETE.json` are required. The accepted run below satisfies
these requirements.

## Accepted full-run result

Polaris job `47411` completed with exit `0` in `01:35:35` on node `lmn01`.
The merged output is
`/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged`.
All 16 shards completed and the merged output contains `COMPLETE.json`; no
`FAILED.json` is present. The report schema is
`gate7-mcd-oracle-report-v2`.

The eligible MCD train cohort is 3,057 clips from 510 subjects and 523,610
hops. Equal-clip MAE is `3.8833134521974753` BPM for Oracle B (`greedy_b`) and
`2.5482841213115375` BPM for Oracle C (`beam_c`), with 222,613 and 204,798
switches respectively. The derived gap is
`dCB = B-C = 1.3350293308859372` BPM.

The verified hashes are:

| Artifact | Path | SHA-256 | Bytes |
|---|---|---|---:|
| Report | `/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged/report.json` | `a5ac3a856639e0bc4f91cf2ad2696c2d96e9f70e954faf31ad29a75d9f1947d8` | 828677567 |
| Hash sidecar | `/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged/artifacts.sha256` | `81cdeac196a66676476614e6ddc275c7d46cd3c52219fffc6c2c4c07d7857be7` | 78 |
| Completion marker | `/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged/COMPLETE.json` | `8fe05103290e8a80ab5f880b9c0ae8e3b7a09892b8bfc9b32cb8da29b79d7a4c` | 2826 |
| Start marker | `/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged/STARTED.json` | `6d19eb0d1cec076827bfd49e61c5bf63abfb63f582a5f2cefc9e12e23dc32707` | 2749 |

The ruler is the current Gate 7 MCD-train oracle protocol: clip-equal MAE
over recomputed teacher rows, with switches counted from the selected action
sequence. These are offline teacher diagnostics. Oracle B uses current GT and
Oracle C uses future full-clip information, so neither is a deployable model
score. They are not directly comparable to legacy test90 or cache values. The
three excluded invalid clips remain exactly as documented above; no other
invalid clip is accepted.

## Current versus legacy boundary

| Boundary | Current Gate 7 | Legacy reference | Why the comparison is limited |
|---|---|---|---|
| Cohort | MCD train inventory: 3,060 clips, 510 subjects; Gate 7 eligible teacher cohort: 3,057 clips, 510 subjects | `/Users/atharva/Desktop/RL for RoI/docs/streaming_control_goal.md` records a historical disjoint 510-subject, 3,057/3,060-clip substrate; `/Users/atharva/Desktop/RL for RoI/results/streaming/phase1/diagnostics/cseq_label_consistency_polaris.txt` records 3,057 labelled clips | Legacy counts are historical context, not current Gate 7 authority or a matched numeric result |
| Input source | Authenticated canonical MCD state CSVs, re-read at merge; POS is rebuilt | `src/headroom.py` loads cached `hr_meas`, `conf`, and GT arrays; `temp/oracle_c_common.py` reuses that cache path | Current POS measurements and legacy cached arrays are different evidence sources |
| Missing RGB | No fill or clipping; missing required RGB invalidates that ROI/window | Legacy `_load_clip` fills missing RGB over the complete clip in `src/headroom.py` before measurement construction | Legacy filling can use later frames; it is not a causal equivalence claim |
| Invalid GT | `read_mcd_labels` requires authenticated, finite, valid labels and rejects invalid GT | Legacy `estimate_clip` keeps only finite derived HR rows (`np.isfinite(gt_hr)`) | Invalid-label handling and source validation are different |
| Ties | B uses `(immediate error, action index)`; C uses `(cumulative error, action sequence)` | Legacy `_run_greedy` uses strict improvement; legacy `_run_beam` sorts cumulative score and then selects the minimum beam | Tie behavior and row-level determinism are not identical by inspection |
| B/C causality | B uses current GT only; C uses full-clip future GT and measurements; both are teachers | Legacy `headroom.py` names B `_run_greedy` and C `_run_beam` with the same semantic distinction; `oracle_c_common.py` calls the shared beam/legality symbols | This is a semantic reference, not an exact numeric reproduction |
| Comparison boundary | Gate 7 compares only recomputed MCD-train B versus C teacher rows | Legacy paths are read-only semantic references | Do not compare a Gate 7 result with a legacy test90/cache result as a matched benchmark |

The legacy belief semantics are documented by
`/Users/atharva/Desktop/RL for RoI/src/ppo/streaming_belief_tracker.py`:
`StreamingBeliefTracker.kalman_step` is the shared constant-velocity Kalman
update and `clone()` supports beam trajectories. The legacy legality and
teacher symbols are `_allowed`, `_run_greedy`, and `_run_beam` in
`/Users/atharva/Desktop/RL for RoI/src/headroom.py`; the diagnostic helper
imports those symbols rather than copying them in
`/Users/atharva/Desktop/RL for RoI/temp/oracle_c_common.py`.

## Repair log

Times are recorded only to the precision present in the local evidence; the
entries below use the UTC date when no clock time is recorded.

| Timestamp | Event | Why it mattered | Repair or status |
|---|---|---|---|
| 2026-08-14 UTC | Earlier merge design was insufficient. | It could trust shard rows and summaries without independently proving that they came from fresh canonical frames/GT, complete shard coverage, legal transitions, and a valid publication state. | Terra review led to source-authoritative merge, exact row/provenance checks, fresh-state C replay, strict marker/report readers, and negative tests. |
| 2026-08-14 UTC | Non-final fixture, Polaris job 47380. | It covered one subject, six train clips, and 1,031 hops only; it was not the 3,060-clip Gate 7 result. | Retained as a fixture check, not a final value or legacy comparison. |
| 2026-08-14 UTC | Attempted full run, Polaris job 47381, cancelled. | A cancelled run cannot establish complete shard coverage or an accepted merged report. | It is not a result. No remote scheduler, log, or output claim is made here. |
| 2026-08-14 UTC | Terra repair completed in the local Gate 7 implementation. | Historical-cache arguments, weak merge acceptance, and incomplete provenance checks would have weakened the train-only evidence boundary. | The launcher now requires a clean BASE snapshot and local BASE Python; the verifier binds the 3,060/510 train plan, re-reads sources, recomputes B/C, validates exact rows, and writes `COMPLETE.json` last. |
| 2026-08-14 UTC | Final marker hardening was verified locally, before the accepted rerun. | Failure cleanup must not hide the experiment failure or publish a misleading state. | The original exception is preserved even if checking `COMPLETE.json`, writing `FAILED.json`, or formatting/writing the diagnostic fails; `FAILED.json` remains skipped after `COMPLETE.json`. Fifteen focused Gate 7 tests and 105 full tests passed locally. This repair work preceded accepted job 47411. |
| 2026-08-14 UTC | Repair rerun status before job 47411: fixture job 47404 completed successfully in 45:24 for 32 subjects and 192 clips; its marker and shard hash were valid. The first full job, 47405, failed before evaluation because Slurm's `PATH` lacked bare `git`, so no result was accepted from 47405. | The launcher depended on an unqualified executable in the Slurm environment. | The launcher now pins and validates `/usr/bin/git`; the local static test plus 15 Gate 7 tests and 105 full tests passed. This repair history is retained; accepted job 47411 is recorded above. |
| 2026-08-14 UTC | Full job 47406 failed before evaluation because the compute-node image lacked `/usr/bin/git`, even though the login node had it; no result was accepted. | The compute-node image did not provide the launcher’s required system executable. | The launcher now computes the declared hash directly from BASE source files using BASE-local Python, excludes `.git`, `.venv`, `__pycache__`, and `.pyc`, rejects other symlinks and nonregular entries, and retains isolated imports. This historical failure is retained; accepted job 47411 is recorded above. |
| 2026-08-14 UTC | Fixture job 47404 remains successful engineering evidence only. Full jobs 47405 and 47406 failed before evaluation and are not results. | 47404 validated the repaired engineering path on 32 subjects and 192 clips; 47405 failed on login-node `PATH` lookup for bare `git`; 47406 failed because the compute-node image lacked `/usr/bin/git`. | Retain 47404 as historical fixture/engineering evidence. Retain 47405/47406 as historical pre-evaluation launcher failures. They do not supersede or alter accepted job 47411. |
| 2026-08-14 UTC | Full job 47407 is `SUPERSEDED` and `INCOMPLETE`. | Shards 0–14 completed under the old v2/3,060-clip plan; shard 15 rejected the 7412 clips. There was no merge and no final result. | The shards cannot be reused because the v3 plan payload changes. Retain job 47407 and its repair history as historical failure/engineering evidence. It is superseded by accepted job 47411. |
| 2026-08-14 UTC | The evaluator's real-manifest check was repaired before rerun. | The accepted MCD inventory must bind both sides of the split, not only the 510-subject train side. | The check now requires 3,060 train clips/510 subjects, 540 evaluation clips/90 subjects, and zero train/evaluation subject overlap before a Gate 7 rerun. |
| 2026-08-14 UTC | Accepted final run, Polaris job 47411. | The run completed all 16 shards and published the merged report only after the required marker and report checks. | Gate 7 is accepted for the MCD-train Oracle B/C teacher diagnostic. The result is not a deployable model score, does not authorize MMPD use, and does not complete Gate 8. |

## Local verification visible in this worktree

The focused test command is:

```sh
PYTHONPATH=src python -m unittest tests.test_gate7_oracles
```

The tests exercise known-answer B/C behavior, current-GT sensitivity, future-GT
dependence of C, tie rules, replay mismatch rejection, row/marker validation,
aggregation mismatch rejection, exclusive completion files, and failure
without `COMPLETE.json`. The worktree contains the test and launcher checks;
no local full MCD computation is claimed by this document.
