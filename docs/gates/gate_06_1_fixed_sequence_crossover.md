# Gate 6.1: fixed-sequence MCD crossover diagnostic

Gate 6.1 is an additive diagnostic. It does not change accepted Gate 6
contracts or semantics and uses MCD evaluation clips only.

For every canonical action, the diagnostic evaluates a fixed sequence that
proposes that same action at every hop. The first action is legal and later
repetitions are legal under the existing minimum-hold rule. The historical arm
reads the corresponding column from the authenticated schema-v3 NPZ; the
current arm uses the existing causal POS measurements and `control_step`.
Each arm reports per-clip MAE, per-subject equal-clip MAE, current-minus-
historical deltas, invalid counts, action/override/switch/hold statistics, and
subject-block bootstrap intervals. ROI-index jumps are descriptive only and
must not be interpreted as learned-policy behavior.

Historical Oracle-B and Oracle-C are reported as unavailable unless exact
authenticated sequence artifacts are supplied. Learned arms are explicitly
`not_run_unavailable`; no Gate 7 model or checkpoint is assumed. The launcher
partitions sorted subjects by modulo into fresh shard directories and merges a
complete shard set deterministically. Shards and merged reports use hashes and
fresh-output checks.

The launcher is `scripts/verify_gate6_1_mcd_crossover.py`. It is intended to be
run inside bounded Slurm subject-disjoint shards; it does not access MMPD or
remote systems itself.

## Repair log

This log keeps the missed issues visible. Times are UTC.

| Timestamp | Issue found | Why it mattered | Fix | Verification state |
|---|---|---|---|---|
| 2026-08-13T03:00Z | Shard merge accepted incomplete or mixed results. | A report could look complete while missing clips, subjects, or ROI arms. | Added strict plan, subject assignment, 533-clip/89-subject, 12-arm, duplicate-row, and provenance checks. | Focused tests passed; final suite rerun pending after the last patch. |
| 2026-08-13T03:10Z | Current and historical rows were initially matched by position. | Different hops or GT targets could be compared as if they were the same hop. | Added exact `(dataset_id, clip_id, hop_idx, hop_time_s)` keys and GT equality checks. | Boundary tests added. |
| 2026-08-13T03:20Z | Historical hop times used frame intervals such as 8.033 seconds. | Gate 6 uses one-second hop times, so the crossover would fail or compare the wrong rows. | Reused the accepted rule: `(round(8*fs) + hop_idx*round(fs))/fs`. | Three-hop timing regression added; final rerun pending. |
| 2026-08-13T03:30Z | Arm names were not fully tied to action numbers and metadata. | Metrics could be labeled as one ROI while scoring another, or mix clips/subjects. | Added `arm_id ↔ action` validation and clip/subject/view/condition consistency checks. | Focused tests passed before the final row-validation patch. |
| 2026-08-13T03:40Z | `COMPLETE.json` could appear before the final validation. | Readers could observe a result that later failed validation. | Moved all report, sidecar, provenance, and file-set checks before the final completion marker; failures retain only STARTED/FAILED. | Marker tests added; final rerun pending. |
| 2026-08-13T03:50Z | Slurm job identity was not preserved in the merged evidence. | The run could not be tied back to a specific compute job and node. | Added job ID, node, command, plan ID, code hash, source hash, and shard provenance to the report. | Launcher requires Slurm variables; final rerun pending. |
| 2026-08-14T03:53:44Z | User requested a detailed record of missed fixes. | Review failures must remain auditable instead of disappearing into the final result. | Added this repair log and kept the failed review findings documented. | Documentation change made; final local verification still required. |
| 2026-08-14T04:05:15Z | Final review found that malformed metrics/statistics and contradictory merged provenance could still pass the publication boundary; one test expected an 11-arm shard to publish. | A syntactically valid JSON file is not enough: bad arithmetic, counts, distributions, or identity fields could contaminate the report, and the stale test did not exercise the real rejection point. | Added strict row identity, delta arithmetic, invalid-count, action-statistics, 12-arm, and subject-list checks before `shard.json`; required merged provenance to match the expected plan and common shard provenance; updated the test to assert `STARTED.json` plus `FAILED.json`. | Fix applied; full local verification and final Sol review pending. |
| 2026-08-14T04:18:43Z | Sol found that ROI jump magnitude was not part of the enforced action-stat schema, merged stats omitted the canonical proposed action/jump magnitude, and aliases such as `00` could represent action `0`. | These values describe the fixed arm’s behavior. If they are not canonical and bounded, the diagnostic can publish internally inconsistent action evidence. | Required exact canonical action keys, nonnegative finite ROI jump magnitude, canonical proposed action, and consistent merged action statistics before the completion marker; added the missing fields to fixtures. | Focused tests pass after the repair; full local verification and final Sol review pending. |
| 2026-08-14T04:24:19Z | Sol reproduced two final boundary failures: a shard could be corrupted between sidecar creation and `COMPLETE.json`, and a null ROI jump magnitude was accepted. | `COMPLETE.json` is the reader’s commit point. If it can be written after an undetected change, the report is not trustworthy. A null jump value also breaks the required numeric action-stat contract. | Added a final pre-COMPLETE reread of serialized shard JSON, digest sidecar, STARTED provenance, schema, and row semantics; required the ROI jump magnitude to be present, finite, and nonnegative in shard and merged validation; added a null-stat regression test. | Focused 8/8 and full-suite checks pass; final Sol review pending. |
| 2026-08-14T04:27:39Z | Sol reproduced an impossible fixed-action row: the arm proposed action 0 but reported action 1, an override, and a large ROI jump. | Every Gate 6.1 arm repeats one action from a fresh state, so those transitions cannot occur. Accepting them would make the crossover report internally false. | Required the distribution to be exactly `{action: hops}`, all override/switch/jump counts and ROI jump magnitude to be zero, and hold statistics to match the fixed sequence; added a contradictory-stat rejection test. | Focused 9/9 and full-suite checks pass; final Sol review pending. |
| 2026-08-14T04:32:17Z | Sol found that merged hold validation treated multiple clips as one continuous sequence, even though every clip starts a fresh control state. | A valid multi-clip merge could be rejected, blocking the actual 533-clip diagnostic. | Changed merged validation to compare against the hop-weighted per-clip hold mean and maximum; changed the successful merge regression to use two clips. | Focused 9/9 and full local checks pass; final Sol review pending. |
| 2026-08-14T04:39:12Z | Polaris fixture job 47320 failed before starting because the copied launcher was one directory above the command path. | This was a packaging/path failure, not a data or scoring result. Without recording it, the failed job could be mistaken for a diagnostic failure. | Moved the launcher into the snapshot `scripts/` directory and will resubmit with the same verified inputs and code hash. | Failure preserved in Polaris log; repair applied; fixture resubmission pending. |
| 2026-08-14T04:47:21Z | Polaris job 47321 reached the launcher but rejected the historical reference mean by `2e-15` on Python 3.10, even though the CSV hash and 533-row Level-A cohort were correct. | Exact float equality made the accepted historical input platform-sensitive. | Replaced exact equality with `math.isclose(..., abs_tol=1e-12)`. This changes only floating-point representation tolerance, not the data, cohort, or metric. | Local suite passed after the compatibility fix. |
| 2026-08-14T04:47:21Z | Polaris job 47322 rejected the cache inventory because the new code snapshot did not contain the previously verified inventory file. | Provenance validation correctly stopped the run instead of using an unverified inventory. | Reused the original verified inventory at `/Users/924254653/adaptive_roi_gate6_snapshot_20260813/streaming_v3nc_cache_inventory.json`. | Failure preserved in Polaris log; corrected run passed. |
| 2026-08-14T04:47:21Z | Polaris job 47323 completed the real 12-clip fixture. | This confirms the repaired code works with real MCD state vectors, GT labels, historical NPZs, Slurm identity, and the publication protocol. | No code change. Verified `COMPLETE.json`, sidecar hash, 144 rows, 12 clips, 2 subjects, 12 arms per clip, and zero fixed-action overrides. | PASS: job 47323, node lmn01, 3:52, exit 0. |
| 2026-08-14T04:51:15Z | The requested 16-way full run was first rejected by `QOSMaxSubmitJobPerUserLimit`; a reduced 4-task array was then accepted as job 47328. One resubmission initially contained a manifest path typo and was cancelled immediately before execution. | Scheduler limits and command typos can create false “experiment failures” if they are not separated from code/data failures. | Reduced the array to four tasks at a time, kept `--shard-count 16` so subject assignment remains the same, cancelled the invalid-path submission, and resubmitted with verified Polaris paths. | Job 47328 was accepted with tasks 0–1 running and 2–3 queued. Subsequent SSH/DNS access failed, so final array state is not yet verified. No result claim made. |
| 2026-08-14T08:35:11Z | Merge job 47347 received only the final shard even though the shell command repeated `--shard-dir` 16 times. The parser used `nargs="+"` without `action="append"`, so repeated options overwrote earlier values. | The 16 shard files were valid, but the merge saw only shard 15 and correctly rejected it as an incomplete deterministic set. | Changed the launcher to use `action="append"` for repeated `--shard-dir` options, made argument parsing testable, and added a regression test covering two repeated shard flags. | Local Gate 6.1 tests 10/10, full test suite, compile check, and `git diff --check` pass. Polaris rerun requires a new snapshot containing this parser fix. |
| 2026-08-14T08:44:00Z | First single-allocation parallel submission, job 47349, launched all 16 tasks but each task could not import `adaptive_roi_rppg`. | The parallel scheduling design was valid, but the wrapper did not propagate the snapshot source path to Python. | Added `PYTHONPATH="$BASE/src"` before `srun`; preserved job 47349 as a launcher/environment failure, not an evaluation result. | Repair submitted as job 47350; initial state RUNNING on `lmn01`, exit code still pending. |
| 2026-08-14T09:00:00Z | The repaired single-allocation launcher completed all 16 shards and merged them automatically. | This confirms future Gate 6.1 runs do not require manually submitting shard batches. | Used one Polaris node with 16 Slurm tasks, one numerical-library thread per task, and an automatic post-shard merge. | PASS: job 47350 completed in 12:50 with exit `0:0`; 16 shard `COMPLETE.json` markers; merged `COMPLETE.json`; 6396 rows, 533 clips, 89 subjects, 12 arms. |

## Final full-run verification

At 2026-08-14T08:42:50Z UTC, Polaris merge job `47348` completed in 12 seconds
with exit code `0`. The output is:
`/Users/924254653/adaptive_roi_gate6_1_runs_20260814/full/merged-rerun`.

The output contains `STARTED.json`, `report.json`, `artifacts.sha256`, and the
final `COMPLETE.json`. The report schema is `gate6.1-crossover-v1` and contains
6396 rows, which is 533 clips times 12 fixed-action arms. The row data contains
89 unique subjects and 533 unique clips. The report's `subjects` field contains
1068 subject-arm summaries, which is expected: 89 subjects times 12 arms, not
1068 unique people. The plan ID is
`gate6-ruler-difference-a27fc3de702d73e80fff6235c30f44f65ed61ca381b3dc13f283c499c16058f2`.

This is a verified Gate 6.1 MCD full-run result. It is a fixed-sequence
crossover diagnostic, not a trained-policy result, and does not change the
MMPD evaluation-only boundary.

The repaired-version Polaris diagnostic is accepted as job `47350`, as recorded
above: it completed in `12:50` with exit `0`, produced all 16 shard completion
markers and the merged `COMPLETE.json`, and published 6,396 rows for 533 clips,
89 subjects, and 12 fixed-action arms. This remains a fixed-sequence crossover
diagnostic, not a trained-policy result, and does not change the accepted Gate
6.1 numbers or the MMPD evaluation-only boundary.
