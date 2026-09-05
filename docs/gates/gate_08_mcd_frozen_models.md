# Gate 8: frozen MCD model evaluation

Status: `ACCEPTED` for the frozen MCD evaluation only. No retraining was run, no MMPD data was accessed, and no transfer or deployment claim follows from this gate.

## Frozen protocol

- MCD evaluation split: 540 original clips, exactly 7 excluded invalid clips, 533 clips and 89 subjects retained.
- Each method has 91,227 valid hops. Metric: equal-clip mean absolute HR error in BPM.
- Arms: full-face action 0 plus nine existing RecurrentPPO checkpoints: three DAgger, three Standard PPO, and three Advantage PPO.
- Bootstrap: 10,000 subject resamples; the report records separate equal-clip and equal-subject estimates and fixed comparison seeds.
- Runtime: committed active repository snapshot, external checkpoint package, Polaris CPU Slurm job. The legacy repository supplied the read-only historical comparison only.

## Final result: 2026-08-15 UTC

Polaris job `47417` completed with exit `0` in `05:44:25` on `lmn01`. The run used 16 subject-disjoint shards, independently replayed each shard and the full source set during merge, rebuilt the CSV aggregates, and wrote `COMPLETE.json` last.

| Method | Equal-clip MAE (BPM) |
|---|---:|
| Full face | 12.6317 |
| DAgger seed 0 / 1 / 2 | 7.8607 / 8.4257 / 9.1284 |
| DAgger seed mean | 8.4716 |
| Standard PPO seed 0 / 1 / 2 | 7.8946 / 7.9689 / 8.1105 |
| Standard PPO seed mean | 7.9913 |
| Advantage PPO seed 0 / 1 / 2 | 7.6375 / 7.5007 / 7.7315 |
| Advantage PPO seed mean | 7.6232 |

All nine frozen models scored below the new full-face baseline. Each family
value is the arithmetic mean over its independent seed checkpoints, not an
ensemble score and not a deployment ensemble. Checkpoint identity and replay
were verified from the sealed evaluation artifacts; the original training
lineage remains descriptive and unverified.

## Matched held-out Oracle headroom: 2026-09-03

Polaris job `48872` evaluated Oracle B and Oracle C on the same retained MCD
evaluation cohort: 533 clips, 89 subjects, and 91,227 hops per Oracle. It used
four subject-disjoint workers and a source-authoritative merge. The job
completed `0:0` in `01:23:59` on `lmn02`. Lower MAE is better.

| Offline diagnostic | Equal-clip MAE (BPM) | 95% subject-block CI (BPM) |
|---|---:|---:|
| Oracle B, greedy current-hop choice | 3.7328 | 3.3331 to 4.1656 |
| Oracle C, width-8 sequence beam | 2.3724 | 2.0348 to 2.7429 |
| B minus C | 1.3604 | 1.2486 to 1.4826 |

The confidence intervals use 10,000 paired subject resamples over the 89
subjects, seed `48872`, and sorted zero-based indices 249 and 9749. Independent
job `48896` rebuilt both equal-clip means from every stored hop row and matched
the report. Job `48897` produced the subject-block intervals. All four shard
hashes and the merged report hash passed.

The accepted Advantage PPO seed mean from E-031 is 7.6232 BPM. Its arithmetic
gap is 3.8904 BPM above Oracle B and 5.2508 BPM above Oracle C. These are
matched descriptive differences, not jointly bootstrapped model-versus-Oracle
comparisons.

Oracle B uses current ground truth to select the legal action with the lowest
post-belief error at each hop. Oracle C searches action sequences using future
ground truth and keeps eight trajectories. Both are offline, GT-informed, and
non-deployable. Oracle C is also noncausal and approximate, so its number is a
headroom diagnostic rather than an achievable policy score.

```text
ROOT=/Users/924254653/adaptive_roi_mcd_eval_oracle_runs_20260903/full-48872/merged
report.json       144,439,998 bytes  sha256 0741a432b0ebf0f2d1f0facb98423e0951596fd4d34d3c5e140f27dc43783bbc
artifacts.sha256           78 bytes  sha256 7b2d3563df71b5b9f64a4c355f748ae989734bd7be99ef0c88fc8d056340f1cd
STARTED.json            1,564 bytes  sha256 a6722992b4cd3aa9fdaef9a09f697e82a00bae30a6762ab1f886313e95f3668f
COMPLETE.json           1,641 bytes  sha256 9be62e2e66eae91b2c35cda13ec6cf45ec9a4c266a8bbcbb26fc4548a3483c8b
```

The runner was isolated at commit `604780f`; the recorded execution snapshot
is `b60805d376007c37e55937ee76460357f5d3a06e1a20855612671c56a3b3c091`.
Job `48871` failed before evaluation because its scheduler log changed the
source-directory hash. It produced no scientific result and was superseded by
job `48872`. Evidence authority is E-033.

## Historical comparison

The legacy values use the legacy causal replay and the same 533-clip names/checkpoint identities where available. They are a reference, not a matched pipeline ablation.

| Method | Historical legacy | New Gate 8 | New minus historical |
|---|---:|---:|---:|
| Full face | 12.3865 | 12.6317 | +0.2452 |
| DAgger seed mean | 8.1157 | 8.4716 | +0.3559 |
| Advantage PPO seed mean | 7.3748 | 7.6232 | +0.2484 |

The difference is consistent with the documented pipeline changes: current POS reconstruction, invalid-hop handling, ground-truth handling, and tie-breaking are not identical to the legacy evaluator. See [the legacy map](../legacy_project_map.md) and evidence IDs E-001–E-004 and E-031.

## Reproducible artifacts

All paths below are on Polaris. The row-level files are the source for recomputation; `report.json` is the final summary.

```text
ROOT=/Users/924254653/adaptive_roi_gate8_runs_20260814_r2/full-47417/merged
per_hop.csv       912,270 rows  sha256 5aa9029b60fbc9d51eb572bb444b0668ec01ef0f297f6dd2c8bc33fd9d871678
per_clip.csv        5,330 rows  sha256 48ac949760678499152034ca7f6ae922cf738fd1b93a29b0b0c530ecb64b73eb
per_subject.csv       890 rows  sha256 a148cffc433c5274378863265bc18d671ee85488a1e1a4325c0571d69d9fc26f
report.json                    sha256 cf5b7da4de7d046b34a5c594ea3972a046ee19e2a185e38a14e1e461b13582da
run_manifest.json              sha256 ef27debbdde8e8fbdce842981f3fdabb74ac45e11190afd0e9628f318bf22a53
COMPLETE.json                  sha256 c09ec7055c130bad23e3b060fb6b81d6b0800ac3aa27357c13c6b58a3922a83c
```

Verification on Polaris:

```bash
sacct -X -j 47417 --format=JobID,State,Elapsed,ExitCode -n -P
find "$ROOT" -maxdepth 1 -name COMPLETE.json -print
sha256sum "$ROOT"/{per_hop.csv,per_clip.csv,per_subject.csv,report.json,run_manifest.json,COMPLETE.json}
```

The three preflight jobs are retained as reproducibility evidence: `47415` completed in `10:53`, repeat `47416` completed in `11:01`, and both wrote `COMPLETE.json` before the full job began.

## Repair log

| Date | Evidence | Issue | Resolution |
|---|---|---|---|
| 2026-08-14 | Sol reviews | Initial implementation had weak production publication, shard ownership, report rebuilding, checkpoint identity, and legacy-path coupling. | Terra rewired strict publication, deterministic ownership, source-authoritative replay, identity-preserving aggregates, and clean external checkpoint locators. |
| 2026-08-14 | Local tests | Final pre-run implementation passed 137/137 tests and Sol gave GO. | Committed as `091ba7a`. |
| 2026-08-15 | Polaris job `47412` | A real invalid POS measurement had blank `selected_confidence` and `selected_ppr`; CSV reread treated them as required floats. Job failed after `11:01`, wrote `FAILED.json`, and produced no result. | Terra allowed those two fields to be null only when `selected_valid=False`; valid selections still require finite numeric values. Added regression tests. |
| 2026-08-15 | Sol review and jobs `47415`–`47417` | Repair needed independent review and a repeat before full evaluation. | Sol GO; local suite 139/139; repair committed as `a39de48`; both preflights and the full run passed. |

The failed `47412` directory remains preserved at `/Users/924254653/adaptive_roi_gate8_runs_20260814/smoke-47412/`. It is not evidence for a model score.

## Code and boundary

- Implementation: `scripts/verify_gate8_mcd_frozen_models.py`, `src/adaptive_roi_rppg/evaluation/model_replay.py`, and `src/adaptive_roi_rppg/evaluation/model_publication.py`.
- Launcher: `slurm/gate8_mcd_frozen_models_full.slurm`.
- Frozen plan: `configs/evaluation/mcd_frozen_models_v1.json`.
- Final code commit: `a39de48`.
- MMPD remained evaluation-only and was not accessed. Checkpoint training-config provenance and original training lineage remain descriptive and unverified; file hash, architecture, checkpoint identity, and replay identity were verified.
