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

All nine frozen models scored below the new full-face baseline. Family means are averages across independent seeds, not ensembles.

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
- MMPD remained evaluation-only and was not accessed. Checkpoint training-config provenance remains descriptive/unverified; file hash, architecture, and replay identity were verified.
