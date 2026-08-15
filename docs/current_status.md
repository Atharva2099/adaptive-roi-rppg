# Current status — 2026-08-14

## Gate tracker

| Gate | Status |
|---|---|
| 1. Contracts and validators | Complete |
| 2. MCD manifest/split adapter | Complete: SOL-reviewed and accepted on full MCD inventory |
| 3. Causal signal builder with synthetic numeric tests | Complete: repaired, SOL-reviewed, and reaccepted on Polaris job `47260` |
| 4. Belief tracker and action legality | Complete: core unchanged and real-data acceptance restored on Polaris job `47260` |
| 5. Frozen evaluator and row-level outputs | Complete: accepted on Polaris job `47300` with reproducible immutable outputs |
| 6. Historical parity replay on a small MCD fixture, then full MCD parity | Complete: SOL-reviewed and accepted on Polaris job `47313` |
| 7. MCD-only training adapters | Complete for the frozen MCD Oracle B/C teacher diagnostic; no deployable model result is claimed |
| 8. Frozen MCD evaluation of existing RecurrentPPO checkpoints | INCOMPLETE / NO-GO: not accepted; no model result and no Polaris fixture/full evaluation allowed |

Gate 2 passed independent SOL review and full-data acceptance on Polaris job `46838`. Gate 3's state reader was repaired so one immutable byte capture is both hashed and parsed, then independently SOL-reviewed and reaccepted on Polaris job `47260`. Gate 4 core was unchanged; job `47260` restored its real-data acceptance through the repaired reader. Gate 5 is now accepted on job `47300`. No training or MMPD access is authorized.

The accepted MCD manifest inventory is fixed at 3,060 train clips from 510 subjects and 540 evaluation clips from 90 subjects, with no train/evaluation subject overlap. For Gate 7, the eligible teacher cohort is 3,057 train clips from the same 510 subjects after excluding exactly `7412_FullHDwebcam_after`, `7412_IriunWebcam_after`, and `7412_USBVideo_after`; all 171 labels in each excluded clip are invalid with `degenerate_variation`. Any other invalid label fails closed.

## What changed

Gate 3 adds a concise NumPy/SciPy implementation of classical POS, fixed trailing-window measurement rules, and a state-only MCD reader. The reader authenticates the state CSV selected by the Gate 2 manifest and exposes no GT path or API. The exact numerical choices, command, evidence hashes, and limitations are kept in [the Gate 3 note](gates/gate_03_causal_pos.md), separate from this overview.

## Verified checks

The accepted Gate 5 run passed 72 Linux tests in 39.970 seconds. Job `47300` completed with `0:0` in `12:12` on `lmn01` using 1 CPU. Its MCD-train-only smoke covered 12 clips, 2 subjects, and 2,059 hops; equal-clip mean MAE was `6.494966476398912`, with 37 invalid selected measurements and zero overrides. Structural verification, SHA-256 member checks, repeat substantive-output comparison, source-bound recomputation, tampered-source failure, and existing-destination preservation all passed. Jobs `47272` and `47273` remain superseded publication attempts; probe `47274` remains a negative publication probe.

## Acceptance result

Polaris job `47260` completed `0:0` in `14:34` on `lmn01`. Gate 3 again emitted exactly 1,031 expected hops and 12,372 valid ROI measurements for subject 1020's six camera-condition clips. Its CSV and global stream hashes are byte-identical to historical E-023. This is structural, authentication, and deterministic evidence only. Exact new artifacts and hashes are E-025.

## Gate 4 acceptance

Gate 4 adds immutable belief/control records, the literal predict-then-update 2x2 NumPy step, the 101-value observation, minimum-hold action resolution, and a state-only Slurm smoke. It does not verify HR accuracy, full-face/Oracle numbers, rewards, learned policy quality, or MMPD transfer. See [the Gate 4 note](gates/gate_04_control_core.md).

The Gate 4 core was unchanged. Job `47260` reran its seven train clips and two action arms through the repaired reader, reproducing 2,404 hops, 600 minimum-hold overrides, 74 selected-invalid transitions, and a CSV byte-identical to E-024. E-025 restores its real-data acceptance.

## Gate 6 acceptance

The exact 12-clip fixture passed on Polaris job `47311` with 2,063 joined hops and zero unclassified rows. The full 533-clip/89-subject MCD replay passed on job `47313` in `2:04:28` on `lmn01`, with 91,227 joined hops and zero unclassified rows. Historical equal-clip MAE was `12.386529570502146` BPM and current equal-clip MAE was `12.631680651991282` BPM. This is observational ruler-difference evidence only. See [the Gate 6 note](gates/gate_06_historical_parity.md). Gate 7 MCD-only training adapter work may now begin under the existing MMPD boundary.

Gate 7 is accepted for the frozen MCD-train Oracle B/C teacher diagnostic on Polaris job `47411`, which completed with exit `0` in `01:35:35` on `lmn01`. The merged output is `/Users/924254653/adaptive_roi_gate7_oracle_runs_20260814_r2/full-47411/merged`; all 16 shards completed, `COMPLETE.json` exists, and no `FAILED.json` exists. The eligible cohort is 3,057 clips from 510 subjects and contains 523,610 hops. Oracle B (`greedy_b`) has clip-equal MAE `3.8833134521974753` BPM and 222,613 switches; Oracle C (`beam_c`) has clip-equal MAE `2.5482841213115375` BPM and 204,798 switches; `dCB = B-C = 1.3350293308859372` BPM. These are current MCD-train teacher diagnostics, not deployable model scores, and are not directly comparable to legacy test90 or cache values. Fixture job `47404` remains successful engineering evidence only. Jobs `47405` and `47406` failed before evaluation and remain historical launcher failures. Job `47407` remains `SUPERSEDED`/`INCOMPLETE`: shards 0–14 used the old v2/3,060-clip plan, shard 15 rejected the three 7412 clips, and no merge or final result exists. Those shards cannot be reused because the v3 plan payload changes.

Gate 8 history is recorded in [the Gate 8 note](gates/gate_08_mcd_frozen_models.md). Sol approved frozen MCD evaluation of nine existing RecurrentPPO checkpoints over 533 valid clips and 89 represented subjects, with no retraining and no MMPD. Terra's checkpoint-load smoke and later local repairs passed their reported local tests, but no Polaris evaluation ran. Sol reviews remained `NO-GO`: the final production runner still does not use strict publication helpers, strict ownership is not proven in production, final report and manifest rebuilds are missing, checkpoint identity can be erased by aggregation, production publish/merge paths are not directly tested, and the manifest references the legacy `RL for RoI` checkout. Gate 8 is not accepted and no model number or completion claim is made.

## Project references

- [System design](system_design.md)
- [Data contract](data_contract.md)
- [Gate 2 adapter](gates/gate_02_mcd_manifest_adapter.md)
- [Gate 3 causal POS](gates/gate_03_causal_pos.md)
- [Gate 4 control core](gates/gate_04_control_core.md)
- [Gate 5 frozen evaluator](gates/gate_05_frozen_evaluator.md)
- [Gate 6 historical parity](gates/gate_06_historical_parity.md)
- [Evidence registry](evidence_registry.md)
