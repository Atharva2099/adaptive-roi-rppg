# Current status — 2026-08-13

## Gate tracker

| Gate | Status |
|---|---|
| 1. Contracts and validators | Complete |
| 2. MCD manifest/split adapter | Complete: SOL-reviewed and accepted on full MCD inventory |
| 3. Causal signal builder with synthetic numeric tests | Complete: repaired, SOL-reviewed, and reaccepted on Polaris job `47260` |
| 4. Belief tracker and action legality | Complete: core unchanged and real-data acceptance restored on Polaris job `47260` |
| 5. Frozen evaluator and row-level outputs | Complete: accepted on Polaris job `47300` with reproducible immutable outputs |
| 6. Historical parity replay on a small MCD fixture, then full MCD parity | Next: approved MCD-only parity work |
| 7. MCD-only training adapters | Blocked by gate order |
| 8. MMPD frozen-evaluation adapter last | Blocked by gate order |

Gate 2 passed independent SOL review and full-data acceptance on Polaris job `46838`. Gate 3's state reader was repaired so one immutable byte capture is both hashed and parsed, then independently SOL-reviewed and reaccepted on Polaris job `47260`. Gate 4 core was unchanged; job `47260` restored its real-data acceptance through the repaired reader. Gate 5 is now accepted on job `47300`. No training or MMPD access is authorized.

## What changed

Gate 3 adds a concise NumPy/SciPy implementation of classical POS, fixed trailing-window measurement rules, and a state-only MCD reader. The reader authenticates the state CSV selected by the Gate 2 manifest and exposes no GT path or API. The exact numerical choices, command, evidence hashes, and limitations are kept in [the Gate 3 note](gates/gate_03_causal_pos.md), separate from this overview.

## Verified checks

The accepted Gate 5 run passed 72 Linux tests in 39.970 seconds. Job `47300` completed with `0:0` in `12:12` on `lmn01` using 1 CPU. Its MCD-train-only smoke covered 12 clips, 2 subjects, and 2,059 hops; equal-clip mean MAE was `6.494966476398912`, with 37 invalid selected measurements and zero overrides. Structural verification, SHA-256 member checks, repeat substantive-output comparison, source-bound recomputation, tampered-source failure, and existing-destination preservation all passed. Jobs `47272` and `47273` remain superseded publication attempts; probe `47274` remains a negative publication probe.

## Acceptance result

Polaris job `47260` completed `0:0` in `14:34` on `lmn01`. Gate 3 again emitted exactly 1,031 expected hops and 12,372 valid ROI measurements for subject 1020's six camera-condition clips. Its CSV and global stream hashes are byte-identical to historical E-023. This is structural, authentication, and deterministic evidence only. Exact new artifacts and hashes are E-025.

## Gate 4 acceptance

Gate 4 adds immutable belief/control records, the literal predict-then-update 2x2 NumPy step, the 101-value observation, minimum-hold action resolution, and a state-only Slurm smoke. It does not verify HR accuracy, full-face/Oracle numbers, rewards, learned policy quality, or MMPD transfer. See [the Gate 4 note](gates/gate_04_control_core.md).

The Gate 4 core was unchanged. Job `47260` reran its seven train clips and two action arms through the repaired reader, reproducing 2,404 hops, 600 minimum-hold overrides, 74 selected-invalid transitions, and a CSV byte-identical to E-024. E-025 restores its real-data acceptance.

## Next action

Proceed to Gate 6: historical parity replay on a small MCD fixture, followed by the predeclared full-MCD parity work if the Gate 6 decision rule passes. Training remains prohibited until Gate 6 passes, and MMPD remains evaluation-only and unavailable for training or development decisions.

## Project references

- [System design](system_design.md)
- [Data contract](data_contract.md)
- [Gate 2 adapter](gates/gate_02_mcd_manifest_adapter.md)
- [Gate 3 causal POS](gates/gate_03_causal_pos.md)
- [Gate 4 control core](gates/gate_04_control_core.md)
- [Gate 5 frozen evaluator](gates/gate_05_frozen_evaluator.md)
- [Evidence registry](evidence_registry.md)
