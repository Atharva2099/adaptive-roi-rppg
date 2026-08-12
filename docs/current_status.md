# Current status — 2026-08-11

## Gate tracker

| Gate | Status |
|---|---|
| 1. Contracts and validators | Complete |
| 2. MCD manifest/split adapter | Complete: SOL-reviewed and accepted on full MCD inventory |
| 3. Causal signal builder with synthetic numeric tests | Complete: SOL-reviewed and accepted on Polaris job `46896` |
| 4. Belief tracker and action legality | Complete: SOL-reviewed and accepted on Polaris job `46927` |
| 5. Frozen evaluator and row-level outputs | Blocked by gate order |
| 6. Historical parity replay on a small MCD fixture, then full MCD parity | Blocked by gate order |
| 7. MCD-only training adapters | Blocked by gate order |
| 8. MMPD frozen-evaluation adapter last | Blocked by gate order |

Gate 2 passed independent SOL review and full-data acceptance on Polaris job `46838`. Gate 3 passed independent SOL review after local and Linux tests plus a real, train-only MCD structural smoke on Polaris job `46896`. Gate 4 passed independent SOL review and a real, train-only MCD structural and failure-path smoke on Polaris job `46927`. No training, label reading, or MMPD access is authorized.

## What changed

Gate 3 adds a concise NumPy/SciPy implementation of classical POS, fixed trailing-window measurement rules, and a state-only MCD reader. The reader authenticates the state CSV selected by the Gate 2 manifest and exposes no GT path or API. The exact numerical choices, command, evidence hashes, and limitations are kept in [the Gate 3 note](gates/gate_03_causal_pos.md), separate from this overview.

## Verified checks

Gate 3 was accepted with 45 local and Linux/Polaris tests. After adding Gate 4, the current local suite has 55 passing tests. The accepted Gate 3 Linux environment used Python 3.13.12, NumPy 2.2.0, and SciPy 1.17.1. Gate 3 checks include a synthetic 1.2 Hz known-answer signal at 24 and 30 FPS, malformed and degenerate inputs, exact hop boundaries, clip reset, fail-closed missing RGB, deterministic output, and authenticated state-file reads.

## Acceptance result

Polaris job `46896` read the first train subject selected by the authenticated split: subject 1020, six clips covering all three cameras and both conditions. Two independent passes produced the same global stream hash. Both emitted exactly 1,031 expected hops and 12,372 valid ROI measurements. This is structural and deterministic evidence only, not an HR-accuracy result. A separate train-wide MCD diagnostic found lower no-fill availability for USBVideo, especially `cheek_upper_right`; it does not authorize imputation. Full values, hashes, and limitations are E-021 through E-023 in the evidence registry.

## Gate 4 acceptance

Gate 4 adds immutable belief/control records, the literal predict-then-update 2x2 NumPy step, the 101-value observation, minimum-hold action resolution, and a state-only Slurm smoke. It does not verify HR accuracy, full-face/Oracle numbers, rewards, learned policy quality, or MMPD transfer. See [the Gate 4 note](gates/gate_04_control_core.md).

Polaris job `46927` completed `0:0` in `11:51` on `lmn01`. The 55-test Linux suite passed. Seven authenticated MCD train clips, two action arms, and two matching passes per arm exercised 2,404 recorded controller hops. The accepted rows include 600 minimum-hold overrides and 74 selected-invalid transitions checked to use predict-only belief propagation. Exact artifacts, hashes, cohort, and limitations are E-024.

## Next action

Plan Gate 5, the frozen evaluator and row-level outputs. Do not begin training until Gates 1–6 pass, and do not access MMPD before its frozen evaluation adapter at Gate 8.

## Project references

- [System design](system_design.md)
- [Data contract](data_contract.md)
- [Gate 2 adapter](gates/gate_02_mcd_manifest_adapter.md)
- [Gate 3 causal POS](gates/gate_03_causal_pos.md)
- [Gate 4 control core](gates/gate_04_control_core.md)
- [Evidence registry](evidence_registry.md)
