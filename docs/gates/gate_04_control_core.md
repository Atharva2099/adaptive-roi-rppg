# Gate 4: shared causal control core

## Purpose and status

Gate 4 adds the small, framework-neutral causal control core used later by the full-face, Oracle, DAgger, and RecurrentPPO adapters. Its unchanged core passed independent SOL review; Polaris job `47260` restored real train-only MCD structural acceptance through the repaired Gate 3 reader. This gate does not claim HR accuracy, controller quality, rewards, learned-policy quality, full-face/Oracle results, or MMPD transfer.

## Frozen configuration

The immutable `CONTROL_CONFIG_PAYLOAD` binds `control-v1-<sha256>`, the exact 12-ROI order, the 101 field names, `dt=1`, `q=.1`, `r0=200`, initial state `(70, 0, diag(400,25))`, confidence threshold `.2`, minimum hold `2`, and all observation normalization/clipping rules. Prediction uses `F=[[1,dt],[0,1]]` and `Q=q[[dt^3/3,dt^2/2],[dt^2/2,dt]]`. It predicts first, then updates only for finite HR and finite positive confidence with `R=r0/max(confidence,1e-6)` and the literal covariance update `(I-KH)P`.

## Observation and action timing

The vector is 12 groups of seven values `(hr-90)/40`, `(hr-belief_mean)/40`, clipped confidence, clipped PPR, clipped coverage, agreement, and valid flag; followed by belief mean/std/velocity/confidence-age, a previous-action one-hot, and normalized hold count. Observations use the pre-action belief, previous executed action, and pre-action hold count. Invalid measurements use the current belief mean for both HR slots, zero confidence/PPR, preserve finite clipped coverage, and set agreement to zero and valid to false. Agreement excludes self and invalid measurements and counts inclusive differences up to 5 BPM.

The first action is free and sets hold to 1. Staying increments hold. A switch before hold 2 executes the previous action with `minimum_hold`; a legal switch resets hold to 1. `control_step` validates clip/hop identity, applies only the executed measurement, and advances the hop. State is explicitly recreated per clip.

## API and framework boundary

`BeliefState`, `ControlState`, `ControllerObservation`, `ActionDecision`, and `ControlTransition` are frozen, validated records. A transition carries the nonempty `MeasurementFrame.provenance_id` as `frame_provenance_id`, bound to the selected measurement, signal configuration, and control configuration. `belief_step`, `build_observation`, `resolve_action`, `control_step`, and `initial_control_state` are pure APIs. NumPy is used for the fixed 2x2 math and observations expose a fresh `float32[101]` array. The core imports no data loaders, labels, evaluation, training, legacy code, MMPD, or Gym/SB3 packages. No reward or label fields are present.

## Verification

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The Gate 4 smoke runner was retired from current HEAD after gate acceptance.
To reproduce the accepted run, check out (or create a worktree at) historical
commit `feecad9`, then run this Slurm-only command there:

```text
PYTHONPATH=src python3 scripts/verify_gate4_mcd_smoke.py --manifest-tree ... --state-root ... --output-dir ... --code-snapshot-sha256 <64 lowercase hex>
```

The smoke requires Slurm metadata, authenticates the Gate 2 bundle, reads only Gate 3 state/POS measurements, and records canonical transition hashes. It runs the first train subject's six clips twice for fixed action 0 and `hop_idx % 12`. Because those six clips had no selected invalid measurements, a bounded sorted search added `1024_USBVideo_after` as the seventh accepted clip. It does not open labels or calculate HR error.

Polaris job `46927` completed with exit `0:0` in `11:51` on compute node `lmn01`. All 55 tests passed under Python 3.13.12, NumPy 2.2.0, and SciPy 1.17.1. The accepted CSV has 14 rows: seven clips by two action arms. Each recorded arm represents two matching complete passes. It contains 2,404 controller hops across arm rows, 600 minimum-hold overrides, and 74 selected-invalid transitions; each invalid transition was checked against the exact predict-only belief step. CSV and JSON rows agree exactly.

After the Gate 3 reader repair, Polaris job `47260` reran 65 Linux tests and the same Gate 4 smoke on `lmn01`, completing `0:0` in `14:34` for the combined Gate 3/4 job. It reproduced all 14 rows, 2,404 hops, 600 overrides, 74 selected-invalid transitions, and CSV SHA-256 `f18f981d7cb8b8ba5c1fa8caf5597637765db65f7f26e58d055af9e0349e8a70`, byte-identical to job `46927`. E-025 is the current real-data acceptance authority.

The code snapshot, Slurm wrapper, scheduler capture, logs, CSV, JSON, and sidecar are covered by `final_evidence.sha256`. The snapshot's controller, Gate 4 test, and smoke files were also compared byte-for-byte with the independently reviewed working tree. Exact paths and hashes are in E-024 of the evidence registry.

This is structural, failure-path, and deterministic evidence only. It does not establish HR accuracy, historical numerical parity, action quality, full-face or Oracle values, deployable-policy quality, transfer, or generalization. Gate 5 is accepted separately; Gate 6 historical parity is next. No training is allowed until Gate 6 passes.
