# Gate 3: causal POS implementation (complete)

Gate 3 implements the corrected Wang POS primitive and a frozen, window-causal measurement profile. After a same-inode content-substitution issue was found, the state reader was repaired to capture one immutable byte stream from one no-follow descriptor and use those exact bytes for both SHA-256 verification and CSV parsing. SOL independently reviewed the repair, and Polaris job `47260` reaccepted it without changing stable numerical output. The reader has no GT root or GT API.

The profile uses NumPy and SciPy: Wang POS uses a `ceil(1.6 * fps)` rolling window, literal paper normalization `window / means`, the corrected combination `S0 + std(S0)/std(S1)*S1`, the projection `[[0,1,-1],[-2,1,1]]`, population standard deviation, and overlap-add without overlap-count division. Each one-second hop uses a trailing eight-second window. POS postprocessing is Butterworth BA order 1 over 0.75–3 Hz, and HR processing is Butterworth BA order 3 over 0.5–3 Hz. Both use `scipy.signal.lfilter` with an independent zero initial state for each ROI and hop. Periodograms use Hann, constant detrending, one-sided density, HR NFFT `max(4096,nextpow2(n))`, and natural-N PPR numerator. The frequency mask is inclusive and ties choose the first ascending bin. No filters or PSDs are reimplemented in tests.

The final legal Wang subwindow is included: for `n=L+1`, starts `0` and `1` both contribute. This corrects the historical omission of the final legal start. The implementation is window-causal, so each hop starts both IIR filters from zero; it is not a continuous IIR stream. Missing RGB is fail-closed with finite coverage and no imputation. A separate causal-fill refinement is pending and must not be inferred from this gate.

The 45 local and Linux checks cover the bare primitive, malformed and degenerate inputs, hop/source boundaries, clip identity and reset behavior, missing RGB, canonical pose serialization, configuration identity, and state-file authentication boundaries. The reader hash, row, token, split, and changed-read claims are backed by the consolidated tests. The `1e-8` threshold is fail-closed for channel means, projection denominator standard deviation, POS postfilter standard deviation, spectral filter standard deviation, and natural in-band total power; Gate 6 must quantify any discrepancy from legacy magnitude behavior.

## Accepted real-MCD smoke and reacceptance

Polaris job `47260` completed `0:0` in `14:34` on `lmn01` with 65 Linux tests. It reproduced the six-clip cohort, 1,031 hops, 12,372 valid ROI measurements, global stream hash `a8996552fd0b61e7338a15a255d8e2acdd3b0da606a853f995c74ed11056c367`, and CSV SHA-256 `bcbcf8b7c87e81daf79adf2d6f458ec5a3359f2ea2d1621a08a507ec3ab03c9b`. The CSV is byte-identical to the historical job below. E-025 is the current acceptance authority.

- Polaris job `46896` completed with exit `0:0` in 3 minutes on `lmn02`.
- Environment: Python 3.13.12, NumPy 2.2.0, SciPy 1.17.1.
- Code snapshot SHA-256: `3d2ab1dcaad32b3faaae4d27668e0855339da670c45a2a942c7f12bf4c565f1d`.
- POS configuration ID: `pos-v1-11ad093b3a2f7be3471d2a10da72ddd9ffd2f7985bcee0646be13e1a1b72d436`.
- Cohort: the first authenticated train subject, 1020; six clips, covering three cameras and before/after conditions.
- Expected and emitted: 1,031 hops. Valid measurements: 12,372, exactly 12 ROIs per hop. Invalid counts were zero in this small cohort.
- Two complete passes produced the same stream SHA-256: `a8996552fd0b61e7338a15a255d8e2acdd3b0da606a853f995c74ed11056c367`.
- CSV SHA-256: `bcbcf8b7c87e81daf79adf2d6f458ec5a3359f2ea2d1621a08a507ec3ab03c9b`. JSON SHA-256: `dbea1cea716d94751ecfba1c37c0999fd8b8f2aee061225275363e7c25e97f91`.
- Full remote path, manifest hashes, logs, wrapper hashes, and limitations are recorded in E-023 of the evidence registry.

A separate train-wide missingness diagnostic, E-021, found that no-fill measurement availability is lower for USBVideo and especially `cheek_upper_right`. This is a recorded limitation, not approval to choose a fill rule.

Gate 3 establishes implementation conformance, synthetic known-answer behavior, authenticated train-only data access, exact timing, and deterministic structure. It does not establish real-MCD HR accuracy, historical parity, controller quality, transfer, or generalization. The six smoke clips had valid measurements at every hop and therefore do not exercise real missing-data failure behavior. No historical code was copied or imported, and no historical repository is a dependency.

The Gate 3 smoke runner was retired from current HEAD after gate acceptance.
The accepted run remains reproducible by checking out (or creating a worktree
at) historical commit `feecad9`, then running the command there:

```text
PYTHONPATH=src python3 scripts/verify_gate3_mcd_smoke.py --manifest-tree PATH --state-root PATH --output-dir PATH --code-snapshot-sha256 SHA256
```

This command executes under Slurm with `SLURM_JOB_ID` and `SLURMD_NODENAME` set. The accepted wrapper supplied the exact environment and execution context recorded in E-023.
