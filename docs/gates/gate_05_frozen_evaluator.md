# Gate 5: frozen evaluator and row-level outputs

Working-tree status: this Gate 5 note is intentionally untracked pending review; this documentation cycle does not stage or commit it.

## Purpose and status

Gate 5 freezes the train-only MCD evaluation ruler, authenticates the Gate 2 `gt_ppg` source, recomputes the Gate 3/4 causal POS and action-0 control path from source roots, and publishes immutable row-level outputs. **Status: accepted.** Polaris job `47300` (`adaptive-roi-gate5`) completed with `0:0` in `12:12` on `lmn01` using 1 CPU. The final snapshot SHA-256 is `5330ae9c25b7ff88468fe2f2cefae3fc2b2041753ea448aabcea5bf49e470de4`; the wrapper SHA-256 is `1cd6b6b751d92de475bc86dfee3bc3343df7e96ac904993fb4f65df5f1176359`. The accepted output is `/Users/924254653/adaptive_roi_gate5_final_20260813.5330ae9/gate5_output`, bound to plan `frozen-plan-023260d9b7cb6c75b07f9e9856ecb85d425582c50c19b2b2da016b49100b9338`. This is structural and reproducibility evidence only; it does not claim parity, uncertainty, controller quality, training value, transfer, or generalization.

## Frozen GT rule and plan

The label ID is `causal_periodogram_peak_no_subharmonic_v3`. The adapter accepts only a complete authenticated Gate 2 bundle, one direct non-symlink `gt/<clip>_ground_truth.csv`, the train split, the exact `frame_idx,gt_ppg` header, canonical frame indices, finite values, the manifest row count, and the manifest SHA-256. For every Gate 3 hop, it uses `win=round(8*fps)`, `hop=round(1*fps)`, `end_exclusive=win+h*hop`, and `hop_time=end_exclusive/fps`; it centers the unpadded segment at `end_exclusive` with `half=round(4*fps)`. It requires eight finite samples, mean-centers, applies the order-3 0.5–3.0 Hz Butterworth `lfilter` with zero initial state, then a Hann/scipy periodogram with `nfft=max(4096,nextpow2(n))`, inclusive band selection, first ascending tie breaking, and no subharmonic correction. It does not use the separate historical `filtfilt` utility.

The plan binds the dataset and train split IDs/hashes, ordered clip cohort and hop counts, method/config/observation/GT IDs, and strict post-update-belief MAE. Clip MAE is the arithmetic mean over sealed hops; run MAE is the arithmetic mean of clip MAEs; subject summaries are arithmetic means of that subject's clip MAEs. Bootstrap is deferred to Gate 6. The smoke selects the first two sorted train subjects, six camera-condition clips each, and no other split.

## Outputs and usage

`read_mcd_labels(bundle, gt_root, clip_id, required_split="train")` returns one `LabelFrame` per expected hop. `build_train_full_face_plan(manifest_tree)` reloads the complete Gate 2 tree and deterministically selects the first two sorted train subjects, requiring exactly six camera-condition clips per subject. `evaluate(manifest_tree, state_root, gt_root, provenance, plan=None)` reloads that tree, reads authenticated canonical frames and GT, constructs POS measurements, and replays `control_step` from the initial state with action 0. It accepts no caller-supplied numerical frames, clips, or transitions; an optional plan must exactly equal the fresh canonical plan. `evaluate_and_publish(...)` creates the destination exclusively, writes `STARTED.json` before evaluation, validates six substantive files, and writes `COMPLETE.json` last. `publish_evaluation(result, destination)` uses the same protocol for precomputed results. No directory rename, relocation, overwrite, or final-directory deletion is used. Existing destinations are refused untouched.

The six substantive files are `per_hop.csv`, `per_clip.csv`, `subject_summary.csv`, `run_summary.json`, `run_manifest.json`, and `artifacts.sha256`. A successful directory contains those six plus `STARTED.json` and `COMPLETE.json`. A caught failure contains `STARTED.json` and `FAILED.json`, with possible partial outputs; started-only directories are interrupted and never accepted. Marker files use schema `gate5-publication-state-v1`, are written exclusively, contain no timestamps, and are never replaced. Readers accept only the exact successful set, direct regular files, valid marker bindings, valid rows, and matching hashes. Structural verification checks those markers and all row/summary semantics; authoritative verification privately recomputes and compares only the six substantive files.

The real-data smoke is Slurm-only. The accepted run covered 12 MCD train clips, 2 subjects, and 2,059 hops. Equal-clip mean MAE was `6.494966476398912`; there were 37 invalid selected measurements and zero overrides. The eight direct files were `per_hop.csv`, `per_clip.csv`, `subject_summary.csv`, `run_summary.json`, `run_manifest.json`, `artifacts.sha256`, `STARTED.json`, and `COMPLETE.json`; all were present and verified. The run passed 72 Linux tests in 39.970 seconds. SHA-256 evidence-member checks and structural verification passed; repeat substantive outputs matched; source-bound recomputation passed; a tampered source produced `STARTED.json` plus `FAILED.json` with no `COMPLETE.json` or substantive outputs; and an existing destination was preserved.

The Gate 5 smoke runner was retired from current HEAD after gate acceptance.
To reproduce the accepted run, check out (or create a worktree at) historical
commit `feecad9`, then run this Slurm-only command there:

```text
PYTHONPATH=src python3 scripts/verify_gate5_mcd_smoke.py --manifest-tree ... --state-root ... --gt-root ... --output-dir ... --code-snapshot-sha256 <64 lowercase hex>
```

It freezes a deterministic full-face action-0 plan, uses `evaluate_and_publish` for two repeat directories and the final directory, compares only the six substantive files, and independently recomputes row counts and equal-clip/subject/run summaries. It then copies only the 12 selected state files into a temporary direct-file root, mutates one named selected source, and calls `evaluate_and_publish` on a negative directory; the result must be `STARTED.json` plus `FAILED.json`, with no `COMPLETE.json`. It also checks marker validation, source-bound recomputation, exclusive destinations, and existing-destination preservation. No accuracy threshold or historical comparison is used. The local fixture checks do not test the target NFS mount.

## Diagnostic protocol

**Question:** Do authenticated MCD train sources reproduce the Gate 3/4 full-face action-0 path and publish reproducibly under the frozen ruler?

**Required arm:** one deterministic full-face arm, action 0, seed and checkpoint null.

**Negative control:** mutate one byte in a copied state source for the named first plan binding; evaluation must fail with a negative destination containing `STARTED.json` and `FAILED.json`, but no `COMPLETE.json` or substantive files. The original source hash must remain unchanged, and no matching publisher staging directory may exist. A second publication attempt against an existing destination must preserve the foreign destination and leave no staging artifact.

The GT known-answer smoke uses a 1.2 Hz sinusoid and records exactly 72.0703125 BPM at both 24 FPS and 30 FPS, including the final clipped edge. This locks the current Butterworth/lfilter and Hann-periodogram ruler, including its 4096-point FFT grid and ascending first-maximum tie behavior. It does not independently prove equivalence to any external toolbox implementation.

**Interpretation limits:** structural verification checks only the six-file publication and its internal hashes, joins, and summaries. Source-bound verification additionally reloads the roots, recomputes the evaluator, and byte-compares all six files. The manifest tree is loaded more than once during plan/evaluation validation, so a same-user manifest replacement between those loads remains an out-of-scope race; this repair does not add a large snapshot mechanism. This is structural and reproducibility evidence only, not an accuracy, parity, controller-quality, training, uncertainty, or transfer result.

**Stop/next rule:** stop on any authentication, key, time, configuration, summary, hash, or publication failure. Gate 5 is accepted. Proceed to Gate 6 historical parity replay on the predeclared MCD fixture, then full-MCD parity only if its decision rule passes. No training is allowed until Gate 6 passes.
