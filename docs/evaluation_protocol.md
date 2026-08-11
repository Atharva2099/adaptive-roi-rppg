# Evaluation protocol

## Frozen development boundary

MCD is the only development and selection domain. MMPD is a one-way external evaluation of already frozen MCD checkpoints. MMPD can support a transfer claim or failure analysis, but cannot choose, create, or prioritize what to train or test next. MCD questions and selection rules are established independently of MMPD observations.

## Method identity

Every learned method reports backbone, learning algorithm, initialization, ensemble/gate, and online/offline status.

- `DAgger`: LSTM; interactive imitation with initial Oracle-C CE clone plus two greedy-B relabel/refit rounds; from scratch before imitation; one model per seed, three seeds reported independently and averaged only for research reporting; online interactive MCD collection followed by offline supervised refitting.
- `Standard PPO`: LSTM; online on-policy PPO with absolute-error reward; warm-start from matching DAgger; one model per seed, three seeds independently; online MCD training.
- `Advantage PPO`: LSTM; online on-policy PPO with advantage-over-matching-DAgger-base reward; warm-start from matching DAgger; one model per seed, three seeds independently; online MCD training.
- `Oracle-C`: no learned backbone; GT-informed offline beam teacher/ruler; no initialization; one deterministic sequence; offline and not deployable.
- `Full-face POS`: no learned backbone; fixed action 0 using the same causal estimator/belief ruler; no initialization or ensemble; offline frozen replay.
- `TS-CAN PURE`: temporal-shift convolutional network; supervised waveform regression; one named frozen public PURE checkpoint whose original initialization is unknown to this package; one checkpoint, no ensemble or fallback; trained offline outside this project and replayed offline with no MMPD fine-tuning.
- `TS-CAN UBFC`: temporal-shift convolutional network; supervised waveform regression; one named frozen public UBFC checkpoint whose original initialization is unknown to this package; one checkpoint, no ensemble or fallback; trained offline outside this project and replayed offline with no MMPD fine-tuning.

Official POS and TS-CAN are context arms, not part of the primary causal-controller rank claim. TS-CAN raw values use its own HR estimates. TS-CAN belief-filtered values use full-face POS confidence as an explicit proxy, not calibrated TS-CAN uncertainty.

## MCD primary ruler

Freeze schema-v3 no-subharmonic test90: 533 clips, 89 represented subjects, trailing 8-second windows, 1-second hops, post-update belief-mean absolute HR error, equal clip weighting, and the same substrate and signal/belief configuration for Full-face, Oracle-C, DAgger, Standard PPO, and Advantage PPO.

Per-clip metric:

`mae_clip = mean_hop(abs(predicted_belief_hr_bpm - gt_hr_bpm))` over the sealed scored-hop set.

The dataset metric is `mean_clip_mae = mean_clip(mae_clip)`, not hop-weighted. Historical parity uses exact authenticated substrate hop keys. Missing/non-finite GT or prediction on any expected key fails that method run, not silently drops it. A new plan must seal expected-hop construction, GT-validity rule, minimum scored hops, and clip-failure policy before predictions are inspected. Paired claims merge exact `(dataset_id, clip_id, hop_idx)` keys and use only a predeclared common cohort.

For three seeds, report every seed, arithmetic seed mean, range, and paired subject-block uncertainty. Seeds are not a deployment ensemble. Every result row carries `seed` and `checkpoint_sha256`; summaries never infer seed identity from filenames.

Exact E-003 reproduction: inner-join the six base/PPO CSVs on the same 533 stems; average the three base values and three PPO values within each clip; define `delta = base_pooled - ppo_pooled`. Resample 89 subject IDs with replacement 10,000 times using NumPy seed `20260709`, carrying all clips and repeated copies. Each replicate is the mean over the concatenated sampled clip deltas, so subjects with more clips retain more weight. Report percentile 2.5/97.5 bounds. This reproduces E-003 only; future runs seal their own seed and replicate count.

## PPO comparisons and diagnostics

Required matched arms are Full-face POS, Oracle-C, matching DAgger initialization before PPO, Standard PPO, Advantage PPO, and a simple action control where relevant. Match cache, GT rule, belief configuration, min-hold rule, checkpoint initialization, seed, rollout budget, evaluation code, and clip cohort.

Every diagnostic states: Question, Required arms, Negative control, Interpretation limits, and Stop/next rule. Do not launch a PPO sweep until data-contract, causal-signal, and parity-evaluator gates pass.

Report median clip MAE and RMSE if predeclared; switch count; proposed versus executed switches; illegal override count/rate; action frequency; first-action distribution; run length; consecutive ROI index distance only as a descriptive index statistic; valid-measurement and predict-only rates; and reward/PPO diagnostics during training. Training diagnostics never replace held-out MAE.

Use paired subject-block bootstrap intervals because clips from one subject are correlated. Predeclare seed, replicate count, and whether the statistic weights subjects or clips. Report subject count, clip count, missingness, point estimate, interval construction, 95% interval, and contrast direction. Subgroups are diagnostic unless predeclared and adequately represented; do not select model changes from MMPD subgroup performance.

## MMPD frozen-transfer protocol

Freeze 15 subjects: 1, 2, 5, 7, 10, 11, 12, 15, 16, 17, 20, 21, 25, 28, 29; 20 clips per subject, 300 clips total. No MMPD fine-tuning, calibration, reward/feature/hyperparameter/checkpoint selection, curriculum, or early stopping.

Keep these conditions separate:

1. **MMPD causal-controller ruler:** Full-face POS, Oracle-C, DAgger, Standard PPO, and Advantage PPO; 8-second windows, 1-second hops, StreamingBeliefTracker, method-appropriate POS confidence, and MMPD-specific corrected GT with subharmonic correction enabled. This is the only primary controller-rank condition.
2. **MMPD matched-grid TS-CAN context:** PURE and UBFC HR predictions aligned to the same grid and corrected GT. Report raw MAE separately. Belief-filtered TS-CAN is context only because it uses full-face POS confidence as a proxy.
3. **MMPD official-POS ruler:** pinned official `POS_WANG` and official FFT metric through the custom wrapper, static first-frame Haar crop enlarged 1.5x and resized 72x72, with no causal belief tracker/smoothing. Record toolbox commit and wrapper provenance from the evidence registry.

Never subtract these rulers and call the difference `effect of smoothing`. For the corrected package, the only permitted description is: **unchanged frozen-controller predictions rescored under the MMPD-specific corrected-GT rule**.

## Failure classification and outputs

Thresholds are `TBD` until chosen on MCD or physical/statistical grounds, never from MMPD outcomes. Categories may overlap:

- Pipeline failure: official POS is good while causal full-face is bad.
- No ROI headroom: Oracle-C is close to causal full-face.
- Controller failure: Oracle-C is materially better but the learned controller fails to recover headroom.
- Transfer failure: a controller with verified MCD benefit loses that benefit on MMPD under the matched causal ruler.

Publication claims require method identity, checkpoint hash, cohort, ruler, GT rule, row count, aggregation, uncertainty, exclusions, and limitations. Stop and fix rather than train for split leakage, GT in observation, future-frame dependency, ruler mismatch, missing comparison arm, non-reproducible checkpoint, incomplete manifest, or summary/CSV disagreement.

Required outputs are immutable per-hop CSV, per-clip CSV, subject summary, run summary, and run manifest. Per-hop rows include run/method/dataset/clip/subject/hop/time, seed, checkpoint SHA-256, frozen-plan ID, GT, selected-ROI HR, pre/post belief, proposed/executed action, legality/override, validity/reason, reward components if training replay, and config/provenance IDs. Per-clip rows include run/method/dataset/clip/subject/view/condition/skin when allowed, seed, checkpoint SHA-256, frozen-plan ID, `mae_clip`, expected/scored/invalid hops, proposed/executed switches, override count, and action counts. Summaries derive from row-level CSVs; prose is not the source of truth.
