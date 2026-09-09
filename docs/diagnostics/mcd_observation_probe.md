# MCD observation-ranking probe

## Contract

**Question:** Do PPO's inputs contain information that can rank the best
immediate ROI, and does that depend on how each candidate is encoded?

**Source:** job `49143`, using the matching shard observation and `per_hop.csv`
files. Subject identity is joined by unique `(clip_id, hop_idx)` keys from
`method_id=advantage_ppo_seed1` rows.

**Feature arms:**

- `full_101`: all 101 observation values plus a 12-column one-hot ROI identity
  (113 features). This preserves the original probe encoding.
- `local_shared`: the candidate ROI's seven-value block `obs_{7*k:7*k+7}`,
  shared/controller values `obs_084:obs_101`, and the same 12-column one-hot
  identity (36 features).

Validity is `obs_{7*k+6}`. Free hops are `hop_idx == 0` or `obs_100 >= 0.2`.
Locked hops and all-invalid hops are excluded and counted.

Both feature arms use identical `HistGradientBoostingRegressor` settings,
subject-grouped five-fold validation, targets, training-only complete-vector
shuffle controls, and subject-block bootstrap. The fixed model settings are
`max_iter=100`, `learning_rate=0.1`, `max_leaf_nodes=31`,
`l2_regularization=0.0`, and configured `random_state=8101`.

The arm names are `full_101_probe`, `full_101_shuffled`, `local_shared_probe`,
`local_shared_shuffled`, `majority`, `ppo_factual`, and `oracle`.

## Results

The combined full-cohort run joined 91,227 PPO hops from 89 subjects. It
excluded 32,000 locked hops and 3,729 free hops with no valid ROI, leaving
55,498 free hops from 530 clips for both encodings. Three of the cohort's 533
clips retained no free hop with a valid ROI and are absent from the analysed
set.

| Arm | Mean immediate regret (BPM) | Subject-block 95% CI | Top-1 | Top-3 | ROIs selected |
|---|---:|---:|---:|---:|---:|
| PPO factual action | 1.6399 | [1.4477, 1.8548] | 30.0% | 48.0% | 12 |
| Full-101 probe | 1.7420 | [1.5808, 1.9099] | 22.9% | 38.5% | 2 |
| Local-plus-shared probe | 1.7417 | [1.5727, 1.9223] | 10.6% | 26.9% | 12 |
| Full-101 shuffled control | 1.8955 | [1.7163, 2.0891] | 6.8% | 24.9% | 2 |
| Local-plus-shared shuffled control | 1.8706 | [1.7008, 2.0500] | 8.3% | 26.5% | 12 |
| Majority rule | 2.7501 | [2.4587, 3.0464] | 34.3% | 43.9% | 1 |
| Oracle | 0.0000 | [0.0000, 0.0000] | 100.0% | 100.0% | 12 |

The paired subject-block results were:

| Contrast | Difference (BPM) | Subject-block 95% CI | Result |
|---|---:|---:|---|
| Full-101 probe minus PPO | +0.1022 | [+0.0125, +0.1706] | PPO lower |
| Local-plus-shared probe minus PPO | +0.1018 | [+0.0474, +0.1508] | PPO lower |
| Full-101 probe minus shuffled | -0.1534 | [-0.2327, -0.0900] | Probe lower |
| Local-plus-shared probe minus shuffled | -0.1289 | [-0.1665, -0.0915] | Probe lower |
| Full-101 probe minus majority | -1.0080 | [-1.2806, -0.7316] | Probe lower |
| Local-plus-shared probe minus majority | -1.0084 | [-1.2812, -0.7264] | Probe lower |

The full-101 probe selected only ROI 0 and ROI 1. The local-plus-shared probe
selected every ROI at least 1,871 times. Changing the candidate-row encoding
therefore removed the action-collapse symptom, but it did not improve immediate
regret: the two point estimates differ by only 0.0004 BPM. The original ROI
0/1 concentration was an encoding effect and must not be interpreted as a
property of the underlying data.

Both encodings beat their matched shuffled controls and the majority rule, so
the 101-value observation contains usable immediate-ranking information under
this fixed model. Neither encoding beat PPO's factual choices. This result does
not support the claim that a fixed supervised probe exposes a substantially
better selector hidden inside the current inputs.

## Metrics and uncertainty

The primary metric is immediate regret in BPM: chosen valid counterfactual
error minus the minimum valid counterfactual error. Secondary metrics are
top-1, top-3, and action histograms. Point estimates aggregate hops to clip
means and then weight clips equally. The 10,000-draw subject-block bootstrap
uses seed `8101`; paired differences are computed within each draw. H1, H2,
and H3 hold only when the corresponding paired CI upper bound is strictly below
zero; otherwise the result is null.

## Interpretation limits

This is held-out immediate-ranking information, not a sequential controller
evaluation. Alternative actions are one-hop counterfactuals and are never
continued, so the output cannot be compared as an MAE against PPO's sequential
`7.623 BPM`. Free hops are conditioned on PPO's own switching and minimum-hold
behavior. MMPD is not used, and no tuning or feature selection is performed on
MMPD.

## Run settings

The authorized run uses the fixed JSON configuration: five subject-grouped
folds and 10,000 bootstrap draws. The single combined run completed on
2026-09-08 local time using Python 3.12.7, NumPy 2.5.3, and scikit-learn 1.9.0.

The retained result is
[`observation_probe_combined_20260908.json`](artifacts/mcd_observation_probe/observation_probe_combined_20260908.json),
with SHA-256
`3ca259ba23e10ff6f112af910896049bc90c52e6cf226f4fb2169ce5f5231073`.
The runner SHA-256 recorded by that result is
`5460891aa068fb1b9f85c66a14e302eea866a95803fd78f9e43a42235958c7f0`,
and the fixed configuration SHA-256 is
`a39d34cf0c74078121ced235cbb148ef1b196810481c5a000d72b7ccdeec479a`.

The direct command was:

```bash
/private/tmp/mcd-observation-probe-venv/bin/python \
  scripts/probe_mcd_observation_ranking.py \
  --shard-root /private/tmp/mcd-observation-probe-49143/shards \
  --config configs/evaluation/mcd_observation_probe_v1.json \
  --output docs/diagnostics/artifacts/mcd_observation_probe/observation_probe_combined_20260908.json
```

## Best-ROI temporal persistence probe (2026-09-08)

### Contract

**Question:** Is the identity of the lowest-error ROI persistent between
consecutive hops within a clip, or is it consistent with independent draws?

**Required arms:** (a) observed adjacent-hop agreement of the best-ROI
identity; (b) mean Spearman rank correlation between the 12-error vectors of
adjacent hops.

**Negative control:** the same two statistics recomputed after shuffling hop
order within each clip (destroys temporal adjacency, preserves each clip's
marginal), plus the analytic chance level `sum_k p_k^2` where `p_k` is the
overall frequency of ROI `k` being best.

**Interpretation limits:** persistence does not isolate its cause and could
reflect slowly varying signal quality or heart-rate autocorrelation; best-ROI
identity is a hard argmin so near-ties flip easily, making the rank
correlation the more stable read; free hops are conditioned on PPO's own
switching behaviour; single-hop counterfactuals were never continued. The
observed statistic uses eligible pairs with `hop_idx` differing by exactly 1,
but the shuffled-control implementation pairs adjacent retained rows after
shuffling rather than rematching the same eligible adjacency positions. The
temporal interpretation is therefore **INCOMPLETE** pending a control with
matched eligible adjacency positions; this run does not establish corrected
or new pair counts.

**Stop/next rule:** if observed agreement is at the chance level and mean
rank correlation is near zero, the best-ROI target carries little temporal
structure and feature work aimed at it has a low ceiling. If agreement and
correlation are clearly above the shuffled control, temporal structure exists
and feature screening is justified.

### Results

The run loaded the same 32 `shard-*/observations_advantage_ppo_seed1.csv`
files (91,227 rows) used above, applied the same free-hop and validity rules,
and retained 55,498 hops from 530 clips (32,000 locked hops and 3,729 hops
with no valid ROI were dropped; zero rows were dropped for non-finite
observations or errors). Subject counts are not available from these files
because the shard root pulled for this probe contains only observation CSVs,
not `per_hop.csv`.

| Statistic | Observed | Shuffled control (mean ± sd, 20 repeats) | Analytic chance |
|---|---:|---:|---:|
| Best-ROI agreement (A) | 0.4624 | 0.1856 ± 0.0017 | 0.1607 |
| Mean Spearman correlation (B) | 0.5145 | 0.0381 ± 0.0013 | not applicable |

The observed statistics were computed over 23,579 adjacent hop pairs (hops
sharing a `clip_id` with `hop_idx` differing by exactly 1). Zero pairs were
skipped from statistic B for having fewer than 3 shared-valid ROIs or for a
non-finite correlation; the shuffled control skipped a mean of 0.0 pairs per
repeat. The best-ROI marginal histogram over the 55,498 retained hops was
`[19013, 3487, 3618, 2433, 2546, 5275, 4690, 4679, 2276, 2250, 2907, 2324]`
for ROIs 0 through 11.

The observed agreement (0.4624) and observed mean rank correlation (0.5145)
are numerically above their respective shuffled-control values (0.1856 ±
0.0017 and 0.0381 ± 0.0013), and observed agreement is above the analytic
chance level (0.1607). However, because the shuffled control does not use
matched eligible adjacency positions, the temporal interpretation is
**INCOMPLETE** pending that recomputation. Do not treat this comparison as a
definitive claim that temporal structure exists or as evidence for a revised
pair count.

### Artifact

[`best_roi_persistence_20260908.json`](artifacts/mcd_observation_probe/best_roi_persistence_20260908.json),
SHA-256 `bffcb9e498368656986affaf57c8dcfc8fee4b7c7bbfa909990ba3f83f39bbd4`.

The direct command was:

```bash
PYTHONPATH=src uv run --no-project --with numpy --with scipy python \
  scripts/probe_best_roi_persistence.py \
  --shard-root temp/probe_data \
  --output docs/diagnostics/artifacts/mcd_observation_probe/best_roi_persistence_20260908.json
```

## Relative feature screen (2026-09-09)

### Motivation

The `local_shared` arm above cannot see the other 11 ROIs at all: its feature
block is the candidate's own 7 values plus 17 shared/controller values. The
`full_101` arm could see all 12 ROIs but did not use that information
effectively, selecting only 2 of them. Relative features supply the
cross-ROI comparison directly, as deterministic functions of the existing
101 observation values, so this screen tests whether re-expressing existing
information changes the fixed model's regret. The screen does not establish
that another representation would behave the same way.

### Contract

**Question:** Do within-hop relative features, derived from the existing 101
observation values, reduce mean immediate regret below the `local_shared`
baseline of 1.7417 BPM?

**Required arms:** `local_shared` (baseline reproduction),
`local_shared_relative`, `local_shared_relative_shuffled`, `ppo_factual`,
`majority`, `oracle`.

**Negative control:** training-target shuffle for the relative arm, matching
the existing probe's control (permute complete 12-error vectors among
training hops only).

**Baseline reproduction check:** the `local_shared` arm must reproduce
1.7417 BPM under the same seed, folds, model settings, and retained-hop set.
Report the reproduced value; if it differs, report the difference and do not
suppress it.

**Interpretation limits:** these seven relative features did not demonstrate
improved regret under this fixed model and cohort; other representations
remain untested. A null result cannot establish that the fixed model already
extracts equivalent information or rule out a useful re-expression;
single-hop counterfactuals are never continued; free hops are conditioned on
PPO's own switching; subject identity is derived from the clip-id prefix
rather than joined from `per_hop.csv`; `scikit-learn`'s
default `early_stopping="auto"` makes `HistGradientBoostingRegressor`
fitted results order-sensitive above 10,000 samples, so both this probe and
`probe_mcd_observation_ranking.py` rely on matched `(clip_id, hop_idx)` row
ordering for their results to be comparable.

**Stop/next rule:** this screen alone does not justify observation regeneration
or retraining. These seven relative features did not demonstrate improved
regret under this fixed model and cohort; other representations remain
untested. Any decision to test another representation requires a separately
specified, fair comparison; do not infer from this screen that re-expression
cannot help.

### Subject-identity derivation check

`temp/probe_data` contains only `shard-*/observations_advantage_ppo_seed1.csv`
files, with no `per_hop.csv` to join `subject_id` from. Subject identity was
instead derived from the `clip_id` prefix before the first underscore (for
example, subject `9940` from clip `9940_IriunWebcam_before`). This rule was
verified directly against the 91,227-row, 533-clip cohort before use: the
prefix rule yields exactly 89 distinct subjects across all 533 clips, and,
after applying the same free-hop and valid-ROI filters used above, exactly
530 clips and 55,498 hops are retained, matching the counts recorded in the
observation-ranking probe above. Fold membership computed from the
prefix-derived subject labels was also compared directly against the
`local_shared` fold membership recorded in
`observation_probe_combined_20260908.json` and found identical (same test
subjects in all 5 folds).

### First run: baseline reproduction failure (negative result, preserved)

The first run of `scripts/probe_relative_feature_screen.py` built its design
matrix rows in shard-read order rather than `(clip_id, hop_idx)` order. It
loaded the same 32 shard files (91,227 rows), applied the same free-hop and
valid-ROI rules, and retained the same 55,498 hops from 530 clips across 89
subjects (32,000 locked hops and 3,729 hops with no valid ROI dropped; zero
rows dropped for non-finite observations or errors), but its reproduced
`local_shared` value was 1.7691 BPM, a difference of +0.0274 BPM from the
recorded 1.7417 BPM in
`observation_probe_combined_20260908.json`. Seed (8101), folds (5,
subject-grouped, verified to produce identical fold membership), model
settings, and the retained-hop set were identical between the two runs.

The identified cause: `HistGradientBoostingRegressor` is constructed without
an explicit `early_stopping` argument, so scikit-learn's default value
`"auto"` applies, which turns early stopping on for datasets above 10,000
samples and takes an internal validation split from the training rows as
given to `.fit()`. That split is order-dependent even under a fixed
`random_state`. `scripts/probe_mcd_observation_ranking.py` iterates
`sorted(observations.items())`, which sorts by `(clip_id, hop_idx)`, while
the first run of `probe_relative_feature_screen.py` appended rows in
shard-read order instead, so the two scripts fit on rows in different order
and produced different fitted models for the same nominal cohort.

The fix: `load_rows` in `scripts/probe_relative_feature_screen.py` now sorts
the retained rows by `(clip_id, hop_idx)` before any design matrix is built
or model is fit, matching `probe_mcd_observation_ranking.py`. No feature
definition, arm, metric, bootstrap setting, fold setting, or model setting
was changed.

The first run's artifact is preserved as a negative result at
[`relative_feature_screen_20260909.json`](artifacts/mcd_observation_probe/relative_feature_screen_20260909.json)
and was not deleted or overwritten.

### Results (ordered run)

The ordered run loaded the same 32 shard files (91,227 rows), applied the
same free-hop and valid-ROI rules, and retained 55,498 hops from 530 clips
across 89 subjects (32,000 locked hops and 3,729 hops with no valid ROI were
dropped; zero rows were dropped for non-finite observations or errors).

**Baseline reproduction:** the reproduced `local_shared` value is 1.7417
BPM (1.7416904151636086), a difference of -0.0000096 BPM from the recorded
1.7417 BPM, within 1e-4.

| Arm | Mean immediate regret (BPM) | Subject-block 95% CI | Top-1 | Top-3 | Hops | Clips |
|---|---:|---:|---:|---:|---:|---:|
| `local_shared` (baseline reproduction) | 1.7417 | [1.5727, 1.9223] | 10.6% | 26.9% | 55,498 | 530 |
| `local_shared_relative` | 1.7360 | [1.5670, 1.9193] | 12.0% | 29.4% | 55,498 | 530 |
| `local_shared_relative_shuffled` control | 1.9121 | [1.7179, 2.1300] | 7.6% | 25.2% | 55,498 | 530 |
| `ppo_factual` | 1.6399 | [1.4477, 1.8548] | 30.0% | 48.0% | 55,498 | 530 |
| `majority` | 2.7501 | [2.4587, 3.0464] | 34.3% | 43.9% | 55,498 | 530 |
| `oracle` | 0.0000 | [0.0000, 0.0000] | 100.0% | 100.0% | 55,498 | 530 |

The paired subject-block results were:

| Contrast | Difference (BPM) | Subject-block 95% CI | Result |
|---|---:|---:|---|
| `local_shared_relative` minus `local_shared` | -0.0057 | [-0.0290, 0.0172] | CI includes zero |
| `local_shared_relative` minus `local_shared_relative_shuffled` | -0.1762 | [-0.2507, -0.1148] | relative lower |
| `local_shared_relative` minus `ppo_factual` | +0.0961 | [+0.0414, +0.1474] | PPO lower |

`local_shared_relative`'s action histogram across ROIs 0-11 was
`[11774, 2526, 2393, 4851, 5162, 6981, 3568, 3940, 4426, 4300, 2026, 3551]`,
selecting every ROI at least 2,026 times.

`local_shared_relative` regret is below the reproduced `local_shared` value
by a point estimate of 0.0057 BPM, but the paired 95% CI, [-0.0290, 0.0172],
includes zero. This does not satisfy the stop/next rule (which requires the
CI upper bound strictly below zero). `local_shared_relative` beats its
matched shuffled control with a CI upper bound strictly below zero. It does
not beat `ppo_factual`; the reported paired CI is entirely above zero,
favoring PPO.

### Metrics and uncertainty

The primary metric is immediate regret in BPM: chosen valid counterfactual
error minus the minimum valid counterfactual error. Secondary metrics are
top-1, top-3, and action histograms. Point estimates aggregate hops to clip
means and then weight clips equally. The 10,000-draw subject-block bootstrap
uses seed `8101`; paired differences are computed within each draw.

### Artifact

[`relative_feature_screen_20260909_ordered.json`](artifacts/mcd_observation_probe/relative_feature_screen_20260909_ordered.json),
SHA-256 `0b4a04a38df2b660f0c552b13f5b081896f82ab4ed902aa62e5c368e00479e2a`.
The runner SHA-256 recorded by that result is
`d5d95981377c0f70acb70fdeef37dcd68ef3e3eb15776be17a2a56ec7ea244e7`, and the
fixed configuration SHA-256 is
`a39d34cf0c74078121ced235cbb148ef1b196810481c5a000d72b7ccdeec479a` (same
configuration file as the observation-ranking probe above, used only for the
fixed model hyperparameters).

The direct command was:

```bash
PYTHONPATH=src uv run --no-project --with numpy --with scipy --with scikit-learn python \
  scripts/probe_relative_feature_screen.py \
  --shard-root temp/probe_data \
  --output docs/diagnostics/artifacts/mcd_observation_probe/relative_feature_screen_20260909_ordered.json
```

The first (unordered) run's command and artifact are preserved unchanged as
a negative result at
[`relative_feature_screen_20260909.json`](artifacts/mcd_observation_probe/relative_feature_screen_20260909.json).
Both JSON reports retain the historical contract wording recorded at
execution; that wording is superseded by the present qualified interpretation
in this document. The reports are preserved byte-for-byte. The runner hash
recorded in the ordered report identifies the execution version before this
wording-only revision; changing the descriptive `CONTRACT` strings changes
the script hash without changing the algorithm or either report.
