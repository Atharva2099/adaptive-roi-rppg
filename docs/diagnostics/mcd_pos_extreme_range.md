# MCD ROI POS extreme-range diagnostic

This note records numerical comparisons between the ground-truth (GT) heart rate and the lowest and highest recovered ROI POS heart-rate estimates. It also records a separate matched comparison of fixed full-face POS and three frozen model families. The two analyses use different jobs and are not combined into one metric.

## Candidate-range question and definitions

For each clip-hop, the analysis recovered up to 12 ROI POS estimates. The primary denominator contains records for which all 12 executed ROI identities were recovered and at least one record has a valid finite HR. The lowest and highest values are computed over valid finite HR values only. A hop with all 12 recovered but no valid finite HR is reported separately and excluded.

* `GT below lowest`: `GT < lowest`; miss magnitude is `lowest - GT` BPM.
* `GT inside range`: `lowest <= GT <= highest`.
* `GT above highest`: `GT > highest`; miss magnitude is `GT - highest` BPM.

The combined output uses seed 0, 1, and 2 files to recover missing stored executed-action measurements by clip-hop and ROI identity. It is a coverage-recovery output, not an ensemble or a new model evaluation. Repeated executed actions are retained only when their measurement tuples agree. Requested actions are not used as ROI identities.

## Candidate-range counts

The full MCD source contains 91,227 clip-hops from 533 clips and 89 subjects. Counts below are raw clip-hop counts. The primary denominator is complete recovery of all 12 executed ROI identities with at least one valid finite HR value.

| Output | Primary denominator | GT below lowest | GT inside range | GT above highest | Outside range |
|---|---:|---:|---:|---:|---:|
| Seed 0 | 60,447 | 509 (0.84%) | 55,637 (92.04%) | 4,301 (7.12%) | 4,810 (7.96%) |
| Seed 1 | 55,498 | 413 (0.74%) | 50,863 (91.65%) | 4,222 (7.61%) | 4,635 (8.35%) |
| Seed 2 | 54,384 | 448 (0.82%) | 49,867 (91.69%) | 4,069 (7.48%) | 4,517 (8.31%) |
| Combined recovery | 81,343 | 620 (0.76%) | 74,564 (91.67%) | 6,159 (7.57%) | 6,779 (8.33%) |

Across the combined recovery, 85,494 hops had complete recovery, 5,733 had partial recovery, and 4,151 complete-recovery hops had zero valid ROI values. These categories are not included in the primary denominator where applicable.

## Miss magnitude distributions

The plot uses the combined recovery and shows one histogram for each outside-range direction. Bins are 1 BPM wide; the visible axis is limited to the common 0–25 BPM region, with the number at 25 BPM and above reported in the plot. The dashed line is the arithmetic mean.

| Output | Direction | n | Mean (BPM) | Median (BPM) | P90 (BPM) | Maximum (BPM) |
|---|---|---:|---:|---:|---:|---:|
| Seed 0 | GT below lowest | 509 | 5.3126 | 3.0762 | 12.7441 | 44.3848 |
| Seed 0 | GT above highest | 4,301 | 7.8844 | 4.9219 | 18.4570 | 88.3301 |
| Seed 1 | GT below lowest | 413 | 5.6493 | 3.0762 | 13.5703 | 44.3848 |
| Seed 1 | GT above highest | 4,222 | 8.0488 | 5.2734 | 18.8965 | 87.8906 |
| Seed 2 | GT below lowest | 448 | 5.4747 | 3.0762 | 13.6230 | 44.3848 |
| Seed 2 | GT above highest | 4,069 | 8.1248 | 5.2734 | 18.8965 | 87.8906 |
| Combined recovery | GT below lowest | 620 | 5.3337 | 3.0762 | 12.7441 | 44.3848 |
| Combined recovery | GT above highest | 6,159 | 8.1062 | 5.2734 | 18.8965 | 88.3301 |

## Matched fixed full-face POS and frozen-model comparison

This separate MCD evaluation used job 47417. The fixed full-face POS row was passed through the same causal belief tracker and scored with the same equal-clip, post-update-belief MAE ruler as the model rows. It is not a raw unsmoothed POS score. Family values are arithmetic means of the three seed values, not ensemble predictions.

| Method | Equal-clip MAE (BPM) |
|---|---:|
| Fixed full-face POS | 12.631680651997897 |
| DAgger, seed mean | 8.471633637516538 |
| Standard PPO, seed mean | 7.991317152305402 |
| Advantage PPO, seed 0 | 7.637492779923465 |
| Advantage PPO, seed 1 | 7.5006634485642305 |
| Advantage PPO, seed 2 | 7.731478074893967 |
| Advantage PPO, seed mean | 7.623211434460554 |

The comparison covers 533 clips, 89 subjects, and 91,227 hops per method. The arithmetic difference between fixed full-face POS and the Advantage PPO seed mean is 5.008469217537 BPM. This is a descriptive difference under the stated ruler; it is not a component or causal attribution.

## Provenance and limits

The candidate source is the completed job 48121 file `/Users/924254653/mcd_failure_eval_full2_20260826/output/full-eval2-20260826/merged/per_hop_counterfactual.csv` (821,043 data rows). The aggregation ran as Polaris job 48715 and wrote `/Users/924254653/mcd_adv_pos_extremes_48121_v1`. The retained local artifacts are the statistics CSV and plot in [the artifact directory](artifacts/mcd_pos_extreme_range/). Per-hop derived CSVs remain at the remote output path because they are about 47 MB in total and are not copied into the repository.

The source rows serialize executed-action measurements, so the analysis cannot recover an ROI value when no corresponding executed action was stored. Partial and zero-valid categories are reported rather than imputed. The combined recovery changes coverage only; it does not remove the missingness or make the three seeds an ensemble. The range classification is a clip-hop diagnostic, not a claim about all possible POS implementations or a causal explanation of model behavior. No MMPD data were used.

The matched comparison source is job 47417, report `/Users/924254653/adaptive_roi_gate8_runs_20260814_r2/full-47417/merged/report.json`, with row-level files recorded in evidence-registry entry E-031.
