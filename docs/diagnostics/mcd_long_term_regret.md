# MCD long-term regret diagnostic

## HISTORICAL / SUPERSEDED worst-54 procedure

The worst-54/v1/v2 material below is retained as a historical record only.
Its selection-freezing scripts, 162-task subset procedure, and referenced
legacy configurations were removed from the current candidate. They are not
current runnable instructions or new verification in this session. The active
full-cohort-only procedure is documented in
[`mcd_repeated_beam.md`](mcd_repeated_beam.md); its MCD-only, offline limits
still apply.

Evidence timeline: source evaluation completed 2026-08-26; worst-54 producer and first aggregation completed 2026-08-27 to 2026-08-28; full-cohort seed-1 producer and aggregation completed 2026-08-31; full-cohort seed-0 and seed-2 producers and aggregations completed 2026-09-01.

Last numerical verification: 2026-09-01 (America/Los_Angeles). On that date, the three full-cohort aggregate jobs were checked through scheduler accounting, all returned `COMPLETED` with exit `0:0`, and the remote `anchors.csv` and `summary.json` SHA-256 values matched the pinned values in this document.

Question: on the 54 MCD evaluation clips with the worst mean post-update-belief MAE in the frozen Advantage PPO replay, how much does the factual action at a genuinely free anchor differ from the best of the 12 one-time-action trajectories over H1, H5, and H15?

Required arms are the factual action and every forced action 0 through 11. A branch forces only the anchor action. At offsets 1 through 14, the same frozen policy selects actions from its post-prediction recurrent state; ordinary minimum-hold overrides remain in effect and are recorded. This is not an exponential all-future-action search.

The source is the completed 48121 failure-audit `per_hop_counterfactual.csv`, its `audit_plan.json`, and `COMPLETE.json`. The selection is rebuilt from row-level `selected_abs_error_bpm` values, never a prose result or pre-aggregated clip CSV. Each Advantage seed must have the same 533 clips and 89 subjects. The cutoff is exactly 54 (`ceil(0.10 * 533)`), sorted by descending three-seed mean clip MAE and `clip_id` on ties.

Freeze that cohort once with `scripts/freeze_mcd_long_term_regret_selection.py`; prepare 162 small clip/seed subsets once with `scripts/prepare_mcd_repeated_beam_subsets.py`. Workers read only their assigned subset and never derive a new cohort.

Eligibility requires `N - anchor_hop_idx >= 15` and every action resolving to itself and legal under the current controller. Locked hops and end-of-clip anchors are excluded separately. Branch traces are unscored before labels are joined. H1 scores offset 0, H5 offsets 0 through 4, and H15 offsets 0 through 14. Positive factual regret means some forced one-time action had lower post-update-belief absolute error than the factual action.

Aggregate each anchor’s factual regret to a clip mean, then equally weight clips within each subject, then use a 10,000-draw subject-block bootstrap. Seed estimates remain separate; only their mean and range are descriptive. This is a conditional worst-54 MCD diagnostic, not a generalization, retraining, reward-design, or MMPD claim.

Negative control: the branch forced to the factual proposed action must be identical to uninterrupted factual replay at H1/H5/H15. Stop if identity, recurrent timing, branch isolation, or label-after-trace checks fail. Continue only when all branch traces, source identities, and subject-level uncertainty are valid.

## Separate repeated-correction offline arm

`mcd_repeated_correction_beam_v1.json` defines a distinct 15-step arm. At the same factual all-12-free anchors, an uninterrupted factual PPO trace is pinned outside the beam capacity. Each active search state calls PPO once, exhaustively finds the legal action with minimum immediate post-update GT absolute error (action-index tie), then expands only that rank-1 child plus the PPO proposal when it is legal and different. The rank-1 choice uses ground-truth information offline and is not a controller result. Primary width is 8; widths 1, 2, 4, 8, 16, and 32 are retained as sensitivity. Candidates are ordered by cumulative AE, endpoint AE, executed sequence, then origin sequence. This arm does not alter the one-time arm or its factual-parity control.

## Observed results: jobs 48193 and 48224 (pending v2 regeneration)

The numerical tables in this section are v1 values and are pending regeneration with the explicit comparison tolerance recorded by the repaired analyzer. They are retained as a provisional record and must not be treated as the v2 output.

The recorded cohort was the selected 54-clip MCD cohort from frozen Advantage PPO replay: 54 clips, 30 subjects, three seeds, and 162 tasks. The output contained 16,724 anchors. Of these, 15,230 were analysis-eligible; 1,494 were excluded because saved single-correction coverage was incomplete. Seed outputs were retained separately and were not pooled or ensembled.

Job 48193 completed with state `COMPLETED`, exit code `0:0`, runtime 06:24:16, from 2026-08-27T20:24:08 to 2026-08-28T02:48:24. Job 48224 completed with state `COMPLETED`, exit code `0:0`, runtime 00:03:46, from 2026-08-28T16:10:48 to 16:14:34. The analyzed output paths were `/Users/924254653/mcd_repeated_beam_analysis_48193_v1/anchors.csv` and `/Users/924254653/mcd_repeated_beam_analysis_48193_v1/summary.json`. The recorded SHA-256 values were `a6770afba3491da9ac4715437dfc2c7d0281a4c7934ab29b39773dc69493fc79` for `anchors.csv` and `65f83c9687a7cfe991543802c619be725656a6d76501fc7657fccc92dcb32934` for `summary.json`.

Definitions used in the tables are as follows. H15 is mean absolute error over the fixed 15 steps. Positive immediate regret is factual PPO immediate error minus the minimum immediate error among legal GT-ranked actions; a positive value means the PPO error was higher for that step. A zero-error tie does not identify which action produced it. Same-step reversal is factual PPO lower immediate error than the best same-anchor non-PPO alternative, while that alternative has lower H15 MAE. Best single is the lowest H15 MAE among saved genuine one-time deviations anywhere from X through X+14 with complete saved coverage. Repeated is a retained width-8 path with at least two real legal deviations from PPO; it is not a globally optimal path. Proposal change rate compares post-correction PPO proposals with untouched factual PPO proposals at matching later offsets.

Anchor values were averaged within clip, clips were equally weighted within subject, and subjects were equally weighted within seed. Intervals are 10,000 subject-block bootstrap draws and are 95% intervals. Proportion raw counts are anchor counts and are shown separately from the subject-weighted estimate and interval. Numeric entries are subject-weighted BPM or rates with the interval in brackets. Displayed values are rounded; the source CSV and JSON retain the underlying values.

| Measure | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Factual H15 MAE (BPM) | 20.63 [18.49, 22.83] | 18.59 [16.13, 21.15] | 21.32 [18.95, 23.83] |
| Positive immediate regret | 60.3% [52.4%, 67.6%]; 2,963/5,130 | 61.6% [53.5%, 69.3%]; 3,037/5,072 | 64.6% [56.5%, 72.2%]; 3,119/5,028 |
| Mean immediate regret (BPM) | 3.07 [2.27, 3.95] | 3.39 [2.37, 4.54] | 3.43 [2.62, 4.34] |
| Same-step reversal | 27.9% [23.4%, 32.2%]; 1,424/5,130 | 25.5% [21.1%, 29.8%]; 1,295/5,072 | 27.0% [21.8%, 32.1%]; 1,355/5,028 |
| Best single lower H15 MAE than factual | 93.0% [86.8%, 97.9%]; 4,576/5,130 | 94.6% [88.9%, 98.7%]; 4,656/5,072 | 92.2% [84.7%, 97.8%]; 4,441/5,028 |
| Best single H15 reduction vs factual (BPM) | 4.79 [3.83, 5.82] | 5.04 [3.85, 6.32] | 5.37 [4.36, 6.47] |
| Best correction after X | 65.8% [59.6%, 70.9%]; 3,326/5,130 | 63.6% [57.3%, 68.9%]; 3,145/5,072 | 65.1% [59.8%, 69.8%]; 3,198/5,028 |
| Repeated path available | 88.4% [80.1%, 95.3%]; 4,382/5,130 | 93.0% [86.6%, 98.0%]; 4,588/5,072 | 86.9% [77.8%, 94.4%]; 4,208/5,028 |
| Repeated lower H15 than best single | 91.3% [86.6%, 95.0%]; 4,049/4,382 | 83.7% [75.6%, 90.5%]; 3,995/4,588 | 92.6% [89.8%, 95.2%]; 3,926/4,208 |
| Repeated H15 reduction vs best single (BPM) | 4.90 [3.97, 5.82] | 3.91 [3.09, 4.76] | 5.05 [4.11, 6.04] |
| Post-correction proposal change rate | 62.8% [60.7%, 64.9%] | 63.4% [61.5%, 65.4%] | 65.6% [63.5%, 67.6%] |

The denominator for best-single comparisons is the complete one-time-coverage set. The denominator for repeated-versus-single comparisons is the repeated-path-available set. The proposal-change rate uses the recorded post-correction decision pairs.

Recovery classification used the pointwise difference between factual error and repeated-path error. Ties were counted as not lower.

| Recovery classification | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Better throughout (`better_throughout`) | 50.5% [43.4%, 57.7%]; 2,155/4,382 | 46.9% [40.3%, 53.9%]; 2,182/4,588 | 52.3% [45.5%, 58.7%]; 2,199/4,208 |
| Briefly better, then ends worse (`briefly_better_then_ends_worse`) | 1.3% [0.7%, 2.1%]; 56/4,382 | 2.0% [1.2%, 2.9%]; 91/4,588 | 1.6% [0.9%, 2.3%]; 66/4,208 |
| Only ends better late (`only_ends_better_late`) | 27.6% [23.1%, 31.9%]; 1,227/4,382 | 22.7% [18.8%, 27.0%]; 1,126/4,588 | 24.5% [20.1%, 29.0%]; 1,027/4,208 |
| Mixed/intermittent (`mixed_intermittent`) | 19.8% [14.0%, 26.1%]; 924/4,382 | 23.3% [17.4%, 29.3%]; 1,039/4,588 | 21.4% [15.0%, 29.2%]; 908/4,208 |
| Never better (`never_better`) | 0.8% [0.0%, 2.3%]; 20/4,382 | 5.1% [0.7%, 10.9%]; 150/4,588 | 0.2% [0.0%, 0.6%]; 8/4,208 |

The recorded limitations are: the cohort is a selected 54-clip conditional subset, not the full 533-clip MCD set; comparisons use offline ground-truth information and are not deployable actions; 1,494 late or otherwise incomplete anchors were excluded; width 8 retains and prunes candidates and is non-exhaustive; final survivors cannot reconstruct intermediate beam winners; no MMPD data were used; and these tables do not identify a PPO training cause or quantify overall MCD performance.

These tables report the specified procedure and cohort only.

## Full-cohort v2 results

The full MCD diagnostic covers all 533 authenticated evaluation clips, 89 subjects, and frozen Advantage PPO seeds 0, 1, and 2. It uses source-plan clip order and `task_index = 3 * clip_position + seed`; it does not reuse the worst-54 selection. The source is `/Users/924254653/mcd_failure_eval_full2_20260826/audit_plan.json`, `merged/COMPLETE.json`, and `merged/per_hop_counterfactual.csv`, revalidated for 533 clips, 89 subjects, 91,227 hops per seed, byte count, hashes, and checkpoint identities.

The three producer runs completed with 533 task summaries and 533 factual-parity passes each. Seed 0 job 48705 completed in `05:42:42`, seed 1 job 48580 completed in `04:46:10`, and seed 2 job 48706 completed in `04:42:09`. The matching v2 aggregations were job 48757 for seed 0 (`00:19:15`), job 48640 for seed 1 (`00:13:29`), and job 48758 for seed 2 (`00:17:05`); each completed with exit `0:0`. Initial aggregation jobs 48754 and 48756 failed in one second before analysis because the snapshot-local default Python executable did not exist. They created no result directory. The successful retries supplied the same Polaris Python 3.10 interpreter used by the seed-1 aggregate.

V2 uses one `1e-6 BPM` tolerance on the reported scale: immediate and pointwise values compare directly; H15 categorical results compare the MAE difference, not the 15-step total. Continuous values are unchanged. Anchor values are averaged within clip, clips are equally weighted within subject, and subjects are equally weighted within seed. Intervals are 10,000 subject-block bootstrap draws and are 95% intervals. Seed means and ranges below are descriptive arithmetic summaries of the three independently aggregated seeds; seed-specific eligible anchor sets are not matched.

| Coverage | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Saved anchors | 59,029 | 54,397 | 53,509 |
| Analysis eligible | 53,806 | 49,640 | 48,858 |
| Excluded: incomplete one-time coverage | 5,223 | 4,757 | 4,651 |
| Retained repeated path available | 51,740 | 48,561 | 46,538 |

| Measure | Seed 0 | Seed 1 | Seed 2 | Seed mean [range] |
|---|---:|---:|---:|---:|
| Factual H15 MAE (BPM) | 7.509 [6.772, 8.262] | 7.355 [6.616, 8.140] | 7.583 [6.811, 8.389] | 7.482 [7.355, 7.583] |
| Positive immediate regret | 64.88% [63.43%, 66.28%]; 34,385/53,806 | 66.66% [65.21%, 68.09%]; 32,748/49,640 | 67.12% [65.64%, 68.58%]; 32,324/48,858 | 66.22% [64.88%, 67.12%] |
| Mean immediate regret (BPM) | 1.498 [1.336, 1.668] | 1.509 [1.346, 1.682] | 1.604 [1.430, 1.785] | 1.537 [1.498, 1.604] |
| Same-step reversal | 29.21% [28.32%, 30.06%]; 15,512/53,806 | 30.59% [29.79%, 31.38%]; 15,024/49,640 | 31.81% [30.84%, 32.74%]; 15,338/48,858 | 30.53% [29.21%, 31.81%] |
| Best saved single lower H15 MAE than factual | 96.71% [95.38%, 97.89%]; 51,731/53,806 | 96.68% [95.34%, 97.88%]; 47,633/49,640 | 96.62% [95.28%, 97.81%]; 46,753/48,858 | 96.67% [96.62%, 96.71%] |
| Best saved single H15 reduction vs factual (BPM) | 2.113 [1.910, 2.325] | 2.114 [1.912, 2.327] | 2.310 [2.075, 2.546] | 2.179 [2.113, 2.310] |
| Repeated path available | 96.72% [95.45%, 97.87%]; 51,740/53,806 | 98.24% [97.29%, 99.05%]; 48,561/49,640 | 96.23% [94.86%, 97.47%]; 46,538/48,858 | 97.06% [96.23%, 98.24%] |
| Repeated lower H15 than best saved single | 92.24% [91.26%, 93.16%]; 48,027/51,740 | 89.65% [88.23%, 91.04%]; 43,719/48,561 | 90.98% [90.13%, 91.82%]; 42,446/46,538 | 90.96% [89.65%, 92.24%] |
| Repeated H15 reduction vs factual (BPM) | 3.752 [3.376, 4.143] | 3.592 [3.228, 3.984] | 3.890 [3.476, 4.310] | 3.745 [3.592, 3.890] |
| Repeated H15 reduction vs best saved single (BPM) | 1.555 [1.366, 1.749] | 1.443 [1.263, 1.641] | 1.463 [1.269, 1.672] | 1.487 [1.443, 1.555] |
| Post-correction PPO proposal change rate | 51.10% [49.71%, 52.40%] | 62.03% [61.42%, 62.63%] | 60.01% [59.26%, 60.71%] | 57.71% [51.10%, 62.03%] |

`Positive immediate regret` asks whether the factual PPO immediate error exceeded the lowest immediate error among legal GT-ranked actions at the anchor. `Best saved single` is retrospective: it is the lowest H15 MAE among genuine one-time deviations saved anywhere from X through X+14 with complete coverage, not necessarily a correction at X. The retained repeated path contains at least two actual legal deviations from PPO. It uses offline GT ranking and width-8 pruning and is not a deployable or globally optimal policy.

Recovery categories use the pointwise factual-error minus repeated-path-error sequence. Ties within `1e-6 BPM` are not lower.

| Recovery classification | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Better throughout | 18.97% [17.22%, 20.81%]; 9,193/51,740 | 18.58% [16.93%, 20.34%]; 8,923/48,561 | 19.85% [18.03%, 21.67%]; 9,003/46,538 |
| Briefly better, then ends worse | 2.55% [2.35%, 2.75%]; 1,344/51,740 | 2.53% [2.34%, 2.73%]; 1,243/48,561 | 2.49% [2.29%, 2.70%]; 1,173/46,538 |
| Only ends better late | 23.45% [22.20%, 24.70%]; 11,872/51,740 | 21.88% [20.67%, 23.12%]; 10,600/48,561 | 22.30% [21.13%, 23.45%]; 10,244/46,538 |
| Mixed/intermittent | 54.28% [51.74%, 56.81%]; 29,105/51,740 | 54.45% [51.96%, 56.91%]; 26,705/48,561 | 55.29% [52.85%, 57.74%]; 26,089/46,538 |
| Never better | 0.75% [0.27%, 1.34%]; 226/51,740 | 2.55% [1.50%, 3.80%]; 1,090/48,561 | 0.08% [0.01%, 0.16%]; 29/46,538 |

The three summaries bind the same beam configuration SHA-256 `11e1de5d3418db6cb570fe6a80998f44027141b4bc3922b8d347decf4ba32a2e`, source SHA-256 `930c1ca3cc414558d044a0f3a9bab4e49830bcc2e6e3da9ec1725b59bd68f715`, source-plan SHA-256 `0d6c4edde239e1cf619cc01d3d59f03c640bc6b84eb68b4425fe047582b50f1c`, and subset-index SHA-256 `9d1bf9d042631eb0ff41305577adc61547a5f0c9b7dd94862238a5ec958ed412`.

| Seed | Aggregate directory | `anchors.csv` SHA-256 | `summary.json` SHA-256 |
|---|---|---|---|
| 0 | `/Users/924254653/mcd_repeated_beam_full_seed0_analysis_48705_v1` | `6f769fcff157c8c9e76df33b07c780ad221d96442c2e7cc95643d99fd9a3c824` | `dcefb1886a305e88c8281d3d3ef3b8eb7bac6dffa3503121dc528b026e473520` |
| 1 | `/Users/924254653/mcd_repeated_beam_full_seed1_analysis_48580_v1` | `5e721116168e93dac002de324ca551afce0f81952131ac73ae0f16c87a5f2cea` | `b0798bf0e2a90a0c5bcc78ae38545c045002976c47c7fcfbef70289297fe78a3` |
| 2 | `/Users/924254653/mcd_repeated_beam_full_seed2_analysis_48706_v1` | `bca404f35e2bfaeb7337b702192780d2d093366f434a5637ec125dcb650b93cb` | `c9f60567b54dbde22cb51e3ec8aebfc5081d7ccce7dfa5c939cb313c7495b471` |

The result is a full-cohort MCD offline headroom diagnostic. It shows that changing a selected measurement can alter later PPO proposals and that saved GT-informed correction schedules often reduce H15 error. It does not establish why PPO chose an action, prove that PPO requires long-horizon planning, define a deployable correction policy, or provide MMPD transfer evidence.
## Human-level MCD case audit protocol

The companion forensic audit uses the fixed selection in
`configs/evaluation/mcd_case_audit_v1.json`: three high-error seed-1 clips from
job 48193 v2 and one predeclared median-error control. It compares raw-video
frames and face/ROI overlays with the frozen semantic-state RGB and coverage,
recomputes POS measurements, and replays the frozen seed-1 policy using the
101-dimensional observation contract. It reports frame counts, provenance,
POS HR/PPR/confidence/coverage/validity, selected ROI, actions, belief, and
error. Rewards, logits, and value estimates are explicitly unavailable because
they were not saved by job 48193. This four-clip audit can identify concrete
pipeline inconsistencies; it cannot establish cohort prevalence or explain all
PPO failures.
