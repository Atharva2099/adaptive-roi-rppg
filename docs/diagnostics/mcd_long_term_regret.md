# MCD long-term regret diagnostic

Question: on the 54 MCD evaluation clips with the worst mean post-update-belief MAE in the frozen Advantage PPO replay, how much does the factual action at a genuinely free anchor differ from the best of the 12 one-time-action trajectories over H1, H5, and H15?

Required arms are the factual action and every forced action 0 through 11. A branch forces only the anchor action. At offsets 1 through 14, the same frozen policy selects actions from its post-prediction recurrent state; ordinary minimum-hold overrides remain in effect and are recorded. This is not an exponential all-future-action search.

The source is the completed 48121 failure-audit `per_hop_counterfactual.csv`, its `audit_plan.json`, and `COMPLETE.json`. The selection is rebuilt from row-level `selected_abs_error_bpm` values, never a prose result or pre-aggregated clip CSV. Each Advantage seed must have the same 533 clips and 89 subjects. The cutoff is exactly 54 (`ceil(0.10 * 533)`), sorted by descending three-seed mean clip MAE and `clip_id` on ties.

Freeze that cohort once with `scripts/freeze_mcd_long_term_regret_selection.py`; prepare 162 small clip/seed subsets once with `scripts/prepare_mcd_repeated_beam_subsets.py`. Workers read only their assigned subset and never derive a new cohort.

Eligibility requires `N - anchor_hop_idx >= 15` and every action resolving to itself and legal under the current controller. Locked hops and end-of-clip anchors are excluded separately. Branch traces are unscored before labels are joined. H1 scores offset 0, H5 offsets 0 through 4, and H15 offsets 0 through 14. Positive factual regret means some forced one-time action had lower post-update-belief absolute error than the factual action.

Aggregate each anchor’s factual regret to a clip mean, then equally weight clips within each subject, then use a 10,000-draw subject-block bootstrap. Seed estimates remain separate; only their mean and range are descriptive. This is a conditional worst-54 MCD diagnostic, not a generalization, retraining, reward-design, or MMPD claim.

Negative control: the branch forced to the factual proposed action must be identical to uninterrupted factual replay at H1/H5/H15. Stop if identity, recurrent timing, branch isolation, or label-after-trace checks fail. Continue only when all branch traces, source identities, and subject-level uncertainty are valid.

## Separate repeated-correction offline headroom arm

`mcd_repeated_correction_beam_v1.json` defines a distinct 15-step arm. At the same factual all-12-free anchors, an uninterrupted factual PPO trace is pinned outside the beam capacity. Each active search state calls PPO once, exhaustively finds the legal action with minimum immediate post-update GT absolute error (action-index tie), then expands only that rank-1 child plus the PPO proposal when it is legal and different. The rank-1 choice is GT-informed offline headroom, not a controller result. Primary width is 8; widths 1, 2, 4, 8, 16, and 32 are retained as sensitivity. Candidates are ordered by cumulative AE, endpoint AE, executed sequence, then origin sequence. This arm does not alter the one-time arm or its factual-parity control.
