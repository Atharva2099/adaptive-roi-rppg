# Sources and result versions

The [evidence registry](../docs/evidence_registry.md) owns research values, artifacts, uncertainty, and supersession. This file tracks sources and writing decisions. These starting entries reflect documentation inspected on 2026-09-07; numerical artifacts were not recomputed for this scaffold. Check for later accepted evidence before drafting.

## Current sources

- [Project scope](../docs/project_scope.md): questions and claim boundaries.
- [System design](../docs/system_design.md): architecture, interfaces, and causality concerns; verify implementation claims against relevant code.
- [Evaluation protocol](../docs/evaluation_protocol.md): comparisons, splits, metrics, and uncertainty.
- [Current status](../docs/current_status.md): reported progress; reconcile older status prose with the registry.
- [Legacy map](../docs/legacy_project_map.md): historical context and routing.

| Evidence | Use | Limits and replacement history |
|---|---|---|
| E-031; [frozen models](../docs/gates/gate_08_mcd_frozen_models.md) | Current MCD full-face, DAgger, Standard PPO, Advantage PPO comparison; 533 clips, 89 subjects, 91,227 hops/method | Nine frozen checkpoints; family means are seed means. Training lineage descriptive/unverified. Supersedes E-029 implementation history. |
| E-033; registry and frozen-model note | Current matched held-out Oracle B/C headroom; 533 clips, 89 subjects | GT-informed; C uses future information and approximate width-8 search. Legacy test90/cache scores cannot substitute for this protocol. |
| E-032; [POS range diagnostic](../docs/diagnostics/mcd_pos_extreme_range.md) | Descriptive measurement-failure analysis | Clip-hop denominators, partial candidate recovery, and remote source limits; does not establish causality. |
| E-030; [train Oracle note](../docs/gates/gate_07_mcd_oracles.md) | Train teacher context; 3,057 clips, 510 subjects | Offline; no published paired subject CI; different split from E-033. Supersedes E-028 run history. |
| E-028 and E-029 | Relevant repair and implementation history | Superseded/incomplete; no accepted final numerical result from these entries. |

## Legacy sources

Root: `/Users/atharva/Desktop/RL for RoI` (read-only). Use these for history, design rationale, failed directions, and potentially matched ablations. They are discovery pointers, not certified numerical claims:

- `docs/streaming_control_goal.md` and `docs/streaming_status_plain_english.md`: original questions, chronology, and explanations.
- `docs/mechanism_results_streaming_ppo.md`: model and critic analyses.
- `docs/streaming_rl_ablation_checklist_2026-06-17.md`: tested factors and comparison rules.
- `docs/t69_online_rl_pivot_decision_2026-06-10.md` and `docs/golden_to_t69_streamlined_report.md`: rejected directions and progression.
- `docs/signal_processing.md`, `docs/architecture.md`, `docs/understanding the model.md`, and `docs/src_pipeline_map_2026-08-09.md`: historical methods; trace claims to corresponding code/configuration.
- `docs/thesis_figures/grid_visualisation/`: figure sources.
- `docs/tables/`, `docs/presentations/`, and `results/`: locate relevant artifacts through the reports above; avoid copying result trees or merging incompatible generations.

Read only sources needed for the section. Full reconciliation of scattered experiments remains pending.

## Claims and version decisions

When drafting a claim, record its exact wording, evidence ID/source, method/checkpoint and protocol, cohort/denominator, metric/aggregation, row-verification command/date, uncertainty, and allowed interpretation. Link to registry details rather than duplicating them. For older results, explain their purpose and controls; for replacements, name the earlier version and reason and retain the historical interpretation.

No numerical thesis claim has been certified here yet. Verify historical missing-data handling before describing the full pipeline as strictly causal; trailing windows alone do not establish this.
