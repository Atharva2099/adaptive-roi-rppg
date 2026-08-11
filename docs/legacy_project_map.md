# Legacy project map

## Snapshot and safety boundary

**HISTORICAL CONTEXT:** The legacy repository is `/Users/atharva/Desktop/RL for RoI`, tracked source commit `28ac407fe495c808b67f5f5288a3c047b63566c3`. Its worktree was dirty during migration, and that commit pins tracked source only. Treat it as read-only. Local results are independently frozen by the evidence registry.

## What may be migrated

Migrate concepts, interfaces, lessons, verified calculations, and provenance references. Start with the evidence IDs in [the evidence registry](evidence_registry.md); do not treat a prose report as row-level proof.

## What must not be copied in this phase

Do not copy code, scripts, configurations, result trees, datasets, checkpoints, caches, virtual environments, logs, temporary files, or old documents wholesale. This package copies knowledge only.

## Top-level directory map

These are historical roles, not an endorsed future structure:

| Directory | Historical role |
|---|---|
| `src/` | Research and evaluation implementation. |
| `scripts/` | Data and analysis launchers. |
| `slurm/` | Remote job descriptions. |
| `configs/` | Experiment and reproducibility settings. |
| `docs/` | Reports, decisions, and evaluation notes. |
| `results/` | Generated metrics and manifests. |
| `tests/` | Probes and regression checks. |
| `temp/` | One-off diagnostics and scratch protocols. |
| `archive/` | Retained historical material. |
| `logs/` | Training and job logs. |

## Authoritative source map

- Active streaming goal and ruler: use E-001–E-005 in [the evidence registry](evidence_registry.md).
- Schema-v3 evaluation artifacts: use the row-level paths owned by E-001–E-004.
- Mechanism and critic diagnostics: inspect E-006 and E-007 only when their limits are understood; this map repeats no numerical conclusions.
- Oracle corrections: inspect E-005 and E-009.
- MMPD official and frozen-controller comparisons: inspect E-010–E-013. They are evaluation-only and remote-limited.
- Golden/fitted-Q history: E-014 identifies it as historical context, not active streaming evidence.

## Polaris conceptual map

**HISTORICAL CONTEXT:** Legacy local launchers historically targeted remote MCD staging and result locations. This map makes no current connectivity or remote-state claim. Do not access remote systems as part of this migration.

## Known traps

- Conflicting old documents may describe retired rulers.
- Schema-v2 subharmonic results are superseded by schema-v3.
- The worktree was dirty, so the tracked commit is not a complete snapshot of local artifacts.
- Multiple signal pipelines make official POS versus causal POS non-comparable as a pure smoothing ablation.
- MMPD row-level results are remote-only and cannot feed MCD development.
- Old temporary scripts are historical context, not production interfaces.

## How to use this map

Start from [the evidence registry](evidence_registry.md), inspect a cited legacy artifact read-only only when necessary, and then redesign the interface for the new repository rather than copying implementation material.
