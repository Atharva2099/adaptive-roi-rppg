# Branch and rollback inventory

Last reviewed: 2026-09-04

`main` is the canonical supported code line. A file being present on `main` does
not make its scientific result verified. Claim status remains controlled by
`docs/evidence_registry.md` and the associated row-level evidence.

## Protected starting points

| Purpose | Commit | Remote reference | Meaning |
| --- | --- | --- | --- |
| Canonical state before integration | `c80122c449f02b41e9be1549841e022189a5bd34` | `rollback/pre-canonical-integration-20260904` | Reviewed lean `main`; 136 local tests passed before the tag was created. |
| Dirty-work preservation snapshot | `990b037` | `codex/pre-canonical-integration-snapshot-20260904` and `rollback/pre-canonical-integration-wip-20260904` | Preservation-only snapshot of MCD diagnostics and MMPD extractor engineering. It is not accepted science. |

The WIP snapshot excludes `.claude_session`, ignored caches and temporary
files, datasets, checkpoints, models, generated outputs, and actual MMPD
material. Do not merge the snapshot branch wholesale.

## Branch status

| Branch | Status | Canonical handling |
| --- | --- | --- |
| `main` | Supported | One true code line. Integrate only reviewed, compatible units. |
| `codex/verification-prune` | Integrated | Its reviewed patch is in `main` as `c80122c`; the commit hash differs because it was applied on top of the latest `origin/main`. |
| `codex/pre-canonical-integration-snapshot-20260904` | WIP preservation | Source for selective recovery only. No scientific claim is accepted by this snapshot. |
| `codex/gate9-dirty-snapshot-20260821` | Superseded snapshot | Preserve by tag until the newer WIP snapshot has been compared; never merge wholesale. |
| `codex/gate9-minimal-rebuild` | Active but incomplete | MMPD evaluation-only engineering reference. `p29_3`, authenticated inputs, and end-to-end scientific evaluation remain unresolved. |
| `codex/mcd-failure-audit` | Experimental / NO-GO | Preserve useful audit code, but causal replay, factual parity, and provenance must pass before results are accepted. |
| `codex/mcd-repeated-beam-smoke` | Experimental diagnostic | Preserve as a source for a lean repeated-correction path. Do not treat branch presence as behavioral evidence. |
| `codex/mcd-eval-oracle-test` | Experimental | Review for distinct scientific value before retaining any held-out Oracle runner. |
| `codex/mmpd-official-extractor-review` | Superseded by WIP snapshot | The tracked tip remains historical; the safe uncommitted additions were preserved in `990b037`. |

## Safe recovery

Recover into a new worktree. Do not reset or overwrite an existing dirty
checkout.

```bash
git fetch origin --tags
git worktree add -b recovery/pre-integration /tmp/adaptive-roi-recovery rollback/pre-canonical-integration-20260904
```

To inspect the preserved WIP instead, use
`rollback/pre-canonical-integration-wip-20260904` as the final argument and a
different worktree path and recovery branch name.

## Integration and retirement rule

Start every integration unit from the latest `origin/main`. Bring over only the
scoped code and documentation, keep evidence status explicit, run focused and
full verification, and review the actual diff before updating `main`.

Before deleting a branch, verify its archive tag, compare any working checkout,
confirm that no job or worktree depends on the branch name, and obtain explicit
approval for the exact remote deletion.
