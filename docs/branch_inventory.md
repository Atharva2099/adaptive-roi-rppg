# Branch and rollback inventory

Last reviewed: 2026-09-05

`main` is the canonical supported code line. A file being present on `main` does
not make its scientific result verified. Claim status remains controlled by
`docs/evidence_registry.md` and the associated row-level evidence.

## Protected starting points

| Purpose | Commit | Remote reference | Meaning |
| --- | --- | --- | --- |
| Canonical state before integration | `c80122c449f02b41e9be1549841e022189a5bd34` | `rollback/pre-canonical-integration-20260904` | Reviewed lean `main`; 136 local tests passed before the tag was created. |
| Current-main rollback | `7cb9fcfdfb2918f351f31101d05801e9587f1949` | `rollback/pre-main-consolidation-20260905` | Protects `main` immediately before the 2026-09-05 consolidation. |
| Dirty-work preservation snapshot | `990b037` | `codex/pre-canonical-integration-snapshot-20260904` and `rollback/pre-canonical-integration-wip-20260904` | Preservation-only snapshot of MCD diagnostics and MMPD extractor engineering. It is not accepted science. |

The WIP snapshot excludes `.claude_session`, ignored caches and temporary
files, datasets, checkpoints, models, generated outputs, and actual MMPD
material. Do not merge the snapshot branch wholesale.

## Branch status

| Branch | Status | Canonical handling |
| --- | --- | --- |
| `main` | Consolidated | One supported code line containing the reviewed selected integration. |
| `codex/verification-prune` | Integrated | Its reviewed patch is in `main` as `c80122c`; the commit hash differs because it was applied on top of the latest `origin/main`. |
| `codex/pre-canonical-integration-snapshot-20260904` | WIP preservation | Source for selective recovery only. No scientific claim is accepted by this snapshot. |
| `codex/gate9-dirty-snapshot-20260821` | Superseded snapshot | Preserve by tag until the newer WIP snapshot has been compared; never merge wholesale. |
| `codex/gate9-minimal-rebuild` | Selectively integrated, incomplete | MMPD evaluation-only engineering reference in `main`; `p29_3`, authenticated inputs, and end-to-end scientific evaluation remain unresolved. |
| `codex/mcd-failure-audit` | Selectively integrated, INCOMPLETE / NO-GO | Compact MCD-only failure-audit primitive retained; dormant runner/launcher omitted. No audit result is accepted; causal replay, factual parity, and provenance remain unresolved. |
| `codex/mcd-repeated-beam-smoke` | Selectively integrated | `main` retains the full-cohort repeated-beam path; the worst-54 prototype is excluded. |
| `codex/mcd-eval-oracle-test` | Selectively integrated | `main` retains the held-out Oracle runner; six untracked companion files remain preserved in the original worktree and `/tmp`, excluded from the consolidation. |
| `codex/mmpd-official-extractor-review` | Documentation selectively recovered | Historical POS/presentation documents were recovered only; the old extractor harness is excluded. |
| `codex/agent-communication-20260905` | Integrated baseline | Its communication guidance is in `origin/main`. |
| `codex/agents-main` | Integrated baseline | Its debugging/evidence guidance is already in the canonical candidate ancestry. |
| `codex/main-integration` | Alias | Points to the protected pre-integration baseline. |
| `codex/canonical-integration` | Reviewed integration line | Same selected code as consolidated `main`. |

## Current integration status

The selected consolidation retains the Gate 9 engineering path, held-out MCD
Oracle runner, full-cohort repeated-beam diagnostic, and documented historical
diagnostics. The MMPD path remains evaluation-only and incomplete. The
worst-54 repeated-beam prototype, dormant failure-audit launcher, duplicate
official-extractor harness, and case-audit configuration are excluded. The
full local suite passed 264 tests and an independent Terra review passed; no
new Slurm or data run was performed.

## Safe recovery

`/private/tmp/adaptive-roi-integration-preservation-20260905` is an ordinary
12-file backup, not a Git worktree. Continue the synchronized candidate for
normal integration. For recovery, create a new worktree; do not reset or
overwrite an existing dirty checkout.

```bash
git fetch origin --tags
git worktree add -b recovery/pre-integration /tmp/adaptive-roi-recovery rollback/pre-canonical-integration-20260904
```

To inspect the preserved WIP instead, use
`rollback/pre-canonical-integration-wip-20260904` as the final argument and a
different worktree path and recovery branch name.

## Integration and retirement rule

Continue the existing synchronized integration branch, incorporating the latest
`origin/main` before final review. Bring over only scoped code and
documentation, keep evidence status explicit, and review the actual diff.

Before deleting a branch, verify its archive tag, compare any working checkout,
confirm that no job or worktree depends on the branch name, and obtain explicit
approval for the exact remote deletion.
