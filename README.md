# Adaptive Semantic ROI Control for Causal rPPG

This project studies whether a lightweight controller can adaptively select semantic regions of a face for causal classical remote photoplethysmography (rPPG). Reinforcement learning (RL) is the research method being investigated, not a guaranteed result: the project may show that RL works, that it works only under specific conditions, or that another controller is better supported by the evidence.

## Current phase

See the [current project status](docs/current_status.md), the [Gate 2 adapter note](docs/gates/gate_02_mcd_manifest_adapter.md), and the [Gate 3 causal POS note](docs/gates/gate_03_causal_pos.md).

## Research question

Can a lightweight adaptive controller improve fixed or full-face classical rPPG extraction, while generalizing as a frozen controller trained only on MCD and evaluated on a separate dataset?

The hard dataset boundary is:

- MCD is used for training and development.
- MMPD is used only for frozen evaluation.
- MMPD labels, metadata, subjects, conditions, and evaluation outcomes must never feed back into design, tuning, checkpoint selection, or training decisions.

## Historical reference

The legacy repository at `/Users/atharva/Desktop/RL for RoI`, at commit `28ac407`, is historical evidence and an implementation reference. It is not a code dependency. We may copy knowledge and documented lessons from it, but we will not copy legacy code, scripts, configurations, checkpoints, datasets, result directories, or old documentation wholesale.

## Planned structure

```text
adaptive-roi-rppg/
├── README.md
├── AGENTS.md
├── docs/          # approved system design, interfaces, protocols, and evidence
├── src/           # production implementation, after design approval
├── scripts/       # small launchers and reproducible utilities
├── slurm/         # approved cluster launchers, if needed later
└── tests/         # tests and known-answer smoke checks
```
