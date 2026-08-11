# Current status — 2026-08-10

## State

The repository is in documentation and migration stage. No implementation, training, or new result has been added.

MCD remains the only training and development dataset. MMPD is evaluation-only for frozen MCD checkpoints. MMPD findings cannot guide MCD training, feature, reward, checkpoint, hyperparameter, or stopping decisions.

## Frozen MCD ruler

Schema-v3 no-subharmonic test90 uses 533 clips from 89 subjects, 8-second trailing windows, 1-second hops, post-update belief-mean error, and equal per-clip aggregation. Full-face is **12.3865 BPM** and Oracle-C K=8 is **2.1084 BPM**. See E-004 and E-005.

## Current result

| Method | Seed 0 | Seed 1 | Seed 2 | Three-seed mean |
|---|---:|---:|---:|---:|
| **DAgger base** | **7.6044** | **7.9429** | **8.7998** | **8.1157 BPM** |
| **Advantage PPO** | **7.3318** | **7.2135** | **7.5792** | **7.3748 BPM** |

Base minus PPO is positive on every seed: **0.2726**, **0.7293**, and **1.2206 BPM**, with mean **0.7409 BPM** and paired subject-block 95% interval **[0.5364, 0.9453] BPM**. Three seeds measure variability; they are not a deployment ensemble. Detailed numerical evidence is in E-001–E-003.

## Transfer status

Official POS and unchanged frozen-controller predictions rescored under the MMPD-specific corrected-GT rule were directly verified on Polaris at the stated host and UTC time. Official POS is **15.8174 BPM** over 300 clips. Causal-controller belief means are Full-face **19.2959**, Oracle-C **4.3396**, DAgger **14.7944**, Standard PPO **14.8840**, and Advantage PPO **14.5870 BPM**. These are transfer diagnostics only and cannot guide MCD training. See E-010, E-011, and E-013.

Official POS and causal-controller rulers are not a smoothing ablation: crop/orchestration, windows, filtering, and HR aggregation differ.

## What is incomplete

- Paired subject-block uncertainty for the MMPD transfer comparisons is missing.
- Predeclared failure thresholds remain `TBD` and cannot be chosen from MMPD outcomes.
- Historical full-clip interpolation can look forward when values are missing; impact is unknown. See E-016.
- The exact legacy DAgger entrypoint is **Unknown**.

## Next action

After these documents are approved, implement contracts and MCD-only causal data/signal parity checks. Do not train PPO and do not implement the MMPD adapter first. Detailed evidence remains in E-010–E-013 and E-016; this status is not a second evidence registry.
