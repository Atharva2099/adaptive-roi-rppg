# Current status — 2026-08-10

## Repository status

**FACT:** The new repository is at the foundation and migration stage. This package adds documentation only. There is no new implementation or new result in this repository.

## Current evaluation ruler

**FACT:** The active local ruler is MCD test90, schema-v3 with the no-subharmonic rule, causal streaming windows of 8 seconds and 1-second hops, frozen-belief mean, and equal per-clip aggregation. The evaluated cohort is 533 clips from 89 subjects. Full-face level A is **12.3865 BPM** and Oracle-C K=8 level C is **2.1084 BPM**; provenance is E-004 and E-005.

## Method definitions

**DAgger base** means an LSTM policy/controller trained with initial cross-entropy cloning of Oracle-C, followed by two self-rollout rounds with greedy one-step level-B relabeling and refitting; it starts from random/from-scratch initialization, uses one model per seed with no deployment ensemble or fallback gate, and collects imitation data interactively before frozen self-rollout evaluation.

**Advantage PPO** means an LSTM policy/controller trained with online PPO and advantage-over-base reward shaping; it is warm-started from the matching DAgger base, uses one model per seed rather than a deployment ensemble or fallback gate, and trains on-policy on MCD before frozen self-rollout evaluation.

**Standard PPO** means an LSTM policy/controller trained with the conventional online PPO objective after DAgger; it is warm-started from its matching DAgger base, uses one model per seed with no deployment ensemble or fallback gate, and trains on-policy on MCD before frozen evaluation. Its MMPD transfer rows are remote-only (E-013).

**Oracle-C** is not a learned model. It is a GT-informed offline beam-search diagnostic; K=8 is a frozen historical reference, not a deployable, globally optimal, or inference-time policy (E-005).

## Current local result

| Method | Seed 0 | Seed 1 | Seed 2 | Three-seed mean |
|---|---:|---:|---:|---:|
| **DAgger base** | **7.6044** | **7.9429** | **8.7998** | **8.1157 BPM** |
| **Advantage PPO** | **7.3318** | **7.2135** | **7.5792** | **7.3748 BPM** |

**FACT:** Base minus PPO is positive on every seed: **0.2726**, **0.7293**, and **1.2206 BPM**, with mean **0.7409 BPM** and paired subject-block 95% interval **[0.5364, 0.9453] BPM** (E-001–E-003). Three seeds measure variability; they are not a deployment ensemble.

## What PPO appears to change

**FACT diagnostic:** Same-state replay found **69.2252%** disagreement across **178,446** decision hops; pooled rank 1 was **31.1635%** and rank 12 was **18.5670%** (E-006). **LIMITATION:** This is behavior evidence, not causal proof of PPO’s mechanism.

## Critic/reward status

**NEGATIVE RESULT:** The audited critic-gap signal added essentially no cross-validated recovery prediction beyond baseline features across the audited seeds (E-007). **LIMITATION:** This closes only that audited explanation; it does not rule out all critics or PPO formulations.

## Oracle-C status

**FACT with LIMITATION:** K=8 is the frozen historical GT-informed reference. A wider offline search improved through K=2048 on the same 533 test90 clips, so K=8 is not globally optimal (E-005). **NEGATIVE RESULT:** The selected wider-search teacher failed the predeclared clone transfer gate against K=8; stop before DAgger or PPO on that teacher line (E-009).

## Cross-dataset status

**REMOTE-ONLY:** Official POS MMPD is documented as 300 clips and **15.8174 BPM** MAE, but its per-clip CSV remains on Polaris and was not locally recomputed (E-010). **REMOTE-ONLY and INCOMPLETE locally:** Unchanged frozen-controller predictions rescored under the MMPD-specific corrected-GT rule were compared with TS-CAN; no value is locally verified (E-011). These are separate rulers, and official POS versus causal POS is not a pure smoothing ablation because crop, filtering, windows, and HR aggregation differ (E-012). MMPD cannot guide MCD decisions.

## Work-state table

| State | Area | Evidence |
|---|---|---|
| Implemented | Migration documents only | Current repository state |
| Locally verified | MCD schema-v3 base, PPO, paired uncertainty, levels A/C | E-001–E-005 |
| Diagnostic / negative | Same-state action behavior; audited critic explanation | E-006–E-007 |
| Superseded | Schema-v2 subharmonic-on headline | E-008 |
| Incomplete / remote-only | MMPD official, frozen-controller/TS-CAN, standard PPO transfer | E-010–E-013 |
| Historical context | Golden/fitted-Q line | E-014 |
| Provenance with limit | MCD train/test separation and checkpoint identity | E-015 |
| Not started | New implementation and training in this repository | Repository status |

## Next decision

**INCOMPLETE:** Do not train. First approve the system design, then rebuild data and signal validation and reproduce the MCD ruler before changing PPO.
