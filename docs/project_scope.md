# Project scope

## Purpose

**FACT:** This project studies adaptive semantic region-of-interest (ROI) selection for causal classical remote photoplethysmography (rPPG). Reinforcement learning (RL) is the method being tested, not a conclusion promised in advance.

## Primary research question

**HYPOTHESIS:** Can a lightweight controller trained on MCD improve full-face or fixed classical extraction and retain value under frozen cross-dataset evaluation?

The claim must be tested with a causal information boundary and a fixed evaluation ruler. A useful result may be positive or negative: the controller may capture useful ROI headroom, or the evidence may show that oracle headroom exists while the tested controller cannot capture it.

## Scientific boundaries

- **FACT:** MCD is the development dataset. Training, feature, reward, checkpoint, and hyperparameter decisions use MCD only.
- **FACT:** MMPD is frozen evaluation-only. MMPD data, labels, metadata, subjects, conditions, and outcomes must not guide MCD design or selection.
- **LIMITATION:** Causal inference uses current and past information only. Future observations and hindsight labels are not controller inputs.
- **LIMITATION:** Cross-dataset results are transfer evidence, not tuning guidance.

## Publication-safe claim ladder

- **FACT:** The strongest current MCD claim is the locally evidenced schema-v3 comparison recorded as E-001 through E-004.
- **INCOMPLETE:** MMPD transfer remains unresolved locally because the relevant row-level packages are remote-only. It cannot support a generalization claim here.
- **NEGATIVE RESULT:** A publishable conclusion can still be that oracle ROI headroom exists but a tested controller fails to capture it, provided the ruler and failure evidence are explicit.

## Primary comparisons

The planned comparison families are full-face and fixed classical references, a learned selector, random or heuristic control, and a GT-informed diagnostic. Deep rPPG models are context, not automatically a fair primary target, because this project does not claim to learn an end-to-end rPPG estimator.

## Non-goals

- **LIMITATION:** No end-to-end learned rPPG estimator claim.
- **LIMITATION:** No MMPD tuning, calibration, checkpoint selection, feature selection, reward design, or early stopping.
- **LIMITATION:** No claim that RL must beat every deep model.
- **HISTORICAL CONTEXT:** No objective to preserve or copy legacy code.

## Decision gates before implementation

**INCOMPLETE:** Before adding `src/`, approve the system interfaces, split and leakage protocol, signal ruler, controller objective, and reproducible evidence package. This migration records knowledge only; it does not approve an implementation architecture.
