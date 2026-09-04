# Gate 10: MCD frozen failure-audit protocol

Status: `INCOMPLETE / NO-GO`. The compact diagnostic primitive is retained, but
no complete MCD audit has been run or accepted. This gate is MCD-only and does
not establish sequential regret, model improvement, or a physiological claim.
MMPD findings cannot drive any decision here.

## Question

For a frozen MCD controller transition, do invalid measurements, a valid
immediately better ROI, or no valid immediate ROI explain the selected action?
The audit evaluates all 12 actions independently from the same pre-hop state,
then joins ground truth only in a separate scoring step.

## Required arms

One frozen MCD controller transition and its 12 same-state action alternatives.
The unscored trace must contain no ground-truth or error fields. Scoring must
use an independently supplied MCD label and preserve the selected versus best
valid immediate comparison.

## Negative control

Use an all-invalid measurement frame. The audit must retain all 12 alternatives,
report no valid immediate action, and must not invent a model or physiological
failure explanation.

## Interpretation limits

This is an immediate one-hop diagnostic. It is not sequential regret, a causal
visual-extraction diagnosis, a deployable policy score, or evidence that a
different model, reward, feature, or tracker would improve results. MMPD is
evaluation-only and must not inform MCD design, tuning, calibration,
checkpoint selection, reward design, feature design, or early stopping.

## Stop/next rule

Stop: do not publish or claim an audit result until a complete MCD-only run has
authenticated its sources, frozen its checkpoint identities, preserved
row-level traces, and reported subject-level uncertainty. Any MCD follow-up
must be justified independently by MCD evidence. If those conditions cannot be
met, retain this gate as `INCOMPLETE / NO-GO`.
