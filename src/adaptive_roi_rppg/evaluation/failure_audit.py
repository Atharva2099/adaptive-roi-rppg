"""One-hop, GT-free counterfactuals for auditing MCD controller choices.

This module evaluates alternatives from one immutable pre-hop state.  It is
deliberately limited to immediate post-belief consequences: it does not claim
sequential regret, visual extraction failure, or causality outside this
deterministic evaluator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import ControlState, ControlTransition, MIN_HOLD, control_step


ACTION_COUNT = len(ROI_NAMES)


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _finite_gt(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail("gt_hr_bpm: must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class UnscoredCounterfactual:
    """One requested action and its actual transition, with no GT surface."""

    requested_action: int
    proposed_action: int
    executed_action: int
    transition: ControlTransition
    post_belief_hr_bpm: float
    selected_measurement: ROIMeasurement

    @property
    def selected_valid(self) -> bool:
        return self.selected_measurement.valid


@dataclass(frozen=True, slots=True)
class UnscoredFailureAudit:
    """All immediate alternatives evaluated from the same pre-hop state."""

    proposed_action: int
    selected: UnscoredCounterfactual
    alternatives: tuple[UnscoredCounterfactual, ...]

    def __post_init__(self) -> None:
        if len(self.alternatives) != ACTION_COUNT:
            _fail("failure audit: alternatives must contain exactly 12 actions")
        if tuple(item.requested_action for item in self.alternatives) != tuple(range(ACTION_COUNT)):
            _fail("failure audit: alternatives must use canonical action order")
        if self.proposed_action != self.selected.proposed_action:
            _fail("failure audit: selected proposed action mismatch")


@dataclass(frozen=True, slots=True)
class ScoredCounterfactual:
    requested_action: int
    proposed_action: int
    executed_action: int
    selected_valid: bool
    post_belief_hr_bpm: float
    absolute_error_bpm: float


@dataclass(frozen=True, slots=True)
class ScoredFailureAudit:
    """GT-joined immediate scores; this is the first structure containing GT."""

    gt_hr_bpm: float
    selected: ScoredCounterfactual
    alternatives: tuple[ScoredCounterfactual, ...]
    category: str
    best_valid_actions: tuple[int, ...]
    best_valid_error_bpm: float | None
    policy_minus_best_immediate_regret_bpm: float | None


def _counterfactual(frame: MeasurementFrame, state: ControlState, requested_action: int) -> UnscoredCounterfactual:
    next_state, transition = control_step(frame, state, requested_action)
    # Keep next_state alive through construction so the transition is visibly
    # produced by the canonical control path; alternatives never chain it.
    del next_state
    return UnscoredCounterfactual(
        requested_action=requested_action,
        proposed_action=transition.action_decision.proposed_action,
        executed_action=transition.action_decision.executed_action,
        transition=transition,
        post_belief_hr_bpm=transition.post_belief.mean_hr,
        selected_measurement=transition.selected_measurement,
    )


def build_failure_audit(frame: MeasurementFrame, state: ControlState, proposed_action: int) -> UnscoredFailureAudit:
    """Build one policy transition plus 12 same-pre-state alternatives.

    The policy transition is run once.  Each alternative is independently
    passed the original ``state`` and ``frame``; no alternative state is fed
    into another alternative.
    """
    if not isinstance(frame, MeasurementFrame) or not isinstance(state, ControlState):
        _fail("failure audit: frame and state types are required")
    selected = _counterfactual(frame, state, proposed_action)
    alternatives = tuple(_counterfactual(frame, state, action) for action in range(ACTION_COUNT))
    return UnscoredFailureAudit(int(proposed_action), selected, alternatives)


def score_failure_audit(audit: UnscoredFailureAudit, gt_hr_bpm: Any) -> ScoredFailureAudit:
    """Join one finite GT value and score immediate post-belief errors."""
    if not isinstance(audit, UnscoredFailureAudit):
        _fail("failure audit: unscored audit is required")
    gt = _finite_gt(gt_hr_bpm)

    def score(item: UnscoredCounterfactual) -> ScoredCounterfactual:
        return ScoredCounterfactual(
            item.requested_action, item.proposed_action, item.executed_action,
            item.selected_valid, item.post_belief_hr_bpm,
            abs(item.post_belief_hr_bpm - gt),
        )

    selected = score(audit.selected)
    alternatives = tuple(score(item) for item in audit.alternatives)
    valid = tuple(item for item in alternatives if item.selected_valid)
    if not valid:
        return ScoredFailureAudit(gt, selected, alternatives, "no_valid_immediate_action", (), None, None)
    best_error = min(item.absolute_error_bpm for item in valid)
    best_actions = tuple(item.requested_action for item in valid if item.absolute_error_bpm == best_error)
    category = "ok" if selected.selected_valid else "selected_measurement_invalid"
    # For a selected-invalid policy action, this is still the error of the
    # canonical predict-only transition. It is useful as an immediate regret
    # diagnostic, but is not evidence of a visual or extraction failure.
    return ScoredFailureAudit(gt, selected, alternatives, category, best_actions, best_error,
                              selected.absolute_error_bpm - best_error)


def random_legal_requested_action(state: ControlState, seed: int) -> int:
    """Return a deterministic random requested action legal under this state.

    This is only a one-hop selector for a negative-control primitive; it is
    not a rollout of a full random policy trajectory.
    """
    if not isinstance(state, ControlState) or isinstance(seed, bool) or not isinstance(seed, int):
        _fail("random selector: state and integer seed are required")
    previous = state.previous_action
    legal = list(range(ACTION_COUNT)) if previous is None or state.hold_count >= MIN_HOLD else [previous]
    return random.Random(seed).choice(legal)


__all__ = [
    "ACTION_COUNT", "ScoredCounterfactual", "ScoredFailureAudit", "UnscoredCounterfactual",
    "UnscoredFailureAudit", "build_failure_audit", "random_legal_requested_action", "score_failure_audit",
]
