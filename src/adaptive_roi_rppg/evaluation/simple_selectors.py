"""Ground-truth-free, parameter-free MCD selector policies.

The policies in this module choose a *requested* action.  Every request is
then passed through :func:`control_step`, so the canonical minimum-hold rule
remains the only switching constraint.  Labels are intentionally absent from
the rollout API; they can be joined later by :func:`score_simple_rollout`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import numpy as np

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import ControlState, ControlTransition, control_step, initial_control_state
from adaptive_roi_rppg.evaluation.failure_audit import UnscoredFailureAudit, build_failure_audit, random_legal_requested_action

ACTION_COUNT = len(ROI_NAMES)
DEFAULT_RANDOM_SEED = 8101


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _valid_score(measurement: Any) -> bool:
    return bool(measurement.valid and measurement.peak_power_ratio is not None and math.isfinite(float(measurement.peak_power_ratio)))


class SimpleSelector(Protocol):
    method_id: str
    family: str
    seed: int | None

    def requested_action(self, frame: MeasurementFrame, state: ControlState) -> int: ...


@dataclass(frozen=True, slots=True)
class MaxPprSelector:
    """Select the valid measurement with maximum POS peak-power ratio.

    Exact ties deliberately keep the current executed ROI; when there is no
    current tied ROI, the lowest canonical index wins (ROI 0 is full_face).
    """

    method_id: str = "max_ppr"
    family: str = "simple_selector"
    seed: int | None = None

    def requested_action(self, frame: MeasurementFrame, state: ControlState) -> int:
        if not isinstance(frame, MeasurementFrame) or not isinstance(state, ControlState):
            _fail("max-ppr selector requires a frame and control state")
        candidates = [m for m in frame.measurements if _valid_score(m)]
        if not candidates:
            return state.previous_action if state.previous_action is not None else 0
        maximum = max(float(m.peak_power_ratio) for m in candidates)
        if state.previous_action is not None:
            current = frame.measurements[state.previous_action]
            if _valid_score(current) and float(current.peak_power_ratio) == maximum:
                return state.previous_action
        return min(m.roi_index for m in candidates if float(m.peak_power_ratio) == maximum)


@dataclass(frozen=True, slots=True)
class FixedFullFaceSelector:
    """Regression-control policy that always requests canonical ROI 0."""

    method_id: str = "full_face"
    family: str = "fixed_full_face"
    seed: int | None = None

    def requested_action(self, frame: MeasurementFrame, state: ControlState) -> int:
        if not isinstance(frame, MeasurementFrame) or not isinstance(state, ControlState):
            _fail("full-face selector requires a frame and control state")
        return 0


def _stable_seed(global_seed: int, clip_id: str, hop_idx: int) -> int:
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        _fail("random selector seed must be an integer")
    if not isinstance(clip_id, str) or not clip_id or isinstance(hop_idx, bool) or not isinstance(hop_idx, int) or hop_idx < 0:
        _fail("random selector identity is invalid")
    digest = hashlib.sha256(f"mcd-simple-random-v1\0{global_seed}\0{clip_id}\0{hop_idx}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True, slots=True)
class RandomLegalSelector:
    """Deterministic sequential random policy over actions legal at a hop."""

    global_seed: int = DEFAULT_RANDOM_SEED
    method_id: str = "random_legal"
    family: str = "random_null"
    seed: int | None = DEFAULT_RANDOM_SEED

    def requested_action(self, frame: MeasurementFrame, state: ControlState) -> int:
        if not isinstance(frame, MeasurementFrame) or not isinstance(state, ControlState):
            _fail("random selector requires a frame and control state")
        return random_legal_requested_action(state, _stable_seed(self.global_seed, frame.clip_id, frame.hop_idx))


def max_ppr_requested_action(frame: MeasurementFrame, state: ControlState) -> int:
    """Functional form of :class:`MaxPprSelector` for small callers."""
    return MaxPprSelector().requested_action(frame, state)


def random_sequential_legal_action(frame: MeasurementFrame, state: ControlState, *, seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Functional deterministic random selector with explicit provenance seed."""
    return RandomLegalSelector(seed).requested_action(frame, state)


@dataclass(frozen=True, slots=True)
class SimpleSelectorHop:
    clip_id: str
    hop_idx: int
    hop_time_s: float
    method_id: str
    family: str
    seed: int | None
    checkpoint_sha256: str | None
    observation_values: tuple[float, ...]
    audit: UnscoredFailureAudit
    transition: ControlTransition

    @property
    def alternative_count(self) -> int:
        return len(self.audit.alternatives)


def rollout_simple_selector(frames: Sequence[MeasurementFrame], policy: SimpleSelector, *, clip_id: str) -> tuple[SimpleSelectorHop, ...]:
    """Replay one clip and retain 101-D observations plus 12 unscored alternatives."""
    if not frames:
        _fail("simple selector rollout rejects an empty clip")
    if not isinstance(clip_id, str) or not clip_id:
        _fail("simple selector clip_id is required")
    state = initial_control_state("mcd", clip_id)
    output: list[SimpleSelectorHop] = []
    for expected_hop, frame in enumerate(frames):
        if frame.dataset_id != "mcd" or frame.clip_id != clip_id or frame.hop_idx != expected_hop:
            _fail("simple selector frames are not contiguous MCD frames")
        requested = policy.requested_action(frame, state)
        audit = build_failure_audit(frame, state, requested)
        transition = audit.selected.transition
        values = tuple(transition.observation.array().tolist())
        if len(values) != 101 or not all(math.isfinite(v) for v in values):
            _fail("simple selector observation must be finite 101-D")
        output.append(SimpleSelectorHop(clip_id, frame.hop_idx, frame.hop_time_s,
                                       policy.method_id, policy.family, policy.seed, getattr(policy, "checkpoint_sha256", None),
                                       values, audit, transition))
        state, _ = control_step(frame, state, requested)
    return tuple(output)


def rollout_diagnostic_policy(frames: Sequence[MeasurementFrame], policy: Any, *, clip_id: str) -> tuple[SimpleSelectorHop, ...]:
    """Shared replay for heuristic and recurrent policies.

    Recurrent policies are identified by ``predict``/``initial_state`` and are
    called once per factual hop.  Alternative transitions never receive their
    recurrent state and therefore cannot affect later predictions.
    """
    if hasattr(policy, "requested_action"):
        return rollout_simple_selector(frames, policy, clip_id=clip_id)
    if not frames or not hasattr(policy, "predict") or not hasattr(policy, "initial_state"):
        _fail("diagnostic policy is invalid or frames are empty")
    state = initial_control_state("mcd", clip_id); recurrent = policy.initial_state(); episode_start = True
    output: list[SimpleSelectorHop] = []
    from adaptive_roi_rppg.control import build_observation
    for expected_hop, frame in enumerate(frames):
        if frame.dataset_id != "mcd" or frame.clip_id != clip_id or frame.hop_idx != expected_hop:
            _fail("diagnostic frames are not contiguous MCD frames")
        observation = build_observation(frame, state)
        action, recurrent = policy.predict(observation.array().reshape(1, 101), recurrent, episode_start=episode_start)
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_COUNT:
            _fail("diagnostic learned policy action is invalid")
        audit = build_failure_audit(frame, state, int(action)); transition = audit.selected.transition
        output.append(SimpleSelectorHop(clip_id, frame.hop_idx, frame.hop_time_s,
                                        policy.identity.method_id, policy.identity.family, policy.identity.seed, policy.identity.checkpoint_sha256,
                                        tuple(transition.observation.array().tolist()), audit, transition))
        state, _ = control_step(frame, state, int(action)); episode_start = False
    return tuple(output)


def score_simple_rollout(rollout: Sequence[SimpleSelectorHop], labels: Sequence[Any], metadata: dict[str, str]) -> tuple[dict[str, Any], ...]:
    """Join finite labels after replay and score factual and alternatives."""
    if not rollout or len(rollout) != len(labels):
        _fail("simple selector rollout/label coverage differs")
    if set(metadata) != {"subject_id", "view", "condition"} or any(not isinstance(v, str) or not v for v in metadata.values()):
        _fail("simple selector metadata is invalid")
    rows = []
    for hop, label in zip(rollout, labels):
        if (label.dataset_id, label.clip_id, label.hop_idx, label.hop_time_s) != ("mcd", hop.clip_id, hop.hop_idx, hop.hop_time_s) or not label.valid or label.gt_hr_bpm is None or not math.isfinite(float(label.gt_hr_bpm)):
            _fail("simple selector labels are missing, invalid, or misaligned")
        gt = float(label.gt_hr_bpm)
        factual = hop.audit.selected
        rows.append({**metadata, "dataset_id": "mcd", "clip_id": hop.clip_id, "hop_idx": hop.hop_idx,
                     "hop_time_s": hop.hop_time_s, "method_id": hop.method_id, "family": hop.family,
                     "seed": hop.seed, "checkpoint_sha256": hop.checkpoint_sha256, "gt_hr_bpm": gt,
                     "abs_error_bpm": abs(factual.post_belief_hr_bpm - gt),
                     "observation_values": hop.observation_values,
                     "requested_action": factual.requested_action,
                     "proposed_action": factual.proposed_action,
                     "executed_action": factual.executed_action,
                     "alternative_abs_error_bpm": tuple(abs(item.post_belief_hr_bpm - gt) for item in hop.audit.alternatives)})
    return tuple(rows)


__all__ = ["ACTION_COUNT", "DEFAULT_RANDOM_SEED", "FixedFullFaceSelector", "MaxPprSelector", "RandomLegalSelector", "SimpleSelectorHop", "max_ppr_requested_action", "random_sequential_legal_action", "rollout_diagnostic_policy", "rollout_simple_selector", "score_simple_rollout"]
