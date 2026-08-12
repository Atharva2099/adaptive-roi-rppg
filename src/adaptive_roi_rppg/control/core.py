from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement, canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError

_DT, _Q, _R0, _THRESHOLD, _MIN_HOLD = 1.0, 0.1, 200.0, 0.2, 2
_ACTION_COUNT = len(ROI_NAMES)
_FIELDS = tuple(f"roi_{roi.value}_{field}" for roi in ROI_NAMES for field in ("hr", "hr_delta", "confidence", "ppr", "coverage", "agreement", "valid")) + ("belief_mean", "belief_std", "belief_velocity", "hops_since_confident",) + tuple(f"previous_action_{i}" for i in range(_ACTION_COUNT)) + ("hold_count",)
OBSERVATION_SCHEMA_ID = "observation-v1-101"
FIELD_NAMES = _FIELDS
_PAYLOAD = {
    "version": 1, "schema_id": OBSERVATION_SCHEMA_ID, "field_names": list(FIELD_NAMES),
    "roi_order": [roi.value for roi in ROI_NAMES], "action_count": _ACTION_COUNT,
    "belief": {"dt": _DT, "q": _Q, "r0": _R0, "initial_mean_hr": 70.0, "initial_velocity": 0.0, "initial_covariance": [[400.0, 0.0], [0.0, 25.0]], "threshold": _THRESHOLD},
    "transition": {"process_matrix": "[[1,dt],[0,1]]", "process_noise": "q*[[dt^3/3,dt^2/2],[dt^2/2,dt]]", "order": "predict_then_update", "update": "finite HR and finite confidence > 0; R=r0/max(confidence,1e-6); literal (I-KH)P"},
    "observation": {"normalization": {"hr": {"center": 90.0, "scale": 40.0}, "velocity": 10.0}, "clipping": {"confidence": [0.0, 1.0], "ppr": [0.0, 1.0], "coverage": [0.0, 1.0], "agreement": [0.0, 1.0], "hold": [0.0, 10.0], "hops": [0.0, 20.0]}, "agreement": "other valid measurements only; inclusive <=5 BPM; denominator other valid count", "invalid": "HR slots use current belief mean; confidence and ppr zero; finite coverage preserved and clipped otherwise zero; agreement and valid follow measurement", "timing": "pre-action belief, previous executed action, pre-action hold"},
    "actions": {"minimum_hold": _MIN_HOLD, "first_action": "free; hold=1", "stay": "increment", "switch": "legal at pre-hold >= 2; post-hold=1", "illegal": "minimum_hold override only"},
    "records": {"state_fields": ["dataset_id", "clip_id", "next_hop_idx", "belief", "previous_action", "hold_count"], "transition_fields": ["dataset_id", "clip_id", "hop_idx", "frame_provenance_id", "observation", "action_decision", "selected_measurement", "pre_belief", "post_belief", "signal_config_id", "control_config_id"]},
}

def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list): return tuple(_freeze(v) for v in value)
    return value

CONTROL_CONFIG_PAYLOAD = _freeze(_PAYLOAD)
CONTROL_CONFIG_ID = "control-v1-" + hashlib.sha256(canonical_json_bytes(_PAYLOAD)).hexdigest()

def _fail(message: str): raise ContractValidationError(message)
def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): _fail(f"{name}: must be finite")
    return float(value)
def _action(value, name: str = "action") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < _ACTION_COUNT: _fail(f"{name}: must be an integer in [0,11]")
    return value

@dataclass(frozen=True, slots=True)
class BeliefState:
    mean_hr: float
    velocity: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    hops_since_confident: int
    def __post_init__(self):
        _finite(self.mean_hr, "mean_hr"); _finite(self.velocity, "velocity")
        if not isinstance(self.covariance, (tuple, list)) or len(self.covariance) != 2 or any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in self.covariance): _fail("covariance: must be 2x2")
        matrix = tuple(tuple(_finite(v, "covariance") for v in row) for row in self.covariance)
        if abs(matrix[0][1] - matrix[1][0]) > 1e-10 or matrix[0][0] < 0 or matrix[1][1] < 0: _fail("covariance: must be symmetric with nonnegative diagonal")
        symmetric = np.asarray(matrix, dtype=np.float64); symmetric = (symmetric + symmetric.T) / 2.0
        scale = max(1.0, float(np.max(np.abs(symmetric))))
        if float(np.min(np.linalg.eigvalsh(symmetric))) < -1e-12 * scale: _fail("covariance: must be positive semidefinite")
        if isinstance(self.hops_since_confident, bool) or not isinstance(self.hops_since_confident, int) or self.hops_since_confident < 0: _fail("hops_since_confident: must be a nonnegative integer")
        object.__setattr__(self, "mean_hr", float(self.mean_hr)); object.__setattr__(self, "velocity", float(self.velocity)); object.__setattr__(self, "covariance", matrix)
    @property
    def covariance_array(self) -> np.ndarray:
        return np.array(self.covariance, dtype=np.float64, copy=True)

@dataclass(frozen=True, slots=True)
class ControlState:
    dataset_id: str
    clip_id: str
    next_hop_idx: int
    belief: BeliefState
    previous_action: int | None
    hold_count: int
    def __post_init__(self):
        if not isinstance(self.dataset_id, str) or not self.dataset_id or not isinstance(self.clip_id, str) or not self.clip_id: _fail("state identity: nonempty strings required")
        if isinstance(self.next_hop_idx, bool) or not isinstance(self.next_hop_idx, int) or self.next_hop_idx < 0: _fail("next_hop_idx: must be nonnegative integer")
        if not isinstance(self.belief, BeliefState): _fail("belief: must be BeliefState")
        if self.previous_action is not None: _action(self.previous_action, "previous_action")
        if isinstance(self.hold_count, bool) or not isinstance(self.hold_count, int) or self.hold_count < 0 or ((self.previous_action is None) != (self.hold_count == 0)): _fail("previous_action and hold_count: inconsistent")

@dataclass(frozen=True, slots=True)
class ControllerObservation:
    schema_id: str
    config_id: str
    field_names: tuple[str, ...]
    values: tuple[float, ...]
    def __post_init__(self):
        if self.schema_id != OBSERVATION_SCHEMA_ID or self.config_id != CONTROL_CONFIG_ID or tuple(self.field_names) != FIELD_NAMES or len(self.values) != 101: _fail("observation: schema/config/field mismatch")
        values = tuple(_finite(v, "observation") for v in self.values)
        object.__setattr__(self, "field_names", FIELD_NAMES); object.__setattr__(self, "values", values)
    def array(self) -> np.ndarray: return np.asarray(self.values, dtype=np.float32).copy()

@dataclass(frozen=True, slots=True)
class ActionDecision:
    proposed_action: int
    executed_action: int
    previous_action: int | None
    pre_hold_count: int
    post_hold_count: int
    legal: bool
    override_reason: str | None
    def __post_init__(self):
        _action(self.proposed_action, "proposed_action"); _action(self.executed_action, "executed_action")
        if self.previous_action is not None: _action(self.previous_action, "previous_action")
        if isinstance(self.legal, bool) is False or isinstance(self.pre_hold_count, bool) or not isinstance(self.pre_hold_count, int) or isinstance(self.post_hold_count, bool) or not isinstance(self.post_hold_count, int) or self.pre_hold_count < 0 or self.post_hold_count < 1: _fail("decision fields: invalid")
        if self.previous_action is None:
            valid = self.pre_hold_count == 0 and self.proposed_action == self.executed_action and self.post_hold_count == 1 and self.legal and self.override_reason is None
        elif self.proposed_action == self.previous_action:
            valid = self.pre_hold_count >= 1 and self.executed_action == self.previous_action and self.post_hold_count == self.pre_hold_count + 1 and self.legal and self.override_reason is None
        elif self.pre_hold_count < _MIN_HOLD:
            valid = 1 <= self.pre_hold_count < _MIN_HOLD and self.executed_action == self.previous_action and self.post_hold_count == self.pre_hold_count + 1 and not self.legal and self.override_reason == "minimum_hold"
        else:
            valid = self.pre_hold_count >= _MIN_HOLD and self.executed_action == self.proposed_action and self.executed_action != self.previous_action and self.post_hold_count == 1 and self.legal and self.override_reason is None
        if not valid: _fail("decision: impossible action/hold combination")

@dataclass(frozen=True, slots=True)
class ControlTransition:
    dataset_id: str
    clip_id: str
    hop_idx: int
    observation: ControllerObservation
    action_decision: ActionDecision
    selected_measurement: ROIMeasurement
    pre_belief: BeliefState
    post_belief: BeliefState
    frame_provenance_id: str
    signal_config_id: str
    control_config_id: str
    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in (self.dataset_id, self.clip_id, self.frame_provenance_id, self.signal_config_id)): _fail("transition provenance: nonempty strings required")
        if isinstance(self.hop_idx, bool) or not isinstance(self.hop_idx, int) or self.hop_idx < 0: _fail("transition identity: invalid")
        if not isinstance(self.observation, ControllerObservation) or not isinstance(self.action_decision, ActionDecision) or not isinstance(self.selected_measurement, ROIMeasurement) or not isinstance(self.pre_belief, BeliefState) or not isinstance(self.post_belief, BeliefState): _fail("transition records: invalid")
        if self.observation.schema_id != OBSERVATION_SCHEMA_ID or self.observation.config_id != CONTROL_CONFIG_ID or self.selected_measurement.roi_index != self.action_decision.executed_action or self.selected_measurement.signal_config_id != self.signal_config_id or self.control_config_id != CONTROL_CONFIG_ID: _fail("transition provenance: invalid")

def initial_control_state(dataset_id: str, clip_id: str) -> ControlState:
    return ControlState(dataset_id, clip_id, 0, BeliefState(70.0, 0.0, ((400.0, 0.0), (0.0, 25.0)), 0), None, 0)

def belief_step(state: BeliefState, measurement_hr, confidence) -> BeliefState:
    if not isinstance(state, BeliefState): _fail("state: must be BeliefState")
    x = np.array([state.mean_hr, state.velocity], dtype=np.float64); p = state.covariance_array
    f = np.array(((1.0, _DT), (0.0, 1.0))); q = _Q * np.array(((_DT**3 / 3, _DT**2 / 2), (_DT**2 / 2, _DT)), dtype=np.float64)
    x = f @ x; p = f @ p @ f.T + q
    valid = isinstance(measurement_hr, (int, float)) and not isinstance(measurement_hr, bool) and math.isfinite(measurement_hr) and isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and confidence > 0
    if valid:
        r = _R0 / max(float(confidence), 1e-6); h = np.array((1.0, 0.0)); y = float(measurement_hr) - h @ x; s = float(h @ p @ h.T + r); k = p @ h / s; x = x + k * y; p = (np.eye(2) - np.outer(k, h)) @ p
    return BeliefState(float(x[0]), float(x[1]), tuple(tuple(float(v) for v in row) for row in p), 0 if valid and confidence >= _THRESHOLD else state.hops_since_confident + 1)

def build_observation(frame: MeasurementFrame, state: ControlState) -> ControllerObservation:
    if not isinstance(frame, MeasurementFrame) or frame.dataset_id != state.dataset_id or frame.clip_id != state.clip_id or frame.hop_idx != state.next_hop_idx: _fail("frame: identity or hop does not match state")
    values = []
    valid_measurements = [m for m in frame.measurements if m.valid and m.hr_bpm is not None]
    for measurement in frame.measurements:
        hr = measurement.hr_bpm if measurement.valid else state.belief.mean_hr
        confidence = measurement.confidence if measurement.valid else 0.0; ppr = measurement.peak_power_ratio if measurement.valid else 0.0
        coverage = measurement.coverage if measurement.coverage is not None and math.isfinite(measurement.coverage) else 0.0
        coverage = min(max(float(coverage), 0.0), 1.0); others = [m for m in valid_measurements if m.roi_index != measurement.roi_index]
        agreement = sum(abs(m.hr_bpm - hr) <= 5.0 for m in others) / len(others) if measurement.valid and others else 0.0
        values.extend(((float(hr) - 90.0) / 40.0, (float(hr) - state.belief.mean_hr) / 40.0, min(max(float(confidence), 0.0), 1.0), min(max(float(ppr), 0.0), 1.0), coverage, agreement, float(measurement.valid)))
    values.extend(((state.belief.mean_hr - 90.0) / 40.0, math.sqrt(max(state.belief.covariance[0][0], 0.0)) / 40.0, state.belief.velocity / 10.0, min(state.belief.hops_since_confident, 20) / 20.0))
    values.extend(float(state.previous_action == i) for i in range(_ACTION_COUNT)); values.append(min(state.hold_count, 10) / 10.0)
    return ControllerObservation(OBSERVATION_SCHEMA_ID, CONTROL_CONFIG_ID, FIELD_NAMES, tuple(values))

def resolve_action(proposed_action: int, state: ControlState) -> ActionDecision:
    _action(proposed_action, "proposed_action"); previous = state.previous_action
    if previous is None: return ActionDecision(proposed_action, proposed_action, None, 0, 1, True, None)
    if proposed_action == previous: return ActionDecision(proposed_action, previous, previous, state.hold_count, state.hold_count + 1, True, None)
    if state.hold_count < _MIN_HOLD: return ActionDecision(proposed_action, previous, previous, state.hold_count, state.hold_count + 1, False, "minimum_hold")
    return ActionDecision(proposed_action, proposed_action, previous, state.hold_count, 1, True, None)

def control_step(frame: MeasurementFrame, state: ControlState, proposed_action: int):
    observation = build_observation(frame, state); decision = resolve_action(proposed_action, state); selected = frame.measurements[decision.executed_action]
    post = belief_step(state.belief, selected.hr_bpm if selected.valid else None, selected.confidence if selected.valid else None)
    new_state = ControlState(state.dataset_id, state.clip_id, state.next_hop_idx + 1, post, decision.executed_action, decision.post_hold_count)
    transition = ControlTransition(state.dataset_id, state.clip_id, frame.hop_idx, observation, decision, selected, state.belief, post, frame.provenance_id, frame.signal_config_id, CONTROL_CONFIG_ID)
    return new_state, transition

__all__ = ["ActionDecision", "BeliefState", "CONTROL_CONFIG_ID", "CONTROL_CONFIG_PAYLOAD", "ControlState", "ControlTransition", "ControllerObservation", "FIELD_NAMES", "OBSERVATION_SCHEMA_ID", "belief_step", "build_observation", "control_step", "initial_control_state", "resolve_action"]
