"""Gate 9-local causal replay; kept separate from the MCD frozen ruler."""
from __future__ import annotations
import hashlib, math
from typing import Any, Sequence
import numpy as np
from adaptive_roi_rppg.contracts import MeasurementFrame
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID, build_observation, control_step, initial_control_state

def _fail(message: str) -> None:
    raise ContractValidationError("Gate 9 replay: " + message)

def rollout_gate9_policy(frames: Sequence[MeasurementFrame], policy: Any, *, clip_id: str) -> tuple[dict[str, Any], ...]:
    """Run one causal clip without exposing labels to the policy."""
    if not frames:
        _fail("empty clip")
    identity = policy.identity
    if identity.observation_schema_id != OBSERVATION_SCHEMA_ID or identity.observation_dim != 101 or identity.action_count != 12:
        _fail("policy identity does not match the controller contract")
    state, recurrent, episode_start, rows = initial_control_state("mmpd", clip_id), policy.initial_state(), True, []
    for expected_hop, frame in enumerate(frames):
        if not isinstance(frame, MeasurementFrame) or frame.dataset_id != "mmpd" or frame.clip_id != clip_id or frame.hop_idx != expected_hop:
            _fail("frame identity is not contiguous")
        observation = build_observation(frame, state).array()
        if observation.shape != (101,) or observation.dtype != np.float32 or not np.isfinite(observation).all():
            _fail("observation is not finite float32[101]")
        proposed, recurrent = policy.predict(observation.reshape(1, 101), recurrent, episode_start=episode_start)
        if isinstance(proposed, np.ndarray):
            proposed = proposed.reshape(-1)[0].item()
        if isinstance(proposed, bool) or not isinstance(proposed, (int, np.integer)) or not 0 <= int(proposed) < 12:
            _fail("policy proposed action is invalid")
        state, transition = control_step(frame, state, int(proposed))
        decision, measurement, belief = transition.action_decision, transition.selected_measurement, transition.post_belief
        rows.append({
            "method_id": identity.method_id, "family": identity.family, "seed": "" if identity.seed is None else str(identity.seed),
            "checkpoint_sha256": identity.checkpoint_sha256 or "", "clip_id": clip_id, "hop_idx": expected_hop,
            "hop_time_s": frame.hop_time_s, "proposed_action": decision.proposed_action, "executed_action": decision.executed_action,
            "legal": decision.legal, "override_reason": decision.override_reason or "", "pre_hold_count": decision.pre_hold_count,
            "post_hold_count": decision.post_hold_count, "selected_valid": measurement.valid,
            "selected_invalid_reason": measurement.invalid_reason or "", "post_belief_hr_bpm": belief.mean_hr,
            "causal_reset": expected_hop == 0, "gt_observation_count": 0,
        })
        episode_start = False
    return tuple(rows)

__all__ = ["rollout_gate9_policy"]
