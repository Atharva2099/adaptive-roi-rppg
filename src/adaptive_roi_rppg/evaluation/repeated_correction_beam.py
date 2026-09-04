"""MCD-only, GT-informed offline repeated-correction beam diagnostic.

This module is intentionally separate from the one-time intervention arm.  A
factual PPO trajectory is replayed independently and is never counted against
beam capacity.  GT is used only to choose the offline rank-1 headroom child;
neither the controller nor the frozen policy receives a label.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import build_observation, control_step, initial_control_state, resolve_action
from adaptive_roi_rppg.evaluation.long_term_regret import ACTION_COUNT, control_state_digest, observation_digest, recurrent_state_digest
from adaptive_roi_rppg.evaluation.model_replay import FrozenPolicy

WINDOW = 15
PRIMARY_WIDTH = 8
SENSITIVITY_WIDTHS = (1, 2, 4, 8, 16, 32)
SCHEMA = "mcd-repeated-correction-beam-v1"


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _action(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < ACTION_COUNT:
        _fail(message)
    return int(value)


def _frames_and_labels(frames: Sequence[MeasurementFrame], labels: Sequence[LabelFrame], clip_id: str) -> None:
    if not frames or len(frames) != len(labels):
        _fail("beam frames/labels have different coverage")
    for index, (frame, label) in enumerate(zip(frames, labels)):
        if (frame.dataset_id, frame.clip_id, frame.hop_idx) != ("mcd", clip_id, index):
            _fail("beam frames are not contiguous MCD frames")
        if (label.dataset_id, label.clip_id, label.hop_idx) != ("mcd", clip_id, index) or not label.valid or label.gt_hr_bpm is None or not math.isfinite(label.gt_hr_bpm):
            _fail("beam labels are invalid or misaligned")


def legal_actions(state: object) -> tuple[int, ...]:
    """Actions that execute as proposed under the current controller state."""
    return tuple(action for action in range(ACTION_COUNT) if (decision := resolve_action(action, state)).legal and decision.executed_action == action)


def rank1_action(frame: MeasurementFrame, label: LabelFrame, state: object) -> tuple[int, float]:
    """Exhaustively choose the immediate GT-AE minimizer, then action index."""
    candidates = []
    for action in legal_actions(state):
        _next, transition = control_step(frame, state, action)
        candidates.append((abs(transition.post_belief.mean_hr - float(label.gt_hr_bpm)), action))
    if not candidates:
        _fail("beam state has no legal controller action")
    error, action = min(candidates, key=lambda value: (value[0], value[1]))
    return action, error


@dataclass(frozen=True, slots=True)
class FactualSnapshot:
    hop_idx: int
    state: object
    recurrent: object | None
    episode_start: bool
    proposed_action: int
    post_prediction_recurrent: object | None
    executed_action: int
    legal: bool
    override_reason: str | None
    post_belief_hr_bpm: float
    abs_error_bpm: float


@dataclass(frozen=True, slots=True)
class BeamHop:
    anchor_hop_idx: int
    width: int
    path_rank: int
    offset: int
    hop_idx: int
    origin_sequence: tuple[int, ...]
    executed_sequence: tuple[int, ...]
    source: str
    observation_sha256: str
    pre_control_state_sha256: str
    pre_predict_recurrent_sha256: str
    post_predict_recurrent_sha256: str
    ppo_action: int
    legal_actions: tuple[int, ...]
    ppo_legal: bool
    rank1_action: int
    proposed_action: int
    executed_action: int
    legal: bool
    override_reason: str | None
    immediate_abs_error_bpm: float
    cumulative_abs_error_bpm: float
    endpoint_abs_error_bpm: float


@dataclass(frozen=True, slots=True)
class BeamResult:
    anchor_hop_idx: int
    width: int
    factual: tuple[BeamHop, ...]
    retained: tuple[tuple[BeamHop, ...], ...]
    generated: int
    deduped: int
    pruned: int
    depth_stats: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _Node:
    state: object
    recurrent: object | None
    origin_sequence: tuple[int, ...]
    executed_sequence: tuple[int, ...]
    hops: tuple[BeamHop, ...]
    cumulative: float
    endpoint: float

    def key(self) -> tuple[float, float, tuple[int, ...], tuple[int, ...]]:
        return (self.cumulative, self.endpoint, self.executed_sequence, self.origin_sequence)


def factual_snapshots(frames: Sequence[MeasurementFrame], labels: Sequence[LabelFrame], policy: FrozenPolicy, *, clip_id: str) -> tuple[FactualSnapshot, ...]:
    """Replay factual PPO once; these snapshots are pinned outside beam capacity."""
    _frames_and_labels(frames, labels, clip_id)
    state, recurrent, episode_start, result = initial_control_state("mcd", clip_id), policy.initial_state(), True, []
    for frame, label in zip(frames, labels):
        values = build_observation(frame, state).array()
        proposed, post_recurrent = policy.predict(values.reshape(1, 101), recurrent, episode_start=episode_start)
        proposed = _action(proposed, "beam factual PPO action is invalid")
        next_state, transition = control_step(frame, state, proposed)
        result.append(FactualSnapshot(frame.hop_idx, state, copy.deepcopy(recurrent), episode_start, proposed, copy.deepcopy(post_recurrent),
            transition.action_decision.executed_action, transition.action_decision.legal, transition.action_decision.override_reason,
            transition.post_belief.mean_hr, abs(transition.post_belief.mean_hr - float(label.gt_hr_bpm))))
        state, recurrent, episode_start = next_state, post_recurrent, False
    return tuple(result)


def eligible_anchors(snapshots: Sequence[FactualSnapshot], *, window: int = WINDOW) -> tuple[int, ...]:
    if window < 1:
        _fail("beam window must be positive")
    return tuple(item.hop_idx for item in snapshots if len(snapshots) - item.hop_idx >= window and len(legal_actions(item.state)) == ACTION_COUNT)


def _factual_window(frames: Sequence[MeasurementFrame], labels: Sequence[LabelFrame], snapshots: Sequence[FactualSnapshot], *, anchor: int, width: int, window: int) -> tuple[BeamHop, ...]:
    output = []
    cumulative = 0.0
    for offset in range(window):
        item, frame = snapshots[anchor + offset], frames[anchor + offset]
        cumulative += item.abs_error_bpm
        output.append(BeamHop(anchor, width, 0, offset, frame.hop_idx, (), (), "factual_ppo",
            observation_digest(build_observation(frame, item.state).array()), control_state_digest(item.state),
            recurrent_state_digest(item.recurrent), recurrent_state_digest(item.post_prediction_recurrent), item.proposed_action,
            legal_actions(item.state), item.proposed_action in legal_actions(item.state), -1, item.proposed_action, item.executed_action,
            item.legal, item.override_reason, item.abs_error_bpm, cumulative, item.abs_error_bpm))
    return tuple(output)


def _dedupe(nodes: Sequence[_Node]) -> tuple[list[_Node], int]:
    """Keep one deterministic representative per future controller/recurrent state."""
    best: dict[tuple[str, str], _Node] = {}
    for node in nodes:
        identity = (control_state_digest(node.state), recurrent_state_digest(node.recurrent))
        prior = best.get(identity)
        if prior is None or node.key() < prior.key():
            best[identity] = node
    retained = sorted(best.values(), key=lambda node: node.key())
    return retained, len(nodes) - len(retained)


def repeated_correction_beam(
    frames: Sequence[MeasurementFrame], labels: Sequence[LabelFrame], policy: FrozenPolicy, *, clip_id: str,
    anchor_hop_idx: int, width: int = PRIMARY_WIDTH, window: int = WINDOW, snapshots: Sequence[FactualSnapshot] | None = None,
) -> BeamResult:
    """Run the offline PPO-plus-rank1 beam from one factual free-choice anchor."""
    _frames_and_labels(frames, labels, clip_id)
    if width < 1 or window < 1 or len(frames) - anchor_hop_idx < window:
        _fail("beam width/window or anchor coverage is invalid")
    if snapshots is None:
        snapshots = factual_snapshots(frames, labels, policy, clip_id=clip_id)
    if anchor_hop_idx not in eligible_anchors(snapshots, window=window):
        _fail("beam anchor is not a factual all-action free choice")
    factual = _factual_window(frames, labels, snapshots, anchor=anchor_hop_idx, width=width, window=window)
    start = snapshots[anchor_hop_idx]
    active = [_Node(start.state, copy.deepcopy(start.recurrent), (), (), (), 0.0, math.inf)]
    generated = deduped = pruned = 0
    depth_stats: list[dict[str, int]] = []
    for offset in range(window):
        frame, label, children = frames[anchor_hop_idx + offset], labels[anchor_hop_idx + offset], []
        for node in active:
            values = build_observation(frame, node.state).array()
            pre_rec = recurrent_state_digest(node.recurrent)
            ppo, post_rec = policy.predict(values.reshape(1, 101), node.recurrent, episode_start=(anchor_hop_idx == 0 and offset == 0))
            ppo = _action(ppo, "beam PPO action is invalid")
            available = legal_actions(node.state); rank1, _rank1_error = rank1_action(frame, label, node.state)
            choices = [rank1]
            if ppo in available and ppo != rank1:
                choices.append(ppo)
            for action in choices:
                next_state, transition = control_step(frame, node.state, action)
                error = abs(transition.post_belief.mean_hr - float(label.gt_hr_bpm))
                hop = BeamHop(anchor_hop_idx, width, -1, offset, frame.hop_idx,
                    node.origin_sequence + (action,), node.executed_sequence + (transition.action_decision.executed_action,),
                    "rank1_gt" if action == rank1 else "ppo_legal", observation_digest(values), control_state_digest(node.state),
                    pre_rec, recurrent_state_digest(post_rec), ppo, available, ppo in available, rank1, action, transition.action_decision.executed_action,
                    transition.action_decision.legal, transition.action_decision.override_reason, error, node.cumulative + error, error)
                children.append(_Node(next_state, copy.deepcopy(post_rec), hop.origin_sequence, hop.executed_sequence,
                    node.hops + (hop,), hop.cumulative_abs_error_bpm, error))
        generated += len(children)
        unique, removed = _dedupe(children)
        deduped += removed
        if len(unique) > width:
            pruned += len(unique) - width
        active = unique[:width]
        if not active:
            _fail("beam lost all legal children")
        best = min(active, key=lambda node: node.key())
        factual_step = snapshots[anchor_hop_idx + offset].abs_error_bpm
        depth_stats.append({"offset": offset, "generated": len(children), "deduped": removed,
                            "retained": len(active), "pruned": max(0, len(unique) - len(active)),
                            "factual_cumulative_abs_error_bpm": sum(snapshots[anchor_hop_idx + i].abs_error_bpm for i in range(offset + 1)),
                            "factual_endpoint_abs_error_bpm": factual_step,
                            "best_cumulative_abs_error_bpm": best.cumulative,
                            "best_endpoint_abs_error_bpm": best.endpoint})
    retained = tuple(tuple(BeamHop(**{field: getattr(hop, field) if field != "path_rank" else rank for field in hop.__dataclass_fields__}) for hop in node.hops) for rank, node in enumerate(active))
    return BeamResult(anchor_hop_idx, width, factual, retained, generated, deduped, pruned, tuple(depth_stats))


def run_widths(frames: Sequence[MeasurementFrame], labels: Sequence[LabelFrame], policy: FrozenPolicy, *, clip_id: str, anchor_hop_idx: int, widths: Sequence[int] = SENSITIVITY_WIDTHS, window: int = WINDOW, snapshots: Sequence[FactualSnapshot] | None = None) -> tuple[BeamResult, ...]:
    if tuple(widths) != tuple(sorted(set(widths))) or any(not isinstance(width, int) or width < 1 for width in widths):
        _fail("beam widths must be unique positive ascending integers")
    return tuple(repeated_correction_beam(frames, labels, policy, clip_id=clip_id, anchor_hop_idx=anchor_hop_idx, width=width, window=window, snapshots=snapshots) for width in widths)


__all__ = ["BeamHop", "BeamResult", "FactualSnapshot", "PRIMARY_WIDTH", "SCHEMA", "SENSITIVITY_WIDTHS", "WINDOW", "eligible_anchors", "factual_snapshots", "legal_actions", "rank1_action", "repeated_correction_beam", "run_widths"]
