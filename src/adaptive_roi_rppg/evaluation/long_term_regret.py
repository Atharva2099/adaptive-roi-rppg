"""MCD-only, one-time-intervention long-term regret diagnostic.

The evaluator is deliberately independent of learned-model backends.  It
reuses the controller for every observation, action decision, and belief
update.  Labels are joined only after unscored branch trajectories exist.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import build_observation, control_step, initial_control_state, resolve_action
from adaptive_roi_rppg.evaluation.model_replay import FrozenPolicy

ACTION_COUNT = 12
HORIZONS = (1, 5, 15)
SCHEMA = "mcd-long-term-regret-v1"
ANCHOR_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "hop_count", "pre_control_state_sha256", "observation_sha256", "pre_predict_recurrent_sha256", "post_predict_recurrent_sha256", "factual_proposed_action", "factual_executed_action", "eligible", "ineligible_reason")
BRANCH_HOP_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "branch_action", "offset", "hop_idx", "observation_sha256", "pre_control_state_sha256", "pre_predict_recurrent_sha256", "post_predict_recurrent_sha256", "proposed_action", "executed_action", "legal", "override_reason", "post_belief_hr_bpm")
BRANCH_SCORE_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "branch_action", "horizon", "mean_abs_error_bpm")
ANCHOR_REGRET_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "horizon", "factual_action", "best_action", "factual_mean_abs_error_bpm", "best_mean_abs_error_bpm", "factual_regret_bpm")


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int): _fail(f"{name} must be an integer")
    return value


def _normalise(value: Any) -> Any:
    """Canonical, non-opaque representation for a recurrent-state digest."""
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float):
        if not math.isfinite(value): _fail("recurrent state contains a non-finite float")
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all(): _fail("recurrent state array is unsupported or non-finite")
        return {"ndarray": True, "dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, (tuple, list)): return [_normalise(item) for item in value]
    if isinstance(value, Mapping): return {str(key): _normalise(value[key]) for key in sorted(value, key=str)}
    _fail("recurrent state has an opaque unsupported type")


def recurrent_state_digest(value: object | None) -> str:
    return hashlib.sha256(canonical_json_bytes(_normalise(value))).hexdigest()


def control_state_digest(state: object) -> str:
    """Digest the immutable controller state without serialising it into output."""
    payload = {"dataset_id": state.dataset_id, "clip_id": state.clip_id, "next_hop_idx": state.next_hop_idx,
               "belief": {"mean_hr": state.belief.mean_hr, "velocity": state.belief.velocity,
                          "covariance": [list(row) for row in state.belief.covariance], "hops_since_confident": state.belief.hops_since_confident},
               "previous_action": state.previous_action, "hold_count": state.hold_count}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def observation_digest(observation: np.ndarray) -> str:
    if observation.shape != (101,) or observation.dtype != np.float32 or not np.isfinite(observation).all(): _fail("observation is not finite float32[101]")
    return hashlib.sha256(observation.tobytes()).hexdigest()


def validate_replay_parity(rows: Sequence[Mapping[str, Any]], snapshots: Sequence[Any], *, clip_id: str, seed: int) -> None:
    """Compare only fields actually persisted by the source replay."""
    expected = [row for row in rows if row.get("clip_id") == clip_id and int(row.get("seed", -1)) == seed and row.get("method_id") == f"advantage_ppo_seed{seed}"]
    expected.sort(key=lambda row: int(row["hop_idx"]))
    if len(expected) != len(snapshots): _fail("replay source parity coverage differs from factual replay")
    for row, snap in zip(expected, snapshots):
        if (int(row["hop_idx"]), int(row["proposed_action"]), int(row["executed_action"])) != (snap.hop_idx, snap.proposed_action, snap.executed_action):
            _fail(f"replay parity mismatch at hop {snap.hop_idx}")
        for key, actual in (("selected_post_belief_hr_bpm", snap.post_belief_hr_bpm), ("selected_abs_error_bpm", snap.abs_error_bpm)):
            if key not in row or not math.isclose(float(row[key]), float(actual), rel_tol=0.0, abs_tol=1e-6): _fail(f"replay parity mismatch for {key} at hop {snap.hop_idx}")




@dataclass(frozen=True, slots=True)
class BranchHop:
    clip_id: str; anchor_hop_idx: int; branch_action: int; offset: int; hop_idx: int
    observation_sha256: str; pre_control_state_sha256: str; pre_predict_recurrent_sha256: str; post_predict_recurrent_sha256: str
    proposed_action: int; executed_action: int; legal: bool; override_reason: str | None; post_belief_hr_bpm: float


def _assert_frames(frames: Sequence[MeasurementFrame], clip_id: str) -> None:
    if not frames or any(frame.dataset_id != "mcd" or frame.clip_id != clip_id or frame.hop_idx != index for index, frame in enumerate(frames)): _fail("branch replay frames must be contiguous MCD frames")


def _run_branch(frames: Sequence[MeasurementFrame], *, anchor: int, action: int, state: object, post_prediction_recurrent: object | None, policy: FrozenPolicy) -> tuple[BranchHop, ...]:
    current_state, recurrent, output = state, copy.deepcopy(post_prediction_recurrent), []
    for offset in range(15):
        frame = frames[anchor + offset]
        pre_state_hash = control_state_digest(current_state)
        if offset == 0:
            proposed = action; pre_recurrent_hash = recurrent_state_digest(recurrent); post_recurrent_hash = pre_recurrent_hash
        else:
            values = build_observation(frame, current_state).array(); pre_recurrent_hash = recurrent_state_digest(recurrent)
            proposed, recurrent = policy.predict(values.reshape(1, 101), recurrent, episode_start=False)
            if isinstance(proposed, bool) or not isinstance(proposed, (int, np.integer)) or not 0 <= int(proposed) < ACTION_COUNT: _fail("policy branch action is invalid")
            proposed = int(proposed); post_recurrent_hash = recurrent_state_digest(recurrent)
        values = build_observation(frame, current_state).array()
        current_state, transition = control_step(frame, current_state, proposed)
        decision = transition.action_decision
        output.append(BranchHop(frame.clip_id, anchor, action, offset, frame.hop_idx, observation_digest(values), pre_state_hash, pre_recurrent_hash, post_recurrent_hash, proposed, decision.executed_action, decision.legal, decision.override_reason, transition.post_belief.mean_hr))
    return tuple(output)


def replay_one_time_interventions(frames: Sequence[MeasurementFrame], policy: FrozenPolicy, *, clip_id: str, snapshots: Sequence[Any] | None = None) -> tuple[tuple[dict[str, Any], ...], tuple[BranchHop, ...]]:
    """Return eligible anchor snapshots and all twelve 15-hop unscored branches."""
    _assert_frames(frames, clip_id)
    if snapshots is None:
        # Kept for the existing full diagnostic.  The combined smoke passes
        # its already-computed factual snapshots to avoid replaying PPO twice.
        state, recurrent, episode_start = initial_control_state("mcd", clip_id), policy.initial_state(), True
        snapshots = []
        for frame in frames:
            values = build_observation(frame, state).array()
            proposed, post_prediction = policy.predict(values.reshape(1, 101), recurrent, episode_start=episode_start)
            if isinstance(proposed, bool) or not isinstance(proposed, (int, np.integer)) or not 0 <= int(proposed) < ACTION_COUNT: _fail("policy factual action is invalid")
            proposed = int(proposed)
            snapshots.append(type("Snapshot", (), {"hop_idx": frame.hop_idx, "state": state, "recurrent": copy.deepcopy(recurrent), "post_prediction_recurrent": copy.deepcopy(post_prediction), "episode_start": episode_start, "proposed_action": proposed, "executed_action": control_step(frame, state, proposed)[1].action_decision.executed_action})())
            state, _transition = control_step(frame, state, proposed)
            recurrent, episode_start = post_prediction, False
    if len(snapshots) != len(frames): _fail("factual snapshots do not cover the clip")
    anchors, branches = [], []
    for anchor, (frame, snapshot) in enumerate(zip(frames, snapshots)):
        values = build_observation(frame, snapshot.state).array(); pre_state = snapshot.state
        proposed = int(snapshot.proposed_action); post_prediction = snapshot.post_prediction_recurrent
        factual = control_step(frame, pre_state, proposed)[1]
        exact_free = all((decision := resolve_action(candidate, pre_state)).legal and decision.executed_action == candidate for candidate in range(ACTION_COUNT))
        eligible = len(frames) - anchor >= 15 and exact_free
        anchor_row = {"anchor_hop_idx": anchor, "hop_count": len(frames), "pre_control_state_sha256": control_state_digest(pre_state), "observation_sha256": observation_digest(values), "pre_predict_recurrent_sha256": recurrent_state_digest(snapshot.recurrent), "post_predict_recurrent_sha256": recurrent_state_digest(post_prediction), "factual_proposed_action": proposed, "factual_executed_action": factual.action_decision.executed_action, "eligible": eligible, "ineligible_reason": None if eligible else ("insufficient_future_hops" if len(frames) - anchor < 15 else "locked_or_nonfree_action_set")}
        anchors.append(anchor_row)
        if eligible:
            for candidate in range(ACTION_COUNT): branches.extend(_run_branch(frames, anchor=anchor, action=candidate, state=pre_state, post_prediction_recurrent=post_prediction, policy=policy))
    return tuple(anchors), tuple(branches)


def score_branches(branches: Sequence[BranchHop], labels: Sequence[LabelFrame]) -> tuple[dict[str, Any], ...]:
    """Join GT after all branch state/action traces were computed."""
    label_by_hop = {label.hop_idx: label for label in labels}
    if len(label_by_hop) != len(labels): _fail("labels have duplicate hops")
    result = []
    by_branch: dict[tuple[int, int], list[BranchHop]] = defaultdict(list)
    for hop in branches: by_branch[(hop.anchor_hop_idx, hop.branch_action)].append(hop)
    for (anchor, action), trace in sorted(by_branch.items()):
        trace.sort(key=lambda item: item.offset)
        if [item.offset for item in trace] != list(range(15)): _fail("branch trace does not cover offsets 0..14")
        errors = []
        for hop in trace:
            label = label_by_hop.get(hop.hop_idx)
            if label is None or label.dataset_id != "mcd" or label.clip_id != hop.clip_id or not label.valid or label.gt_hr_bpm is None: _fail("branch labels are invalid or misaligned")
            errors.append(abs(hop.post_belief_hr_bpm - label.gt_hr_bpm))
        for horizon in HORIZONS: result.append({"anchor_hop_idx": anchor, "branch_action": action, "horizon": horizon, "mean_abs_error_bpm": sum(errors[:horizon]) / horizon})
    return tuple(result)


def build_anchor_regret(scores: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scores: grouped[(int(row["anchor_hop_idx"]), int(row["horizon"]))].append(row)
    result = []
    for (anchor, horizon), values in sorted(grouped.items()):
        if len(values) != ACTION_COUNT or {int(value["branch_action"]) for value in values} != set(range(ACTION_COUNT)): _fail("anchor lacks all twelve branches")
        factual_action = next((int(value["branch_action"]) for value in values if value.get("is_factual")), None)
        if factual_action is None: _fail("scores must identify one factual action per anchor")
        factual = next(float(value["mean_abs_error_bpm"]) for value in values if int(value["branch_action"]) == factual_action)
        best = min(values, key=lambda value: (float(value["mean_abs_error_bpm"]), int(value["branch_action"])))
        result.append({"anchor_hop_idx": anchor, "horizon": horizon, "factual_action": factual_action, "best_action": int(best["branch_action"]), "factual_mean_abs_error_bpm": factual, "best_mean_abs_error_bpm": float(best["mean_abs_error_bpm"]), "factual_regret_bpm": factual - float(best["mean_abs_error_bpm"])})
    return tuple(result)


def subject_block_bootstrap(anchor_rows: Sequence[Mapping[str, Any]], *, replicates: int = 10_000, seed: int = 8101) -> dict[str, Any]:
    """Clip mean then equal-clips-within-subject, with subject-block resampling."""
    if replicates < 1: _fail("bootstrap replicates must be positive")
    clips: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in anchor_rows:
        value = float(row["factual_regret_bpm"])
        if not math.isfinite(value) or not row.get("subject_id") or not row.get("clip_id"): _fail("anchor regret row is invalid")
        clips[(str(row["subject_id"]), str(row["clip_id"]))].append(value)
    by_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _clip), values in clips.items(): by_subject[subject].append(sum(values) / len(values))
    if not by_subject: _fail("no eligible anchors for bootstrap")
    subject_values = np.asarray([sum(values) / len(values) for _, values in sorted(by_subject.items())], dtype=np.float64)
    point = float(np.mean(subject_values)); rng = np.random.default_rng(seed)
    draws = np.mean(subject_values[rng.integers(0, len(subject_values), size=(replicates, len(subject_values)))], axis=1); draws.sort()
    return {"estimand": "mean_factual_regret_bpm_clip_then_equal_clip_within_subject", "point_estimate_bpm": point, "ci_95_percentile_bpm": [float(draws[int(.025 * (replicates - 1))]), float(draws[int(.975 * (replicates - 1))])], "replicates": replicates, "seed": seed, "resampling_unit": "subject"}


def csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    import io
    handle = io.StringIO(newline=""); writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows: writer.writerow({field: row.get(field) for field in fields})
    return handle.getvalue().encode("utf-8")


__all__ = ["ACTION_COUNT", "ANCHOR_FIELDS", "ANCHOR_REGRET_FIELDS", "BRANCH_HOP_FIELDS", "BRANCH_SCORE_FIELDS", "HORIZONS", "SCHEMA", "BranchHop", "build_anchor_regret", "control_state_digest", "csv_bytes", "observation_digest", "recurrent_state_digest", "replay_one_time_interventions", "score_branches", "subject_block_bootstrap", "validate_replay_parity"]
