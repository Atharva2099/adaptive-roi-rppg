"""Deterministic, MCD-train-only Oracle B/C teacher evaluation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from adaptive_roi_rppg.contracts import ROI_NAMES, canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import MIN_HOLD, ControlState, control_step, initial_control_state
from adaptive_roi_rppg.labels.mcd import GT_RULE_ID

ORACLE_SCHEMA = "gate7-mcd-oracle-shard-v2"
ORACLE_REPORT_SCHEMA = "gate7-mcd-oracle-report-v2"
ACTION_COUNT = len(ROI_NAMES)
ROW_FIELDS = (
    "dataset_id", "clip_id", "subject_id", "view", "condition", "hop_idx", "hop_time_s",
    "gt_hr_bpm", "gt_rule_id", "frame_provenance_id", "signal_config_id", "control_config_id",
    "proposed_action", "executed_action", "pre_hold_count", "post_hold_count", "override_reason",
    "selected_valid", "selected_invalid_reason", "pre_belief_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm",
)


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _error(state: ControlState, frame: Any, gt_hr: float, action: int) -> tuple[float, ControlState, Any]:
    next_state, transition = control_step(frame, state, action)
    value = float(next_state.belief.mean_hr)
    if not math.isfinite(value) or not math.isfinite(gt_hr):
        _fail("oracle received non-finite scored HR")
    return abs(value - gt_hr), next_state, transition


def legal_actions(state: ControlState) -> tuple[int, ...]:
    if state.previous_action is not None and state.hold_count < MIN_HOLD:
        return (state.previous_action,)
    return tuple(range(ACTION_COUNT))


def _row(clip_id: str, subject_id: str, view: str, condition: str, hop_idx: int,
         hop_time_s: float, gt_hr: float, action: int, error: float, transition: Any) -> dict[str, Any]:
    decision = transition.action_decision
    if (transition.dataset_id != "mcd" or transition.clip_id != clip_id or transition.hop_idx != hop_idx
            or transition.selected_measurement.roi_index != decision.executed_action
            or transition.control_config_id != transition.observation.config_id):
        _fail("oracle transition identity/configuration mismatch")
    return {
        "dataset_id": "mcd", "clip_id": clip_id, "subject_id": subject_id, "view": view,
        "condition": condition, "hop_idx": hop_idx, "hop_time_s": hop_time_s, "gt_hr_bpm": gt_hr,
        "gt_rule_id": GT_RULE_ID, "frame_provenance_id": transition.frame_provenance_id,
        "signal_config_id": transition.signal_config_id, "control_config_id": transition.control_config_id,
        "proposed_action": decision.proposed_action, "executed_action": decision.executed_action,
        "pre_hold_count": decision.pre_hold_count, "post_hold_count": decision.post_hold_count,
        "override_reason": decision.override_reason, "selected_valid": transition.selected_measurement.valid,
        "selected_invalid_reason": transition.selected_measurement.invalid_reason,
        "pre_belief_hr_bpm": transition.pre_belief.mean_hr, "post_belief_hr_bpm": transition.post_belief.mean_hr,
        "abs_error_bpm": error,
    }


def replay_action_sequence(frames: Sequence[Any], labels: Sequence[Any], rows: Sequence[Mapping[str, Any]],
                           *, clip_id: str) -> None:
    """Replay stored actions from a fresh state and compare every derived field."""
    state = initial_control_state("mcd", clip_id)
    if len(rows) != len(frames) or len(labels) != len(frames):
        _fail("oracle replay row/frame/label counts differ")
    for frame, label, row in zip(frames, labels, rows):
        if row.get("executed_action") != row.get("proposed_action") and row.get("override_reason") != "minimum_hold":
            _fail("oracle replay row has an impossible override")
        error, state, transition = _error(state, frame, float(label.gt_hr_bpm), int(row["proposed_action"]))
        expected = _row(row["clip_id"], row["subject_id"], row["view"], row["condition"], frame.hop_idx,
                        frame.hop_time_s, float(label.gt_hr_bpm), int(row["proposed_action"]), error, transition)
        for field in ROW_FIELDS:
            left, right = row.get(field), expected.get(field)
            if isinstance(left, float) or isinstance(right, float):
                if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                    _fail(f"oracle replay mismatch: {field}")
            elif left != right:
                _fail(f"oracle replay mismatch: {field}")


def _validate_labels(frames: Sequence[Any], labels: Sequence[Any], clip_id: str) -> dict[tuple[int, float], Any]:
    if len(frames) != len(labels):
        _fail("oracle frame/label hop counts differ")
    if any(label.dataset_id != "mcd" or label.clip_id != clip_id or label.gt_rule_id != GT_RULE_ID
           or not label.valid or label.gt_hr_bpm is None for label in labels):
        _fail("oracle label identity or GT rule mismatch")
    keys = [(label.hop_idx, label.hop_time_s) for label in labels]
    if len(set(keys)) != len(keys) or any((frame.hop_idx, frame.hop_time_s) != key for frame, key in zip(frames, keys)):
        _fail("oracle frame/label identity mismatch")
    return dict(zip(keys, labels))


def run_oracle_clip(frames: Sequence[Any], labels: Sequence[Any], *, clip_id: str, subject_id: str,
                    view: str, condition: str, beam_width: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if beam_width < 1:
        _fail("beam_width must be positive")
    label_by_hop = _validate_labels(frames, labels, clip_id)
    greedy_state = initial_control_state("mcd", clip_id)
    greedy_rows: list[dict[str, Any]] = []
    beam: list[tuple[float, tuple[int, ...], ControlState, list[dict[str, Any]]]] = [
        (0.0, (), initial_control_state("mcd", clip_id), [])
    ]
    for frame in frames:
        label = label_by_hop[(frame.hop_idx, frame.hop_time_s)]
        gt = float(label.gt_hr_bpm)
        candidates = [(_error(greedy_state, frame, gt, action) + (action,)) for action in legal_actions(greedy_state)]
        score, greedy_state, transition, action = min(candidates, key=lambda item: (item[0], item[3]))
        greedy_rows.append(_row(clip_id, subject_id, view, condition, frame.hop_idx, frame.hop_time_s, gt, action, score, transition))
        expanded = []
        for cumulative, sequence, state, rows in beam:
            for proposed in legal_actions(state):
                step_score, next_state, step = _error(state, frame, gt, proposed)
                expanded.append((cumulative + step_score, sequence + (proposed,), next_state,
                                 rows + [_row(clip_id, subject_id, view, condition, frame.hop_idx, frame.hop_time_s,
                                              gt, proposed, step_score, step)]))
        expanded.sort(key=lambda item: (item[0], item[1]))
        beam = expanded[:beam_width]
    if not beam:
        _fail("oracle beam is empty")
    beam_rows = beam[0][3]
    replay_action_sequence(frames, labels, beam_rows, clip_id=clip_id)
    return greedy_rows, beam_rows


def _clip_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    result = []
    for clip_id, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["hop_idx"])
        result.append({
            "clip_id": clip_id, "subject_id": values[0]["subject_id"], "view": values[0]["view"],
            "condition": values[0]["condition"], "hop_count": len(values),
            "mae_bpm": sum(float(v["abs_error_bpm"]) for v in values) / len(values),
            "switch_count": sum(v["executed_action"] != values[i - 1]["executed_action"] for i, v in enumerate(values) if i),
            "override_count": sum(v["override_reason"] is not None for v in values),
        })
    return result


def _summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    clips = _clip_rows(rows)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clips:
        by_subject[row["subject_id"]].append(row)
        by_bucket[(row["view"], row["condition"])].append(row)
    subject_rows = [{"subject_id": key, "clip_count": len(value), "mae_bpm": sum(v["mae_bpm"] for v in value) / len(value)}
                    for key, value in sorted(by_subject.items())]
    bucket_rows = [{"view": key[0], "condition": key[1], "clip_count": len(value),
                    "mae_bpm": sum(v["mae_bpm"] for v in value) / len(value)}
                   for key, value in sorted(by_bucket.items())]
    return {"method": method, "clip_count": len(clips), "subject_count": len(subject_rows),
            "equal_clip_mae_bpm": sum(v["mae_bpm"] for v in clips) / len(clips),
            "total_hops": sum(v["hop_count"] for v in clips),
            "total_switches": sum(v["switch_count"] for v in clips),
            "total_overrides": sum(v["override_count"] for v in clips),
            "clip_rows": clips, "subject_rows": subject_rows, "view_condition_rows": bucket_rows}


def evaluate_oracle_shard(frames_by_clip: Mapping[str, Sequence[Any]], labels_by_clip: Mapping[str, Sequence[Any]],
                          metadata: Mapping[str, Mapping[str, str]], *, beam_width: int = 8) -> dict[str, Any]:
    greedy: list[dict[str, Any]] = []
    beam: list[dict[str, Any]] = []
    for clip_id in sorted(frames_by_clip):
        if clip_id not in labels_by_clip or clip_id not in metadata:
            _fail("oracle shard clip inputs are incomplete")
        meta = metadata[clip_id]
        b, c = run_oracle_clip(frames_by_clip[clip_id], labels_by_clip[clip_id], clip_id=clip_id,
                                subject_id=meta["subject_id"], view=meta["view"], condition=meta["condition"],
                                beam_width=beam_width)
        greedy.extend(b); beam.extend(c)
    return {"schema": ORACLE_SCHEMA, "beam_width": beam_width, "rows": {"greedy_b": greedy, "beam_c": beam},
            "summary": {"greedy_b": _summary(greedy, "oracle_b_greedy"), "beam_c": _summary(beam, "oracle_c_beam")}}


def aggregate_report(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    b, c = rows["greedy_b"], rows["beam_c"]
    summaries = {"greedy_b": _summary(b, "oracle_b_greedy"), "beam_c": _summary(c, "oracle_c_beam")}
    return {"methods": summaries, "comparison": {
        "method_b": "oracle_b_greedy", "method_c": "oracle_c_beam", "equal_clip_mae_bpm": {
            "oracle_b_greedy": summaries["greedy_b"]["equal_clip_mae_bpm"],
            "oracle_c_beam": summaries["beam_c"]["equal_clip_mae_bpm"]},
        "dCB_b_mae_minus_c_mae_bpm": summaries["greedy_b"]["equal_clip_mae_bpm"] - summaries["beam_c"]["equal_clip_mae_bpm"],
        "total_hops": {method: summary["total_hops"] for method, summary in summaries.items()},
        "total_switches": {method: summary["total_switches"] for method, summary in summaries.items()},
        "total_overrides": {method: summary["total_overrides"] for method, summary in summaries.items()},
    }}


def validate_aggregation(rows: Mapping[str, Sequence[Mapping[str, Any]]], aggregation: Mapping[str, Any]) -> None:
    """Reject a reported aggregate unless it is the exact aggregate of its hop rows."""
    if set(rows) != {"greedy_b", "beam_c"}:
        _fail("oracle aggregation methods are incomplete")
    expected = aggregate_report(rows)
    if canonical_json_bytes(aggregation) != canonical_json_bytes(expected):
        _fail("oracle aggregation does not match hop rows")


def write_json_with_hash(path: str, payload: Mapping[str, Any]) -> str:
    from pathlib import Path
    target = Path(path)
    target.write_text(__import__("json").dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return sha256_file(target)


__all__ = ["ORACLE_SCHEMA", "ORACLE_REPORT_SCHEMA", "ROW_FIELDS", "legal_actions", "replay_action_sequence",
           "run_oracle_clip", "evaluate_oracle_shard", "aggregate_report", "validate_aggregation", "write_json_with_hash"]
