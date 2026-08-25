"""The single real Gate 9 MMPD extraction and replay callable.

This module is deliberately MMPD-local.  It consumes only authenticated raw
MAT/video sources, an authenticated metadata coding file, and frozen policy
identities supplied by Gate 9 preflight.  Labels are joined after policy
replay; they are never present in observations or policy state.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import control_step, initial_control_state
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from .extraction import extract_mmpd_measurements
from .raw_source import capture_source_bytes
from .publication import HOP_FIELDS
from .replay import rollout_gate9_policy
from .rules import MMPD_FPS, MMPD_GT_RULE_ID, build_mmpd_labels

ROI_MAPPING_ID = "engineering_face_relative_rectangles_v1"


def _fail(message: str) -> None:
    raise ContractValidationError("Gate 9 production: " + message)


class _FullFacePolicy:
    identity = CheckpointIdentity("full_face_pos", "fixed_full_face", None, None)

    def initial_state(self) -> None:
        return None

    def predict(self, observation: np.ndarray, recurrent_state: None, *, episode_start: bool) -> tuple[int, None]:
        if observation.shape != (1, 101) or observation.dtype != np.float32 or not np.isfinite(observation).all():
            _fail("full-face policy received an invalid observation")
        return 0, None


def _safe_locator(root: Path, locator: str, label: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        _fail(f"{label} locator is unsafe")
    target = root / relative
    try:
        if target.resolve().parent != (root / relative.parent).resolve():
            _fail(f"{label} locator escapes its root")
    except OSError as exc:
        raise ContractValidationError(f"Gate 9 production: cannot resolve {label} locator") from exc
    if target.is_symlink() or not target.is_file():
        _fail(f"{label} must be a regular file")
    return target


def _metadata(provenance: Mapping[str, Any], plan: Any) -> dict[str, dict[str, str]]:
    root = Path(str(provenance.get("rule_root", "")))
    locator = provenance.get("metadata_coding_locator")
    if not root or not locator:
        _fail("authenticated metadata coding input is unavailable")
    path = _safe_locator(root, str(locator), "metadata coding")
    expected_sha256 = provenance.get("metadata_coding_sha256")
    expected_bytes = provenance.get("metadata_coding_bytes")
    if not isinstance(expected_sha256, str) or not isinstance(expected_bytes, int):
        _fail("metadata coding identity is incomplete")
    try:
        raw = capture_source_bytes(path, expected_sha256, expected_bytes).data
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("Gate 9 production: metadata coding input is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"clips"} or not isinstance(payload["clips"], list):
        _fail("metadata coding input must contain exactly clips")
    expected = {clip["clip_id"]: clip["subject_id"] for clip in plan.payload["clips"]}
    result: dict[str, dict[str, str]] = {}
    for item in payload["clips"]:
        if not isinstance(item, dict) or set(item) != {"clip_id", "subject_id", "view", "condition"}:
            _fail("metadata coding record fields are not exact")
        clip_id = item["clip_id"]
        if clip_id in result or expected.get(clip_id) != item["subject_id"]:
            _fail("metadata coding cohort binding is invalid")
        if any(not isinstance(item[key], str) or not item[key] for key in ("subject_id", "view", "condition")):
            _fail("metadata coding values must be nonempty strings")
        result[clip_id] = {key: item[key] for key in ("subject_id", "view", "condition")}
    if set(result) != set(expected):
        _fail("metadata coding does not cover the exact plan cohort")
    return result


def _row(plan: Any, clip: Mapping[str, str], method: str, identity: Mapping[str, Any], hop: Mapping[str, Any], label: LabelFrame, *, validity_reason: str = "") -> dict[str, Any]:
    belief = float(hop["post_belief_hr_bpm"])
    gt = float(label.gt_hr_bpm)
    return {"plan_id": plan.plan_id, "plan_sha256": plan.plan_sha256, "dataset_id": "mmpd", "clip_id": clip["clip_id"], "subject_id": clip["subject_id"], "view": clip["view"], "condition": clip["condition"], "method_id": method, "family": identity["family"], "seed": identity["seed"], "checkpoint_sha256": identity["sha256"], "hop_idx": int(hop["hop_idx"]), "hop_time_s": float(hop["hop_time_s"]), "gt_hr_bpm": gt, "post_belief_hr_bpm": belief, "abs_error_bpm": abs(belief - gt), "selected_valid": bool(hop["selected_valid"]), "selected_invalid_reason": hop["selected_invalid_reason"], "proposed_action": int(hop["proposed_action"]), "executed_action": int(hop["executed_action"]), "legal": bool(hop["legal"]), "override_reason": hop["override_reason"], "pre_hold_count": int(hop["pre_hold_count"]), "post_hold_count": int(hop["post_hold_count"]), "causal_reset": int(hop["hop_idx"]) == 0, "gt_observation_count": 0, "validity_reason": validity_reason or ("" if hop["selected_valid"] else hop["selected_invalid_reason"] or "invalid_selected_measurement")}


def _oracle_rows(frames: Sequence[Any], labels: Sequence[LabelFrame], clip: Mapping[str, str], plan: Any, method: str, *, beam_width: int = 8) -> list[dict[str, Any]]:
    if len(frames) != len(labels) or len(frames) != 53:
        _fail("Oracle input does not cover the exact 53-hop lattice")
    if any(not label.valid or label.gt_hr_bpm is None for label in labels):
        _fail("Oracle input contains an invalid GT label")
    # B is deterministic one-step greedy; C is deterministic finite beam search.
    candidates: list[tuple[float, tuple[int, ...], Any, list[dict[str, Any]]]] = [(0.0, (), initial_control_state("mmpd", clip["clip_id"]), [])]
    greedy_state = initial_control_state("mmpd", clip["clip_id"])
    greedy: list[dict[str, Any]] = []
    for frame, label in zip(frames, labels):
        if method == "oracle_b_greedy":
            legal = ((greedy_state.previous_action,) if greedy_state.previous_action is not None and greedy_state.hold_count < 2 else tuple(range(12)))
            expanded = []
            for action in legal:
                next_state, transition = control_step(frame, greedy_state, int(action))
                error = abs(float(next_state.belief.mean_hr) - float(label.gt_hr_bpm))
                row = {"hop_idx": frame.hop_idx, "hop_time_s": frame.hop_time_s, "post_belief_hr_bpm": next_state.belief.mean_hr, "selected_valid": transition.selected_measurement.valid, "selected_invalid_reason": transition.selected_measurement.invalid_reason or "", "proposed_action": transition.action_decision.proposed_action, "executed_action": transition.action_decision.executed_action, "legal": transition.action_decision.legal, "override_reason": transition.action_decision.override_reason or "", "pre_hold_count": transition.action_decision.pre_hold_count, "post_hold_count": transition.action_decision.post_hold_count}
                expanded.append((error, (int(action),), next_state, [row]))
            chosen = min(expanded, key=lambda item: (item[0], item[1]))
            greedy_state, row = chosen[2], chosen[3][0]
            greedy.append(_row(plan, clip, method, {"family": "oracle_b", "seed": "", "sha256": ""}, row, label))
        else:
            # Re-expand the retained beam with cumulative scores and action paths.
            next_beam: list[tuple[float, tuple[int, ...], Any, list[dict[str, Any]]]] = []
            for cumulative, sequence, state, rows in candidates:
                legal = ((state.previous_action,) if state.previous_action is not None and state.hold_count < 2 else tuple(range(12)))
                for action in legal:
                    next_state, transition = control_step(frame, state, int(action))
                    error = abs(float(next_state.belief.mean_hr) - float(label.gt_hr_bpm))
                    row = {"hop_idx": frame.hop_idx, "hop_time_s": frame.hop_time_s, "post_belief_hr_bpm": next_state.belief.mean_hr, "selected_valid": transition.selected_measurement.valid, "selected_invalid_reason": transition.selected_measurement.invalid_reason or "", "proposed_action": transition.action_decision.proposed_action, "executed_action": transition.action_decision.executed_action, "legal": transition.action_decision.legal, "override_reason": transition.action_decision.override_reason or "", "pre_hold_count": transition.action_decision.pre_hold_count, "post_hold_count": transition.action_decision.post_hold_count}
                    next_beam.append((cumulative + error, sequence + (int(action),), next_state, rows + [row]))
            candidates = sorted(next_beam, key=lambda item: (item[0], item[1]))[:beam_width]
    if method == "oracle_b_greedy":
        return greedy
    if not candidates:
        _fail("Oracle C beam is empty")
    return [_row(plan, clip, method, {"family": "oracle_c", "seed": "", "sha256": ""}, row, label) for row, label in zip(candidates[0][3], labels)]


def _csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HOP_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded = dict(row)
        for field in ("selected_valid", "legal", "causal_reset"):
            encoded[field] = "true" if encoded[field] is True else "false" if encoded[field] is False else encoded[field]
        writer.writerow(encoded)
    return stream.getvalue().encode()


def extract_gate9(plan: Any, provenance: Mapping[str, Any]) -> Mapping[str, bytes]:
    """Production callable loaded by ``scripts/evaluate_mmpd_gate9.py``."""
    if plan.payload.get("dataset_id") != "mmpd" or plan.clip_count != 299:
        _fail("callable requires the exact engineering MMPD plan")
    if provenance.get("corrected_gt_rule_id") != MMPD_GT_RULE_ID:
        _fail("authenticated corrected-GT rule is not the repository-owned MMPD rule")
    if provenance.get("corrected_gt_rule_source_sha256") != provenance.get("corrected_gt_rule_sha256"):
        _fail("executed MMPD rule source is not bound to the authenticated rule identity")
    rule_source = Path(__file__).with_name("rules.py")
    if (provenance.get("corrected_gt_rule_source_locator") != str(rule_source)
            or provenance.get("corrected_gt_rule_source_bytes") != rule_source.stat().st_size):
        _fail("executed MMPD rule source path or size is not authenticated")
    metadata = _metadata(provenance, plan)
    raw_root = Path(str(provenance.get("raw_root", "")))
    checkpoint_root = Path(str(provenance.get("checkpoint_root", "")))
    model_asset = provenance.get("face_landmarker_model_path")
    if not model_asset:
        _fail("authenticated MediaPipe face-landmarker model path is unavailable")
    model_path = Path(str(model_asset))
    try:
        capture_source_bytes(model_path, provenance.get("face_landmarker_model_sha256"), provenance.get("face_landmarker_model_bytes"))
    except ContractValidationError as exc:
        raise ContractValidationError("Gate 9 production: face-landmarker model authentication failed") from exc
    identities = {item["method_id"]: item for item in provenance.get("checkpoint_identities", [])}
    if set(identities) != set(plan.payload["arms"]):
        _fail("authenticated checkpoint identities are incomplete")
    policies: dict[str, Any] = {"full_face_pos": _FullFacePolicy()}
    for method in plan.payload["arms"]:
        if method == "full_face_pos":
            continue
        spec = dict(identities[method])
        spec["locator"] = str(_safe_locator(checkpoint_root, spec["locator"], f"checkpoint {method}"))
        spec["byte_size"] = spec.pop("bytes")
        policies[method] = load_frozen_recurrent_policy(spec)
    raw_records = {item["clip_id"]: item for item in provenance.get("raw_inventory_records", [])}
    if set(raw_records) != set(plan.expected_clip_ids):
        _fail("authenticated raw inventory records are unavailable")
    primary: list[dict[str, Any]] = []
    oracle_b: list[dict[str, Any]] = []
    oracle_c: list[dict[str, Any]] = []
    for clip in plan.payload["clips"]:
        item = raw_records[clip["clip_id"]]
        source = _safe_locator(raw_root, item["locator"], f"raw source {clip['clip_id']}")
        frames, gt, _ = extract_mmpd_measurements(str(source), clip_id=clip["clip_id"], expected_sha256=item["sha256"], expected_bytes=item["bytes"], model_asset_path=str(model_path), fps=MMPD_FPS)
        labels = build_mmpd_labels(gt, clip_id=clip["clip_id"])
        if len(frames) != 53 or len(labels) != 53:
            _fail(f"{clip['clip_id']} does not produce exactly 53 hops")
        meta = metadata[clip["clip_id"]]
        bound_clip = {**clip, **meta}
        if any(not label.valid for label in labels):
            _fail(f"{clip['clip_id']} has an invalid MMPD GT label")
        for method, policy in policies.items():
            for hop, label in zip(rollout_gate9_policy(frames, policy, clip_id=clip["clip_id"]), labels):
                primary.append(_row(plan, bound_clip, method, {"family": hop["family"], "seed": hop["seed"], "sha256": hop["checkpoint_sha256"]}, hop, label))
        oracle_b.extend(_oracle_rows(frames, labels, bound_clip, plan, "oracle_b_greedy"))
        oracle_c.extend(_oracle_rows(frames, labels, bound_clip, plan, "oracle_c_beam"))
    return {"per_hop": _csv(primary), "oracle_b": _csv(oracle_b), "oracle_c": _csv(oracle_c)}


__all__ = ["MMPD_GT_RULE_ID", "ROI_MAPPING_ID", "build_mmpd_labels", "extract_gate9"]
