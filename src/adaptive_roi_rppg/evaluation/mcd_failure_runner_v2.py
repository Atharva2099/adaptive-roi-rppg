"""Deterministic, restartable MCD frozen-policy failure audit.

This is MCD-only and CPU-first.  Its logical-shard/COMPLETE convention is
derived from the clean Gate 8 evaluator (docs/gates/gate_08_mcd_frozen_models.md,
lines 7--15); canonical action order is the clean Gate 6.1 contract.  It is
not a trainer and it never reads MMPD.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import CanonicalFrame, LabelFrame, MeasurementFrame, ROI_NAMES, canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import build_observation, control_step, initial_control_state
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.evaluation.failure_audit import ACTION_COUNT, build_failure_audit, random_legal_requested_action, score_failure_audit
from adaptive_roi_rppg.evaluation.model_plan import EXCLUSIONS, load_frozen_model_plan
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels, read_mcd_labels
from adaptive_roi_rppg.signal import build_pos_measurements

SCHEMA = "mcd-frozen-failure-audit-v2"
PLAN_SCHEMA = "mcd-frozen-failure-audit-plan-v1"
BENCHMARK_SCHEMA = "mcd-frozen-failure-audit-benchmark-v2"
SAFETY_FACTOR = 1.5
FIXED_RESERVE_SECONDS = 300.0
HOP_FIELDS = ("subject_id", "clip_id", "camera_id", "view", "condition", "pose_status", "pose_yaw", "pose_pitch", "pose_roll", "hop_idx", "hop_time_s", "method_id", "family", "seed", "checkpoint_sha256", "proposed_action", "executed_action", "previous_action", "legal", "override_reason", "pre_hold_count", "post_hold_count", "selected_measurement_valid", "selected_measurement_invalid_reason", "selected_measurement_hr_bpm", "selected_post_belief_hr_bpm", "gt_rule_id", "gt_hr_bpm", "selected_abs_error_bpm", "failure_category", "best_valid_actions_json", "best_valid_error_bpm", "policy_minus_best_immediate_regret_bpm", "any_valid_immediate_action", "better_actionable_alternative", "random_control_requested_action", "alternatives_json")
CLIP_FIELDS = ("method_id", "family", "seed", "checkpoint_sha256", "subject_id", "clip_id", "camera_id", "view", "condition", "hop_count", "selected_mae_bpm", "selected_invalid_hop_count", "no_valid_immediate_action_hop_count", "any_valid_immediate_action_hop_count", "better_actionable_alternative_hop_count", "mean_immediate_regret_bpm")
SUBJECT_FIELDS = ("method_id", "family", "seed", "checkpoint_sha256", "subject_id", "clip_count", "equal_clip_selected_mae_bpm", "selected_invalid_hop_count", "no_valid_immediate_action_hop_count", "any_valid_immediate_action_hop_count", "better_actionable_alternative_hop_count", "equal_clip_mean_immediate_regret_bpm")

def _fail(message: str) -> None: raise ContractValidationError(f"MCD failure audit: {message}")
def _sha(value: object) -> bool: return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
def _json(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle: handle.write(canonical_json_bytes(value))
def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink(): _fail("output must be a fresh directory under a real parent")
    path.mkdir()
def _forbid_mmpd(*values: str | Path) -> None:
    if any("mmpd" in str(value).lower() for value in values): _fail("MMPD paths are forbidden")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
def _safe_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."} or _SAFE_TOKEN.fullmatch(value) is None: _fail(f"{label} is unsafe")
    return value
def _artifact_map(root: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file(): _fail(f"missing regular output {name}")
        result[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result
def _marker(root: Path, state: str, body: Mapping[str, Any], outputs: Mapping[str, Any] | None = None) -> None:
    value = {"schema": SCHEMA, "state": state, **body}
    if outputs is not None: value["outputs"] = dict(outputs)
    _write_json(root / f"{state}.json", value)

def expected_hops_for_clip(clip: Any) -> int:
    count = max(0, (int(clip.state_row_count) - round(8 * float(clip.camera_fps))) // round(float(clip.camera_fps)) + 1)
    if count < 1: _fail("planned clip has no causal hop")
    return count

def balanced_subject_assignment(subject_work: Mapping[str, int], logical_shards: int) -> tuple[tuple[str, ...], ...]:
    """Stable longest-processing-time assignment with lexical shard tie break."""
    if not isinstance(logical_shards, int) or isinstance(logical_shards, bool) or logical_shards < 1: _fail("logical shard count is invalid")
    if not subject_work or any(not isinstance(key, str) or not key or not isinstance(value, int) or isinstance(value, bool) or value < 1 for key, value in subject_work.items()): _fail("subject work is invalid")
    heap = [(0, index) for index in range(logical_shards)]; heapq.heapify(heap); values = [[] for _ in range(logical_shards)]
    for subject, work in sorted(subject_work.items(), key=lambda item: (-item[1], item[0])):
        load, index = heapq.heappop(heap); values[index].append(subject); heapq.heappush(heap, (load + work, index))
    return tuple(tuple(sorted(value)) for value in values)
def frozen_worker_assignment(logical_shard_count: int, logical_indices: Sequence[int], worker_count: int) -> dict[str, Any]:
    """Freeze rank ownership once; workers must never redistribute local IDs."""
    if not isinstance(logical_shard_count, int) or logical_shard_count < 1 or not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count < 1:
        _fail("worker assignment shape is invalid")
    values = tuple(logical_indices)
    if not values or len(set(values)) != len(values) or any(not isinstance(index, int) or isinstance(index, bool) or index not in range(logical_shard_count) for index in values):
        _fail("worker assignment batch is invalid")
    return {"worker_count": worker_count, "global_logical_shard_indices": list(values), "worker_logical_shard_indices": [list(values[rank::worker_count]) for rank in range(worker_count)]}
def _worker_shape(args: Any) -> tuple[int, int]:
    workers, allocated = getattr(args, "worker_count", 1), getattr(args, "allocated_task_count", 1)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (workers, allocated)) or not 1 <= workers <= allocated:
        _fail("worker count must satisfy 1 <= W <= allocated Slurm tasks")
    return workers, allocated

def bind_mcd_clips(manifest_tree: str | Path, split: str, *, max_subjects: int | None = None) -> tuple[Any, tuple[Any, ...]]:
    _forbid_mmpd(manifest_tree)
    if split not in {"train", "eval"}: _fail("split must be train or eval")
    if split == "eval" and max_subjects is not None: _fail("partial held-out eval is forbidden")
    bundle = load_mcd_manifest_tree(manifest_tree); ids = set(bundle.split_manifest.train_clip_ids if split == "train" else bundle.split_manifest.eval_clip_ids)
    if set(bundle.split_manifest.train_subject_ids) & set(bundle.split_manifest.eval_subject_ids): _fail("train/eval subject overlap")
    clips = tuple(sorted((clip for clip in bundle.clip_manifests if clip.clip_id in ids and (split == "train" or clip.clip_id not in EXCLUSIONS)), key=lambda clip: clip.clip_id))
    subjects = sorted({clip.subject_id for clip in clips})
    if split == "eval":
        if len(ids) != 540 or len(clips) != 533 or len(subjects) != 89 or ids - {clip.clip_id for clip in clips} != EXCLUSIONS: _fail("eval must bind exact 540/7/533/89 Gate 8 cohort")
        if sum(expected_hops_for_clip(clip) for clip in clips) != 91_227: _fail("eval must bind 91,227 hops per checkpoint")
    if max_subjects is not None:
        if not isinstance(max_subjects, int) or isinstance(max_subjects, bool) or max_subjects < 1: _fail("max subjects is invalid")
        keep = set(subjects[:max_subjects]); clips = tuple(clip for clip in clips if clip.subject_id in keep)
    if not clips: _fail("bound cohort is empty")
    return bundle, clips

def _runtime_checkpoints(root: str | Path, model: Any) -> tuple[dict[str, Any], ...]:
    base = Path(root)
    if base.is_symlink() or not base.is_dir(): _fail("checkpoint root must be real")
    base = base.resolve(); values = []
    for item in model.checkpoints:
        relative = Path(item["locator"])
        if relative.is_absolute() or ".." in relative.parts: _fail("checkpoint locator is invalid")
        path = (base / relative).resolve()
        if base not in path.parents or path.is_symlink() or not path.is_file(): _fail("checkpoint path is invalid")
        next_item = dict(item); next_item["locator"] = str(path); values.append(next_item)
    if len(values) != 9: _fail("requires exactly nine Gate 8 checkpoints")
    return tuple(values)
def _identities(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [{"method_id": str(item["method_id"]), "family": str(item["family"]), "seed": int(item["seed"]), "checkpoint_sha256": str(item["sha256"])} for item in values]
    if len(result) != 9 or len({_json(item) for item in result}) != 9: _fail("checkpoint identities are not nine unique values")
    return result
def _environment() -> dict[str, str]:
    installed = {value.metadata["Name"] for value in package_metadata.distributions()}; names = ("numpy", "scipy", "stable-baselines3", "sb3-contrib", "gymnasium", "torch", "cloudpickle")
    return {name: package_metadata.version(name) if name in installed else "not-installed" for name in names}
def _inventory(manifest_tree: str | Path, plan_path: str | Path, checkpoints: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    paths = [Path(manifest_tree) / name for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")] + [Path(plan_path)] + [Path(item["locator"]) for item in checkpoints]
    if any(path.is_symlink() or not path.is_file() for path in paths): _fail("source inventory is missing or unsafe")
    source = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(paths, key=str)]
    return hashlib.sha256(canonical_json_bytes(source)).hexdigest(), hashlib.sha256(canonical_json_bytes(_environment())).hexdigest()

def build_audit_plan(args: Any) -> dict[str, Any]:
    _forbid_mmpd(args.manifest_tree, args.state_root, args.gt_root, args.audit_root)
    if not _sha(args.code_snapshot_sha256): _fail("code hash is invalid")
    _safe_token(args.run_id, "run ID")
    if not isinstance(args.logical_shards, int) or args.logical_shards < 1: _fail("logical shard count is invalid")
    model = load_frozen_model_plan(args.plan); checkpoints = _runtime_checkpoints(args.checkpoint_root, model); _, clips = bind_mcd_clips(args.manifest_tree, args.split, max_subjects=getattr(args, "max_subjects", None)); source_hash, environment_hash = _inventory(args.manifest_tree, args.plan, checkpoints)
    expected = {clip.clip_id: expected_hops_for_clip(clip) for clip in clips}; subject_work = {subject: sum(expected[clip.clip_id] for clip in clips if clip.subject_id == subject) for subject in sorted({clip.subject_id for clip in clips})}
    if args.logical_shards > len(subject_work): _fail("logical shard count exceeds available subjects")
    assignment = balanced_subject_assignment(subject_work, args.logical_shards)
    shards = []
    for index, subject_ids in enumerate(assignment):
        clip_ids = [clip.clip_id for clip in clips if clip.subject_id in set(subject_ids)]
        shards.append({"logical_shard_index": index, "subject_ids": list(subject_ids), "clip_ids": clip_ids, "expected_hops_per_checkpoint": sum(expected[value] for value in clip_ids)})
    workers, _ = _worker_shape(args)
    worker_assignment = frozen_worker_assignment(args.logical_shards, tuple(args.logical_shard_indices), workers)
    payload = {"schema": PLAN_SCHEMA, "run_id": args.run_id, "split": args.split, "code_snapshot_sha256": args.code_snapshot_sha256, "source_inventory_sha256": source_hash, "environment_sha256": environment_hash, "plan_id": model.plan_id, "checkpoint_identities": _identities(checkpoints), "canonical_roi_names": list(ROI_NAMES), "random_control_rule": "one-hop deterministic random_legal_requested_action", "resource_shape": {"cpus_per_worker": 1, "gpu": False, "logical_shards": args.logical_shards, "worker_count": workers}, "worker_assignment": worker_assignment, "logical_shard_count": args.logical_shards, "cohort": {"clip_ids": sorted(expected), "subject_ids": sorted(subject_work), "expected_hops": dict(sorted(expected.items()))}, "logical_shards": shards}
    if args.split == "eval" and (len(expected), len(subject_work), sum(expected.values()), len(checkpoints)) != (533, 89, 91_227, 9): _fail("exact Gate 8 evaluation guard differs")
    payload["audit_plan_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest(); return payload
def write_audit_plan(args: Any) -> Path:
    payload = build_audit_plan(args); root = Path(args.audit_root) / args.run_id; root.mkdir(parents=True, exist_ok=True); path = root / "audit_plan.json"
    if path.exists():
        if path.is_symlink() or read_json_object(path) != payload: _fail("immutable audit plan exists with different bindings")
    else: _write_json(path, payload)
    return path
def load_audit_plan(path: str | Path) -> dict[str, Any]:
    value = read_json_object(path); actual = value.pop("audit_plan_sha256", None)
    if value.get("schema") != PLAN_SCHEMA or not _sha(actual) or actual != hashlib.sha256(canonical_json_bytes(value)).hexdigest(): _fail("audit plan hash/schema is invalid")
    value["audit_plan_sha256"] = actual
    if tuple(value.get("canonical_roi_names", ())) != ROI_NAMES or len(value.get("checkpoint_identities", ())) != 9 or len(value.get("logical_shards", ())) != value.get("logical_shard_count") or value.get("worker_assignment") != frozen_worker_assignment(value["logical_shard_count"], value.get("worker_assignment", {}).get("global_logical_shard_indices", ()), value.get("worker_assignment", {}).get("worker_count", 0)):
        _fail("audit plan contract differs")
    return value
def _assert_runtime(args: Any, plan: Mapping[str, Any]) -> tuple[Any, tuple[dict[str, Any], ...], Any, tuple[Any, ...]]:
    if (args.run_id, args.split, args.code_snapshot_sha256) != (plan["run_id"], plan["split"], plan["code_snapshot_sha256"]): _fail("runtime differs from immutable audit plan")
    model = load_frozen_model_plan(args.plan); checkpoints = _runtime_checkpoints(args.checkpoint_root, model); source_hash, env_hash = _inventory(args.manifest_tree, args.plan, checkpoints)
    if (model.plan_id, source_hash, env_hash, _identities(checkpoints)) != (plan["plan_id"], plan["source_inventory_sha256"], plan["environment_sha256"], plan["checkpoint_identities"]): _fail("runtime source/environment/checkpoints differ from plan")
    bundle, clips = bind_mcd_clips(args.manifest_tree, args.split, max_subjects=getattr(args, "max_subjects", None))
    if {clip.clip_id: expected_hops_for_clip(clip) for clip in clips} != plan["cohort"]["expected_hops"]: _fail("runtime cohort differs from plan")
    return model, checkpoints, bundle, clips

def _pose(frames: Sequence[CanonicalFrame], measurement: MeasurementFrame) -> tuple[str, float | None, float | None, float | None]:
    index = round(measurement.hop_time_s * frames[0].camera_fps) - 1
    if index < 0 or index >= len(frames): _fail("measurement has no source frame")
    frame = frames[index]
    if any(value is None for value in (frame.head_yaw, frame.head_pitch, frame.head_roll)): return "unavailable_in_source", None, None, None
    return "mcd_state_end_frame", float(frame.head_yaw), float(frame.head_pitch), float(frame.head_roll)
@dataclass(frozen=True, slots=True)
class _Replay: clip: Any; frame: MeasurementFrame; pose: tuple[str, float | None, float | None, float | None]; identity: Any; audit: Any; random_action: int
def replay_measurements_without_labels(measurements: Sequence[MeasurementFrame], frames: Sequence[CanonicalFrame], policy: Any, clip: Any) -> tuple[_Replay, ...]:
    """Replay one policy from already-built causal measurements, without labels."""
    if not measurements: _fail("clip has no causal POS hops")
    state, recurrent, episode, values = initial_control_state("mcd", clip.clip_id), policy.initial_state(), True, []
    for expected, frame in enumerate(measurements):
        if frame.hop_idx != expected: _fail("noncontiguous measurements")
        observation = build_observation(frame, state).array(); action, recurrent = policy.predict(observation.reshape(1, 101), recurrent, episode_start=episode)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_COUNT: _fail("policy action invalid")
        audit = build_failure_audit(frame, state, int(action)); seed = int.from_bytes(hashlib.sha256(f"{clip.clip_id}:{frame.hop_idx}:{policy.identity.method_id}".encode()).digest()[:8], "big")
        values.append(_Replay(clip, frame, _pose(frames, frame), policy.identity, audit, random_legal_requested_action(state, seed))); state, _ = control_step(frame, state, int(action)); episode = False
    return tuple(values)
def replay_clip_without_labels(frames: Sequence[CanonicalFrame], policy: Any, clip: Any) -> tuple[_Replay, ...]:
    """Compatibility entry point for one policy; builds POS once for this clip."""
    return replay_measurements_without_labels(build_pos_measurements(frames), frames, policy, clip)
def _replay_all_policies_without_labels(frames: Sequence[CanonicalFrame], policies: Sequence[Any], clip: Any) -> tuple[tuple[_Replay, ...], ...]:
    """Build canonical causal POS once, then replay every frozen checkpoint."""
    measurements = build_pos_measurements(frames)
    return tuple(replay_measurements_without_labels(measurements, frames, policy, clip) for policy in policies)
def score_replay_after_labels(replay: Sequence[_Replay], labels: Sequence[LabelFrame]) -> list[dict[str, Any]]:
    if not replay or len(replay) != len(labels): _fail("replay/label coverage differs")
    rows = []
    for item, label in zip(replay, labels, strict=True):
        if (label.dataset_id, label.clip_id, label.hop_idx, label.hop_time_s) != ("mcd", item.frame.clip_id, item.frame.hop_idx, item.frame.hop_time_s) or not label.valid or label.gt_hr_bpm is None: _fail("labels invalid/misaligned")
        scored = score_failure_audit(item.audit, label.gt_hr_bpm); decision = item.audit.selected.transition.action_decision; selected = item.audit.selected.selected_measurement
        alternatives = [{"requested_action": raw.requested_action, "proposed_action": value.proposed_action, "executed_action": value.executed_action, "measurement_valid": value.selected_valid, "measurement_invalid_reason": raw.selected_measurement.invalid_reason, "measurement_hr_bpm": raw.selected_measurement.hr_bpm, "post_belief_hr_bpm": value.post_belief_hr_bpm, "abs_error_bpm": value.absolute_error_bpm} for raw, value in zip(item.audit.alternatives, scored.alternatives, strict=True)]
        better = any(value["executed_action"] != decision.executed_action and value["measurement_valid"] and value["abs_error_bpm"] < scored.selected.absolute_error_bpm for value in alternatives); status, yaw, pitch, roll = item.pose
        row = {"subject_id": item.clip.subject_id, "clip_id": item.clip.clip_id, "camera_id": item.clip.camera_id, "view": item.clip.view, "condition": item.clip.condition, "pose_status": status, "pose_yaw": yaw, "pose_pitch": pitch, "pose_roll": roll, "hop_idx": item.frame.hop_idx, "hop_time_s": item.frame.hop_time_s, "method_id": item.identity.method_id, "family": item.identity.family, "seed": item.identity.seed, "checkpoint_sha256": item.identity.checkpoint_sha256, "proposed_action": decision.proposed_action, "executed_action": decision.executed_action, "previous_action": decision.previous_action, "legal": decision.legal, "override_reason": decision.override_reason, "pre_hold_count": decision.pre_hold_count, "post_hold_count": decision.post_hold_count, "selected_measurement_valid": selected.valid, "selected_measurement_invalid_reason": selected.invalid_reason, "selected_measurement_hr_bpm": selected.hr_bpm, "selected_post_belief_hr_bpm": scored.selected.post_belief_hr_bpm, "gt_rule_id": label.gt_rule_id, "gt_hr_bpm": scored.gt_hr_bpm, "selected_abs_error_bpm": scored.selected.absolute_error_bpm, "failure_category": scored.category, "best_valid_actions_json": _json(list(scored.best_valid_actions)), "best_valid_error_bpm": scored.best_valid_error_bpm, "policy_minus_best_immediate_regret_bpm": scored.policy_minus_best_immediate_regret_bpm, "any_valid_immediate_action": any(value["measurement_valid"] for value in alternatives), "better_actionable_alternative": better, "random_control_requested_action": item.random_action, "alternatives_json": _json(alternatives)}
        if tuple(row) != HOP_FIELDS: _fail("compact row schema differs")
        rows.append(row)
    return rows

def _alternatives(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    try: values, best = json.loads(str(row["alternatives_json"])), json.loads(str(row["best_valid_actions_json"]))
    except json.JSONDecodeError as exc: raise ContractValidationError("MCD failure audit: invalid compact JSON") from exc
    keys = {"requested_action", "proposed_action", "executed_action", "measurement_valid", "measurement_invalid_reason", "measurement_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm"}
    if _json(values) != row["alternatives_json"] or _json(best) != row["best_valid_actions_json"] or not isinstance(values, list) or len(values) != ACTION_COUNT or [value.get("requested_action") for value in values] != list(range(ACTION_COUNT)) or any(not isinstance(value, dict) or set(value) != keys for value in values) or any(not isinstance(value, int) or value not in range(ACTION_COUNT) for value in best): _fail("invalid compact alternative/order")
    return values
def validate_hop_rows(rows: Sequence[Mapping[str, Any]], expected_hops: Mapping[str, int] | None = None, checkpoint_identities: Sequence[Mapping[str, Any]] | None = None) -> None:
    expected = {str(key): int(value) for key, value in (expected_hops or {}).items()}; allowed = {(item["method_id"], item["family"], int(item["seed"]), item["checkpoint_sha256"]) for item in checkpoint_identities or []}; seen = set()
    for row in rows:
        if tuple(row) != HOP_FIELDS: _fail("hop row schema differs")
        values = _alternatives(row); identity = (row["method_id"], row["family"], row["seed"], row["checkpoint_sha256"]); key = (identity, row["clip_id"], row["hop_idx"])
        if key in seen or (allowed and identity not in allowed) or (expected and (row["clip_id"] not in expected or row["hop_idx"] not in range(expected[row["clip_id"]]))): _fail("duplicate/unplanned compact row")
        if bool(row["any_valid_immediate_action"]) != any(value["measurement_valid"] for value in values): _fail("any valid flag differs")
        better = any(value["executed_action"] != row["executed_action"] and value["measurement_valid"] and value["abs_error_bpm"] < row["selected_abs_error_bpm"] for value in values)
        if bool(row["better_actionable_alternative"]) != better: _fail("actionable flag differs")
        seen.add(key)
    if expected and len(seen) != sum(expected.values()) * len(allowed): _fail("compact coverage differs")

def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> int:
    count = 0
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            if tuple(row) != tuple(fields): _fail("CSV row schema differs")
            writer.writerow(row); count += 1
    return count
def _convert(raw: Mapping[str, str | None], fields: Sequence[str]) -> dict[str, Any]:
    integers = {"hop_idx", "seed", "proposed_action", "executed_action", "previous_action", "pre_hold_count", "post_hold_count", "random_control_requested_action", "hop_count", "selected_invalid_hop_count", "no_valid_immediate_action_hop_count", "any_valid_immediate_action_hop_count", "better_actionable_alternative_hop_count", "clip_count"}; booleans = {"legal", "selected_measurement_valid", "any_valid_immediate_action", "better_actionable_alternative"}; floats = {value for value in fields if value.endswith("_bpm") or value in {"hop_time_s", "pose_yaw", "pose_pitch", "pose_roll"}}; nullable = {"previous_action", "override_reason", "selected_measurement_invalid_reason", "selected_measurement_hr_bpm", "best_valid_error_bpm", "policy_minus_best_immediate_regret_bpm", "pose_yaw", "pose_pitch", "pose_roll", "mean_immediate_regret_bpm", "equal_clip_mean_immediate_regret_bpm"}; out = {}
    for key, value in raw.items():
        if key is None or value is None: _fail("malformed CSV")
        if value == "" and key in nullable: out[key] = None
        elif key in integers: out[key] = int(value)
        elif key in booleans:
            if value not in {"True", "False"}: _fail("invalid boolean")
            out[key] = value == "True"
        elif key in floats: out[key] = float(value)
        else: out[key] = value
    if tuple(out) != tuple(fields): _fail("CSV header differs")
    return out
def _stream(path: Path, fields: Sequence[str]):
    if path.is_symlink() or not path.is_file(): _fail("missing CSV")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields): _fail("CSV header differs")
        for row in reader: yield _convert(row, fields)

class _Aggregate:
    def __init__(self) -> None: self.values: dict[tuple[Any, ...], dict[str, Any]] = {}
    def add(self, row: Mapping[str, Any]) -> None:
        key = tuple(row[field] for field in ("method_id", "family", "seed", "checkpoint_sha256", "subject_id", "clip_id")); item = self.values.setdefault(key, {"first": row, "n": 0, "error": 0., "invalid": 0, "no_valid": 0, "any_valid": 0, "better": 0, "regret": 0., "regret_n": 0}); item["n"] += 1; item["error"] += row["selected_abs_error_bpm"]; item["invalid"] += int(not row["selected_measurement_valid"]); item["no_valid"] += int(not row["any_valid_immediate_action"]); item["any_valid"] += int(row["any_valid_immediate_action"]); item["better"] += int(row["better_actionable_alternative"])
        if row["policy_minus_best_immediate_regret_bpm"] is not None: item["regret"] += row["policy_minus_best_immediate_regret_bpm"]; item["regret_n"] += 1
    def rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clips = []
        for key, item in sorted(self.values.items()):
            first = item["first"]; clips.append({"method_id": key[0], "family": key[1], "seed": key[2], "checkpoint_sha256": key[3], "subject_id": key[4], "clip_id": key[5], "camera_id": first["camera_id"], "view": first["view"], "condition": first["condition"], "hop_count": item["n"], "selected_mae_bpm": item["error"] / item["n"], "selected_invalid_hop_count": item["invalid"], "no_valid_immediate_action_hop_count": item["no_valid"], "any_valid_immediate_action_hop_count": item["any_valid"], "better_actionable_alternative_hop_count": item["better"], "mean_immediate_regret_bpm": item["regret"] / item["regret_n"] if item["regret_n"] else None})
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in clips: grouped[tuple(row[key] for key in ("method_id", "family", "seed", "checkpoint_sha256", "subject_id"))].append(row)
        subjects = []
        for key, rows in sorted(grouped.items()):
            regrets = [row["mean_immediate_regret_bpm"] for row in rows if row["mean_immediate_regret_bpm"] is not None]
            subjects.append({"method_id": key[0], "family": key[1], "seed": key[2], "checkpoint_sha256": key[3], "subject_id": key[4], "clip_count": len(rows), "equal_clip_selected_mae_bpm": sum(row["selected_mae_bpm"] for row in rows) / len(rows), "selected_invalid_hop_count": sum(row["selected_invalid_hop_count"] for row in rows), "no_valid_immediate_action_hop_count": sum(row["no_valid_immediate_action_hop_count"] for row in rows), "any_valid_immediate_action_hop_count": sum(row["any_valid_immediate_action_hop_count"] for row in rows), "better_actionable_alternative_hop_count": sum(row["better_actionable_alternative_hop_count"] for row in rows), "equal_clip_mean_immediate_regret_bpm": sum(regrets) / len(regrets) if regrets else None})
        return clips, subjects
def _validate_stream(path: Path, expected: Mapping[str, int], identities: Sequence[Mapping[str, Any]]) -> tuple[int, _Aggregate]:
    aggregate = _Aggregate(); count = 0; seen = set(); allowed = {(item["method_id"], item["family"], item["seed"], item["checkpoint_sha256"]) for item in identities}
    for row in _stream(path, HOP_FIELDS):
        values = _alternatives(row); identity = (row["method_id"], row["family"], row["seed"], row["checkpoint_sha256"]); key = (identity, row["clip_id"], row["hop_idx"])
        if identity not in allowed or row["clip_id"] not in expected or row["hop_idx"] not in range(expected[row["clip_id"]]) or key in seen: _fail("invalid shard compact coverage")
        if bool(row["any_valid_immediate_action"]) != any(value["measurement_valid"] for value in values): _fail("invalid any-valid field")
        better = any(value["executed_action"] != row["executed_action"] and value["measurement_valid"] and value["abs_error_bpm"] < row["selected_abs_error_bpm"] for value in values)
        if bool(row["better_actionable_alternative"]) != better: _fail("invalid actionable field")
        seen.add(key); aggregate.add(row); count += 1
    if len(seen) != sum(expected.values()) * len(allowed): _fail("missing shard compact coverage")
    return count, aggregate

def _resource(args: Any) -> dict[str, Any]:
    seconds, cpus = getattr(args, "audit_walltime_seconds", None), getattr(args, "cpus_per_worker", 1)
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1 or cpus != 1: _fail("requires AUDIT_WALLTIME_SECONDS and one CPU per worker")
    workers, allocated = _worker_shape(args)
    return {"cpus_per_worker": 1, "gpu": False, "worker_count": workers, "allocated_task_count": allocated, "walltime_seconds": seconds}
def _matching_completed(root: Path, plan_hash: str, index: int) -> Path | None:
    attempts = root / "attempts"
    if not attempts.is_dir(): return None
    values = []
    for child in sorted(attempts.iterdir(), key=lambda item: item.name):
        marker = child / "COMPLETE.json"
        if child.is_dir() and marker.is_file() and not marker.is_symlink():
            value = read_json_object(marker)
            if value.get("state") == "COMPLETE" and value.get("audit_plan_sha256") == plan_hash and value.get("logical_shard_index") == index: values.append(child)
    if len(values) > 1: _fail("multiple completed attempts for logical shard")
    return values[0] if values else None
def _validate_completed_attempt(root: Path, plan: Mapping[str, Any], index: int) -> None:
    """Validate a reusable completion before skip; never repair or overwrite it."""
    logical = plan["logical_shards"][index]
    marker = read_json_object(root / "COMPLETE.json")
    manifest = read_json_object(root / "shard_manifest.json")
    expected_body = {"schema": SCHEMA, "state": "COMPLETE", "audit_plan_sha256": plan["audit_plan_sha256"], "run_id": plan["run_id"], "split": plan["split"], "logical_shard_index": index, "logical_shard_count": plan["logical_shard_count"]}
    if any(marker.get(key) != value for key, value in expected_body.items()) or marker.get("outputs") != _artifact_map(root, ("per_hop_counterfactual.csv", "shard_manifest.json")):
        _fail("completed shard marker/artifacts are corrupt")
    if manifest.get("schema") != SCHEMA or any(manifest.get(key) != value for key, value in expected_body.items() if key not in {"state"}) or manifest.get("clip_ids") != logical["clip_ids"]:
        _fail("completed shard manifest is corrupt or misowned")
    expected = {clip: plan["cohort"]["expected_hops"][clip] for clip in logical["clip_ids"]}
    count, _ = _validate_stream(root / "per_hop_counterfactual.csv", expected, plan["checkpoint_identities"])
    if manifest.get("work_units") != count or manifest.get("per_hop_bytes") != (root / "per_hop_counterfactual.csv").stat().st_size:
        _fail("completed shard work/byte accounting differs")
def _projection(record: Mapping[str, Any], plan: Mapping[str, Any], *, shard_work_units: int | None = None) -> dict[str, float]:
    if record.get("schema") != BENCHMARK_SCHEMA or record.get("status") != "complete": _fail("matching completed train benchmark required")
    compatibility = record.get("compatibility", {})
    for key in ("code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "plan_id", "checkpoint_identities", "canonical_roi_names", "resource_shape"):
        if compatibility.get(key) != plan.get(key): _fail("benchmark is stale/incompatible")
    measured = record.get("measured", {}); rates = (measured.get("worker_seconds_per_hop_checkpoint"), measured.get("bytes_per_hop_checkpoint"), measured.get("merge_seconds_per_byte"))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in rates): _fail("benchmark rates invalid")
    if shard_work_units is None: shard_work_units = max(item["expected_hops_per_checkpoint"] for item in plan["logical_shards"]) * 9
    if not isinstance(shard_work_units, int) or isinstance(shard_work_units, bool) or shard_work_units < 1: _fail("projected shard work is invalid")
    total = sum(plan["cohort"]["expected_hops"].values()) * 9
    return {"shard_seconds": rates[0] * shard_work_units * SAFETY_FACTOR + FIXED_RESERVE_SECONDS, "merge_seconds": rates[2] * rates[1] * total * SAFETY_FACTOR + FIXED_RESERVE_SECONDS}
def _local_shard_work(plan: Mapping[str, Any], logical_indices: Sequence[int]) -> int:
    if not logical_indices or len(set(logical_indices)) != len(logical_indices) or any(not isinstance(index, int) or index not in range(plan["logical_shard_count"]) for index in logical_indices): _fail("local logical shard list is invalid")
    return sum(int(plan["logical_shards"][index]["expected_hops_per_checkpoint"]) * 9 for index in logical_indices)
def _validate_worker_assignment(args: Any, plan: Mapping[str, Any], local_indices: Sequence[int]) -> None:
    workers, _ = _worker_shape(args); rank = getattr(args, "worker_rank", 0); assignment = plan["worker_assignment"]
    if not isinstance(rank, int) or isinstance(rank, bool) or rank not in range(workers) or assignment["worker_count"] != workers or tuple(local_indices) != tuple(assignment["worker_logical_shard_indices"][rank]):
        _fail("worker local shard list differs from frozen assignment")
def _require_budget(args: Any, plan: Mapping[str, Any], stage: str, *, logical_indices: Sequence[int] | None = None) -> dict[str, float]:
    if args.split == "train": return {}
    if not getattr(args, "benchmark_record", None): _fail("held-out stage requires completed train benchmark")
    resource = _resource(args)
    work = _local_shard_work(plan, logical_indices or ()) if stage == "shard" else None
    projected = _projection(read_json_object(args.benchmark_record), plan, shard_work_units=work); required = projected["shard_seconds" if stage == "shard" else "merge_seconds"]
    if resource["walltime_seconds"] < math.ceil(required): _fail(f"requested walltime unsafe; need {math.ceil(required)} seconds")
    return projected

def _evaluate(args: Any, bundle: Any, clips: Sequence[Any], checkpoints: Sequence[Mapping[str, Any]]):
    policies = tuple(load_frozen_recurrent_policy(item) for item in checkpoints)
    for clip in clips:
        frames = read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, args.split)
        # Labels are deliberately read only after all nine GT-free replays.
        replays = _replay_all_policies_without_labels(frames, policies, clip)
        labels = read_mcd_labels(bundle, args.gt_root, clip.clip_id, "train", clip) if args.split == "train" else read_mcd_eval_labels(bundle, args.gt_root, clip.clip_id, clip)
        for replay in replays: yield from score_replay_after_labels(replay, labels)
def publish_shard(args: Any) -> None:
    _safe_token(args.attempt_id, "attempt ID")
    plan = load_audit_plan(args.audit_plan); selected = tuple(args.logical_shard_indices)
    _resource(args); _validate_worker_assignment(args, plan, selected); _require_budget(args, plan, "shard", logical_indices=selected)
    _, checkpoints, bundle, clips = _assert_runtime(args, plan)
    base = Path(args.audit_root) / args.run_id / "logical_shards"; base.mkdir(parents=True, exist_ok=True)
    for index in selected:
        logical = plan["logical_shards"][index]; logical_root = base / f"shard-{index:03d}"
        completed = _matching_completed(logical_root, plan["audit_plan_sha256"], index)
        if completed is not None:
            _validate_completed_attempt(completed, plan, index)
            continue
        attempt = logical_root / "attempts" / f"{args.attempt_id}-s{index:03d}"; attempt.parent.mkdir(parents=True, exist_ok=True); _fresh(attempt); body = {"audit_plan_sha256": plan["audit_plan_sha256"], "run_id": args.run_id, "split": args.split, "logical_shard_index": index, "logical_shard_count": plan["logical_shard_count"], "resource": _resource(args)}; _marker(attempt, "STARTED", body); complete = False; started = time.monotonic()
        try:
            owned = tuple(clip for clip in clips if clip.clip_id in set(logical["clip_ids"])); expected = {clip.clip_id: expected_hops_for_clip(clip) for clip in owned}; count = _write_csv(attempt / "per_hop_counterfactual.csv", _evaluate(args, bundle, owned, checkpoints), HOP_FIELDS)
            if count != sum(expected.values()) * 9: _fail("worker row count differs")
            _validate_stream(attempt / "per_hop_counterfactual.csv", expected, plan["checkpoint_identities"]); manifest = {"schema": SCHEMA, **body, "elapsed_seconds": time.monotonic() - started, "work_units": count, "per_hop_bytes": (attempt / "per_hop_counterfactual.csv").stat().st_size, "clip_ids": logical["clip_ids"]}; _write_json(attempt / "shard_manifest.json", manifest); _marker(attempt, "COMPLETE", body, _artifact_map(attempt, ("per_hop_counterfactual.csv", "shard_manifest.json"))); complete = True
        except BaseException:
            if not complete:
                try: _marker(attempt, "FAILED", body)
                except BaseException: pass
            raise
def _completed(args: Any, plan: Mapping[str, Any]) -> list[Path]:
    base = Path(args.audit_root) / args.run_id / "logical_shards"; values = []
    for index, logical in enumerate(plan["logical_shards"]):
        root = _matching_completed(base / f"shard-{index:03d}", plan["audit_plan_sha256"], index)
        if root is None: _fail("merge requires each completed logical shard")
        _validate_completed_attempt(root, plan, index)
        values.append(root)
    return values
def _merge(paths: Sequence[Path], target: Path, plan: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    streams = [iter(_stream(path / "per_hop_counterfactual.csv", HOP_FIELDS)) for path in paths]; heap = []
    for index, stream in enumerate(streams):
        try: row = next(stream)
        except StopIteration: _fail("empty completed shard")
        heapq.heappush(heap, ((row["clip_id"], row["method_id"], row["hop_idx"]), index, row))
    aggregate = _Aggregate(); count = 0
    with target.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HOP_FIELDS, lineterminator="\n"); writer.writeheader()
        while heap:
            _, index, row = heapq.heappop(heap); writer.writerow(row); aggregate.add(row); count += 1
            try: next_row = next(streams[index])
            except StopIteration: continue
            heapq.heappush(heap, ((next_row["clip_id"], next_row["method_id"], next_row["hop_idx"]), index, next_row))
    _validate_stream(target, plan["cohort"]["expected_hops"], plan["checkpoint_identities"]); clips, subjects = aggregate.rows(); return count, clips, subjects
def _benchmark(path: Path, plan: Mapping[str, Any], shards: Sequence[Path], report: Path) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir(): _fail("benchmark path must be fresh")
    manifests = [read_json_object(value / "shard_manifest.json") for value in shards]; work = sum(value["work_units"] for value in manifests); bytes_ = sum(value["per_hop_bytes"] for value in manifests); merge_elapsed = read_json_object(report)["merge_elapsed_seconds"]
    if min(work, bytes_, merge_elapsed) <= 0 or any(not isinstance(value.get("elapsed_seconds"), (int, float)) or value["elapsed_seconds"] <= 0 or not isinstance(value.get("work_units"), int) or value["work_units"] <= 0 for value in manifests): _fail("benchmark measurements invalid")
    compatibility = {key: plan[key] for key in ("code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "plan_id", "checkpoint_identities", "canonical_roi_names", "resource_shape")}
    # A parallel benchmark is bounded by its slowest shard rate, not max elapsed
    # divided by all workers' combined work.
    worker_rate = max(float(value["elapsed_seconds"]) / int(value["work_units"]) for value in manifests)
    _write_json(path, {"schema": BENCHMARK_SCHEMA, "status": "complete", "compatibility": compatibility, "measured": {"worker_seconds_per_hop_checkpoint": worker_rate, "bytes_per_hop_checkpoint": bytes_ / work, "merge_seconds_per_byte": merge_elapsed / bytes_}, "safety_factor": SAFETY_FACTOR, "fixed_reserve_seconds": FIXED_RESERVE_SECONDS, "train_plan_sha256": plan["audit_plan_sha256"], "train_report_sha256": sha256_file(report)})
def merge_shards(args: Any) -> None:
    plan = load_audit_plan(args.audit_plan); _assert_runtime(args, plan); _resource(args); projection = _require_budget(args, plan, "merge"); shards = _completed(args, plan); root = Path(args.audit_root) / args.run_id / "merged"; _fresh(root); body = {"audit_plan_sha256": plan["audit_plan_sha256"], "run_id": args.run_id, "split": args.split, "resource": _resource(args)}; _marker(root, "STARTED", body); complete = False; started = time.monotonic()
    try:
        count, clips, subjects = _merge(shards, root / "per_hop_counterfactual.csv", plan); _write_csv(root / "per_clip_failure.csv", clips, CLIP_FIELDS); _write_csv(root / "per_subject_failure.csv", subjects, SUBJECT_FIELDS)
        if count != sum(plan["cohort"]["expected_hops"].values()) * 9: _fail("merged exact hop count differs")
        report = {"schema": SCHEMA, **body, "hop_counterfactual_row_count": count, "clip_row_count": len(clips), "subject_row_count": len(subjects), "merge_elapsed_seconds": time.monotonic() - started, "benchmark_projection": projection, "limitations": ["One-hop alternatives are not sequential regret.", "CPU-first: any GPU profile needs separate deterministic benchmark evidence.", "Frozen MCD audit only; no training or MMPD access."]}; _write_json(root / "report.json", report); _marker(root, "COMPLETE", body, _artifact_map(root, ("per_hop_counterfactual.csv", "per_clip_failure.csv", "per_subject_failure.csv", "report.json"))); complete = True
        if args.split == "train" and getattr(args, "benchmark_output", None): _benchmark(Path(args.benchmark_output), plan, shards, root / "report.json")
    except BaseException:
        if not complete:
            try: _marker(root, "FAILED", body)
            except BaseException: pass
        raise

__all__ = ["BENCHMARK_SCHEMA", "CLIP_FIELDS", "FIXED_RESERVE_SECONDS", "HOP_FIELDS", "SCHEMA", "SUBJECT_FIELDS", "balanced_subject_assignment", "bind_mcd_clips", "build_audit_plan", "expected_hops_for_clip", "load_audit_plan", "merge_shards", "publish_shard", "replay_clip_without_labels", "score_replay_after_labels", "validate_hop_rows", "write_audit_plan"]
