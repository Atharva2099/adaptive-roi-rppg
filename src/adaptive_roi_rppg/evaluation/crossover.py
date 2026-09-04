"""Gate 6.1 fixed-action crossover diagnostic for MCD.

This is additive to Gate 6.  It compares the same fixed, legal action arm
under the historical NPZ ruler and the current causal POS ruler.  It does not
load MMPD data, learned models, or historical Oracle sequences.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import ROI_NAMES, canonical_json_bytes, publish_directory, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import belief_step, control_step, initial_control_state
from adaptive_roi_rppg.evaluation.parity import HistoricalCacheBinding

ARM_IDS = tuple(f"fixed_{name.value}" for name in ROI_NAMES)
ACTION_COUNT = len(ROI_NAMES)
LEARNED_ARM_STATUS = "not_run_unavailable"
HISTORICAL_ORACLE_STATUS = "unavailable_exact_authenticated_sequence_artifacts"


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        _fail("historical NPZ SHA-256 mismatch")


def _action_distribution(actions: Sequence[int]) -> dict[str, int]:
    return {str(action): actions.count(action) for action in sorted(set(actions))}


def _add_jumps(rows: list[dict[str, Any]]) -> None:
    previous_hr = previous_belief = None
    for row in rows:
        row["hr_jump_abs_bpm"] = abs(row["hr_bpm"] - previous_hr) if row["hr_bpm"] is not None and previous_hr is not None else 0.0
        row["belief_jump_abs_bpm"] = abs(row["belief_hr_bpm"] - previous_belief) if previous_belief is not None else 0.0
        previous_hr = row["hr_bpm"]
        previous_belief = row["belief_hr_bpm"]


def _stats(proposed: Sequence[int], executed: Sequence[int], holds: Sequence[int], overrides: int) -> dict[str, Any]:
    if not (len(proposed) == len(executed) == len(holds)):
        _fail("action statistics have inconsistent lengths")
    return {
        "hops": len(executed),
        "override_count": int(overrides),
        "switch_count": sum(a != b for a, b in zip(executed[1:], executed[:-1])),
        "roi_index_jump_count": sum(a != b for a, b in zip(executed[1:], executed[:-1])),
        "roi_index_jump_mean_abs": float(np.mean([abs(a - b) for a, b in zip(executed[1:], executed[:-1]) if a != b])) if any(a != b for a, b in zip(executed[1:], executed[:-1])) else 0.0,
        "hold_mean": float(np.mean(holds)) if holds else None,
        "hold_max": max(holds, default=None),
        "proposed_action": proposed[0] if proposed and len(set(proposed)) == 1 else None,
        "executed_action_distribution": _action_distribution(list(executed)),
    }


def _score(values: Sequence[tuple[float | None, float]], gt: Sequence[float], action: int) -> tuple[dict[str, Any], ...]:
    if len(values) != len(gt):
        _fail("historical measurements and GT have different hop counts")
    state = initial_control_state("mcd", "gate6_1")
    rows = []
    proposed: list[int] = []
    executed: list[int] = []
    holds: list[int] = []
    overrides = 0
    for hop_idx, ((hr, confidence), target) in enumerate(zip(values, gt)):
        valid = _finite(hr) and _finite(confidence) and float(confidence) > 0
        measurement = float(hr) if valid else None
        belief = belief_step(state.belief, measurement, float(confidence) if valid else None)
        previous = state.previous_action
        if previous is None:
            executed_action, hold, legal = action, 1, True
        elif previous == action:
            executed_action, hold, legal = action, state.hold_count + 1, True
        elif state.hold_count < 2:
            executed_action, hold, legal = previous, state.hold_count + 1, False
        else:
            executed_action, hold, legal = action, 1, True
        overrides += int(executed_action != action)
        proposed.append(action); executed.append(executed_action); holds.append(hold)
        rows.append({"hop_idx": hop_idx, "gt_hr_bpm": float(target), "hr_bpm": measurement,
                     "confidence": float(confidence), "valid": valid,
                     "belief_hr_bpm": belief.mean_hr, "abs_error_bpm": abs(belief.mean_hr - float(target)),
                     "proposed_action": action, "executed_action": executed_action,
                     "legal": legal, "override": executed_action != action, "hold_count": hold})
        state = type(state)("mcd", "gate6_1", hop_idx + 1, belief, executed_action, hold)
    _add_jumps(rows)
    stats = _stats(proposed, executed, holds, overrides)
    stats.update({"hr_jump_mean_abs_bpm": float(np.mean([r["hr_jump_abs_bpm"] for r in rows])) if rows else None,
                  "belief_jump_mean_abs_bpm": float(np.mean([r["belief_jump_abs_bpm"] for r in rows])) if rows else None})
    return tuple(rows), stats


def score_historical_arm(npz_path: str | Path, action: int, *, expected_sha256: str, clip_id: str) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        _fail("action must be in the canonical ROI range")
    path = Path(npz_path)
    _sha(path, expected_sha256)
    with np.load(path, allow_pickle=False) as data:
        hr = np.asarray(data["hr_meas"], dtype=float)[:, action]
        confidence = np.asarray(data["conf"], dtype=float)[:, action]
        gt = np.asarray(data["gt_hr"], dtype=float)
        fs = float(np.asarray(data["fs"]).item())
        dataset_id = "mcd"; stored_clip_id = str(np.asarray(data["stem"]).item())
    if stored_clip_id != clip_id or fs not in (24.0, 30.0): _fail("historical NPZ identity is invalid")
    rows, stats = _score(tuple(zip(hr.tolist(), confidence.tolist())), gt.tolist(), action)
    start = round(8 * fs); hop = round(fs)
    hop_seconds = (start + np.arange(len(rows), dtype=int) * hop) / fs
    for row, hop_time in zip(rows, hop_seconds):
        row.update({"dataset_id": dataset_id, "clip_id": clip_id, "hop_time_s": float(hop_time)})
    return rows, stats


def score_current_arm(measurements: Sequence[Any], labels: Sequence[Any], action: int, clip_id: str) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        _fail("action must be in the canonical ROI range")
    if len(measurements) != len(labels):
        _fail("current measurements and labels have different hop counts")
    state = initial_control_state("mcd", clip_id)
    rows = []; proposed: list[int] = []; executed: list[int] = []; holds: list[int] = []; overrides = 0
    for expected_hop_idx, (frame, label) in enumerate(zip(measurements, labels)):
        if any(getattr(frame, field, None) != getattr(label, field, None) for field in ("dataset_id", "clip_id", "hop_idx")):
            _fail("current measurement and label identity differs")
        if not math.isclose(float(frame.hop_time_s), float(label.hop_time_s), rel_tol=0.0, abs_tol=1e-9):
            _fail("current measurement and label hop time differs")
        if frame.hop_idx != expected_hop_idx or not math.isclose(float(label.hop_time_s), 8.0 + expected_hop_idx, rel_tol=0.0, abs_tol=1e-9):
            _fail("current hop timing does not follow the accepted one-second schedule")
        if frame.dataset_id != "mcd" or frame.clip_id != clip_id or not label.valid or label.gt_hr_bpm is None:
            _fail("current MCD frame/label identity or GT validity is invalid")
        state, transition = control_step(frame, state, action)
        decision = transition.action_decision; selected = transition.selected_measurement
        proposed.append(action); executed.append(decision.executed_action); holds.append(decision.post_hold_count)
        overrides += int(decision.executed_action != action)
        target = float(label.gt_hr_bpm)
        rows.append({"dataset_id": "mcd", "clip_id": clip_id, "hop_idx": int(label.hop_idx), "hop_time_s": float(label.hop_time_s), "gt_hr_bpm": target,
                     "hr_bpm": selected.hr_bpm if selected.valid else None,
                     "confidence": selected.confidence if selected.valid else 0.0,
                     "valid": bool(selected.valid), "belief_hr_bpm": transition.post_belief.mean_hr,
                     "abs_error_bpm": abs(transition.post_belief.mean_hr - target),
                     "proposed_action": action, "executed_action": decision.executed_action,
                     "legal": decision.legal, "override": decision.executed_action != action,
                     "hold_count": decision.post_hold_count, "invalid_reason": selected.invalid_reason or ""})
    _add_jumps(rows)
    stats = _stats(proposed, executed, holds, overrides)
    stats.update({"hr_jump_mean_abs_bpm": float(np.mean([r["hr_jump_abs_bpm"] for r in rows])) if rows else None,
                  "belief_jump_mean_abs_bpm": float(np.mean([r["belief_jump_abs_bpm"] for r in rows])) if rows else None})
    return tuple(rows), stats


def _clip_row(binding: HistoricalCacheBinding, arm_id: str, action: int, historical: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]], historical_stats: Mapping[str, Any], current_stats: Mapping[str, Any]) -> dict[str, Any]:
    hkeys = {(r["dataset_id"], r["clip_id"], r["hop_idx"], r["hop_time_s"]): r for r in historical}
    ckeys = {(r["dataset_id"], r["clip_id"], r["hop_idx"], r["hop_time_s"]): r for r in current}
    if len(hkeys) != len(historical) or len(ckeys) != len(current) or set(hkeys) != set(ckeys):
        _fail(f"{binding.clip_id}/{arm_id}: exact keyed hop alignment failed")
    for key in hkeys:
        if hkeys[key]["gt_hr_bpm"] != ckeys[key]["gt_hr_bpm"]:
            _fail(f"{binding.clip_id}/{arm_id}: historical/current GT differs at {key}")
    h_mae = float(np.mean([r["abs_error_bpm"] for r in historical]))
    c_mae = float(np.mean([r["abs_error_bpm"] for r in current]))
    return {"clip_id": binding.clip_id, "subject_id": binding.subject_id, "view": binding.view,
            "condition": binding.condition, "arm_id": arm_id, "action": action,
            "historical_mae_bpm": h_mae, "current_mae_bpm": c_mae,
            "current_minus_historical_bpm": c_mae - h_mae,
            "historical_invalid_count": sum(not r["valid"] for r in historical),
            "current_invalid_count": sum(not r["valid"] for r in current),
            "historical_action_stats": dict(historical_stats), "current_action_stats": dict(current_stats)}


def build_crossover_rows(bindings: Sequence[HistoricalCacheBinding], current_by_clip: Mapping[str, tuple[Sequence[Any], Sequence[Any]]]) -> list[dict[str, Any]]:
    rows = []
    for binding in bindings:
        if binding.clip_id not in current_by_clip:
            _fail(f"missing current MCD clip {binding.clip_id}")
        measurements, labels = current_by_clip[binding.clip_id]
        for action, arm_id in enumerate(ARM_IDS):
            historical, hs = score_historical_arm(binding.npz_path, action, expected_sha256=binding.npz_sha256, clip_id=binding.clip_id)
            current, cs = score_current_arm(measurements, labels, action, binding.clip_id)
            rows.append(_clip_row(binding, arm_id, action, historical, current, hs, cs))
    return rows


def aggregate_crossover(rows: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int = 2000, seed: int = 6101) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_arm_subject: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows: by_arm_subject[row["arm_id"]][row["subject_id"]].append(row)
    subjects = []; arms = []
    for arm_id in ARM_IDS:
        for subject_id, clip_rows in sorted(by_arm_subject[arm_id].items()):
            h = float(np.mean([r["historical_mae_bpm"] for r in clip_rows])); c = float(np.mean([r["current_mae_bpm"] for r in clip_rows]))
            subjects.append({"arm_id": arm_id, "subject_id": subject_id, "clip_count": len(clip_rows), "historical_equal_clip_mae_bpm": h, "current_equal_clip_mae_bpm": c, "current_minus_historical_bpm": c - h})
        arm_subjects = [r for r in subjects if r["arm_id"] == arm_id]
        deltas = np.asarray([r["current_minus_historical_bpm"] for r in arm_subjects], dtype=float)
        if deltas.size:
            rng = np.random.default_rng(seed + ARM_IDS.index(arm_id)); sample = rng.integers(0, deltas.size, size=(bootstrap_replicates, deltas.size)); means = deltas[sample].mean(axis=1)
            ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
            h_mean = float(np.mean([r["historical_equal_clip_mae_bpm"] for r in arm_subjects])); c_mean = float(np.mean([r["current_equal_clip_mae_bpm"] for r in arm_subjects]))
        else: ci = [None, None]; h_mean = c_mean = None
        clip_rows = [r for r in rows if r["arm_id"] == arm_id]
        def combine(side: str) -> dict[str, Any]:
            values = [r[f"{side}_action_stats"] for r in clip_rows]
            distribution: dict[str, int] = defaultdict(int)
            for value in values:
                for key, count in value["executed_action_distribution"].items(): distribution[key] += count
            hops = sum(value["hops"] for value in values)
            return {"hops": hops, "override_count": sum(value["override_count"] for value in values),
                    "switch_count": sum(value["switch_count"] for value in values),
                    "roi_index_jump_count": sum(value["roi_index_jump_count"] for value in values),
                    "proposed_action": ARM_IDS.index(arm_id),
                    "hold_mean": float(sum(value["hold_mean"] * value["hops"] for value in values if value["hold_mean"] is not None) / hops) if hops else None,
                    "hold_max": max((value["hold_max"] for value in values), default=None),
                    "executed_action_distribution": dict(sorted(distribution.items())),
                    "roi_index_jump_mean_abs": float(sum(value["roi_index_jump_mean_abs"] * value["roi_index_jump_count"] for value in values) / sum(value["roi_index_jump_count"] for value in values)) if sum(value["roi_index_jump_count"] for value in values) else 0.0,
                    "hr_jump_mean_abs_bpm": float(sum(value["hr_jump_mean_abs_bpm"] * value["hops"] for value in values if value["hr_jump_mean_abs_bpm"] is not None) / hops) if hops else None,
                    "belief_jump_mean_abs_bpm": float(sum(value["belief_jump_mean_abs_bpm"] * value["hops"] for value in values if value["belief_jump_mean_abs_bpm"] is not None) / hops) if hops else None}
        arms.append({"arm_id": arm_id, "action": ARM_IDS.index(arm_id), "clip_count": len(clip_rows), "subject_count": len(arm_subjects), "historical_equal_clip_mae_bpm": h_mean, "current_equal_clip_mae_bpm": c_mean, "current_minus_historical_bpm": c_mean - h_mean if h_mean is not None else None, "subject_block_bootstrap_ci95_bpm": ci, "invalid_historical": sum(r["historical_invalid_count"] for r in clip_rows), "invalid_current": sum(r["current_invalid_count"] for r in clip_rows), "historical_action_stats": combine("historical"), "current_action_stats": combine("current")})
    summary = {"dataset_id": "mcd", "arms": [*arms], "learned_arms": {"status": LEARNED_ARM_STATUS}, "historical_oracle_b": {"status": HISTORICAL_ORACLE_STATUS}, "historical_oracle_c": {"status": HISTORICAL_ORACLE_STATUS}, "bootstrap": {"unit": "subject", "replicates": bootstrap_replicates, "seed": seed}, "interpretation": "diagnostic ruler crossover only; no controller-quality or transfer claim"}
    return list(rows), subjects, summary


_SHARD_FILES = {"STARTED.json", "shard.json", "artifacts.sha256", "COMPLETE.json"}


def _validate_marker_tree(root: Path, complete: bool) -> None:
    expected = _SHARD_FILES if complete else {"STARTED.json", "FAILED.json"}
    if set(p.name for p in root.iterdir()) != expected:
        _fail("Gate 6.1 marker file set is invalid")


def _validate_shard_precomplete(root: Path) -> None:
    if {p.name for p in root.iterdir()} != {"STARTED.json", "shard.json", "artifacts.sha256"}:
        _fail("Gate 6.1 shard file set is invalid before COMPLETE")


def _validate_shard_precomplete_integrity(root: Path, payload: Mapping[str, Any], digest: str) -> None:
    try:
        serialized = json.loads((root / "shard.json").read_text(encoding="utf-8"))
        started = json.loads((root / "STARTED.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("shard changed before COMPLETE") from exc
    if serialized != dict(payload) or sha256_file(root / "shard.json") != digest:
        _fail("shard payload changed before COMPLETE")
    if (root / "artifacts.sha256").read_text(encoding="utf-8") != f"{digest}  shard.json\n":
        _fail("shard sidecar changed before COMPLETE")
    fields = ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "command", "node", "job_id", "shard_index", "shard_count")
    if started.get("state") != "started" or any(started.get(field) != payload.get(field) for field in fields):
        _fail("shard STARTED marker changed before COMPLETE")
    if payload.get("schema") != "gate6.1-crossover-shard-v2":
        _fail("shard schema changed before COMPLETE")
    _validate_crossover_rows(payload["rows"], payload["subject_ids"])


def _validate_merged_tree(root: Path, complete: bool) -> None:
    expected = {"STARTED.json", "report.json", "artifacts.sha256"} | ({"COMPLETE.json"} if complete else set())
    if {p.name for p in root.iterdir()} != expected:
        _fail("Gate 6.1 merged file set is invalid")


def _validate_shard_markers(root: Path, payload: Mapping[str, Any], digest: str) -> None:
    started = json.loads((root / "STARTED.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    fields = ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "command", "node", "job_id", "shard_index", "shard_count")
    if started.get("state") != "started" or complete.get("state") != "complete" or complete.get("sha256") != digest:
        _fail("Gate 6.1 shard terminal marker is invalid")
    if any(started.get(key) != payload.get(key) for key in fields) or any(complete.get(key) != payload.get(key) for key in fields):
        _fail("Gate 6.1 shard marker provenance mismatch")


def _validate_crossover_rows(rows: Sequence[Mapping[str, Any]], subject_ids: Sequence[str]) -> None:
    if not rows:
        _fail("empty shard")
    required = ("clip_id", "subject_id", "view", "condition", "arm_id", "action",
                "historical_mae_bpm", "current_mae_bpm", "current_minus_historical_bpm",
                "historical_invalid_count", "current_invalid_count",
                "historical_action_stats", "current_action_stats")
    stat_fields = ("hops", "override_count", "switch_count", "roi_index_jump_count",
                   "proposed_action",
                   "hold_mean", "hold_max", "executed_action_distribution", "roi_index_jump_mean_abs",
                   "hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm")

    def validate_stats(stats: Any, action: int) -> None:
        if not isinstance(stats, Mapping) or any(field not in stats for field in stat_fields):
            _fail("action statistics schema is incomplete")
        hops = stats["hops"]
        if type(hops) is not int or hops <= 0:
            _fail("action statistics hops must be a positive integer")
        for field in ("override_count", "switch_count", "roi_index_jump_count"):
            value = stats[field]
            if type(value) is not int or not 0 <= value <= hops:
                _fail("action statistic count is invalid")
        if stats["proposed_action"] != action:
            _fail("action statistics proposed action does not match row action")
        if stats["switch_count"] > hops - 1 or stats["roi_index_jump_count"] > hops - 1:
            _fail("action switch count is impossible")
        distribution = stats["executed_action_distribution"]
        if not isinstance(distribution, Mapping) or any(
                not isinstance(key, str) or key not in {str(index) for index in range(ACTION_COUNT)}
                or type(value) is not int or value < 0
                for key, value in distribution.items()):
            _fail("executed action distribution is invalid")
        if sum(distribution.values()) != hops:
            _fail("executed action distribution does not match hops")
        if distribution != {str(action): hops}:
            _fail("fixed-action distribution contains an unexpected action")
        if stats["override_count"] != 0 or stats["switch_count"] != 0 or stats["roi_index_jump_count"] != 0:
            _fail("fixed-action statistics report an impossible transition")
        if stats["override_count"] != sum(value for key, value in distribution.items() if int(key) != action):
            _fail("override count does not match executed actions")
        for field in ("hold_mean", "hold_max", "roi_index_jump_mean_abs", "hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm"):
            if stats[field] is not None and not _finite(stats[field]):
                _fail("action statistic value is non-finite")
        if stats["hold_mean"] is not None and stats["hold_mean"] < 1:
            _fail("hold mean is invalid")
        if stats["hold_max"] is not None and (stats["hold_max"] < 1 or stats["hold_max"] > hops):
            _fail("hold maximum is invalid")
        if stats["hold_max"] != hops or not math.isclose(stats["hold_mean"], (hops + 1) / 2, rel_tol=0.0, abs_tol=1e-12):
            _fail("fixed-action hold statistics are inconsistent")
        if stats["roi_index_jump_mean_abs"] is None or not _finite(stats["roi_index_jump_mean_abs"]):
            _fail("ROI-index jump magnitude is required and must be finite")
        for field in ("roi_index_jump_mean_abs", "hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm"):
            if stats[field] is not None and stats[field] < 0:
                _fail("jump statistic is invalid")

    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(field not in row for field in required):
            _fail("crossover row schema is incomplete")
        if any(not isinstance(row[field], str) or not row[field] for field in ("clip_id", "subject_id", "view", "condition", "arm_id")):
            _fail("crossover row identity is invalid")
        if row["arm_id"] not in ARM_IDS or type(row["action"]) is not int or row["action"] != ARM_IDS.index(row["arm_id"]):
            _fail("invalid arm/action binding")
        if any(not _finite(row[field]) for field in ("historical_mae_bpm", "current_mae_bpm", "current_minus_historical_bpm")):
            _fail("non-finite crossover metric")
        if row["historical_mae_bpm"] < 0 or row["current_mae_bpm"] < 0:
            _fail("MAE cannot be negative")
        if not math.isclose(row["current_minus_historical_bpm"], row["current_mae_bpm"] - row["historical_mae_bpm"], rel_tol=0.0, abs_tol=1e-12):
            _fail("crossover delta does not match MAE values")
        for field in ("historical_invalid_count", "current_invalid_count"):
            if type(row[field]) is not int or row[field] < 0:
                _fail("invalid hop count")
        validate_stats(row["historical_action_stats"], row["action"])
        validate_stats(row["current_action_stats"], row["action"])
        for field, stats_key in (("historical_invalid_count", "historical_action_stats"), ("current_invalid_count", "current_action_stats")):
            if row[field] > row[stats_key]["hops"]:
                _fail("invalid hop count exceeds action-stat hops")
        by_clip[row["clip_id"]].append(row)
    if any(not isinstance(subject, str) or not subject for subject in subject_ids):
        _fail("subject list contains an invalid identity")
    if {row["subject_id"] for row in rows} != set(subject_ids):
        _fail("shard subject list does not match rows")
    for clip_id, clip_rows in by_clip.items():
        if len(clip_rows) != ACTION_COUNT or {row["arm_id"] for row in clip_rows} != set(ARM_IDS):
            _fail(f"{clip_id}: expected exactly 12 unique arms")
        if len({(row["subject_id"], row["view"], row["condition"]) for row in clip_rows}) != 1:
            _fail(f"{clip_id}: inconsistent clip metadata")


def publish_crossover_shard(rows: Sequence[Mapping[str, Any]], destination: str | Path, *, shard_index: int, shard_count: int, plan_id: str = "gate6.1", phase: str = "full", code_snapshot_sha256: str = "", source_inventory_sha256: str = "", subject_ids: Sequence[str] = (), command: str = "", node: str = "", job_id: str = "") -> None:
    started = {"state": "started", "plan_id": plan_id, "phase": phase, "shard_index": shard_index, "shard_count": shard_count, "code_snapshot_sha256": code_snapshot_sha256, "source_inventory_sha256": source_inventory_sha256, "command": command, "node": node, "job_id": job_id}
    failed = {"state": "failed", **{key: value for key, value in started.items() if key != "state"}}
    def write_artifacts(root):
        if shard_count < 1 or not 0 <= shard_index < shard_count: _fail("invalid shard index/count")
        if len(code_snapshot_sha256) != 64 or len(source_inventory_sha256) != 64: _fail("missing provenance hashes")
        if not command or not node or not job_id: _fail("missing command/node/job provenance")
        _validate_crossover_rows(rows, subject_ids)
        payload = {"schema": "gate6.1-crossover-shard-v2", "plan_id": plan_id, "phase": phase, "code_snapshot_sha256": code_snapshot_sha256, "source_inventory_sha256": source_inventory_sha256, "command": command, "node": node, "job_id": job_id, "shard_index": shard_index, "shard_count": shard_count, "subject_ids": sorted(subject_ids), "rows": list(rows)}
        (root / "shard.json").write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        digest = sha256_file(root / "shard.json")
        (root / "artifacts.sha256").write_text(f"{digest}  shard.json\n", encoding="utf-8")
        _validate_shard_precomplete(root)
        _validate_shard_precomplete_integrity(root, payload, digest)
    def write_marker(root, payload):
        with (root / ("STARTED.json" if payload["state"] == "started" else "COMPLETE.json" if payload["state"] == "complete" else "FAILED.json")).open("x") as handle:
            json.dump(payload, handle, sort_keys=True)
    publish_directory(destination, write_started=lambda root: write_marker(root, started), write_artifacts=write_artifacts,
                      validate_precomplete=lambda root: None, write_complete=lambda root: write_marker(root, {"state": "complete", **{key: value for key, value in started.items() if key != "state"}, "sha256": sha256_file(root / "shard.json")}),
                      write_failed=lambda root: write_marker(root, failed), substantive_files=("shard.json", "artifacts.sha256"))


def _merge_crossover_shards_impl(shard_dirs: Sequence[str | Path], destination: str | Path, *, bootstrap_replicates: int = 2000, seed: int = 6101, expected_phase: str = "full", expected_plan_id: str | None = None, expected_code_snapshot_sha256: str | None = None, expected_source_inventory_sha256: str | None = None, expected_clip_subjects: Mapping[str, str] | None = None, expected_clip_metadata: Mapping[str, Mapping[str, str]] | None = None, merged_provenance: Mapping[str, str] | None = None) -> dict[str, Any]:
    payloads = []
    for directory in shard_dirs:
        root = Path(directory)
        if not (root / "COMPLETE.json").is_file(): _fail("incomplete shard")
        _validate_marker_tree(root, True)
        try:
            payload = json.loads((root / "shard.json").read_text(encoding="utf-8")); marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError("tampered shard JSON") from exc
        sidecar = (root / "artifacts.sha256").read_text(encoding="utf-8").strip()
        if marker.get("sha256") != sha256_file(root / "shard.json") or sidecar != f"{marker['sha256']}  shard.json" or payload.get("schema") != "gate6.1-crossover-shard-v2" or marker.get("state") != "complete": _fail("tampered shard")
        _validate_shard_markers(root, payload, marker["sha256"])
        for key in ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "command", "node", "job_id", "shard_index", "shard_count"):
            if marker.get(key) != payload.get(key): _fail(f"shard marker {key} mismatch")
        for key, expected in (("phase", expected_phase), ("plan_id", expected_plan_id), ("code_snapshot_sha256", expected_code_snapshot_sha256), ("source_inventory_sha256", expected_source_inventory_sha256)):
            if expected is not None and payload.get(key) != expected: _fail(f"shard {key} mismatch")
        if not payload.get("rows") or len(set(payload.get("subject_ids", ()))) != len(payload.get("subject_ids", ())): _fail("empty or duplicate-subject shard")
        _validate_crossover_rows(payload["rows"], payload["subject_ids"])
        if set(payload["subject_ids"]) != {row["subject_id"] for row in payload["rows"]}: _fail("shard subject binding mismatch")
        payloads.append(payload)
    if not payloads: _fail("no shards")
    provenance = {(p.get("plan_id"), p.get("phase"), p.get("code_snapshot_sha256"), p.get("source_inventory_sha256")) for p in payloads}
    if len(provenance) != 1: _fail("mixed shard plan/code/source provenance")
    counts = {p["shard_count"] for p in payloads}; indices = [p["shard_index"] for p in payloads]
    if len(counts) != 1 or sorted(indices) != list(range(next(iter(counts)))) or len(indices) != len(set(indices)): _fail("shards are not a complete deterministic set")
    rows = [row for payload in sorted(payloads, key=lambda p: p["shard_index"]) for row in payload["rows"]]
    required_row_fields = ("clip_id", "subject_id", "view", "condition", "arm_id", "action")
    for row in rows:
        if any(field not in row for field in required_row_fields): _fail("clip row metadata is incomplete")
        if row["arm_id"] not in ARM_IDS or isinstance(row["action"], bool) or not isinstance(row["action"], int) or row["action"] != ARM_IDS.index(row["arm_id"]): _fail("invalid arm/action binding")
    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: by_clip[row["clip_id"]].append(row)
    if any(len(clip_rows) != ACTION_COUNT or {r["arm_id"] for r in clip_rows} != set(ARM_IDS) for clip_rows in by_clip.values()): _fail("each clip must have exactly 12 unique arms")
    if len({(r["clip_id"], r["arm_id"]) for r in rows}) != len(rows): _fail("duplicate clip/arm row")
    for row in rows:
        clip_rows = by_clip[row["clip_id"]]
        if len({(r["subject_id"], r["view"], r["condition"]) for r in clip_rows}) != 1: _fail("mixed subject/clip metadata")
        if expected_clip_metadata is not None:
            expected = expected_clip_metadata.get(row.get("clip_id"))
            if expected is None or any(row.get(key) != expected[key] for key in ("subject_id", "view", "condition")): _fail("clip metadata does not match expected plan")
    if expected_clip_subjects is not None:
        expected_subjects = set(expected_clip_subjects.values())
        if expected_phase == "full" and (len(expected_clip_subjects) != 533 or len(expected_subjects) != 89): _fail("full phase cohort is not 533 clips/89 subjects")
        assigned: set[str] = set()
        for payload in payloads:
            subjects = set(payload["subject_ids"])
            if subjects & assigned: _fail("duplicate subject across shards")
            assigned |= subjects
            ordered = sorted(expected_subjects)
            if subjects != {subject for index, subject in enumerate(ordered) if index % next(iter(counts)) == payload["shard_index"]}: _fail("wrong shard subject assignment")
        if assigned != expected_subjects: _fail("missing subject shard")
        if set(by_clip) != set(expected_clip_subjects) or any(expected_clip_subjects[c] != rows_for_clip[0]["subject_id"] for c, rows_for_clip in by_clip.items()): _fail("missing, extra, or misbound clip")
    _, subjects, summary = aggregate_crossover(rows, bootstrap_replicates=bootstrap_replicates, seed=seed)
    for arm in summary["arms"]:
        for side in ("historical_action_stats", "current_action_stats"):
            stats = arm[side]
            if stats.get("proposed_action") != arm["action"]:
                _fail("merged action statistics proposed action mismatch")
            if type(stats.get("hops")) is not int or stats["hops"] <= 0:
                _fail("merged action statistics hops are invalid")
            if any(type(stats.get(field)) is not int or not 0 <= stats[field] <= stats["hops"] for field in ("override_count", "switch_count", "roi_index_jump_count")):
                _fail("merged action statistic count is invalid")
            if stats["switch_count"] > stats["hops"] - 1 or stats["roi_index_jump_count"] > stats["hops"] - 1:
                _fail("merged action switch count is impossible")
            distribution = stats.get("executed_action_distribution")
            if not isinstance(distribution, Mapping) or any(key not in {str(index) for index in range(ACTION_COUNT)} or type(value) is not int or value < 0 for key, value in distribution.items()) or sum(distribution.values()) != stats["hops"]:
                _fail("merged executed action distribution is invalid")
            if stats["override_count"] != sum(value for key, value in distribution.items() if int(key) != arm["action"]):
                _fail("merged override count is inconsistent")
            if distribution != {str(arm["action"]): stats["hops"]} or stats["switch_count"] != 0 or stats["roi_index_jump_count"] != 0 or stats["roi_index_jump_mean_abs"] != 0.0:
                _fail("merged fixed-action statistics are inconsistent")
            for field in ("hold_mean", "hold_max", "roi_index_jump_mean_abs", "hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm"):
                if stats.get(field) is not None and (not _finite(stats[field]) or stats[field] < 0):
                    _fail("merged action statistic value is invalid")
            if stats.get("roi_index_jump_mean_abs") is None:
                _fail("merged ROI-index jump magnitude is required")
            if stats["hold_mean"] is not None and stats["hold_mean"] < 1:
                _fail("merged hold mean is invalid")
            if stats["hold_max"] is not None and (stats["hold_max"] < 1 or stats["hold_max"] > stats["hops"]):
                _fail("merged hold maximum is invalid")
            clip_stats = [row[side] for row in rows if row["arm_id"] == arm["arm_id"]]
            expected_hold_mean = sum(value["hold_mean"] * value["hops"] for value in clip_stats) / sum(value["hops"] for value in clip_stats)
            expected_hold_max = max(value["hold_max"] for value in clip_stats)
            if stats["hold_max"] != expected_hold_max or not math.isclose(stats["hold_mean"], expected_hold_mean, rel_tol=0.0, abs_tol=1e-12):
                _fail("merged fixed-action hold statistics are inconsistent")
    root = Path(destination)
    if not merged_provenance or any(not merged_provenance.get(key) for key in ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "command", "node", "job_id")):
        _fail("merged provenance is incomplete")
    common = payloads[0]
    for key, expected in (("plan_id", expected_plan_id), ("phase", expected_phase),
                          ("code_snapshot_sha256", expected_code_snapshot_sha256),
                          ("source_inventory_sha256", expected_source_inventory_sha256)):
        if expected is not None and merged_provenance.get(key) != expected:
            _fail(f"merged provenance {key} mismatch")
        if merged_provenance.get(key) != common.get(key):
            _fail(f"merged provenance {key} disagrees with shard provenance")
    result = {"schema": "gate6.1-crossover-v1", "rows": rows, "subjects": subjects, "summary": summary, "provenance": dict(merged_provenance), "shard_provenance": [{"shard_index": p["shard_index"], "job_id": p["job_id"], "node": p["node"], "command": p["command"]} for p in payloads]}
    (root / "report.json").write_text(json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    digest = sha256_file(root / "report.json"); (root / "artifacts.sha256").write_text(f"{digest}  report.json\n", encoding="utf-8")
    if sha256_file(root / "report.json") != digest or (root / "artifacts.sha256").read_text(encoding="utf-8") != f"{digest}  report.json\n": _fail("merged report sidecar validation failed")
    return result


def merge_crossover_shards(shard_dirs: Sequence[str | Path], destination: str | Path, *, bootstrap_replicates: int = 2000, seed: int = 6101, expected_phase: str = "full", expected_plan_id: str | None = None, expected_code_snapshot_sha256: str | None = None, expected_source_inventory_sha256: str | None = None, expected_clip_subjects: Mapping[str, str] | None = None, expected_clip_metadata: Mapping[str, Mapping[str, str]] | None = None, merged_provenance: Mapping[str, str] | None = None) -> dict[str, Any]:
    result_box = {}
    provenance = dict(merged_provenance or {})
    def write_artifacts(root):
        result_box["result"] = _merge_crossover_shards_impl(shard_dirs, root, bootstrap_replicates=bootstrap_replicates, seed=seed, expected_phase=expected_phase, expected_plan_id=expected_plan_id, expected_code_snapshot_sha256=expected_code_snapshot_sha256, expected_source_inventory_sha256=expected_source_inventory_sha256, expected_clip_subjects=expected_clip_subjects, expected_clip_metadata=expected_clip_metadata, merged_provenance=merged_provenance)
        expected_report = {"STARTED.json", "report.json", "artifacts.sha256"}
        if {p.name for p in root.iterdir()} != expected_report: _fail("merged output file set is invalid before COMPLETE")
        started = json.loads((root / "STARTED.json").read_text(encoding="utf-8"))
        if started.get("state") != "started": _fail("merged STARTED marker is invalid")
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if report.get("provenance") != dict(merged_provenance or {}): _fail("merged report provenance mismatch")
        if any(started.get(key) != (merged_provenance or {}).get(key) for key in ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "command", "node", "job_id")): _fail("merged STARTED provenance mismatch")
    def write_marker(root, state):
        name = "STARTED.json" if state == "started" else "COMPLETE.json" if state == "complete" else "FAILED.json"
        with (root / name).open("x") as handle:
            json.dump({"state": state, **provenance, **({"sha256": sha256_file(root / "report.json")} if state == "complete" else {})}, handle, sort_keys=True)
    publish_directory(destination, write_started=lambda root: write_marker(root, "started"), write_artifacts=write_artifacts,
                      validate_precomplete=lambda root: None, write_complete=lambda root: write_marker(root, "complete"),
                      write_failed=lambda root: write_marker(root, "failed"), substantive_files=("report.json", "artifacts.sha256"))
    return result_box["result"]


__all__ = ["ACTION_COUNT", "ARM_IDS", "LEARNED_ARM_STATUS", "HISTORICAL_ORACLE_STATUS", "score_historical_arm", "score_current_arm", "build_crossover_rows", "aggregate_crossover", "publish_crossover_shard", "merge_crossover_shards"]
