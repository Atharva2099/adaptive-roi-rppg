"""Authenticated, observational Gate 6 ruler-difference report for MCD.

The frozen inputs contain outputs, not the historical source windows, masks, or
spectra.  This module therefore classifies observed value differences only; it
does not claim a causal decomposition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, belief_step, control_step, initial_control_state
from adaptive_roi_rppg.data.mcd import MCD_DATASET_ID, load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.labels import GT_RULE_ID
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import POS_CONFIG_ID, build_pos_measurements

HISTORICAL_CACHE_MANIFEST_SHA256 = "d9c63ab03582ae254e067ce103e6ddb563019c65f1087ee723bcd9614d55aadf"
HISTORICAL_CACHE_INVENTORY_SHA256 = "ad2a2cfa8d11ac6978d72958a4cadd8027e7eb9502a9482b53df8cf3f77c2f4c"
HISTORICAL_PER_CLIP_SHA256 = "7481c4e5bcecda77dd954e78d49d7e6cb435a5353bf89f6a4a30187ce2169f2e"
HISTORICAL_TEST90_CSV_SHA256 = HISTORICAL_PER_CLIP_SHA256
HISTORICAL_LEVEL_A_MEAN_MAE = 12.386529570502146
HISTORICAL_EXCLUDED_CLIPS = frozenset({
    "4952_FullHDwebcam_after", "4952_FullHDwebcam_before",
    "4952_IriunWebcam_after", "4952_IriunWebcam_before",
    "4952_USBVideo_after", "4952_USBVideo_before", "6066_USBVideo_before",
})
OBSERVATIONAL_CATEGORIES = (
    "exact_observed_match", "hop_historical_only", "hop_current_only",
    "gt_value_difference", "validity_difference", "measurement_value_difference",
    "confidence_difference", "belief_value_difference", "numerical_difference",
    "unclassified",
)
_NPZ_KEYS = frozenset({"hr_meas", "conf", "ppr", "cov", "gt_hr", "view", "condition", "fs", "stem", "cache_schema_version", "gt_subharmonic_labels"})
PARITY_HOP_FIELDS = ("run_id", "plan_id", "dataset_id", "clip_id", "subject_id", "view", "condition", "hop_idx", "hop_time_s", "historical_hop_count", "current_hop_count", "hop_presence", "historical_full_face_hr_bpm", "current_full_face_hr_bpm", "full_face_hr_delta_bpm", "historical_confidence", "current_confidence", "confidence_delta", "historical_gt_hr_bpm", "current_gt_hr_bpm", "gt_delta_bpm", "historical_post_belief_hr_bpm", "current_post_belief_hr_bpm", "belief_delta_bpm", "historical_abs_error_bpm", "current_abs_error_bpm", "abs_error_delta_bpm", "current_selected_valid", "current_invalid_reason", "source_window_complete", "diagnostic_status", "discrepancy_category", "signal_config_id", "control_config_id", "gt_rule_id", "source_inventory_sha256")
PARITY_CLIP_FIELDS = ("run_id", "plan_id", "dataset_id", "clip_id", "subject_id", "view", "condition", "expected_hops", "historical_hops", "current_hops", "joined_hops", "historical_only_hops", "current_only_hops", "historical_mae_clip", "current_mae_clip", "mae_delta_bpm", "historical_mae_hop_count", "current_mae_hop_count", "delta_joined_hop_count", "max_abs_belief_delta_bpm", "current_invalid_count", "unclassified_count", "fixture_member", "clip_status", "source_inventory_sha256")
_OUTPUTS = ("parity_per_hop.csv", "parity_per_clip.csv", "parity_by_subject.csv", "parity_by_category.csv", "parity_summary.json", "plan.json", "run_manifest.json", "source_inventory.json", "artifacts.sha256")

def _fail(message: str) -> None: raise ContractValidationError(message)
def _sha(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected: _fail(f"{field}: SHA-256 mismatch")
def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): _fail(f"{field}: invalid SHA-256")
    return value

@dataclass(frozen=True, slots=True)
class HistoricalCacheBinding:
    clip_id: str; subject_id: str; view: str; condition: str; npz_filename: str; npz_sha256: str; npz_path: str; expected_hops: int; reference_mae_clip: float
@dataclass(frozen=True, slots=True)
class Gate6ParityPlan:
    plan_id: str; dataset_id: str; phase: str; fixture_subject_ids: tuple[str, ...]; bindings: tuple[HistoricalCacheBinding, ...]; source_inventory_sha256: str = ""
@dataclass(frozen=True, slots=True)
class HistoricalFullFaceHop:
    hop_idx: int; hop_time_s: float; hr_bpm: float | None; confidence: float; gt_hr_bpm: float; post_belief_hr_bpm: float
@dataclass(frozen=True, slots=True)
class ParityHopRow: values: Mapping[str, Any]
@dataclass(frozen=True, slots=True)
class ParityClipRow: values: Mapping[str, Any]
@dataclass(frozen=True, slots=True)
class ParitySummary: values: Mapping[str, Any]

def _inventory_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if set(payload) != {"cache_directory", "files", "n_files", "schema"} or payload.get("schema") != "streaming-v3nc-cache-inventory-v1": _fail("historical inventory schema is not exact")
    records = payload["files"]
    if type(payload["n_files"]) is not int or not isinstance(records, list) or payload["n_files"] != len(records) or not records: _fail("historical inventory file count is invalid")
    out = []
    for record in records:
        if set(record) != {"path", "sha256", "size"} or Path(record["path"]).name != record["path"] or Path(record["path"]).suffix != ".npz": _fail("historical inventory record is invalid")
        if type(record["size"]) is not int or record["size"] <= 0: _fail("historical inventory size is invalid")
        _digest(record["sha256"], "historical NPZ SHA-256"); out.append(dict(record))
    if len({r["path"] for r in out}) != len(out): _fail("historical inventory has duplicate filenames")
    return out

def _read_npz_metadata(path: Path, expected_sha256: str) -> dict[str, Any]:
    _sha(path, expected_sha256, "historical NPZ")
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _NPZ_KEYS: _fail("historical NPZ keys are not the exact 11-key envelope")
            arrays = {key: np.asarray(data[key]) for key in data.files}
            def scalar(key):
                if arrays[key].ndim != 0: _fail(f"historical {key} must be scalar")
                return arrays[key].item()
            stem, view, condition = scalar("stem"), scalar("view"), scalar("condition")
            if not all(isinstance(v, str) and v for v in (stem, view, condition)) or int(scalar("cache_schema_version")) != 3 or bool(scalar("gt_subharmonic_labels")): _fail("historical NPZ metadata/correction binding is invalid")
            fs = float(scalar("fs")); hr = arrays["hr_meas"]
            if fs not in (24.0, 30.0) or hr.ndim != 2 or hr.shape[1] != 12: _fail("historical NPZ shape/FPS is invalid")
            n = hr.shape[0]
            if not np.isfinite(hr[~np.isnan(hr)]).all() or arrays["gt_hr"].shape != (n,) or not np.isfinite(arrays["gt_hr"]).all(): _fail("historical NPZ HR/GT is invalid")
            for key in ("conf", "ppr", "cov"):
                if arrays[key].shape != (n, 12) or not np.isfinite(arrays[key]).all(): _fail("historical NPZ array shape/finiteness is invalid")
            if np.any(arrays["conf"] < 0) or np.any(np.isnan(hr) & (arrays["conf"] > 0)): _fail("historical NaN HR requires zero confidence")
            return {"stem": stem, "view": view, "condition": condition, "fs": fs, "expected_hops": n}
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ContractValidationError): raise
        raise ContractValidationError("historical NPZ cannot be read") from exc

def _validate_test90(path: Path) -> tuple[set[str], dict[str, float]]:
    _sha(path, HISTORICAL_TEST90_CSV_SHA256, "historical per_clip/test90 CSV")
    try:
        with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        if len(rows) != 3198: _fail("historical per_clip must contain 3198 rows")
        fields = rows[0].keys(); arm_field = next((f for f in ("policy_arm", "arm", "level") if f in fields), None); stem_field = next((f for f in ("clip_id", "stem", "clip") if f in fields), None); mae_field = next((f for f in ("mae_clip", "clip_mae", "mae") if f in fields), None)
        if not arm_field or not stem_field or not mae_field: _fail("historical per_clip lacks arm/stem/MAE fields")
        counts = Counter(row[arm_field] for row in rows)
        if set(counts) != {"A", "Aprime", "B", "C", "D", "Dv2"} or any(v != 533 for v in counts.values()): _fail("historical per_clip is not six complete arms")
        a = [row for row in rows if row[arm_field] == "A"]; values = [float(row[mae_field]) for row in a]
        if len({row[stem_field] for row in a}) != 533 or not all(math.isfinite(v) for v in values) or not math.isclose(sum(values) / len(values), HISTORICAL_LEVEL_A_MEAN_MAE, rel_tol=0.0, abs_tol=1e-12): _fail("historical Level-A cohort/mean is not exact")
        return {row[stem_field] for row in a}, {row[stem_field]: float(row[mae_field]) for row in a}
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        if isinstance(exc, ContractValidationError): raise
        raise ContractValidationError("historical per_clip is malformed") from exc

def load_historical_cache_inventory(cache_manifest, cache_inventory, historical_per_clip=None, historical_test90_csv=None):
    _sha(Path(cache_manifest), HISTORICAL_CACHE_MANIFEST_SHA256, "historical cache manifest"); _sha(Path(cache_inventory), HISTORICAL_CACHE_INVENTORY_SHA256, "historical cache inventory")
    records = _inventory_records(read_json_object(Path(cache_inventory)))
    if historical_test90_csv is not None: _validate_test90(Path(historical_test90_csv))
    elif historical_per_clip is not None: _validate_test90(Path(historical_per_clip))
    return tuple(records)

def _plan(manifest_tree, cache_root, cache_manifest, cache_inventory, per_clip, phase, test90_csv=None):
    if phase not in ("fixture", "full"): _fail("phase must be fixture or full")
    stems, refs = _validate_test90(Path(test90_csv or per_clip))
    records = load_historical_cache_inventory(cache_manifest, cache_inventory, historical_test90_csv=test90_csv or per_clip)
    bundle = load_mcd_manifest_tree(manifest_tree); root = Path(cache_root)
    if bundle.dataset_manifest.dataset_id != MCD_DATASET_ID or root.is_symlink() or not root.is_dir(): _fail("Gate 6 source identity is invalid")
    current = {c.clip_id: c for c in bundle.clip_manifests if c.clip_id in bundle.split_manifest.eval_clip_ids and c.clip_id not in HISTORICAL_EXCLUDED_CLIPS}
    bindings = []
    for record in sorted(records, key=lambda r: r["path"]):
        clip_id = Path(record["path"]).stem
        if clip_id not in stems or clip_id not in current: _fail(f"historical cohort contains unbound clip {clip_id}")
        path = root / record["path"]; meta = _read_npz_metadata(path, record["sha256"]); clip = current[clip_id]
        if meta["stem"] != clip_id or meta["view"] != clip.view or meta["condition"] != clip.condition: _fail(f"historical metadata does not bind {clip_id}")
        bindings.append(HistoricalCacheBinding(clip_id, clip.subject_id, clip.view, clip.condition, record["path"], record["sha256"], str(path), meta["expected_hops"], refs[clip_id]))
    if {b.clip_id for b in bindings} != stems or {p.name for p in root.iterdir() if p.is_file() and p.suffix == ".npz"} != {b.npz_filename for b in bindings}: _fail("historical NPZ/stem cohort is not exact")
    subjects = tuple(sorted({b.subject_id for b in bindings})); selected = ("1107", "1314") if phase == "fixture" else subjects; chosen = tuple(b for b in bindings if b.subject_id in selected)
    if phase == "fixture" and (len(chosen) != 12 or selected != ("1107", "1314")): _fail("fixture cohort must be 12 clips from subjects 1107 and 1314")
    if phase == "full" and (len(chosen) != 533 or len(selected) != 89 or sum(s != "6066" for s in selected) != 88 or sum(b.subject_id == "6066" for b in chosen) != 5): _fail("full cohort must be 533 clips, 89 subjects, with 6066 at five clips")
    identity = {"phase": phase, "dataset_id": MCD_DATASET_ID, "clips": [(b.clip_id, b.expected_hops) for b in chosen], "exclusions": sorted(HISTORICAL_EXCLUDED_CLIPS)}
    return Gate6ParityPlan("gate6-ruler-difference-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest(), MCD_DATASET_ID, phase, selected, chosen)
def build_gate6_fixture_plan(manifest_tree, cache_root, cache_manifest, cache_inventory, per_clip, historical_test90_csv=None): return _plan(manifest_tree, cache_root, cache_manifest, cache_inventory, per_clip, "fixture", historical_test90_csv)
def build_gate6_full_plan(manifest_tree, cache_root, cache_manifest, cache_inventory, per_clip, historical_test90_csv=None): return _plan(manifest_tree, cache_root, cache_manifest, cache_inventory, per_clip, "full", historical_test90_csv)

def replay_historical_full_face(npz_path, expected_sha256=None):
    path = Path(npz_path); digest = sha256_file(path) if path.is_file() and not path.is_symlink() else ""; _sha(path, expected_sha256 or digest, "historical NPZ")
    meta = _read_npz_metadata(path, digest)
    with np.load(path, allow_pickle=False) as data: hr, conf, gt = np.asarray(data["hr_meas"], float)[:, 0], np.asarray(data["conf"], float)[:, 0], np.asarray(data["gt_hr"], float)
    state = initial_control_state(MCD_DATASET_ID, path.stem); rows = []
    for i, value in enumerate(hr):
        measurement = float(value) if math.isfinite(value) and conf[i] > 0 else None; belief = belief_step(state.belief, measurement, float(conf[i])); rows.append(HistoricalFullFaceHop(i, (round(8 * meta["fs"]) + i * round(meta["fs"])) / meta["fs"], measurement, float(conf[i]), float(gt[i]), belief.mean_hr)); state = type(state)(state.dataset_id, state.clip_id, i + 1, belief, 0, state.hold_count + 1 if state.previous_action == 0 else 1)
    return tuple(rows)

def evaluate_current_full_face(manifest_tree, state_root, gt_root, plan):
    bundle = load_mcd_manifest_tree(manifest_tree); out = []
    for binding in plan.bindings:
        clip = next((c for c in bundle.clip_manifests if c.clip_id == binding.clip_id), None)
        if clip is None or clip.subject_id != binding.subject_id: _fail("current clip identity mismatch")
        measurements = build_pos_measurements(read_mcd_canonical_frames(bundle, state_root, binding.clip_id, "eval")); labels = read_mcd_eval_labels(bundle, gt_root, binding.clip_id, clip)
        if len(measurements) != len(labels): _fail("current signal/GT hop counts differ")
        state = initial_control_state(MCD_DATASET_ID, binding.clip_id)
        for measurement_frame, label in zip(measurements, labels):
            state, transition = control_step(measurement_frame, state, 0); selected = transition.selected_measurement
            out.append({"dataset_id": MCD_DATASET_ID, "clip_id": binding.clip_id, "subject_id": binding.subject_id, "view": binding.view, "condition": binding.condition, "hop_idx": label.hop_idx, "hop_time_s": label.hop_time_s, "hr": selected.hr_bpm, "confidence": selected.confidence if selected.valid else 0.0, "gt": label.gt_hr_bpm, "valid": selected.valid, "reason": selected.invalid_reason or "", "belief": transition.post_belief.mean_hr, "source_window_complete": True})
    return tuple(out)

def classify_discrepancy(historical, current, *, identity_ok=True, gt_equal=None, **_ignored):
    if not identity_ok: return "hop_historical_only" if historical is not None and current is None else "hop_current_only" if current is not None else "unclassified"
    if historical is None: return "hop_current_only"
    if current is None: return "hop_historical_only"
    if gt_equal is False or current.get("gt") != historical.gt_hr_bpm: return "gt_value_difference"
    h_valid = historical.hr_bpm is not None and historical.confidence > 0; c_valid = bool(current.get("valid"))
    if h_valid != c_valid: return "validity_difference"
    if h_valid and current.get("hr") != historical.hr_bpm: return "measurement_value_difference"
    if current.get("confidence") != historical.confidence: return "confidence_difference"
    if current.get("belief") != historical.post_belief_hr_bpm: return "belief_value_difference"
    if current.get("gt") == historical.gt_hr_bpm and current.get("hr") == historical.hr_bpm and current.get("confidence") == historical.confidence: return "exact_observed_match"
    return "numerical_difference"

def compare_full_face_trajectories(historical, current, *, run_id="gate6", plan_id="gate6", clip_id=None, subject_id=None, view=None, condition=None, source_inventory_sha256=""):
    inferred_clip_id = clip_id or (current[0].get("clip_id", "") if current else "")
    current_map = {(r.get("dataset_id", MCD_DATASET_ID), r.get("clip_id", inferred_clip_id), int(r["hop_idx"]), float(r["hop_time_s"])): r for r in current}; historical_map = {(MCD_DATASET_ID, inferred_clip_id, h.hop_idx, h.hop_time_s): h for h in historical}
    if len(current_map) != len(current): _fail("current trajectory has duplicate dataset/clip/hop/time keys")
    keys = sorted(set(historical_map) | set(current_map), key=lambda x: x[2:]); rows = []
    for key in keys:
        h, c = historical_map.get(key), current_map.get(key); presence = "joined" if h and c else "historical_only" if h else "current_only"; category = classify_discrepancy(h, c, identity_ok=presence == "joined")
        ident = c or {}; hid = {"clip_id": inferred_clip_id, "subject_id": subject_id or "", "view": view or "", "condition": condition or ""}; ident = {**hid, **ident}
        rows.append({"run_id": run_id, "plan_id": plan_id, "dataset_id": key[0], "clip_id": ident.get("clip_id", ""), "subject_id": ident.get("subject_id", ""), "view": ident.get("view", ""), "condition": ident.get("condition", ""), "hop_idx": key[2], "hop_time_s": key[3], "historical_hop_count": len(historical), "current_hop_count": len(current), "hop_presence": presence, "historical_full_face_hr_bpm": h.hr_bpm if h else None, "current_full_face_hr_bpm": c.get("hr") if c else None, "full_face_hr_delta_bpm": c.get("hr") - h.hr_bpm if h and c and h.hr_bpm is not None and c.get("hr") is not None else None, "historical_confidence": h.confidence if h else None, "current_confidence": c.get("confidence") if c else None, "confidence_delta": c.get("confidence") - h.confidence if h and c else None, "historical_gt_hr_bpm": h.gt_hr_bpm if h else None, "current_gt_hr_bpm": c.get("gt") if c else None, "gt_delta_bpm": c.get("gt") - h.gt_hr_bpm if h and c else None, "historical_post_belief_hr_bpm": h.post_belief_hr_bpm if h else None, "current_post_belief_hr_bpm": c.get("belief") if c else None, "belief_delta_bpm": c.get("belief") - h.post_belief_hr_bpm if h and c else None, "historical_abs_error_bpm": abs(h.post_belief_hr_bpm - h.gt_hr_bpm) if h else None, "current_abs_error_bpm": abs(c["belief"] - c["gt"]) if c else None, "abs_error_delta_bpm": abs(c["belief"] - c["gt"]) - abs(h.post_belief_hr_bpm - h.gt_hr_bpm) if h and c else None, "current_selected_valid": c.get("valid", False) if c else False, "current_invalid_reason": c.get("reason", "missing_hop") if c else "missing_hop", "source_window_complete": c.get("source_window_complete", False) if c else False, "diagnostic_status": "not_supported_by_frozen_inputs", "discrepancy_category": category, "signal_config_id": POS_CONFIG_ID, "control_config_id": CONTROL_CONFIG_ID, "gt_rule_id": GT_RULE_ID, "source_inventory_sha256": source_inventory_sha256})
    return tuple(rows)

def summarize_gate6_rows(hops, clips):
    hops, clips = list(hops), list(clips); categories = Counter(r["discrepancy_category"] for r in hops); subjects = sorted({r["subject_id"] for r in clips})
    def nums(field, predicate=lambda r: True): return [float(r[field]) for r in hops if predicate(r) and r.get(field) not in (None, "")]
    he, ce, de = nums("historical_abs_error_bpm"), nums("current_abs_error_bpm"), nums("abs_error_delta_bpm", lambda r: r["hop_presence"] == "joined")
    subject_rows = [{"subject_id": s, "clip_count": sum(r["subject_id"] == s for r in clips), "mean_clip_mae": sum(float(r["current_mae_clip"]) for r in clips if r["subject_id"] == s and r.get("current_mae_clip") not in (None, "")) / max(1, sum(r["subject_id"] == s and r.get("current_mae_clip") not in (None, "") for r in clips))} for s in subjects]
    category_rows = [{"discrepancy_category": c, "hop_count": categories.get(c, 0)} for c in OBSERVATIONAL_CATEGORIES]
    hist_clip = [float(r["historical_mae_clip"]) for r in clips if r.get("historical_mae_clip") not in (None, "")]
    curr_clip = [float(r["current_mae_clip"]) for r in clips if r.get("current_mae_clip") not in (None, "")]
    paired_clip = [float(r["mae_delta_bpm"]) for r in clips if r.get("mae_delta_bpm") not in (None, "")]
    return subject_rows, category_rows, {
        "clip_count": len(clips), "subject_count": len(subject_rows), "hop_count": len(hops),
        "historical_scored_hops": len(he), "current_scored_hops": len(ce), "joined_delta_hops": len(de),
        "historical_mae_bpm": sum(he) / len(he) if he else None,
        "current_mae_bpm": sum(ce) / len(ce) if ce else None,
        "joined_mae_delta_bpm": sum(de) / len(de) if de else None,
        "historical_equal_clip_mae_bpm": sum(hist_clip) / len(hist_clip) if hist_clip else None,
        "current_equal_clip_mae_bpm": sum(curr_clip) / len(curr_clip) if curr_clip else None,
        "joined_equal_clip_delta_bpm": sum(paired_clip) / len(paired_clip) if paired_clip else None,
        "equal_clip_overall_mae_bpm": sum(curr_clip) / len(curr_clip) if curr_clip else None,
        "category_counts": dict(categories),
    }

def _write_csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=fields, extrasaction="raise").writeheader(); csv.DictWriter(handle, fieldnames=fields, extrasaction="raise").writerows(rows)

def publish_gate6_report(result, destination):
    root = Path(destination)
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink(): _fail("destination must be fresh with a real parent")
    root.mkdir(); run_id = result.get("run_id", "gate6"); plan = result.get("plan"); plan_id = plan.plan_id if isinstance(plan, Gate6ParityPlan) else result.get("plan_id", "gate6"); hops, clips = list(result.get("hop_rows", ())), list(result.get("clip_rows", ()))
    try:
        # STARTED is the first publication marker. Every later failure is
        # recoverable by a reader as incomplete unless COMPLETE exists.
        with (root / "STARTED.json").open("x") as h: json.dump({"state": "started", "run_id": run_id, "plan_id": plan_id}, h, sort_keys=True)
        if not hops or not clips or any(r.get("discrepancy_category") in ("unclassified", None) for r in hops): _fail("Gate 6 requires nonempty rows and no unclassified rows")
        diagnostics = result.get("diagnostics", {}); inv = result.get("source_inventory", {}); inv_sha = result.get("source_inventory_sha256", "") or hashlib.sha256(canonical_json_bytes(inv)).hexdigest()
        if diagnostics.get("causal_decomposition_status") != "not_supported_by_frozen_inputs" or not inv: _fail("Gate 6 accepted limitation/source inventory is missing")
        _write_csv(root / "parity_per_hop.csv", PARITY_HOP_FIELDS, hops); _write_csv(root / "parity_per_clip.csv", PARITY_CLIP_FIELDS, clips); subject, category, summary = summarize_gate6_rows(hops, clips)
        _write_csv(root / "parity_by_subject.csv", ("subject_id", "clip_count", "mean_clip_mae"), subject); _write_csv(root / "parity_by_category.csv", ("discrepancy_category", "hop_count"), category)
        summary.update({"run_id": run_id, "plan_id": plan_id, "dataset_id": result.get("dataset_id", MCD_DATASET_ID), "phase": result.get("phase", "unknown"), "causal_decomposition_status": diagnostics["causal_decomposition_status"], "source_inventory_sha256": inv_sha})
        (root / "parity_summary.json").write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        (root / "plan.json").write_text(json.dumps(result.get("plan_payload", {}), sort_keys=True, separators=(",", ":")), encoding="utf-8")
        manifest = {"state": "complete", "run_id": run_id, "plan_id": plan_id, "source_inventory_sha256": inv_sha, "sources": inv, "diagnostics": diagnostics}; (root / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        (root / "source_inventory.json").write_bytes(canonical_json_bytes(inv))
        (root / "artifacts.sha256").write_text("".join(f"{sha256_file(root / n)}  {n}\n" for n in _OUTPUTS[:-1]), encoding="utf-8"); _verify_gate6_tree(root, False)
        with (root / "COMPLETE.json").open("x") as h: json.dump({"state": "complete", "run_id": run_id, "plan_id": plan_id}, h, sort_keys=True)
        _verify_gate6_tree(root, True)
    except Exception:
        # Never leave contradictory terminal markers. COMPLETE is only
        # written after the non-complete tree has passed verification.
        if (root / "COMPLETE.json").exists():
            raise
        try:
            with (root / "FAILED.json").open("x") as h: json.dump({"state": "failed", "run_id": run_id, "plan_id": plan_id}, h, sort_keys=True)
        except FileExistsError: pass
        raise

def _verify_gate6_tree(destination, require_complete):
    root = Path(destination); expected = set(_OUTPUTS) | {"STARTED.json"} | ({"COMPLETE.json"} if require_complete else set())
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != expected: _fail("Gate 6 publication file state is invalid")
    with (root / "parity_per_hop.csv").open(newline="", encoding="utf-8") as h: reader = csv.DictReader(h); rows = list(reader)
    if tuple(reader.fieldnames or ()) != PARITY_HOP_FIELDS or not rows: _fail("Gate 6 hop schema is invalid")
    if any(r.get("discrepancy_category") not in OBSERVATIONAL_CATEGORIES[:-1] for r in rows): _fail("Gate 6 has an invalid or unclassified category")
    summary = read_json_object(root / "parity_summary.json"); _, _, recomputed = summarize_gate6_rows(rows, list(csv.DictReader((root / "parity_per_clip.csv").open(newline="", encoding="utf-8"))))
    for key, value in recomputed.items():
        if summary.get(key) != value: _fail(f"Gate 6 summary field {key} was not recomputed")
    manifest = read_json_object(root / "run_manifest.json"); plan = read_json_object(root / "plan.json")
    started = read_json_object(root / "STARTED.json")
    if started.get("state") != "started" or started.get("run_id") != manifest.get("run_id") or started.get("plan_id") != manifest.get("plan_id"): _fail("Gate 6 STARTED marker is invalid")
    if require_complete:
        complete = read_json_object(root / "COMPLETE.json")
        if complete.get("state") != "complete" or complete.get("run_id") != manifest.get("run_id") or complete.get("plan_id") != manifest.get("plan_id"): _fail("Gate 6 COMPLETE marker is invalid")
    if manifest.get("source_inventory_sha256") != summary.get("source_inventory_sha256") or manifest.get("plan_id") != plan.get("plan_id", manifest.get("plan_id")) or manifest.get("diagnostics", {}).get("causal_decomposition_status") != "not_supported_by_frozen_inputs": _fail("Gate 6 provenance binding is invalid")
    if sha256_file(root / "source_inventory.json") != manifest.get("source_inventory_sha256"): _fail("Gate 6 source inventory hash is invalid")
    side = {}; 
    for line in (root / "artifacts.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1); side[name] = digest
    if set(side) != set(_OUTPUTS[:-1]) or any(side[n] != sha256_file(root / n) for n in side): _fail("Gate 6 artifact hashes are invalid")

def verify_gate6_publication(destination): _verify_gate6_tree(destination, True)

__all__ = ["HistoricalCacheBinding", "Gate6ParityPlan", "HistoricalFullFaceHop", "ParityHopRow", "ParityClipRow", "ParitySummary", "OBSERVATIONAL_CATEGORIES", "load_historical_cache_inventory", "build_gate6_fixture_plan", "build_gate6_full_plan", "replay_historical_full_face", "evaluate_current_full_face", "compare_full_face_trajectories", "classify_discrepancy", "summarize_gate6_rows", "publish_gate6_report", "verify_gate6_publication"]
