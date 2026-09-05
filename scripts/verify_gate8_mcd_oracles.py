"""Run or merge the canonical held-out MCD Gate 8 Oracle B/C evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.oracles import ORACLE_REPORT_SCHEMA as _ORACLE_REPORT_SCHEMA, ORACLE_SCHEMA as _ORACLE_SCHEMA, ROW_FIELDS, aggregate_report, evaluate_oracle_shard, validate_aggregation
from adaptive_roi_rppg.labels.mcd import GT_RULE_ID, read_mcd_eval_labels
from adaptive_roi_rppg.signal import POS_CONFIG_ID, build_pos_measurements

MARKERS = {"started": "STARTED.json", "complete": "COMPLETE.json", "failed": "FAILED.json"}
ORACLE_SCHEMA = "gate8-mcd-oracle-shard-v1"
ORACLE_REPORT_SCHEMA = "gate8-mcd-oracle-report-v1"


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-dir", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-tree", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--merge-workers", type=int, default=16,
                        help="bounded source-authoritative merge workers (1..16)")
    args = parser.parse_args(argv)
    if args.beam_width != 8:
        parser.error("Gate 8 accepted protocol requires --beam-width 8")
    return args


def _require_slurm():
    job, node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
    if not job or not node:
        _fail("Gate 8 requires Slurm")
    return job, node


def _plan(args):
    bundle = load_mcd_manifest_tree(args.manifest_tree)
    split = bundle.split_manifest
    train_ids, eval_ids = set(split.train_clip_ids), set(split.eval_clip_ids)
    manifest_train_clips = tuple(sorted((c for c in bundle.clip_manifests if c.clip_id in train_ids), key=lambda c: c.clip_id))
    manifest_eval_clips = tuple(sorted((c for c in bundle.clip_manifests if c.clip_id in eval_ids), key=lambda c: c.clip_id))
    manifest_subjects = sorted({c.subject_id for c in manifest_train_clips})
    eval_subjects = {c.subject_id for c in manifest_eval_clips}
    if (len(train_ids) != 3060 or len(eval_ids) != 540
            or len(manifest_train_clips) != 3060 or len(manifest_subjects) != 510
            or len(manifest_eval_clips) != 540 or len(eval_subjects) != 90
            or set(manifest_subjects) & eval_subjects):
        _fail("Gate 8 requires 3060 train clips and 510 train subjects, plus 540 held-out eval clips across 90 eval subjects, with no overlap")
    excluded_ids = {"4952_FullHDwebcam_after","4952_FullHDwebcam_before","4952_IriunWebcam_after","4952_IriunWebcam_before","4952_USBVideo_after","4952_USBVideo_before","6066_USBVideo_before"}
    clips = tuple(clip for clip in manifest_eval_clips if clip.clip_id not in excluded_ids)
    subjects = sorted({clip.subject_id for clip in clips})
    if len(clips) != 533 or len(subjects) != 89 or sum(max(0, (c.state_row_count-round(8*c.camera_fps))//round(c.camera_fps)+1) for c in clips) != 91227:
        _fail("Gate 8 requires exactly 533 eligible eval clips across 89 subjects and 91,227 hops")
    dataset_path, split_path = Path(args.manifest_tree) / "dataset_manifest.json", Path(args.manifest_tree) / "split_manifest.json"
    inventory_path = Path(args.manifest_tree) / "source_inventory.json"
    payload = {"schema": "gate8-mcd-oracle-plan-v1", "dataset_id": split.dataset_id,
               "dataset_manifest_id": bundle.dataset_manifest.manifest_id, "dataset_manifest_sha256": sha256_file(dataset_path),
               "split_id": split.split_id, "split_manifest_sha256": sha256_file(split_path),
               "source_inventory_sha256": sha256_file(inventory_path), "split": "eval",
               "manifest_counts": {"train_subject_count": len(manifest_subjects), "train_clip_count": len(manifest_train_clips),
                                   "eval_clip_count": len(manifest_eval_clips), "eval_subject_count": len(eval_subjects)},
               "eligible_eval_counts": {"eval_subject_count": len(subjects), "eval_clip_count": len(clips)},
               "eval_exclusions": sorted(excluded_ids),
               "beam_width": args.beam_width, "signal_config_id": POS_CONFIG_ID, "control_config_id": CONTROL_CONFIG_ID,
               "gt_rule_id": GT_RULE_ID,
               "clips": [{"clip_id": c.clip_id, "subject_id": c.subject_id, "view": c.view, "condition": c.condition,
                          "camera_fps": c.camera_fps, "clip_manifest_id": c.clip_manifest_id,
                          "state_sha256": c.state_sha256, "gt_sha256": c.gt_sha256,
                          "expected_hops": max(0, (c.state_row_count - round(8 * c.camera_fps)) // round(c.camera_fps) + 1)} for c in clips]}
    payload["plan_id"] = "gate8-plan-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return SimpleNamespace(plan_id=payload["plan_id"], bindings=clips, subjects=subjects, eval_subjects=eval_subjects, payload=payload)


def _fresh_root(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        _fail(f"{label} output must be a fresh real directory path")
    path.mkdir()


def _write_marker(root: Path, state: str, common: dict, **extra) -> None:
    _write_json_exclusive(root / MARKERS[state], {"state": state, **common, **extra})


def _best_effort_failed_marker(root: Path, common: dict) -> None:
    """Record failure without ever replacing the exception being handled."""
    try:
        if not (root / MARKERS["complete"]).exists():
            try:
                _write_marker(root, "failed", common)
            except BaseException:
                try:
                    print("could not write FAILED marker", file=sys.stderr)
                except BaseException:
                    pass
    except BaseException:
        pass


def _write_json_exclusive(path: Path, payload: dict) -> None:
    """Create a state/artifact once. COMPLETE relies on O_EXCL as its commit point."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def _write_text_exclusive(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def _common(args, plan, job, node, *, phase, shard_index=None):
    result = {"plan_id": plan.plan_id, "phase": phase, "code_snapshot_sha256": args.code_snapshot_sha256,
              "source_inventory_sha256": plan.payload["source_inventory_sha256"],
              "signal_config_id": POS_CONFIG_ID, "control_config_id": CONTROL_CONFIG_ID, "gt_rule_id": GT_RULE_ID,
              "beam_width": args.beam_width, "job_id": job, "node": node, "command": shlex.join(sys.argv)}
    if shard_index is not None:
        result.update({"shard_index": shard_index, "shard_count": args.shard_count})
    return result


def _validate_file_set(root: Path, expected: set[str], label: str) -> None:
    if {path.name for path in root.iterdir()} != expected:
        _fail(f"{label} file set is invalid")


def _expected_hop(clip: dict, hop_idx: int) -> tuple[float, str]:
    fps = float(clip["camera_fps"])
    start = hop_idx * round(fps)
    end = start + round(8.0 * fps) - 1
    payload = {"dataset_id": "mcd", "clip_id": clip["clip_id"], "source_provenance_id": clip["clip_manifest_id"],
               "source_frame_start": start, "source_frame_end": end, "signal_config_id": POS_CONFIG_ID}
    return (8.0 + hop_idx, "pos-measurement-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest())


def _validate_rows(rows, plan_payload, method, expected_clip_ids=None):
    expected = {c["clip_id"]: c for c in plan_payload["clips"] if expected_clip_ids is None or c["clip_id"] in expected_clip_ids}
    keys = set(); per_clip = {clip_id: [] for clip_id in expected}; completed_clips = set(); current_clip = None
    previous_action = None; previous_hold = 0; previous_post_belief = None
    for row in rows:
        if set(row) != set(ROW_FIELDS) or row["dataset_id"] != "mcd" or row["gt_rule_id"] != GT_RULE_ID:
            _fail(f"{method} row identity/schema is invalid")
        clip = expected.get(row["clip_id"])
        if clip is None or any(row[key] != clip[key] for key in ("subject_id", "view", "condition")):
            _fail(f"{method} row clip identity is invalid")
        if row["clip_id"] != current_clip:
            if row["clip_id"] in completed_clips:
                _fail(f"{method} rows are not deterministically ordered by clip")
            if current_clip is not None:
                completed_clips.add(current_clip)
            current_clip = row["clip_id"]
            previous_action = None; previous_hold = 0; previous_post_belief = None
        if not isinstance(row["proposed_action"], int) or isinstance(row["proposed_action"], bool) or not 0 <= row["proposed_action"] < 12:
            _fail(f"{method} proposed action is invalid")
        if not isinstance(row["executed_action"], int) or isinstance(row["executed_action"], bool) or not 0 <= row["executed_action"] < 12:
            _fail(f"{method} executed action is invalid")
        key = (row["clip_id"], row["hop_idx"])
        if key in keys or not isinstance(row["hop_idx"], int) or row["hop_idx"] < 0 or not isinstance(row["hop_time_s"], (int, float)):
            _fail(f"{method} row hop identity is invalid or duplicated")
        keys.add(key)
        per_clip[row["clip_id"]].append(row)
        expected_time, expected_provenance = _expected_hop(clip, row["hop_idx"])
        if row["hop_idx"] >= clip["expected_hops"] or not math.isclose(float(row["hop_time_s"]), expected_time, rel_tol=0.0, abs_tol=1e-12) or row["frame_provenance_id"] != expected_provenance:
            _fail(f"{method} hop time or frame provenance is invalid")
        if row["signal_config_id"] != POS_CONFIG_ID or row["control_config_id"] != CONTROL_CONFIG_ID:
            _fail(f"{method} row configuration provenance is invalid")
        if not all(isinstance(row[name], (int, float)) and not isinstance(row[name], bool) and math.isfinite(float(row[name])) for name in ("hop_time_s", "gt_hr_bpm", "pre_belief_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm")) or row["hop_time_s"] < 0:
            _fail(f"{method} row numeric values are invalid")
        if not isinstance(row["selected_valid"], bool) or (row["selected_valid"] and row["selected_invalid_reason"] is not None) or (not row["selected_valid"] and not isinstance(row["selected_invalid_reason"], str)):
            _fail(f"{method} selected validity/reason is invalid")
        if row["abs_error_bpm"] != abs(row["post_belief_hr_bpm"] - row["gt_hr_bpm"]):
            _fail(f"{method} absolute error is not derived from post belief and GT")
        if previous_post_belief is not None and row["pre_belief_hr_bpm"] != previous_post_belief:
            _fail(f"{method} belief chain is discontinuous")
        if previous_action is None:
            if row["pre_hold_count"] != 0 or row["executed_action"] != row["proposed_action"] or row["post_hold_count"] != 1 or row["override_reason"] is not None:
                _fail(f"{method} first action/hold transition is invalid")
        elif row["pre_hold_count"] != previous_hold:
            _fail(f"{method} pre-hold chain is discontinuous")
        elif row["proposed_action"] == previous_action:
            if row["executed_action"] != previous_action or row["post_hold_count"] != row["pre_hold_count"] + 1 or row["override_reason"] is not None:
                _fail(f"{method} stay transition is invalid")
        elif row["pre_hold_count"] < 2:
            if row["executed_action"] != previous_action or row["post_hold_count"] != row["pre_hold_count"] + 1 or row["override_reason"] != "minimum_hold":
                _fail(f"{method} minimum-hold transition is invalid")
        elif row["executed_action"] != row["proposed_action"] or row["post_hold_count"] != 1 or row["override_reason"] is not None:
            _fail(f"{method} legal switch transition is invalid")
        previous_action, previous_hold, previous_post_belief = row["executed_action"], row["post_hold_count"], row["post_belief_hr_bpm"]
    expected_count = sum(c["expected_hops"] for c in expected.values())
    if len(rows) != expected_count:
        _fail(f"{method} row coverage is incomplete")
    for clip_id, clip_rows in per_clip.items():
        expected_keys = {(clip_id, hop_idx) for hop_idx in range(expected[clip_id]["expected_hops"])}
        if {(row["clip_id"], row["hop_idx"]) for row in clip_rows} != expected_keys:
            _fail(f"{method} per-clip hop coverage is incomplete")
    return keys


def _validate_payload(payload, common, plan, *, expected_index=None):
    if payload.get("schema") != ORACLE_SCHEMA or payload.get("plan_id") != plan.plan_id or payload.get("plan_payload") != plan.payload:
        _fail("oracle shard plan provenance mismatch")
    for key in ("phase", "code_snapshot_sha256", "source_inventory_sha256", "signal_config_id", "control_config_id", "gt_rule_id", "beam_width"):
        if payload.get(key) != common[key]:
            _fail(f"oracle shard {key} provenance mismatch")
    if expected_index is not None and payload.get("shard_index") != expected_index:
        _fail("oracle shard index mismatch")
    if payload.get("shard_count") != common.get("shard_count"):
        _fail("oracle shard count mismatch")
    if set(payload.get("rows", {})) != {"greedy_b", "beam_c"}:
        _fail("oracle shard methods are incomplete")
    expected_clip_ids = set(payload.get("clip_ids", ()))
    _validate_rows(payload["rows"]["greedy_b"], plan.payload, "greedy_b", expected_clip_ids)
    _validate_rows(payload["rows"]["beam_c"], plan.payload, "beam_c", expected_clip_ids)
    expected_summary = aggregate_report(payload["rows"])["methods"]
    if payload.get("summary") != expected_summary:
        _fail("oracle shard summary does not match hop rows")
    actual_clip_sets = [{r["clip_id"] for r in payload["rows"][method]} for method in ("greedy_b", "beam_c")]
    if actual_clip_sets != [expected_clip_ids, expected_clip_ids]:
        _fail("oracle shard clip coverage is invalid")


def _validate_complete_marker(marker, expected, digest, *, shard=False):
    required = ("plan_id", "phase", "code_snapshot_sha256", "source_inventory_sha256", "signal_config_id",
                "control_config_id", "gt_rule_id", "beam_width", "job_id", "node", "command")
    if marker.get("state") != "complete" or marker.get("sha256") != digest or any(marker.get(key) != expected.get(key) for key in required):
        _fail("COMPLETE marker provenance is invalid")
    if shard and any(marker.get(key) != expected.get(key) for key in ("shard_index", "shard_count")):
        _fail("shard COMPLETE marker index provenance is invalid")


def _validate_report_artifacts(root: Path, common: dict, plan, *, require_complete: bool = False) -> dict:
    expected_files = {"STARTED.json", "report.json", "artifacts.sha256"} | ({"COMPLETE.json"} if require_complete else set())
    _validate_file_set(root, expected_files, "oracle merge")
    started = json.loads((root / "STARTED.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    digest = sha256_file(root / "report.json")
    if started.get("state") != "started" or any(started.get(key) != common.get(key) for key in common):
        _fail("merged STARTED provenance is invalid")
    if report.get("schema") != ORACLE_REPORT_SCHEMA or report.get("plan_payload") != plan.payload:
        _fail("merged report schema or plan is invalid")
    if any(report.get(key) != common.get(key) for key in common):
        _fail("merged report provenance is invalid")
    if set(report.get("rows", {})) != {"greedy_b", "beam_c"}:
        _fail("merged report methods are incomplete")
    for method in ("greedy_b", "beam_c"):
        _validate_rows(report["rows"][method], plan.payload, method)
    if report.get("summary") != report.get("aggregation", {}).get("methods"):
        _fail("merged report summary does not match aggregation")
    validate_aggregation(report["rows"], report.get("aggregation", {}))
    provenance = report.get("shard_provenance")
    if not isinstance(provenance, list) or len(provenance) != report.get("shard_count", -1):
        _fail("merged report shard provenance is invalid")
    required_shard_keys = {"shard_index", "shard_count", "job_id", "node", "command", "plan_id",
                           "code_snapshot_sha256", "source_inventory_sha256", "signal_config_id",
                           "control_config_id", "gt_rule_id", "beam_width"}
    if ({entry.get("shard_index") for entry in provenance} != set(range(report["shard_count"]))
            or any(set(entry) != required_shard_keys for entry in provenance)
            or any(entry.get(key) != common.get(key) for entry in provenance
                   for key in required_shard_keys - {"shard_index", "shard_count", "job_id", "node", "command"})):
        _fail("merged report shard provenance is incomplete")
    if (root / "artifacts.sha256").read_text(encoding="utf-8") != f"{digest}  report.json\n":
        _fail("merged report sidecar is invalid")
    if require_complete:
        marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
        _validate_complete_marker(marker, common, digest)
    return report


def _load_and_evaluate_bindings(args, bindings):
    """Load only authenticated canonical eval sources and recompute Oracle B/C."""
    bundle = load_mcd_manifest_tree(args.manifest_tree)
    clips = {clip.clip_id: clip for clip in bundle.clip_manifests}
    frames, labels, metadata = {}, {}, {}
    for binding in bindings:
        clip = clips.get(binding.clip_id)
        if clip != binding:
            _fail("canonical manifest no longer matches exact Gate 8 binding")
        frames[binding.clip_id] = build_pos_measurements(
            read_mcd_canonical_frames(bundle, args.state_root, binding.clip_id, "eval"))
        labels[binding.clip_id] = read_mcd_eval_labels(bundle, args.gt_root, binding.clip_id, clip)
        metadata[binding.clip_id] = {"subject_id": binding.subject_id, "view": binding.view,
                                      "condition": binding.condition}
    result = evaluate_oracle_shard(frames, labels, metadata, beam_width=args.beam_width)
    result["schema"] = ORACLE_SCHEMA
    return result


def _recompute_bound_shard(args, bindings):
    return _load_and_evaluate_bindings(args, bindings)


def _same_semantics(left, right, label: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        _fail(f"{label} differs from authenticated canonical recomputation")


def _run(args):
    job, node = _require_slurm(); plan = _plan(args)
    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        _fail("invalid deterministic shard index")
    selected = {s for i, s in enumerate(plan.subjects) if i % args.shard_count == args.shard_index}
    bindings = [b for b in plan.bindings if b.subject_id in selected]
    root = Path(args.output_dir); _fresh_root(root, "oracle shard")
    common = _common(args, plan, job, node, phase="full", shard_index=args.shard_index)
    _write_marker(root, "started", common)
    try:
        result = _load_and_evaluate_bindings(args, bindings)
        payload = {**common, **result, "subject_ids": sorted(selected), "clip_ids": sorted(b.clip_id for b in bindings), "plan_payload": plan.payload}
        _write_json_exclusive(root / "shard.json", payload)
        _validate_payload(payload, common, plan, expected_index=args.shard_index)
        digest = sha256_file(root / "shard.json")
        _write_text_exclusive(root / "artifacts.sha256", f"{digest}  shard.json\n")
        _validate_file_set(root, {"STARTED.json", "shard.json", "artifacts.sha256"}, "oracle shard pre-complete")
        complete = {"state": "complete", **common, "sha256": digest}
        _validate_complete_marker(complete, common, digest, shard=True)
        _write_json_exclusive(root / MARKERS["complete"], complete)
    except BaseException:
        _best_effort_failed_marker(root, common)
        raise


def _merge(args):
    job, node = _require_slurm(); plan = _plan(args); dirs = args.shard_dir or []
    if len(dirs) != args.shard_count:
        _fail("exactly one directory is required for every shard index")
    if not 1 <= args.merge_workers <= 16:
        _fail("merge workers must be between 1 and 16")
    root = Path(args.output_dir); _fresh_root(root, "oracle merge")
    common = _common(args, plan, job, node, phase="full")
    _write_marker(root, "started", common)
    try:
        payloads = []
        for directory in dirs:
            d = Path(directory)
            if d.is_symlink() or not d.is_dir(): _fail("oracle shard directory must be a real directory")
            marker_names = {p.name for p in d.iterdir()}
            if marker_names != {"STARTED.json", "shard.json", "artifacts.sha256", "COMPLETE.json"}:
                _fail("oracle shard marker/file state is invalid")
            marker = json.loads((d / "COMPLETE.json").read_text(encoding="utf-8"))
            started = json.loads((d / "STARTED.json").read_text(encoding="utf-8"))
            payload = json.loads((d / "shard.json").read_text(encoding="utf-8"))
            if started.get("state") != "started": _fail("oracle shard marker state is invalid")
            if any(started.get(key) != payload.get(key) for key in common if key != "job_id" and key != "node"): _fail("oracle shard STARTED provenance mismatch")
            if any(started.get(key) != payload.get(key) for key in ("shard_index", "shard_count")): _fail("oracle shard STARTED index provenance mismatch")
            shard_digest = sha256_file(d / "shard.json")
            _validate_complete_marker(marker, payload, shard_digest, shard=True)
            if (d / "artifacts.sha256").read_text(encoding="utf-8") != f"{shard_digest}  shard.json\n": _fail("oracle shard hash sidecar mismatch")
            payloads.append(payload)
        indices = [p.get("shard_index") for p in payloads]
        if sorted(indices) != list(range(args.shard_count)) or len(set(indices)) != len(indices): _fail("oracle shards have nondeterministic indexes")
        by_index = {p["shard_index"]: p for p in payloads}
        shard_common = {**common, "shard_count": args.shard_count}
        for index, payload in by_index.items(): _validate_payload(payload, shard_common, plan, expected_index=index)
        expected_subjects = {s for s in plan.subjects}; assigned = set()
        for index, payload in by_index.items():
            expected = {s for i, s in enumerate(plan.subjects) if i % args.shard_count == index}
            if set(payload["subject_ids"]) != expected or assigned & expected: _fail("oracle shard subject assignment is invalid")
            assigned |= expected
            if set(payload["clip_ids"]) != {c.clip_id for c in plan.bindings if c.subject_id in expected}: _fail("oracle shard clip assignment is invalid")
        if assigned != expected_subjects: _fail("oracle shard subject coverage is incomplete")
        bindings_by_index = {
            index: tuple(binding for binding in plan.bindings if binding.subject_id in set(payload["subject_ids"]))
            for index, payload in by_index.items()
        }
        # Merge is source-authoritative: each bounded worker reloads its exact
        # canonical binding, rebuilds POS measurements, and reruns B/C.
        with ProcessPoolExecutor(max_workers=min(args.shard_count, args.merge_workers)) as pool:
            futures = {index: pool.submit(_recompute_bound_shard, args, bindings_by_index[index])
                       for index in range(args.shard_count)}
            recomputed = {index: futures[index].result() for index in range(args.shard_count)}
        for index in range(args.shard_count):
            serialized = by_index[index]
            fresh = recomputed[index]
            _same_semantics(serialized["rows"], fresh["rows"], f"oracle shard {index} rows")
            _same_semantics(serialized["summary"], fresh["summary"], f"oracle shard {index} summary")
        rows = {"greedy_b": [], "beam_c": []}
        for result in (recomputed[i] for i in range(args.shard_count)):
            for method in rows:
                rows[method].extend(result["rows"][method])
        for method in rows: _validate_rows(rows[method], plan.payload, method)
        by_method_keys = {method: {(r["clip_id"], r["hop_idx"]) for r in rows[method]} for method in rows}
        if by_method_keys["greedy_b"] != by_method_keys["beam_c"] or len(by_method_keys["greedy_b"]) != len(rows["greedy_b"]): _fail("oracle B/C exact hop coverage differs")
        aggregation = aggregate_report(rows)
        validate_aggregation(rows, aggregation)
        for left, right in zip(rows["greedy_b"], rows["beam_c"]):
            for field in ("dataset_id", "clip_id", "subject_id", "view", "condition", "hop_idx", "hop_time_s", "gt_hr_bpm", "gt_rule_id", "frame_provenance_id", "signal_config_id", "control_config_id"):
                if left[field] != right[field]: _fail("oracle B/C GT or hop identity differs")
        report = {"schema": ORACLE_REPORT_SCHEMA, **common, "shard_count": args.shard_count,
                  "plan_payload": plan.payload, "rows": rows,
                  "aggregation": aggregation, "summary": aggregation["methods"],
                  "shard_provenance": [{k: p[k] for k in ("shard_index", "shard_count", "job_id", "node", "command", "plan_id", "code_snapshot_sha256", "source_inventory_sha256", "signal_config_id", "control_config_id", "gt_rule_id", "beam_width")} for p in (by_index[i] for i in range(args.shard_count))]}
        _write_json_exclusive(root / "report.json", report)
        digest = sha256_file(root / "report.json")
        _write_text_exclusive(root / "artifacts.sha256", f"{digest}  report.json\n")
        reread = _validate_report_artifacts(root, common, plan)
        if reread != report: _fail("merged report reread changed the report")
        complete = {"state": "complete", **common, "sha256": digest}
        _validate_complete_marker(complete, common, digest)
        _write_json_exclusive(root / MARKERS["complete"], complete)
    except BaseException:
        _best_effort_failed_marker(root, common)
        raise


def main():
    args = _args()
    _merge(args) if args.merge else _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
