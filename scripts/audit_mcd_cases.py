#!/usr/bin/env python3
"""Fail-closed human forensic audit for four fixed MCD seed-1 clips."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, stat, sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import ROI_NAMES, canonical_json_bytes
from adaptive_roi_rppg.control import OBSERVATION_SCHEMA_ID, build_observation, control_step, initial_control_state
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.model_replay import rollout_frozen_policy, score_frozen_rollout
from adaptive_roi_rppg.evaluation.model_plan import load_frozen_model_plan
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements

CONFIG_SCHEMA = "mcd-case-audit-v1"
REQUIRED_CASES = ("9940_IriunWebcam_before", "2651_USBVideo_after", "9825_IriunWebcam_after", "3448_USBVideo_after")
UNAVAILABLE = "UNAVAILABLE: not saved by job 48193"

def _fail(message: str) -> None: raise ValueError(message)

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def file_provenance(path: str | Path) -> dict[str, Any]:
    path = Path(path); return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}

def discover_raw_video(video_root: str | Path, clip_id: str) -> Path:
    root = Path(video_root)
    if root.is_symlink() or not root.is_dir(): _fail("video root must be a real directory")
    found = []
    for suffix in (".avi", ".mp4", ".mov"):
        candidate = root / f"{clip_id}{suffix}"
        try: mode = candidate.lstat().st_mode
        except FileNotFoundError: continue
        if stat.S_ISREG(mode) and not candidate.is_symlink(): found.append(candidate)
    if len(found) != 1: _fail(f"{clip_id}: expected exactly one direct raw video, found {len(found)}")
    return found[0]

def verify_raw_video(path: Path, record: Mapping[str, Any]) -> None:
    if path.name != record.get("locator") or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
        _fail(f"{path.name}: raw video provenance does not match audit config")

def load_selection(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != CONFIG_SCHEMA or value.get("dataset_id") != "mcd" or value.get("seed") != 1: _fail("case-audit config identity is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or tuple(x.get("clip_id") for x in cases) != REQUIRED_CASES: _fail("case selection is not the fixed four-clip contract")
    if {x.get("role") for x in cases} != {"high_error", "median_control"}: _fail("case roles are invalid")
    if value.get("observation_dim") != 101 or value.get("action_count") != 12: _fail("controller dimensions are not frozen")
    if "mmpd" in json.dumps(value, sort_keys=True).lower(): _fail("MMPD is forbidden in the audit config")
    return value

def _status(name: str, passed: bool | None, detail: str = "") -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed is True else "FAIL" if passed is False else "UNAVAILABLE", "detail": detail}

def check_frame_alignment(raw_count: int, state_frames: Sequence[Any], raw_frames: Sequence[Any]) -> list[dict[str, Any]]:
    checks = [_status("raw_vs_state_frame_count", raw_count == len(state_frames), f"raw={raw_count} state={len(state_frames)}"), _status("raw_extraction_frame_count", raw_count == len(raw_frames), f"raw={raw_count} extracted={len(raw_frames)}"), _status("state_contiguous_frame_idx", [x.frame_idx for x in state_frames] == list(range(len(state_frames)))), _status("raw_contiguous_frame_idx", [x.frame_idx for x in raw_frames] == list(range(len(raw_frames)))), _status("state_identity", all(x.dataset_id == "mcd" and x.frame_idx == i for i, x in enumerate(state_frames))), _status("raw_identity", all(x.dataset_id == "mcd" and x.frame_idx == i for i, x in enumerate(raw_frames)))]
    checks.append(_status("aligned_frame_times", len(state_frames) == len(raw_frames) and all(abs(a.timestamp_s - b.timestamp_s) <= 1e-12 for a, b in zip(state_frames, raw_frames))))
    return checks

def compare_dataclass_fields(left: Any, right: Any, *, ignore: Sequence[str] = ()) -> list[str]:
    if is_dataclass(left) and is_dataclass(right):
        ignored = set(ignore); out = []
        for item in fields(left):
            if item.name in ignored or not hasattr(right, item.name): continue
            out.extend(f"{item.name}.{x}" if x else item.name for x in compare_dataclass_fields(getattr(left, item.name), getattr(right, item.name)))
        return out
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right): return ["length"]
        return [f"[{i}].{x}" if x else f"[{i}]" for i, (a, b) in enumerate(zip(left, right)) for x in compare_dataclass_fields(a, b)]
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray): return [] if np.array_equal(left, right) else ["value"]
    try: return [] if bool(left == right) else ["value"]
    except Exception: return ["value"]

def compare_factual_rows(current: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-12) -> list[str]:
    names = ("clip_id", "hop_idx", "proposed_action", "executed_action", "selected_post_belief_hr_bpm", "selected_hr_bpm", "selected_confidence", "selected_ppr", "selected_coverage", "post_belief_hr_bpm", "abs_error_bpm", "observation_sha256")
    if len(current) != len(source): return [f"row_count current={len(current)} source={len(source)}"]
    out = []
    for i, (a, b) in enumerate(zip(current, source)):
        for name in names:
            if name not in a or name not in b: continue
            try:
                if name in {"hop_idx", "proposed_action", "executed_action"}:
                    equal = int(a[name]) == int(b[name])
                elif name not in {"clip_id", "observation_sha256"}:
                    equal = abs(float(a[name]) - float(b[name])) <= tolerance
                else: equal = a[name] == b[name]
            except (TypeError, ValueError): equal = a[name] == b[name]
            if not equal: out.append(f"row={i} field={name}")
    return out

def load_factual_source(root: str | Path, clip_id: str) -> list[Mapping[str, Any]] | None:
    """Find exactly one seed-1 factual source subset for a fixed clip."""
    root = Path(root)
    if not root.is_dir(): _fail("48193 factual-results root is missing")
    matches = []
    for path in root.rglob("factual_hops.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error): continue
        rows = [row for row in rows if row.get("clip_id") == clip_id and str(row.get("seed")) == "1"]
        if rows: matches.append((path, rows))
    if len(matches) != 1: return None
    return matches[0][1]

def load_authenticated_subset(root: str | Path, clip_id: str, *, task_index: int) -> list[Mapping[str, Any]]:
    from adaptive_roi_rppg.evaluation.source_subsets import read_task_subset
    entry, rows = read_task_subset(root, task_index, task_count=162)
    if entry.get("clip_id") != clip_id or int(entry.get("seed", -1)) != 1 or int(entry.get("task_index", -1)) != task_index:
        _fail(f"{clip_id}: authenticated source subset task binding failed")
    if not rows or [int(row["hop_idx"]) for row in rows] != list(range(len(rows))): _fail(f"{clip_id}: source subset hops are not contiguous")
    return rows

def _read_video(path: Path) -> np.ndarray:
    try: import cv2
    except ImportError as exc: raise RuntimeError("OpenCV is required for raw MCD video audit") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened(): raise RuntimeError(f"cannot open raw video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally: capture.release()
    if not frames: raise RuntimeError(f"raw video has no frames: {path}")
    return np.asarray(frames)

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields_ = tuple(rows[0]) if rows else ("empty",)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)

def _extract(video: np.ndarray, clip: Any, video_path: Path, model_path: str) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    # Shared extractor is dataset-generic; this import performs no MMPD I/O.
    from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import extract_face_relative_canonical_frames
    diagnostics: list[dict[str, Any]] = []
    frames = extract_face_relative_canonical_frames(video, dataset_id="mcd", clip_id=clip.clip_id, source_provenance_id=sha256_file(video_path), model_asset_path=model_path, fps=clip.camera_fps, diagnostics=diagnostics)
    return frames, diagnostics

def _contact_sheet(path: Path, video: np.ndarray, diagnostics: Sequence[Mapping[str, Any]], frames: Sequence[Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont
    panels = []; font = ImageFont.load_default(); indices = sorted(set((0, len(video)//4, len(video)//2, 3*len(video)//4, len(video)-1)))
    for i in indices:
        image = Image.fromarray(np.clip(video[i], 0, 255).astype(np.uint8)).convert("RGB"); draw = ImageDraw.Draw(image); diag = diagnostics[i]
        for j, box in enumerate([diag.get("face_box"), *diag.get("roi_boxes", [])]):
            if box is None: continue
            color = "yellow" if j == 0 else ("lime" if frames[i].roi_values[j-1].valid else "red"); draw.rectangle(tuple(box), outline=color, width=2); draw.text((box[0]+2, box[1]+2), "F" if j == 0 else str(j-1), fill=color, font=font)
        draw.text((4, 4), f"frame={i} geometry={diag.get('geometry_source')}", fill="white", stroke_width=2, stroke_fill="black", font=font); panels.append(image)
    sheet = Image.new("RGB", (max(x.width for x in panels), sum(x.height for x in panels)), "black"); y = 0
    for panel in panels: sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(path)

def audit_case(*, bundle: Any, clip: Any, state_root: str, gt_root: str, video_path: Path, model_path: str, policy: Any, output: Path, factual_source: Sequence[Mapping[str, Any]] | None = None, source_subset: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    state = read_mcd_canonical_frames(bundle, state_root, clip.clip_id, "eval"); labels = read_mcd_eval_labels(bundle, gt_root, clip.clip_id, clip); video = _read_video(video_path); raw, diagnostics = _extract(video, clip, video_path, model_path)
    checks = check_frame_alignment(len(video), state, raw)
    if any(x["status"] == "FAIL" for x in checks): _fail(f"{clip.clip_id}: frame/alignment stop gate failed")
    state_measurements, raw_measurements = build_pos_measurements(state), build_pos_measurements(raw)
    replay = rollout_frozen_policy(state_measurements, policy, clip_id=clip.clip_id); again = rollout_frozen_policy(state_measurements, policy, clip_id=clip.clip_id)
    mismatch = [f"hop={i}.{x}" for i, (a, b) in enumerate(zip(replay, again)) for x in compare_dataclass_fields(a, b)]
    checks += [_status("deterministic_replay_parity", not mismatch, "; ".join(mismatch[:5])), _status("observation_dimension", all(build_observation(x, initial_control_state("mcd", clip.clip_id)).array().shape == (101,) for x in state_measurements))]
    scored = score_frozen_rollout(replay, labels, {"subject_id": clip.subject_id, "view": clip.view, "condition": clip.condition})
    if factual_source is None: checks.append(_status("factual_job_parity", False, "authenticated 48193 source rows missing"))
    else: checks.append(_status("factual_job_parity", not compare_factual_rows(scored, factual_source), "factual rows differ"))
    if source_subset is None: checks.append(_status("source_subset_parity", False, "authenticated source subset missing"))
    else: checks.append(_status("source_subset_parity", not compare_factual_rows(scored, source_subset), "source subset rows differ"))
    if any(x["status"] == "FAIL" for x in checks): _fail(f"{clip.clip_id}: replay/parity stop gate failed")
    frame_rows = []; pos_rows = []
    for i, (rv, sv) in enumerate(zip(raw, state)):
        for j, (r, s) in enumerate(zip(rv.roi_values, sv.roi_values)):
            frame_rows.append({"clip_id": clip.clip_id, "frame_idx": i, "roi_index": j, "roi_name": ROI_NAMES[j], "raw_r": r.r_mean, "raw_g": r.g_mean, "raw_b": r.b_mean, "raw_coverage": r.coverage, "raw_valid": r.valid, "raw_invalid_reason": r.invalid_reason, "state_r": s.r_mean, "state_g": s.g_mean, "state_b": s.b_mean, "state_coverage": s.coverage, "state_valid": s.valid, "state_invalid_reason": s.invalid_reason, "face_box": json.dumps(diagnostics[i].get("face_box")), "roi_box": json.dumps(diagnostics[i].get("roi_boxes", [None]*12)[j])})
    for source, measurements in (("raw", raw_measurements), ("frozen_state", state_measurements)):
        for frame in measurements:
            for m in frame.measurements: pos_rows.append({"clip_id": clip.clip_id, "source": source, "hop_idx": frame.hop_idx, "hop_time_s": frame.hop_time_s, "roi_index": m.roi_index, "roi_name": m.roi_name.value, "hr_bpm": m.hr_bpm, "ppr": m.peak_power_ratio, "confidence": m.confidence, "coverage": m.coverage, "valid": m.valid, "invalid_reason": m.invalid_reason})
    hop_rows = []; causal_state = initial_control_state("mcd", clip.clip_id)
    for row, transition in zip(scored, replay):
        frame = state_measurements[row["hop_idx"]]; obs = build_observation(frame, causal_state).array()
        if hashlib.sha256(obs.tobytes()).hexdigest() != transition.observation_sha256: _fail(f"{clip.clip_id}: replay observation reconstruction mismatch at hop {row['hop_idx']}")
        causal_state, _ = control_step(frame, causal_state, transition.proposed_action)
        hop_rows.append({**row, "selected_roi_index": row["executed_action"], "selected_roi_name": ROI_NAMES[row["executed_action"]], "observation_dim": obs.size, "observation_values": json.dumps([float(x) for x in obs]), "observation_sha256": transition.observation_sha256})
    output.mkdir(parents=True, exist_ok=False); _write_csv(output / "per_frame_roi.csv", frame_rows); _write_csv(output / "per_pos_roi.csv", pos_rows); _write_csv(output / "per_hop.csv", hop_rows); _contact_sheet(output / "raw_contact_sheet.png", video, diagnostics, raw)
    statuses = checks + [_status("raw_state_geometry_parity", None, "current diagnostic rectangle geometry is not authenticated as historical extractor geometry")] + [_status(x, None, UNAVAILABLE) for x in ("rewards", "logits", "values", "raw_pos_waveform")]
    return {"clip_id": clip.clip_id, "subject_id": clip.subject_id, "video_provenance": file_provenance(video_path), "state_provenance": {"clip_manifest_id": clip.clip_manifest_id, "state_locator": clip.state_locator, "state_sha256": clip.state_sha256, "state_bytes": Path(state_root, Path(clip.state_locator).name).stat().st_size}, "face_landmarker_provenance": file_provenance(model_path), "raw_frame_count": len(video), "state_frame_count": len(state), "observation_dim": 101, "status": statuses, "pipeline_status": "PASS" if all(x["status"] != "FAIL" for x in statuses) else "FAIL", "rewards_logits_values": UNAVAILABLE}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest-tree", "state-root", "gt-root", "video-root", "face-landmarker-model", "checkpoint-index", "plan", "checkpoint-root", "output-dir", "source-subsets-dir", "job-48193-results"): parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_selection(args.config); bundle = load_mcd_manifest_tree(args.manifest_tree); plan = load_frozen_model_plan(args.plan); index = json.loads(Path(args.checkpoint_index).read_text(encoding="utf-8")); hashes = {int(k): v for k, v in index["checkpoint_sha256"].items()}; matches = [x for x in plan.checkpoints if x["family"] == "advantage_ppo" and x["seed"] == 1]
        if not Path(args.source_subsets_dir).is_dir(): _fail("source-subsets directory is missing")
        if not Path(args.job_48193_results).is_dir(): _fail("48193 factual-results directory is missing")
        if len(matches) != 1 or matches[0]["sha256"] != hashes.get(1): _fail("seed-1 checkpoint identity is not authenticated")
        checkpoint = dict(matches[0]); checkpoint_path = (Path(args.checkpoint_root) / checkpoint["locator"]).resolve()
        if Path(args.checkpoint_root).resolve() not in checkpoint_path.parents or checkpoint_path.is_symlink() or not checkpoint_path.is_file(): _fail("checkpoint path is outside checkpoint root")
        if sha256_file(checkpoint_path) != hashes[1]: _fail("seed-1 checkpoint bytes do not match authenticated hash")
        checkpoint["locator"] = str(checkpoint_path)
        from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
        policy = load_frozen_recurrent_policy(checkpoint); clips = {x.clip_id: x for x in bundle.clip_manifests}; root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=False); source_root = Path(args.job_48193_results); summaries = []
        for item in config["cases"]:
            clip = clips.get(item["clip_id"])
            if clip is None or clip.split_id != bundle.split_manifest.split_id or clip.clip_id not in bundle.split_manifest.eval_clip_ids: _fail(f"{item['clip_id']}: not authenticated MCD eval clip")
            video = discover_raw_video(args.video_root, clip.clip_id); verify_raw_video(video, config["raw_videos"][clip.clip_id]); task_map = {"9940_IriunWebcam_before": 1, "2651_USBVideo_after": 34, "9825_IriunWebcam_after": 4, "3448_USBVideo_after": 109}; subset = load_authenticated_subset(args.source_subsets_dir, clip.clip_id, task_index=task_map[clip.clip_id]); factual = load_factual_source(source_root, clip.clip_id)
            if factual is None: _fail(f"{clip.clip_id}: authenticated 48193 factual_hops.csv missing")
            summaries.append({"role": item["role"], **audit_case(bundle=bundle, clip=clip, state_root=args.state_root, gt_root=args.gt_root, video_path=video, model_path=args.face_landmarker_model, policy=policy, output=root / clip.clip_id, factual_source=factual, source_subset=subset)})
        status = "PASS" if all(x["pipeline_status"] == "PASS" for x in summaries) else "FAIL"; (root / "summary.json").write_bytes(canonical_json_bytes({"schema": CONFIG_SCHEMA, "dataset_id": "mcd", "source_job_id": 48193, "seed": 1, "cases": summaries, "status": status, "limitations": ["Four clips cannot establish pipeline cleanliness or prevalence.", "Rewards, logits, values, and raw POS waveforms were not saved by job 48193."]}))
    except Exception as exc:
        # Preserve a machine-readable diagnosis without touching any source run.
        try:
            failure_root = Path(args.output_dir)
            failure_root.mkdir(parents=True, exist_ok=True)
            (failure_root / "summary.json").write_bytes(canonical_json_bytes({"schema": CONFIG_SCHEMA, "dataset_id": "mcd", "source_job_id": 48193, "seed": 1, "status": "FAIL", "stop_reason": str(exc), "limitations": ["Behavioral interpretation is stopped after a hard-gate failure.", "Rewards, logits, values, and raw POS waveforms are unavailable."]}))
        except Exception:
            pass
        print(f"audit_mcd_cases: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
