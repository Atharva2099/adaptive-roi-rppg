#!/usr/bin/env python3
"""Render and fail-closed audit of current MCD/MMPD causal extraction.

This is a Polaris diagnostic, not a training or evaluation runner.  MCD raw
video values are deliberately labelled face-relative diagnostic values; they
are not asserted equal to the precomputed MCD state values.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_roi_rppg.contracts import ROI_NAMES
from adaptive_roi_rppg.data.mcd.schema import MCD_STATE_COLUMNS
from adaptive_roi_rppg.control import control_step, initial_control_state
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import (
    extract_face_relative_canonical_frames,
    load_mmpd_mat,
)
from adaptive_roi_rppg.signal import build_pos_measurements


def invalid_diagnostic_failure(frames, measurements, transitions, state_rows=None) -> dict[str, Any] | None:
    """Return the first hard failure, or ``None`` when the audit passed."""
    for row in state_rows or ():
        if not row["_state_valid"]:
            return {"kind": "state", "frame": row["_state_frame_idx"], "roi": row["_state_invalid_roi"], "reason": row["_state_reason"]}
    for frame in frames:
        for roi in frame.roi_values:
            if not roi.valid:
                return {"kind": "source_roi", "frame": frame.frame_idx, "roi": roi.roi_name.value, "reason": roi.invalid_reason}
    for hop in measurements:
        selected = hop.measurements[0]
        if not selected.valid:
            return {"kind": "selected_pos", "hop": hop.hop_idx, "roi": selected.roi_name.value, "reason": selected.invalid_reason}
    if not any(t.selected_measurement.valid and (t.selected_measurement.confidence or 0.0) > 0.0 and t.post_belief != t.pre_belief for t in transitions):
        return {"kind": "controller_update", "hop": transitions[0].hop_idx if transitions else None, "roi": ROI_NAMES[0].value, "reason": "no valid selected measurement caused a belief update"}
    return None


def format_failure(dataset: str, clip: str, failure: dict[str, Any]) -> str:
    location = f"frame {failure['frame']}" if "frame" in failure else f"hop {failure['hop']}"
    return f"{dataset} clip {clip}: {failure['kind']} failure at {location}, ROI {failure['roi']}: {failure['reason']}"


def _read_avi(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required on Polaris to read the MCD AVI") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open MCD AVI: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"MCD AVI has no frames: {path}")
    return np.asarray(frames)


def _finite_state_number(raw: str, field: str) -> float | None:
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"MCD state {field} is not numeric") from exc
    if not np.isfinite(value):
        raise RuntimeError(f"MCD state {field} is not finite")
    return value


def _read_mcd_state(path: Path, clip_id: str) -> list[dict[str, Any]]:
    expected_name = f"{clip_id}_semantic_state_vectors.csv"
    if path.name != expected_name:
        raise RuntimeError(f"MCD state filename must be exactly {expected_name}, got {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader, ())) != MCD_STATE_COLUMNS:
            raise RuntimeError(f"MCD state CSV has unexpected header: {path}")
        rows = []
        for frame_idx, raw in enumerate(reader):
            if len(raw) != len(MCD_STATE_COLUMNS):
                raise RuntimeError(f"MCD state row {frame_idx} has width {len(raw)}, expected {len(MCD_STATE_COLUMNS)}")
            try:
                parsed_frame_idx = int(raw[0])
            except ValueError as exc:
                raise RuntimeError(f"MCD state row {frame_idx} has non-integer frame_idx") from exc
            if parsed_frame_idx != frame_idx:
                raise RuntimeError(f"MCD state frame_idx is not canonical at row {frame_idx}: {raw[0]}")
            state_valid = True
            reasons = []
            pose = tuple(_finite_state_number(value, f"row {frame_idx} pose") for value in raw[1:4])
            if any(value is None for value in pose) and not all(value is None for value in pose):
                state_valid = False; reasons.append("pose must be all blank or all finite")
            roi_values = []
            invalid_roi = "state"
            for roi_index, roi in enumerate(ROI_NAMES):
                offset = 4 + roi_index * 5
                values = tuple(_finite_state_number(value, f"row {frame_idx} {roi.value}") for value in raw[offset:offset + 4])
                coverage = _finite_state_number(raw[offset + 4], f"row {frame_idx} {roi.value}.coverage")
                valid = True; reason = None
                if coverage is None or not 0.0 <= coverage <= 1.0:
                    valid = False; reason = "coverage must be finite in [0,1]"
                elif coverage == 0.0:
                    if any(value is not None for value in values):
                        valid = False; reason = "zero coverage requires all ROI values blank"
                    else:
                        valid = False; reason = "source_missing"
                elif coverage > 0.0 and any(value is None for value in values):
                    valid = False; reason = "positive coverage requires all ROI values finite"
                roi_values.append({"values": values, "coverage": coverage, "valid": valid, "reason": reason})
                if not valid and state_valid:
                    invalid_roi = roi.value
                    state_valid = False
                    reasons.append(f"{roi.value}: {reason}")
            rows.append({**dict(zip(MCD_STATE_COLUMNS, raw)), "_state_frame_idx": parsed_frame_idx, "_state_pose": pose, "_state_rois": roi_values, "_state_valid": state_valid, "_state_reason": "; ".join(reasons) if reasons else None, "_state_invalid_roi": invalid_roi})
    return rows


def _state_roi(row: dict[str, str], index: int) -> tuple[str, str, str, str, str]:
    offset = 4 + index * 5
    parsed = row["_state_rois"][index]
    return (*parsed["values"], parsed["coverage"])


def _write_outputs(out: Path, dataset: str, frames, diagnostics, state_rows, measurements, transitions) -> None:
    out.mkdir(parents=True, exist_ok=True)
    per_frame = out / f"{dataset}_per_frame.csv"
    fields = ["dataset", "frame_idx", "timestamp_s", "face_valid", "face_box", "geometry_source", "geometry_source_frame_idx", "geometry_bridged", "current_frame_evidence", "recovery_attempts"]
    if dataset == "mcd":
        fields.extend(["state_frame_idx", "state_yaw", "state_pitch", "state_roll", "state_valid", "state_reason"])
    for roi in ROI_NAMES:
        fields.extend([f"{roi.value}_raw_r", f"{roi.value}_raw_g", f"{roi.value}_raw_b", f"{roi.value}_raw_coverage", f"{roi.value}_raw_valid", f"{roi.value}_raw_reason"])
        if dataset == "mcd":
            fields.extend([f"{roi.value}_state_r", f"{roi.value}_state_g", f"{roi.value}_state_b", f"{roi.value}_state_std", f"{roi.value}_state_coverage", f"{roi.value}_state_valid", f"{roi.value}_state_reason"])
    with per_frame.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for frame, diag, state in zip(frames, diagnostics, state_rows or [None] * len(frames)):
            row = {"dataset": dataset, "frame_idx": frame.frame_idx, "timestamp_s": frame.timestamp_s, "face_valid": diag["face_valid"], "face_box": diag["face_box"], "geometry_source": diag["geometry_source"], "geometry_source_frame_idx": diag["geometry_source_frame_idx"], "geometry_bridged": diag["geometry_bridged"], "current_frame_evidence": diag["current_frame_evidence"], "recovery_attempts": json.dumps(diag["recovery_attempts"], separators=(",", ":"))}
            if state is not None:
                row.update({"state_frame_idx": state["_state_frame_idx"], "state_yaw": state["_state_pose"][0], "state_pitch": state["_state_pose"][1], "state_roll": state["_state_pose"][2], "state_valid": state["_state_valid"], "state_reason": state["_state_reason"]})
            for i, roi in enumerate(frame.roi_values):
                prefix = roi.roi_name.value
                row.update({f"{prefix}_raw_r": roi.r_mean, f"{prefix}_raw_g": roi.g_mean, f"{prefix}_raw_b": roi.b_mean, f"{prefix}_raw_coverage": roi.coverage, f"{prefix}_raw_valid": roi.valid, f"{prefix}_raw_reason": roi.invalid_reason})
                if state is not None:
                    values = _state_roi(state, i)
                    state_roi = state["_state_rois"][i]
                    row.update(dict(zip((f"{prefix}_state_r", f"{prefix}_state_g", f"{prefix}_state_b", f"{prefix}_state_std", f"{prefix}_state_coverage"), values)))
                    row.update({f"{prefix}_state_valid": state_roi["valid"], f"{prefix}_state_reason": state_roi["reason"]})
            writer.writerow(row)
    per_hop = out / f"{dataset}_per_hop.csv"
    hop_fields = ["dataset", "hop_idx", "hop_time_s", "roi_index", "roi_name", "roi_hr_bpm", "roi_confidence", "roi_ppr", "roi_coverage", "roi_valid", "roi_reason", "selected_roi", "selected_hr_bpm", "selected_confidence", "selected_ppr", "selected_coverage", "selected_valid", "selected_reason", "pre_belief_mean_hr", "post_belief_mean_hr", "belief_update_occurred"]
    with per_hop.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=hop_fields); writer.writeheader()
        for hop, transition in zip(measurements, transitions):
            selected = transition.selected_measurement
            for roi in hop.measurements:
                writer.writerow({"dataset": dataset, "hop_idx": hop.hop_idx, "hop_time_s": hop.hop_time_s, "roi_index": roi.roi_index, "roi_name": roi.roi_name.value, "roi_hr_bpm": roi.hr_bpm, "roi_confidence": roi.confidence, "roi_ppr": roi.peak_power_ratio, "roi_coverage": roi.coverage, "roi_valid": roi.valid, "roi_reason": roi.invalid_reason, "selected_roi": selected.roi_name.value, "selected_hr_bpm": selected.hr_bpm, "selected_confidence": selected.confidence, "selected_ppr": selected.peak_power_ratio, "selected_coverage": selected.coverage, "selected_valid": selected.valid, "selected_reason": selected.invalid_reason, "pre_belief_mean_hr": transition.pre_belief.mean_hr, "post_belief_mean_hr": transition.post_belief.mean_hr, "belief_update_occurred": transition.post_belief != transition.pre_belief})


def _display_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a frame for Pillow display without changing extraction inputs."""
    scale = 255.0 if float(np.nanmax(frame)) <= 1.0 else 1.0
    return np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)


def _contact_sheet(out: Path, dataset: str, video: np.ndarray, diagnostics: list[dict[str, Any]], frames, frame_indices: tuple[int, ...] | list[int] | None = None) -> None:
    from PIL import Image, ImageDraw, ImageFont
    indices = sorted(set(frame_indices if frame_indices is not None else [0, len(video) // 2, len(video) - 1]))
    if not indices or indices[0] < 0 or indices[-1] >= len(video):
        raise ValueError("contact-sheet frame indices must select existing video frames")
    panels = []
    font = ImageFont.load_default()
    for index in indices:
        image = Image.fromarray(_display_rgb(video[index])).convert("RGB"); draw = ImageDraw.Draw(image)
        diag = diagnostics[index]; boxes = [diag["face_box"], *diag["roi_boxes"]]
        for box_index, box in enumerate(boxes):
            if box is None: continue
            color = "yellow" if box_index == 0 else ("lime" if frames[index].roi_values[box_index - 1].valid else "red")
            draw.rectangle(tuple(box), outline=color, width=2)
            draw.text((box[0] + 2, box[1] + 2), "F" if box_index == 0 else str(box_index - 1), fill=color, font=font)
        draw.text((4, 4), f"{dataset} frame={index} face={diag['face_valid']} geometry={diag['geometry_source']} source_frame={diag['geometry_source_frame_idx']}", fill="white", stroke_width=2, stroke_fill="black", font=font)
        panels.append(image)
    width = max(image.width for image in panels); height = sum(image.height for image in panels)
    sheet = Image.new("RGB", (width, height), "black"); y = 0
    for image in panels: sheet.paste(image, (0, y)); y += image.height
    sheet.save(out / f"{dataset}_contact_sheet.png")


def _run_one(dataset: str, video: np.ndarray, clip: str, provenance: str, model: str, fps: float, out: Path, state_rows=None, contact_sheet_frame_indices: tuple[int, ...] | list[int] | None = None) -> str | None:
    diagnostics: list[dict[str, Any]] = []
    is_mmpd = dataset == "mmpd"
    frames = extract_face_relative_canonical_frames(video, dataset_id=dataset, clip_id=clip, source_provenance_id=provenance, model_asset_path=model, fps=fps, diagnostics=diagnostics, reject_exact_zero_raw_frames=is_mmpd)
    measurements = build_pos_measurements(frames)
    state = initial_control_state(dataset, clip); transitions = []
    for measurement in measurements:
        state, transition = control_step(measurement, state, 0); transitions.append(transition)
    _write_outputs(out, dataset, frames, diagnostics, state_rows, measurements, transitions)
    _contact_sheet(out, dataset, video, diagnostics, frames, frame_indices=contact_sheet_frame_indices)
    failure = invalid_diagnostic_failure(frames, measurements, transitions, state_rows)
    return format_failure(dataset, clip, failure) if failure else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcd-video", type=Path, required=True); parser.add_argument("--mcd-state-csv", type=Path, required=True); parser.add_argument("--mcd-clip-id", required=True)
    parser.add_argument("--mmpd-mat", type=Path, required=True); parser.add_argument("--mmpd-clip-id", required=True)
    parser.add_argument("--face-landmarker-model", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcd-fps", type=float, default=30.0); parser.add_argument("--mmpd-fps", type=float, default=30.0)
    parser.add_argument("--mmpd-contact-frame-indices", nargs="+", type=int, metavar="FRAME_IDX", help="explicit MMPD contact-sheet frames, for example: 1310 1311 1312")
    args = parser.parse_args(argv)
    try:
        failures = []
        mcd_video = _read_avi(args.mcd_video); mcd_state = _read_mcd_state(args.mcd_state_csv, args.mcd_clip_id)
        if len(mcd_video) != len(mcd_state): raise RuntimeError(f"MCD clip {args.mcd_clip_id}: video/state frame count mismatch ({len(mcd_video)} vs {len(mcd_state)})")
        failure = _run_one("mcd", mcd_video, args.mcd_clip_id, str(args.mcd_video), str(args.face_landmarker_model), args.mcd_fps, args.output_dir, mcd_state)
        if failure: failures.append(failure)
        source = load_mmpd_mat(str(args.mmpd_mat))
        failure = _run_one("mmpd", source["video"], args.mmpd_clip_id, source["source_sha256"], str(args.face_landmarker_model), args.mmpd_fps, args.output_dir, contact_sheet_frame_indices=args.mmpd_contact_frame_indices)
        if failure: failures.append(failure)
        if failures:
            raise RuntimeError("; ".join(failures))
    except Exception as exc:
        print(f"diagnose_current_extraction: {exc}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
