#!/usr/bin/env python3
"""Run one MMPD-only extraction audit task for a Slurm array index."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from adaptive_roi_rppg.contracts import ROI_NAMES
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import (
    extract_face_relative_canonical_frames,
    load_mmpd_mat,
)
from adaptive_roi_rppg.evaluation.adapters.mmpd.plan import build_engineering_plan


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _located(root: Path, locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else root / path


def _failure_frame(reason: str) -> int | None:
    match = re.search(r"frame_idx=(\d+)", reason)
    return int(match.group(1)) if match else None


def _stable_json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _require_alignment(frames, diagnostics) -> None:
    if len(frames) != len(diagnostics):
        raise ValueError(f"audit frame/diagnostic length mismatch: {len(frames)} != {len(diagnostics)}")
    for index in range(len(frames)):
        if frames[index].frame_idx != diagnostics[index].get("frame_idx"):
            raise ValueError(f"audit frame/diagnostic index mismatch at position {index}")


def _write_rows(path: Path, frames, diagnostics) -> None:
    _require_alignment(frames, diagnostics)
    fields = ["frame_idx", "face_box", "roi_boxes", "geometry_source", "geometry_source_frame_idx", "geometry_bridged", "current_frame_evidence", "recovery_attempts"]
    for roi in ROI_NAMES:
        prefix = roi.value
        fields.extend((f"{prefix}_r", f"{prefix}_g", f"{prefix}_b", f"{prefix}_std", f"{prefix}_coverage", f"{prefix}_valid", f"{prefix}_reason"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(frames)):
            frame, diagnostic = frames[index], diagnostics[index]
            row = {"frame_idx": frame.frame_idx, "face_box": _stable_json(diagnostic["face_box"]),
                   "roi_boxes": _stable_json(diagnostic["roi_boxes"]),
                   "geometry_source": diagnostic["geometry_source"],
                   "geometry_source_frame_idx": diagnostic["geometry_source_frame_idx"],
                   "geometry_bridged": diagnostic["geometry_bridged"],
                   "current_frame_evidence": diagnostic["current_frame_evidence"],
                   "recovery_attempts": _stable_json(diagnostic["recovery_attempts"])}
            for roi in frame.roi_values:
                prefix = roi.roi_name.value
                row.update({f"{prefix}_r": roi.r_mean, f"{prefix}_g": roi.g_mean, f"{prefix}_b": roi.b_mean,
                            f"{prefix}_std": roi.std, f"{prefix}_coverage": roi.coverage,
                            f"{prefix}_valid": roi.valid, f"{prefix}_reason": roi.invalid_reason})
            writer.writerow(row)


def _inventory_record(inventory, *, clip_id: str, subject_id: str) -> dict:
    matches = [item for item in inventory if isinstance(item, dict) and item.get("clip_id") == clip_id]
    if len(matches) != 1 or matches[0].get("subject_id") != subject_id or "locator" not in matches[0]:
        raise ValueError(f"raw inventory must contain exactly one clip/subject record for {clip_id}/{subject_id}")
    return matches[0]


def audit_clip(*, raw_path: Path, clip_id: str, subject_id: str, source_locator: str, model_path: str, output_dir: Path, fps: float = 30.0) -> dict:
    diagnostics = []
    failed_diagnostics = []
    emitted_frames = []
    frames = ()
    source = {}
    failure_reason = ""
    try:
        source = load_mmpd_mat(str(raw_path))
        frames = extract_face_relative_canonical_frames(
            source["video"], dataset_id="mmpd", clip_id=clip_id,
            source_provenance_id=source["source_sha256"], model_asset_path=model_path,
            fps=fps, diagnostics=diagnostics,
            reject_exact_zero_raw_frames=True, emitted_frames=emitted_frames,
            failed_diagnostics=failed_diagnostics,
        )
    except Exception as exc:
        failure_reason = str(exc)
        frames = tuple(emitted_frames)

    _require_alignment(frames, diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / f"{clip_id}_per_frame.csv", frames, diagnostics)
    (output_dir / f"{clip_id}_failed_frames.json").write_text(
        json.dumps(failed_diagnostics, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    invalid_values = sum(1 for frame in frames for roi in frame.roi_values if not roi.valid)
    video_detections = sum(d.get("geometry_source") == "video" for d in diagnostics)
    image_recoveries = sum(d.get("geometry_source") == "image" for d in diagnostics)
    bridges = sum(bool(d.get("geometry_bridged")) for d in diagnostics)
    if bridges:
        raise ValueError("audit found forbidden stale geometry bridges")
    emitted_frames = len(frames)
    summary = {
        "clip_id": clip_id,
        "subject_id": subject_id,
        "source_locator": source_locator,
        "original_frame_count": source.get("original_frame_count"),
        "processed_frame_count": source.get("processed_frame_count", emitted_frames),
        "trimmed_tail_frames": source.get("trimmed_tail_frames", 0),
        "video_detections": video_detections,
        "image_recoveries": image_recoveries,
        "one_frame_bridges": bridges,
        "emitted_frame_count": emitted_frames,
        "emitted_roi_count": emitted_frames * len(ROI_NAMES),
        "invalid_roi_value_count": invalid_values,
        "first_failure_frame": _failure_frame(failure_reason) if failure_reason else None,
        "first_failure_reason": failure_reason or None,
        "classification": "still_failed" if failure_reason or invalid_values else ("repaired" if source.get("trimmed_tail_frames", 0) or image_recoveries else "clean"),
    }
    (output_dir / f"{clip_id}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("."))
    parser.add_argument("--face-landmarker-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    clips = build_engineering_plan()["clips"]
    if args.array_index < 0 or args.array_index >= len(clips):
        parser.error(f"--array-index must be in [0, {len(clips) - 1}]")
    clip = clips[args.array_index]
    inventory = _load_json(args.raw_inventory).get("clips", [])
    record = _inventory_record(inventory, clip_id=clip["clip_id"], subject_id=clip["subject_id"])
    summary = audit_clip(raw_path=_located(args.raw_root, record["locator"]), clip_id=clip["clip_id"], subject_id=clip["subject_id"], source_locator=str(record["locator"]), model_path=args.face_landmarker_model, output_dir=args.output_dir)
    if summary["classification"] == "still_failed":
        print(f"{clip['clip_id']}: {summary['first_failure_reason']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
