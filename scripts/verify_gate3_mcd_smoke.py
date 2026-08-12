"""Run the state-only, local Gate 3 MCD smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import shlex
import sys
from pathlib import Path

import numpy
import scipy

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.signal import POS_CONFIG_ID, POS_CONFIG_PAYLOAD, build_pos_measurements

CSV_FIELDS = ("clip_id", "camera_id", "condition", "fps", "state_sha256", "state_rows", "expected_hops", "emitted_hops", "valid_hops", "valid_roi_measurements", "stream_sha256")
REASONS = ("missing_required_rgb", "degenerate_channel_mean", "degenerate_projection", "degenerate_pos_postprocess", "degenerate_spectrum")


def _stream_hash(frames) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(canonical_json_bytes(frame.to_dict()) + b"\n")
    return digest.hexdigest()


def _provenance(dataset_id: str, clip_id: str, source_id: str, start: int, end: int) -> str:
    payload = {"dataset_id": dataset_id, "clip_id": clip_id, "source_provenance_id": source_id, "source_frame_start": start, "source_frame_end": end, "signal_config_id": POS_CONFIG_ID}
    return "pos-measurement-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _global_hash(per_clip: list[str]) -> str:
    digest = hashlib.sha256()
    for value in per_clip:
        digest.update(value.encode("ascii") + b"\n")
    return digest.hexdigest()


def _assert_clip(clip, frames, measurements, expected: int) -> None:
    fps = float(clip.camera_fps); window = round(8 * fps); hop = round(fps)
    if not measurements or len(measurements) != expected: raise ContractValidationError(f"{clip.clip_id}: hop count or nonempty assertion failed")
    if not any(frame.valid for frame in measurements) or not any(item.valid for frame in measurements for item in frame.measurements): raise ContractValidationError(f"{clip.clip_id}: no valid frame or ROI")
    for ordinal, frame in enumerate(measurements):
        start = ordinal * hop; end = start + window - 1
        if frame.hop_idx != ordinal or frame.hop_time_s != (window + ordinal * hop) / fps: raise ContractValidationError(f"{clip.clip_id}: hop timing mismatch")
        if frame.signal_config_id != POS_CONFIG_ID or frame.provenance_id != _provenance(clip.dataset_id, clip.clip_id, frames[0].provenance_id, start, end): raise ContractValidationError(f"{clip.clip_id}: frame provenance mismatch")
        if frame.valid != any(item.valid for item in frame.measurements) or (frame.valid and frame.invalid_reason is not None) or (not frame.valid and frame.invalid_reason != "no_valid_roi_measurement"): raise ContractValidationError(f"{clip.clip_id}: whole-frame validity mismatch")
        if len(frame.measurements) != 12: raise ContractValidationError(f"{clip.clip_id}: ROI count mismatch")
        for index, item in enumerate(frame.measurements):
            if item.roi_index != index or item.roi_name != frames[0].roi_values[index].roi_name or item.source_frame_start != start or item.source_frame_end != end or item.signal_config_id != POS_CONFIG_ID: raise ContractValidationError(f"{clip.clip_id}: ROI boundary mismatch")
            if item.source_frame_end >= (window + ordinal * hop): raise ContractValidationError(f"{clip.clip_id}: future source frame")
            if item.coverage is None or not numpy.isfinite(item.coverage): raise ContractValidationError(f"{clip.clip_id}: nonfinite coverage")
            if item.valid:
                if not all(numpy.isfinite(value) for value in (item.hr_bpm, item.confidence, item.peak_power_ratio)) or abs(item.confidence - item.peak_power_ratio * item.coverage) > 1e-12: raise ContractValidationError(f"{clip.clip_id}: invalid valid ROI fields")
            elif any(value is not None for value in (item.hr_bpm, item.confidence, item.peak_power_ratio)): raise ContractValidationError(f"{clip.clip_id}: invalid ROI fields are not null")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tree", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    args = parser.parse_args()
    if len(args.code_snapshot_sha256) != 64 or any(char not in "0123456789abcdef" for char in args.code_snapshot_sha256): raise SystemExit("--code-snapshot-sha256 must be 64 lowercase hexadecimal characters")
    job_id, node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
    if not job_id or not node: raise SystemExit("SLURM_JOB_ID and SLURMD_NODENAME are required")
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit("output-dir must not exist or must be empty")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_mcd_manifest_tree(args.manifest_tree)
    train = sorted((clip for clip in bundle.clip_manifests if clip.subject_id in bundle.split_manifest.train_subject_ids), key=lambda clip: clip.subject_id)
    if not train: raise SystemExit("no train subject")
    subject = train[0].subject_id; clips = sorted((clip for clip in train if clip.subject_id == subject), key=lambda clip: clip.clip_id)
    expected_keys = {(camera, condition) for camera in ("FullHDwebcam", "USBVideo", "IriunWebcam") for condition in ("before", "after")}
    if len(clips) != 6 or {(clip.camera_id, clip.condition) for clip in clips} != expected_keys or {clip.camera_fps for clip in clips} != {24.0, 30.0}: raise SystemExit("first train subject does not have the exact six-clip inventory")
    rows, per_reason, run_hashes = [], {reason: 0 for reason in REASONS}, [[], []]
    for clip in clips:
        frames = read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "train")
        outputs = [build_pos_measurements(frames), build_pos_measurements(frames)]
        expected = max(0, (len(frames) - round(8 * clip.camera_fps)) // round(clip.camera_fps) + 1)
        for run_index, measurements in enumerate(outputs):
            _assert_clip(clip, frames, measurements, expected)
            run_hashes[run_index].append(_stream_hash(measurements))
            for frame in measurements:
                for item in frame.measurements:
                    if run_index == 0 and not item.valid:
                        if item.invalid_reason not in per_reason: raise ContractValidationError(f"{clip.clip_id}: unknown invalid reason")
                        per_reason[item.invalid_reason] += 1
        if run_hashes[0][-1] != run_hashes[1][-1]: raise ContractValidationError(f"{clip.clip_id}: run hashes differ")
        first = outputs[0]
        rows.append({"clip_id": clip.clip_id, "camera_id": clip.camera_id, "condition": clip.condition, "fps": clip.camera_fps, "state_sha256": clip.state_sha256, "state_rows": clip.state_row_count, "expected_hops": expected, "emitted_hops": len(first), "valid_hops": sum(frame.valid for frame in first), "valid_roi_measurements": sum(item.valid for frame in first for item in frame.measurements), "stream_sha256": run_hashes[0][-1]})
    csv_path = output / "gate3_mcd_smoke.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(rows)
    manifest_paths = {name: sha256_file(Path(args.manifest_tree) / name) for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")}
    global_one, global_two = _global_hash(run_hashes[0]), _global_hash(run_hashes[1])
    payload = {"subject_id": subject, "dataset_id": bundle.dataset_manifest.dataset_id, "split_id": bundle.split_manifest.split_id, "inventory_id": bundle.source_inventory.get("inventory_id"), "dataset_manifest_id": bundle.dataset_manifest.manifest_id, "manifest_hashes": manifest_paths, "state_root_locator": str(Path(args.state_root)), "signal_config_payload": POS_CONFIG_PAYLOAD, "signal_config_id": POS_CONFIG_ID, "signal_config_payload_sha256": hashlib.sha256(canonical_json_bytes(POS_CONFIG_PAYLOAD)).hexdigest(), "command": shlex.join(sys.argv), "code_snapshot_sha256": args.code_snapshot_sha256, "slurm_job_id": job_id, "slurm_nodename": node, "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "stream_sha256_run1": global_one, "stream_sha256_run2": global_two, "per_reason_invalid_counts": per_reason, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "assertions": "passed", "clips": rows}
    (output / "gate3_mcd_smoke.json").write_bytes(canonical_json_bytes(payload))
    return 0


if __name__ == "__main__": raise SystemExit(main())
