#!/usr/bin/env python3
"""Evaluate one MMPD clip; intended for one Slurm array task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_roi_rppg.evaluation.adapters.mmpd.minimal import (
    FullFacePolicy, evaluate_minimal_clip, load_minimal_recurrent_policy, validate_checkpoint_manifest,
    write_clip_csv,
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


def clip_for_array_index(index: int) -> dict:
    clips = build_engineering_plan()["clips"]
    if index < 0 or index >= len(clips):
        raise IndexError(index)
    return clips[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("."))
    parser.add_argument("--face-landmarker-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    clips = build_engineering_plan()["clips"]
    if args.array_index < 0 or args.array_index >= len(clips):
        parser.error(f"--array-index must be in [0, {len(clips) - 1}]")
    clip = clip_for_array_index(args.array_index)
    inventory = _load_json(args.raw_inventory).get("clips", [])
    record = next((item for item in inventory if item.get("clip_id") == clip["clip_id"]), None)
    if not isinstance(record, dict) or "locator" not in record:
        raise ValueError(f"raw inventory has no locator for {clip['clip_id']}")
    checkpoint_manifest = _load_json(args.checkpoint_manifest).get("checkpoints", [])
    learned_checkpoint_manifest = [
        item for item in checkpoint_manifest
        if isinstance(item, dict) and item.get("method_id") != "full_face_pos"
    ]
    validate_checkpoint_manifest(learned_checkpoint_manifest)
    policies = {"full_face_pos": FullFacePolicy()}
    for item in learned_checkpoint_manifest:
        method = item.get("method_id")
        if method == "full_face_pos":
            continue
        if not method or "locator" not in item:
            raise ValueError("checkpoint manifest record lacks method_id or locator")
        policies[method] = load_minimal_recurrent_policy(
            _located(args.checkpoint_root, item["locator"]), method_id=method,
            family=str(item.get("family", "ppo")), seed=item.get("seed", ""))
    rows = evaluate_minimal_clip(
        _located(args.raw_root, record["locator"]), clip_id=clip["clip_id"],
        model_asset_path=args.face_landmarker_model, policies=policies,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_clip_csv(rows, args.output_dir / f"{clip['clip_id']}.csv")
    invalid_hops = sorted({int(row["hop_idx"]) for row in rows if not row["selected_valid"]})
    if invalid_hops:
        raise SystemExit(
            f"{clip['clip_id']}: invalid selected measurement at hops "
            + ",".join(str(hop) for hop in invalid_hops)
        )


if __name__ == "__main__":
    main()
