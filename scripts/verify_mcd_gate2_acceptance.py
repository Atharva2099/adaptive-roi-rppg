#!/usr/bin/env python3
"""One-time canonical acceptance check for a published MCD bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_roi_rppg.contracts import read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--expectation", default="configs/data/mcd_gate2_acceptance_v1.json")
    args = parser.parse_args()
    expected = read_json_object(args.expectation)
    expectation_path = Path(args.expectation).resolve()
    expectation_sha256 = sha256_file(expectation_path)
    fields = {"split_sha256", "train_subjects", "eval_subjects", "total_subjects", "total_clips", "camera_condition_combinations"}
    if set(expected) != fields or not isinstance(expected["split_sha256"], str) or not isinstance(expected["camera_condition_combinations"], list):
        raise ValueError("expectation has wrong fields or types")
    bundle = load_mcd_manifest_tree(args.output_dir)
    inventory = read_json_object(Path(args.output_dir) / "source_inventory.json")
    split = bundle.split_manifest
    if inventory["split_source_sha256"] != expected["split_sha256"]:
        raise ValueError("split source hash does not match canonical expectation")
    if (split.train_subject_count, split.eval_subject_count, bundle.dataset_manifest.subject_count, bundle.dataset_manifest.clip_count) != (expected["train_subjects"], expected["eval_subjects"], expected["total_subjects"], expected["total_clips"]):
        raise ValueError("subject or clip counts do not match canonical expectation")
    combinations = {tuple(item) for item in expected["camera_condition_combinations"]}
    by_subject: dict[str, set[tuple[str, str]]] = {}
    for clip in bundle.clip_manifests:
        by_subject.setdefault(clip.subject_id, set()).add((clip.camera_id, clip.condition))
    if set(by_subject) != set(split.train_subject_ids) | set(split.eval_subject_ids) or any(value != combinations for value in by_subject.values()):
        raise ValueError("canonical camera-condition combinations are incomplete")
    replay = load_mcd_manifest_tree(args.output_dir)
    if replay != bundle:
        raise ValueError("final loader replay differs")
    root = Path(args.output_dir)
    result = {"status": "accepted", "expectation_path": str(expectation_path), "expectation_sha256": expectation_sha256,
              "dataset_manifest_id": bundle.dataset_manifest.manifest_id,
              "dataset_manifest_hash": sha256_file(root / "dataset_manifest.json"),
              "inventory_id": inventory["inventory_id"], "inventory_hash": sha256_file(root / "source_inventory.json"),
              "split_id": split.split_id, "split_hash": sha256_file(root / "split_manifest.json"),
              "total_subjects": bundle.dataset_manifest.subject_count, "train_subjects": split.train_subject_count,
              "eval_subjects": split.eval_subject_count, "total_clips": bundle.dataset_manifest.clip_count,
              "train_clips": split.train_clip_count, "eval_clips": split.eval_clip_count,
              "overlap_result": split.overlap_result.value}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
