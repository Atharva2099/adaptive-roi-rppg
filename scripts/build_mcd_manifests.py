#!/usr/bin/env python3
"""Build and publish the MCD Gate 2 manifest tree."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from adaptive_roi_rppg.contracts import sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import build_mcd_manifest_bundle, write_mcd_manifest_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()
    state = Path(args.state_root).resolve()
    gt = Path(args.gt_root).resolve()
    output = Path(args.output_dir).resolve()
    parent = output.parent
    if not parent.is_dir() or output == state or output == gt or state in output.parents or gt in output.parents or parent == state or parent == gt:
        raise ContractValidationError("output_dir: output and parent must be outside source roots")
    bundle = build_mcd_manifest_bundle(state, gt, args.split_csv, created_at_utc=args.created_at_utc,
                                       producer_command=shlex.join(sys.argv))
    write_mcd_manifest_bundle(bundle, output)
    dataset_path = output / "dataset_manifest.json"
    split_path = output / "split_manifest.json"
    inventory_path = output / "source_inventory.json"
    result = {
        "output_dir": str(output),
        "dataset_manifest_id": bundle.dataset_manifest.manifest_id,
        "dataset_manifest_hash": sha256_file(dataset_path),
        "inventory_id": bundle.source_inventory["inventory_id"],
        "inventory_hash": sha256_file(inventory_path),
        "split_id": bundle.split_manifest.split_id,
        "split_hash": sha256_file(split_path),
        "total_subjects": bundle.dataset_manifest.subject_count,
        "train_subjects": bundle.split_manifest.train_subject_count,
        "eval_subjects": bundle.split_manifest.eval_subject_count,
        "total_clips": bundle.dataset_manifest.clip_count,
        "train_clips": bundle.split_manifest.train_clip_count,
        "eval_clips": bundle.split_manifest.eval_clip_count,
        "overlap_result": bundle.split_manifest.overlap_result.value,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
