"""Run or merge the bounded MCD-only Gate 6.1 crossover shards."""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
from pathlib import Path

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.data.mcd import resolve_mcd_locator
from adaptive_roi_rppg.evaluation import (
    build_crossover_rows, merge_crossover_shards,
    publish_crossover_shard,
)
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--shard-dir", action="append", help="complete shard directory; repeat for --merge")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--phase", choices=("fixture", "full"), default="full")
    parser.add_argument("--code-snapshot-sha256", required=False)
    for name in ("manifest_tree", "state_root", "gt_root", "historical_cache_root", "historical_cache_manifest", "historical_cache_inventory", "historical_test90_csv"):
        parser.add_argument(f"--{name.replace('_', '-')}")
    return parser.parse_args(argv)


def _require_slurm() -> tuple[str, str]:
    job = os.environ.get("SLURM_JOB_ID"); node = os.environ.get("SLURMD_NODENAME")
    if not job or not node: raise ValueError("Gate 6.1 requires SLURM_JOB_ID and SLURMD_NODENAME")
    return job, node


def _plan(args):
    from adaptive_roi_rppg.evaluation import build_gate6_fixture_plan, build_gate6_full_plan
    builder = build_gate6_fixture_plan if args.phase == "fixture" else build_gate6_full_plan
    return builder(args.manifest_tree, args.historical_cache_root, args.historical_cache_manifest, args.historical_cache_inventory, args.historical_test90_csv, args.historical_test90_csv)


def _source_inventory(args, plan) -> tuple[dict[str, object], str]:
    bundle = load_mcd_manifest_tree(args.manifest_tree)
    paths = [Path(args.manifest_tree) / name for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")]
    paths += [Path(args.historical_cache_manifest), Path(args.historical_cache_inventory), Path(args.historical_test90_csv)]
    for binding in plan.bindings:
        clip = next(c for c in bundle.clip_manifests if c.clip_id == binding.clip_id)
        paths += [Path(args.manifest_tree) / "clips" / f"{clip.clip_manifest_id}.json", Path(args.historical_cache_root) / binding.npz_filename]
        paths += [resolve_mcd_locator(f"state/{binding.clip_id}_semantic_state_vectors.csv", state_root=args.state_root, gt_root=args.gt_root), resolve_mcd_locator(f"gt/{binding.clip_id}_ground_truth.csv", state_root=args.state_root, gt_root=args.gt_root)]
    records = [{"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size} for path in sorted(set(paths), key=str)]
    payload = {"schema": "gate6.1-source-inventory-v1", "files": records}
    return payload, hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def main() -> int:
    args = _args()
    job_id, node = _require_slurm()
    if not args.code_snapshot_sha256 or len(args.code_snapshot_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.code_snapshot_sha256): raise ValueError("--code-snapshot-sha256 must be lowercase SHA-256")
    if args.merge:
        if not args.shard_dir: raise ValueError("--merge requires --shard-dir")
        plan = _plan(args); _, source_hash = _source_inventory(args, plan)
        command = shlex.join(__import__("sys").argv)
        provenance = {"plan_id": plan.plan_id, "phase": args.phase, "code_snapshot_sha256": args.code_snapshot_sha256, "source_inventory_sha256": source_hash, "command": command, "node": node, "job_id": job_id}
        merge_crossover_shards(args.shard_dir, args.output_dir, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed, expected_phase=args.phase, expected_plan_id=plan.plan_id, expected_code_snapshot_sha256=args.code_snapshot_sha256, expected_source_inventory_sha256=source_hash, expected_clip_subjects={b.clip_id: b.subject_id for b in plan.bindings}, expected_clip_metadata={b.clip_id: {"subject_id": b.subject_id, "view": b.view, "condition": b.condition} for b in plan.bindings}, merged_provenance=provenance)
        return 0
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count: raise ValueError("invalid shard index/count")
    required = (args.manifest_tree, args.state_root, args.gt_root, args.historical_cache_root, args.historical_cache_manifest, args.historical_cache_inventory, args.historical_test90_csv)
    if any(value is None for value in required): raise ValueError("run mode requires all MCD and historical input paths")
    plan = _plan(args); _, source_hash = _source_inventory(args, plan)
    selected_subjects = [subject for index, subject in enumerate(sorted({b.subject_id for b in plan.bindings})) if index % args.shard_count == args.shard_index]
    bindings = tuple(binding for binding in plan.bindings if binding.subject_id in selected_subjects)
    bundle = load_mcd_manifest_tree(args.manifest_tree); current = {}
    for binding in bindings:
        clip = next(c for c in bundle.clip_manifests if c.clip_id == binding.clip_id)
        frames = build_pos_measurements(read_mcd_canonical_frames(bundle, args.state_root, binding.clip_id, "eval"))
        labels = read_mcd_eval_labels(bundle, args.gt_root, binding.clip_id, clip)
        current[binding.clip_id] = (frames, labels)
    rows = build_crossover_rows(bindings, current)
    publish_crossover_shard(rows, args.output_dir, shard_index=args.shard_index, shard_count=args.shard_count, plan_id=plan.plan_id, phase=args.phase, code_snapshot_sha256=args.code_snapshot_sha256, source_inventory_sha256=source_hash, subject_ids=selected_subjects, command=shlex.join(__import__("sys").argv), node=node, job_id=job_id)
    return 0


if __name__ == "__main__": raise SystemExit(main())
