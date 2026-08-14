"""Run the authenticated MCD-only Gate 6 ruler-difference consumer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, resolve_mcd_locator
from adaptive_roi_rppg.evaluation import (
    build_gate6_fixture_plan, build_gate6_full_plan, compare_full_face_trajectories,
    evaluate_current_full_face, publish_gate6_report, replay_historical_full_face,
    summarize_gate6_rows, verify_gate6_publication,
)


def _source_file(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source is not a direct regular file: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _provenance(args, plan) -> dict[str, object]:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise ValueError("Gate 6 requires Slurm: SLURM_JOB_ID and SLURMD_NODENAME are missing")
    snapshot = args.code_snapshot_sha256
    if len(snapshot) != 64 or any(c not in "0123456789abcdef" for c in snapshot):
        raise ValueError("--code-snapshot-sha256 must be a lowercase 64-hex SHA-256")
    manifest = Path(args.manifest_tree); cache = Path(args.historical_cache_root); bundle = load_mcd_manifest_tree(manifest)
    consumed = [_source_file(manifest / name) for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")]
    for binding in plan.bindings:
        # The manifest contract stores clip files by clip_manifest_id, not clip_id.
        bundle_clip = next(c for c in bundle.clip_manifests if c.clip_id == binding.clip_id)
        consumed.append(_source_file(manifest / "clips" / f"{bundle_clip.clip_manifest_id}.json"))
        consumed.append(_source_file(resolve_mcd_locator(f"state/{binding.clip_id}_semantic_state_vectors.csv", state_root=args.state_root, gt_root=args.gt_root)))
        consumed.append(_source_file(resolve_mcd_locator(f"gt/{binding.clip_id}_ground_truth.csv", state_root=args.state_root, gt_root=args.gt_root)))
        consumed.append(_source_file(cache / binding.npz_filename))
    consumed.extend(_source_file(Path(value)) for value in (args.historical_cache_manifest, args.historical_cache_inventory, args.historical_test90_csv))
    return {
        "canonical_roots": {"manifest_tree": str(manifest), "state_root": str(Path(args.state_root)), "gt_root": str(Path(args.gt_root)), "historical_cache_root": str(cache)},
        "consumed_files": consumed, "code_snapshot_sha256": snapshot, "command": " ".join(sys.argv),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurmd_nodename": os.environ["SLURMD_NODENAME"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fixture", "full"), required=True)
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True)
    parser.add_argument("--historical-cache-root", required=True); parser.add_argument("--historical-cache-manifest", required=True); parser.add_argument("--historical-cache-inventory", required=True)
    parser.add_argument("--historical-test90-csv", required=True); parser.add_argument("--historical-per-clip", default=None)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--code-snapshot-sha256", required=True)
    args = parser.parse_args()
    builder = build_gate6_fixture_plan if args.phase == "fixture" else build_gate6_full_plan
    plan = builder(args.manifest_tree, args.historical_cache_root, args.historical_cache_manifest, args.historical_cache_inventory, args.historical_test90_csv, args.historical_test90_csv)
    sources = _provenance(args, plan); source_inventory_sha256 = hashlib.sha256(canonical_json_bytes(sources)).hexdigest()
    current = evaluate_current_full_face(args.manifest_tree, args.state_root, args.gt_root, plan); by_clip = defaultdict(list)
    for row in current: by_clip[row["clip_id"]].append(row)
    hops = []; clips = []
    for binding in plan.bindings:
        historical = replay_historical_full_face(binding.npz_path, binding.npz_sha256)
        historical_mae = sum(abs(row.post_belief_hr_bpm - row.gt_hr_bpm) for row in historical) / len(historical)
        if abs(historical_mae - binding.reference_mae_clip) > 1e-12 + 1e-12 * abs(binding.reference_mae_clip):
            raise ValueError(f"historical Level-A replay disagrees with frozen per-clip reference: {binding.clip_id}")
        rows = compare_full_face_trajectories(historical, by_clip[binding.clip_id], plan_id=plan.plan_id, clip_id=binding.clip_id, subject_id=binding.subject_id, view=binding.view, condition=binding.condition, source_inventory_sha256=source_inventory_sha256); hops.extend(rows)
        joined = [row for row in rows if row["hop_presence"] == "joined"]
        he = [float(row["historical_abs_error_bpm"]) for row in rows if row["historical_abs_error_bpm"] is not None]; ce = [float(row["current_abs_error_bpm"]) for row in rows if row["current_abs_error_bpm"] is not None]; de = [float(row["abs_error_delta_bpm"]) for row in joined if row["abs_error_delta_bpm"] is not None]
        clips.append({"run_id": "gate6", "plan_id": plan.plan_id, "dataset_id": plan.dataset_id, "clip_id": binding.clip_id, "subject_id": binding.subject_id, "view": binding.view, "condition": binding.condition, "expected_hops": binding.expected_hops, "historical_hops": len(historical), "current_hops": len(by_clip[binding.clip_id]), "joined_hops": len(joined), "historical_only_hops": sum(r["hop_presence"] == "historical_only" for r in rows), "current_only_hops": sum(r["hop_presence"] == "current_only" for r in rows), "historical_mae_clip": sum(he) / len(he) if he else None, "current_mae_clip": sum(ce) / len(ce) if ce else None, "mae_delta_bpm": sum(de) / len(de) if de else None, "historical_mae_hop_count": len(he), "current_mae_hop_count": len(ce), "delta_joined_hop_count": len(de), "max_abs_belief_delta_bpm": max((abs(float(r["belief_delta_bpm"])) for r in joined), default=None), "current_invalid_count": sum(not r["current_selected_valid"] for r in rows), "unclassified_count": sum(r["discrepancy_category"] == "unclassified" for r in rows), "fixture_member": binding.subject_id in plan.fixture_subject_ids, "clip_status": "complete" if not any(r["hop_presence"] != "joined" for r in rows) else "variable_hops", "source_inventory_sha256": source_inventory_sha256})
    subject_rows, category_rows, summary = summarize_gate6_rows(hops, clips)
    result = {"run_id": "gate6", "plan": plan, "plan_id": plan.plan_id, "dataset_id": plan.dataset_id, "phase": plan.phase, "source_inventory": sources, "source_inventory_sha256": source_inventory_sha256, "plan_payload": {"plan_id": plan.plan_id, "dataset_id": plan.dataset_id, "phase": plan.phase, "fixture_subject_ids": list(plan.fixture_subject_ids), "source_inventory_sha256": source_inventory_sha256, "clips": [{"clip_id": b.clip_id, "subject_id": b.subject_id, "view": b.view, "condition": b.condition, "npz_filename": b.npz_filename, "npz_sha256": b.npz_sha256, "expected_hops": b.expected_hops, "reference_mae_clip": b.reference_mae_clip} for b in plan.bindings]}, "hop_rows": hops, "clip_rows": clips, "subject_rows": subject_rows, "category_rows": category_rows, "summary": summary, "diagnostics": {"causal_decomposition_status": "not_supported_by_frozen_inputs", "inputs": ["historical NPZ full-face Level-A replay", "current MCD full-face POS action-0 replay", "joined per-hop rows"], "arms": ["historical_Level_A_all_hops", "current_all_hops", "paired_joined_hops"]}}
    publish_gate6_report(result, args.output_dir); verify_gate6_publication(args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
