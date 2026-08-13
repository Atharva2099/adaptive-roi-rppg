"""Slurm-only train-MCD Gate 5 structural smoke; never runs on real data locally."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

import numpy
import scipy

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.data.mcd import MCD_STATE_SUFFIX
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation import (RunProvenance, build_train_full_face_plan,
    evaluate_and_publish, verify_publication_against_sources,
    verify_publication_structure)


def _sha(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise SystemExit("--code-snapshot-sha256 must be lowercase SHA-256")


def _substantive_hashes(root: Path):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256")}


def _independent_csv_check(root: Path, plan, provenance):
    with (root / "per_hop.csv").open(newline="", encoding="utf-8") as handle: hops = list(csv.DictReader(handle))
    with (root / "per_clip.csv").open(newline="", encoding="utf-8") as handle: clips_rows = list(csv.DictReader(handle))
    with (root / "subject_summary.csv").open(newline="", encoding="utf-8") as handle: subjects = list(csv.DictReader(handle))
    run = read_json_object(root / "run_summary.json"); manifest = read_json_object(root / "run_manifest.json")
    if manifest["plan_id"] != plan.plan_id or canonical_json_bytes(manifest["frozen_plan"]) != canonical_json_bytes(plan.to_dict()): raise ContractValidationError("independent smoke manifest identity check failed")
    if run["run_id"] != provenance.run_id or manifest["run_provenance"]["run_id"] != provenance.run_id or manifest["run_provenance"]["slurm_job_id"] != provenance.slurm_job_id or manifest["run_provenance"]["slurm_node"] != provenance.slurm_node or len(hops) != sum(int(row["scored_hops"]) for row in clips_rows): raise ContractValidationError("independent smoke row identity/count check failed")
    for name, entry in manifest["outputs"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != entry["sha256"] or (root / name).stat().st_size != entry["bytes"]: raise ContractValidationError("independent smoke manifest hash check failed")
    sidecar = {line.split("  ")[1]: line.split("  ")[0] for line in (root / "artifacts.sha256").read_text(encoding="utf-8").splitlines()}
    if set(sidecar) != {"per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json"} or any(sidecar[name] != hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sidecar): raise ContractValidationError("independent smoke sidecar hash check failed")
    errors = {}
    for row in hops:
        if row["run_id"] != provenance.run_id or row["plan_id"] != plan.plan_id or row["method_id"] != plan.method_id or row["signal_config_id"] != plan.signal_config_id or row["control_config_id"] != plan.control_config_id or row["observation_schema_id"] != plan.observation_schema_id or row["gt_rule_id"] != plan.gt_rule_id or row["proposed_action"] != "0" or row["executed_action"] != "0": raise ContractValidationError("independent smoke identity/action check failed")
        prediction, gt, absolute = float(row["prediction_post_belief_hr_bpm"]), float(row["gt_hr_bpm"]), float(row["abs_error_bpm"])
        if not numpy.isfinite((prediction, gt, absolute)).all() or abs(abs(prediction - gt) - absolute) > 1e-12: raise ContractValidationError("independent smoke error check failed")
        errors.setdefault(row["clip_id"], []).append(absolute)
    for row in clips_rows:
        values = errors.get(row["clip_id"], [])
        if int(row["expected_hops"]) != len(values) or int(row["scored_hops"]) != len(values) or abs(float(row["mae_clip"]) - sum(values) / len(values)) > 1e-12 or sum(int(row[f"action_count_{i:02d}"]) for i in range(12)) != len(values): raise ContractValidationError("independent smoke clip aggregation failed")
    if abs(float(run["equal_clip_mean_mae"]) - sum(float(row["mae_clip"]) for row in clips_rows) / len(clips_rows)) > 1e-12: raise ContractValidationError("independent smoke run aggregation failed")
    for subject in subjects:
        values = [float(row["mae_clip"]) for row in clips_rows if row["subject_id"] == subject["subject_id"]]
        if int(subject["clip_count"]) != len(values) or abs(float(subject["mean_clip_mae"]) - sum(values) / len(values)) > 1e-12: raise ContractValidationError("independent smoke subject aggregation failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--code-snapshot-sha256", required=True)
    args = parser.parse_args(); _sha(args.code_snapshot_sha256)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"): raise SystemExit("SLURM_JOB_ID and SLURMD_NODENAME are required")
    output = Path(args.output_dir)
    if output.is_symlink() or output.exists(): raise SystemExit("output-dir must not already exist")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir(): raise SystemExit("output-dir parent must be an existing real directory")
    manifest_root = Path(args.manifest_tree)
    plan = build_train_full_face_plan(manifest_root)
    provenance = RunProvenance("gate5-mcd-smoke", args.code_snapshot_sha256, shlex.join(["verify_gate5_mcd_smoke.py", *sys.argv[1:]]), {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}, os.environ["SLURM_JOB_ID"], os.environ["SLURMD_NODENAME"])
    scratch = []; scratch_parent = Path(tempfile.mkdtemp(prefix="gate5-scratch-", dir=parent))
    try:
        for name in ("repeat-a", "repeat-b"):
            path = scratch_parent / name; scratch.append(path); evaluate_and_publish(manifest_root, args.state_root, args.gt_root, provenance, path, plan)
        if _substantive_hashes(scratch[0]) != _substantive_hashes(scratch[1]): raise ContractValidationError("repeat publication differs")
        _independent_csv_check(scratch[0], plan, provenance)
        verify_publication_structure(scratch[0])
        verify_publication_against_sources(scratch[0], manifest_root, args.state_root, args.gt_root)
        tampered_state = scratch_parent / "tampered-state"; tampered_state.mkdir()
        source_root = Path(args.state_root)
        if source_root.is_symlink() or not source_root.is_dir(): raise ContractValidationError("state source root must be a real directory")
        copied = {}
        for binding in plan.clip_bindings:
            source = source_root / f"{binding.clip_id}{MCD_STATE_SUFFIX}"
            destination = tampered_state / source.name
            if source.is_symlink() or not source.is_file() or not destination.parent.is_dir(): raise ContractValidationError("state tamper source/destination must be direct regular files")
            data = source.read_bytes(); copied[binding.clip_id] = (source, destination, hashlib.sha256(data).hexdigest())
            destination.write_bytes(data)
        selected_id = plan.clip_bindings[0].clip_id
        source, first_state, original_sha = copied[selected_id]
        before = first_state.read_bytes()
        offset = before.find(b",0.0")
        if offset < 0: offset = before.find(b"0")
        if offset < 0: raise ContractValidationError("selected state source has no mutable byte")
        mutated = before[:offset] + (b"1" if before[offset:offset + 1] != b"1" else b"2") + before[offset + 1:]
        first_state.write_bytes(mutated)
        if first_state.read_bytes() == before or hashlib.sha256(first_state.read_bytes()).hexdigest() == original_sha: raise ContractValidationError("state mutation did not change bytes/hash")
        negative = scratch_parent / "negative"; scratch.append(negative)
        try: evaluate_and_publish(manifest_root, tampered_state, args.gt_root, provenance, negative, plan)
        except ContractValidationError: pass
        else: raise ContractValidationError("tampered state source was accepted")
        if hashlib.sha256(source.read_bytes()).hexdigest() != original_sha: raise ContractValidationError("original state source changed")
        if {path.name for path in negative.iterdir()} != {"STARTED.json", "FAILED.json"}: raise ContractValidationError("negative publication state is not STARTED+FAILED")
        existing = scratch_parent / "existing"; existing.mkdir(); (existing / "foreign").write_text("keep", encoding="utf-8")
        try: evaluate_and_publish(manifest_root, args.state_root, args.gt_root, provenance, existing, plan)
        except ContractValidationError: pass
        else: raise ContractValidationError("existing destination was replaced")
        if (existing / "foreign").read_text(encoding="utf-8") != "keep": raise ContractValidationError("foreign destination was altered")
        evaluate_and_publish(manifest_root, args.state_root, args.gt_root, provenance, output, plan)
    finally:
        shutil.rmtree(scratch_parent, ignore_errors=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
