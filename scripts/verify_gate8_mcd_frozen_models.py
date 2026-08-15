"""Evaluate the nine frozen controllers on canonical MCD data.

Each run writes one canonical subject shard.  Merge independently replays its
sources, validates every shard, and commits COMPLETE.json only after rebuilding
every published aggregate from the CSV bytes on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.evaluation.model_plan import EXCLUSIONS, load_frozen_model_plan
from adaptive_roi_rppg.evaluation.model_publication import (
    CLIP_FIELDS, HOP_FIELDS, MARKER_NAMES, REPORT_SCHEMA, RUN_MANIFEST_SCHEMA,
    SHARD_SCHEMA, SUBJECT_FIELDS, artifact_map, canonical_subject_shard,
    csv_bytes, expected_shard_clip_ids, marker_payload, read_strict_csv,
    validate_directory, validate_exact_json, validate_hop_rows, validate_precomplete_directory,
    validate_shard_manifest, write_csv_exclusive, write_marker,
)
from adaptive_roi_rppg.evaluation.model_replay import (
    paired_subject_bootstrap, rollout_frozen_policy, score_frozen_rollout,
    summarize_model_rows,
)
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements


def _fail(message: str) -> None: raise ContractValidationError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle: handle.write(canonical_json_bytes(value))


def _tree_hash(base: str | Path) -> str:
    base = Path(base).resolve(); ignored = {".git", ".venv", "__pycache__"}; files: list[Path] = []
    def walk(path: Path) -> None:
        for entry in sorted(path.iterdir(), key=lambda p: p.name.encode()):
            if entry.name in ignored: continue
            if entry.is_symlink(): _fail("code snapshot has a symlink")
            if entry.is_dir(): walk(entry)
            elif entry.is_file() and entry.suffix != ".pyc": files.append(entry.relative_to(base))
            else: _fail("code snapshot has an unsupported entry")
    walk(base); digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode() + b"\0"); digest.update((base / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true"); parser.add_argument("--shard-dir", action="append")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--manifest-tree", required=True)
    parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--code-snapshot-sha256", required=True)
    return parser.parse_args()


def _runtime_checkpoints(args: argparse.Namespace, plan: Any) -> tuple[dict[str, Any], ...]:
    root = Path(args.checkpoint_root)
    if root.is_symlink() or not root.is_dir(): _fail("checkpoint root must be an existing non-symlink directory")
    root = root.resolve(); resolved = []
    for original in plan.checkpoints:
        locator = Path(original["locator"])
        if locator.is_absolute() or ".." in locator.parts: _fail("frozen checkpoint locator must be package-relative")
        path = (root / locator).resolve()
        if root not in path.parents or path.is_symlink() or not path.is_file(): _fail("frozen checkpoint package path is invalid")
        spec = dict(original); spec["locator"] = str(path); resolved.append(spec)
    return tuple(resolved)


def _identities(plan: Any) -> list[dict[str, Any]]:
    return [{key: spec[key] for key in ("method_id", "family", "seed", "sha256", "byte_size", "provenance_status", "training_config_provenance")} for spec in plan.checkpoints]


def _inventory(args: argparse.Namespace, checkpoints: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str, str]:
    paths = [Path(args.manifest_tree) / name for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")] + [Path(args.plan)] + [Path(spec["locator"]) for spec in checkpoints]
    if any(path.is_symlink() or not path.is_file() for path in paths): _fail("source/checkpoint inventory path is invalid")
    files = [{"path": str(path), "sha256": sha256_file(path), "byte_size": path.stat().st_size} for path in sorted(set(paths), key=str)]
    installed = {dist.metadata["Name"] for dist in package_metadata.distributions()}
    environment = {"runtime": {name: package_metadata.version(name) if name in installed else "not-installed" for name in ("numpy", "scipy", "stable-baselines3", "sb3-contrib", "gymnasium", "torch", "cloudpickle")}}
    source = {"files": files}; inventory = {"source": source, "environment": environment}
    return inventory, hashlib.sha256(canonical_json_bytes(source)).hexdigest(), hashlib.sha256(canonical_json_bytes(environment)).hexdigest()


def _marker_provenance(args: argparse.Namespace, plan: Any, checkpoints: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(args.code_snapshot_sha256) != 64 or set(args.code_snapshot_sha256) - set("0123456789abcdef"): _fail("code snapshot hash must be lowercase SHA-256")
    job, node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
    if not job or not node: _fail("Gate 8 requires Slurm")
    if _tree_hash(Path(__file__).resolve().parents[1]) != args.code_snapshot_sha256: _fail("declared code snapshot hash does not match runtime source")
    inventory, source_hash, environment_hash = _inventory(args, checkpoints)
    # Shard and merge invocations necessarily have different CLI arguments.
    # Bind their shared publication provenance to the immutable runner/plan
    # contract, while Slurm's launcher records the full shell command.
    provenance = {"plan_id": plan.plan_id, "code_snapshot_sha256": args.code_snapshot_sha256, "source_inventory_sha256": source_hash, "environment_sha256": environment_hash, "job_id": job, "node": node, "command": f"gate8-frozen-models:{plan.plan_id}"}
    return provenance, inventory


def _bindings(args: argparse.Namespace) -> tuple[Any, tuple[Any, ...]]:
    bundle = load_mcd_manifest_tree(args.manifest_tree); ids = set(bundle.split_manifest.eval_clip_ids)
    if len(ids) != 540: _fail("Gate 8 requires the original 540-clip eval inventory before exclusions")
    clips = tuple(sorted((clip for clip in bundle.clip_manifests if clip.clip_id in ids and clip.clip_id not in EXCLUSIONS), key=lambda clip: clip.clip_id))
    if len(clips) != 533 or len({clip.subject_id for clip in clips}) != 89 or ids - {clip.clip_id for clip in clips} != EXCLUSIONS: _fail("Gate 8 requires exactly the Gate 6 533-clip/89-subject cohort")
    if {clip.subject_id for clip in clips} & {clip.subject_id for clip in bundle.clip_manifests if clip.clip_id in bundle.split_manifest.train_clip_ids}: _fail("train/evaluation subject overlap")
    if sum(max(0, (clip.state_row_count-round(8*clip.camera_fps))//round(clip.camera_fps)+1) for clip in clips) != 91227: _fail("Gate 8 expected hop cohort is not 91,227")
    return bundle, clips


def _expected_hops(clips: Sequence[Any]) -> dict[str, int]:
    return {clip.clip_id: max(0, (clip.state_row_count-round(8*clip.camera_fps))//round(clip.camera_fps)+1) for clip in clips}


def _evaluate(args: argparse.Namespace, clips: Sequence[Any], checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity
    class FullFace:
        identity = CheckpointIdentity("full_face", "fixed_full_face", None, None)
        def initial_state(self): return None
        def predict(self, observation, recurrent_state, *, episode_start): return 0, None
    bundle, _ = _bindings(args); policies = [FullFace(), *[load_frozen_recurrent_policy(spec) for spec in checkpoints]]; rows = []
    for clip in clips:
        frames = build_pos_measurements(read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "eval")); labels = read_mcd_eval_labels(bundle, args.gt_root, clip.clip_id, clip)
        if len(frames) != _expected_hops([clip])[clip.clip_id] or len(labels) != len(frames): _fail("canonical source does not match expected hop coverage")
        for policy in policies: rows.extend(score_frozen_rollout(rollout_frozen_policy(frames, policy, clip_id=clip.clip_id), labels, {"subject_id": clip.subject_id, "view": clip.view, "condition": clip.condition}))
    return sorted(rows, key=lambda row: (row["method_id"], row["family"], str(row["seed"]), str(row["checkpoint_sha256"]), row["clip_id"], row["hop_idx"]))


def _paired(plan: Any, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    schedule = plan.payload["bootstrap"]["comparison_seeds"]
    pairs = []
    for seed in range(3):
        pairs.extend(((f"dagger_seed{seed}", "full_face"), (f"standard_ppo_seed{seed}", f"dagger_seed{seed}"), (f"advantage_ppo_seed{seed}", f"dagger_seed{seed}"), (f"advantage_ppo_seed{seed}", f"standard_ppo_seed{seed}")))
    if len(schedule) != len(pairs): _fail("frozen bootstrap comparison seed schedule is invalid")
    for pair, value in zip(pairs, schedule, strict=True):
        out.extend((paired_subject_bootstrap(summary["clip_rows"], *pair, estimand="equal_clip", seed=value), paired_subject_bootstrap(summary["clip_rows"], *pair, estimand="equal_subject", seed=value)))
    return out


def _cohort() -> dict[str, Any]: return {"original_eval_clip_count": 540, "clip_count": 533, "subject_count": 89, "expected_hops_per_checkpoint": 91227, "exclusions": sorted(EXCLUSIONS)}


def _validate_learned(summary: Mapping[str, Any], plan: Any) -> None:
    expected = {(spec["method_id"], spec["family"], spec["seed"], spec["sha256"]) for spec in plan.checkpoints}
    learned = [row for row in summary["clip_rows"] if row["method_id"] != "full_face"]
    actual = {(row["method_id"], row["family"], row["seed"], row["checkpoint_sha256"]) for row in learned}
    if actual != expected or len(learned) != 533 * 9 or len({row["subject_id"] for row in learned}) != 89 or sum(row["hop_count"] for row in learned) != 91227 * 9: _fail("merged learned checkpoint cohort is incomplete or has conflicting identities")


def _report(provenance: Mapping[str, Any], inventory: Mapping[str, Any], plan: Any, summary: Mapping[str, Any]) -> dict[str, Any]:
    _validate_learned(summary, plan)
    return {"schema": REPORT_SCHEMA, **provenance, "source_inventory": inventory, "cohort": _cohort(), "checkpoint_results": summary["checkpoint_results"], "family_results": summary["family_results"], "subject_rows": summary["subject_rows"], "view_rows": summary["view_rows"], "condition_rows": summary["condition_rows"], "view_condition_rows": summary["view_condition_rows"], "paired_comparisons": _paired(plan, summary), "checkpoint_identities": _identities(plan), "limitations": ["Frozen MCD evaluation only; no training or MMPD access.", "Historical and canonical POS pipelines are not identical.", "Checkpoint training-config provenance is descriptive_unverified; file/hash/architecture identity is verified."]}


def _run_manifest(provenance: Mapping[str, Any], inventory: Mapping[str, Any], plan: Any) -> dict[str, Any]:
    return {"schema": RUN_MANIFEST_SCHEMA, **provenance, "source_inventory": inventory, "cohort": _cohort(), "checkpoint_identities": _identities(plan), "status": "complete"}


def _fresh_output(path: Path) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink(): _fail("output must be a fresh path under an existing real parent")
    path.mkdir()


def _validate_shard_args(args: argparse.Namespace) -> None:
    if not isinstance(args.shard_count, int) or not isinstance(args.shard_index, int) or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        _fail("invalid shard assignment")


def _publish_shard(args: argparse.Namespace, plan: Any, clips: Sequence[Any], checkpoints: Sequence[Mapping[str, Any]]) -> None:
    _validate_shard_args(args)
    provenance, _ = _marker_provenance(args, plan, checkpoints); root = Path(args.output_dir); _fresh_output(root)
    write_marker(root, marker_payload("STARTED", "shard", provenance, shard_index=args.shard_index, shard_count=args.shard_count))
    complete_written = False
    try:
        expected_subjects = canonical_subject_shard(sorted({clip.subject_id for clip in clips}), args.shard_index, args.shard_count)
        expected_clips = expected_shard_clip_ids([{"subject_id": clip.subject_id, "clip_id": clip.clip_id} for clip in clips], args.shard_index, args.shard_count)
        selected = tuple(clip for clip in clips if clip.subject_id in set(expected_subjects))
        if tuple(clip.clip_id for clip in selected) != expected_clips: _fail("canonical shard selection differs from frozen clip order")
        rows = _evaluate(args, selected, checkpoints); validate_hop_rows(rows, expected_clips, _expected_hops(selected)); summary = summarize_model_rows(rows)
        write_csv_exclusive(root / "per_hop.csv", rows, HOP_FIELDS); write_csv_exclusive(root / "per_clip.csv", summary["clip_rows"], CLIP_FIELDS)
        manifest = {"schema": SHARD_SCHEMA, **provenance, "shard_index": args.shard_index, "shard_count": args.shard_count, "subject_ids": list(expected_subjects), "clip_ids": list(expected_clips), "checkpoint_identities": _identities(plan), "expected_hops": _expected_hops(selected)}
        _write_json(root / "shard_manifest.json", manifest)
        validate_shard_manifest(read_json_object(root / "shard_manifest.json"), provenance=provenance, expected_subject_ids=expected_subjects, expected_clip_ids=expected_clips, shard_index=args.shard_index, shard_count=args.shard_count, checkpoint_identities=_identities(plan))
        if read_strict_csv(root / "per_hop.csv", HOP_FIELDS) != rows or (root / "per_clip.csv").read_bytes() != csv_bytes(summary["clip_rows"], CLIP_FIELDS): _fail("shard output reread differs before completion")
        shard_outputs = ("per_hop.csv", "per_clip.csv", "shard_manifest.json")
        output_map = validate_precomplete_directory(root, shard_outputs, kind="shard", provenance=provenance)
        complete = marker_payload("COMPLETE", "shard", provenance, shard_index=args.shard_index, shard_count=args.shard_count, outputs=output_map)
        write_marker(root, complete)
        complete_written = True
    except BaseException:
        if not complete_written:
            try: write_marker(root, marker_payload("FAILED", "shard", provenance, shard_index=args.shard_index, shard_count=args.shard_count))
            except BaseException: pass
        raise


def _merge(args: argparse.Namespace, plan: Any, clips: Sequence[Any], checkpoints: Sequence[Mapping[str, Any]]) -> None:
    _validate_shard_args(args)
    if not args.shard_dir or len(args.shard_dir) != args.shard_count: _fail("merge requires every deterministic shard")
    provenance, inventory = _marker_provenance(args, plan, checkpoints); root = Path(args.output_dir); _fresh_output(root)
    write_marker(root, marker_payload("STARTED", "run", provenance))
    complete_written = False
    try:
        for index, name in enumerate(args.shard_dir):
            expected_subjects = canonical_subject_shard(sorted({clip.subject_id for clip in clips}), index, args.shard_count)
            expected_clips = expected_shard_clip_ids([{"subject_id": clip.subject_id, "clip_id": clip.clip_id} for clip in clips], index, args.shard_count)
            selected = tuple(clip for clip in clips if clip.subject_id in set(expected_subjects)); directory = Path(name)
            validate_directory(directory, {"STARTED.json", "COMPLETE.json", "per_hop.csv", "per_clip.csv", "shard_manifest.json"}, kind="shard", provenance=provenance, complete_outputs=("per_hop.csv", "per_clip.csv", "shard_manifest.json"))
            validate_shard_manifest(read_json_object(directory / "shard_manifest.json"), provenance=provenance, expected_subject_ids=expected_subjects, expected_clip_ids=expected_clips, shard_index=index, shard_count=args.shard_count, checkpoint_identities=_identities(plan))
            stored = read_strict_csv(directory / "per_hop.csv", HOP_FIELDS); validate_hop_rows(stored, expected_clips, _expected_hops(selected))
            expected = _evaluate(args, selected, checkpoints)
            if (directory / "per_hop.csv").read_bytes() != csv_bytes(expected, HOP_FIELDS) or (directory / "per_clip.csv").read_bytes() != csv_bytes(summarize_model_rows(stored)["clip_rows"], CLIP_FIELDS): _fail("shard rows do not match source-authoritative replay")
        rows = _evaluate(args, clips, checkpoints); validate_hop_rows(rows, [clip.clip_id for clip in clips], _expected_hops(clips)); summary = summarize_model_rows(rows)
        write_csv_exclusive(root / "per_hop.csv", rows, HOP_FIELDS); write_csv_exclusive(root / "per_clip.csv", summary["clip_rows"], CLIP_FIELDS); write_csv_exclusive(root / "per_subject.csv", summary["subject_rows"], SUBJECT_FIELDS)
        report = _report(provenance, inventory, plan, summary); manifest = _run_manifest(provenance, inventory, plan); _write_json(root / "report.json", report); _write_json(root / "run_manifest.json", manifest)
        disk_rows = read_strict_csv(root / "per_hop.csv", HOP_FIELDS); validate_hop_rows(disk_rows, [clip.clip_id for clip in clips], _expected_hops(clips)); rebuilt = summarize_model_rows(disk_rows)
        if (root / "per_clip.csv").read_bytes() != csv_bytes(rebuilt["clip_rows"], CLIP_FIELDS) or (root / "per_subject.csv").read_bytes() != csv_bytes(rebuilt["subject_rows"], SUBJECT_FIELDS): _fail("disk CSV aggregates differ from rebuilt rows")
        validate_exact_json(read_json_object(root / "report.json"), schema=REPORT_SCHEMA, expected=_report(provenance, inventory, plan, rebuilt)); validate_exact_json(read_json_object(root / "run_manifest.json"), schema=RUN_MANIFEST_SCHEMA, expected=_run_manifest(provenance, inventory, plan))
        outputs = ("per_hop.csv", "per_clip.csv", "per_subject.csv", "report.json", "run_manifest.json")
        output_map = validate_precomplete_directory(root, outputs, kind="run", provenance=provenance)
        complete = marker_payload("COMPLETE", "run", provenance, outputs=output_map)
        write_marker(root, complete)
        complete_written = True
    except BaseException:
        if not complete_written:
            try: write_marker(root, marker_payload("FAILED", "run", provenance))
            except BaseException: pass
        raise


def main() -> int:
    args = _args(); plan = load_frozen_model_plan(args.plan); _, clips = _bindings(args); checkpoints = _runtime_checkpoints(args, plan)
    _validate_shard_args(args)
    if args.merge: _merge(args, plan, clips, checkpoints)
    else: _publish_shard(args, plan, clips, checkpoints)
    return 0


if __name__ == "__main__": raise SystemExit(main())
