#!/usr/bin/env python3
"""Run the MCD-only conditional worst-54 long-term regret diagnostic.

This is a diagnostic runner, not a training or selection tool. It accepts an
already completed MCD replay source and derives the selected clips from rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.evaluation.long_term_regret import (
    ANCHOR_FIELDS, ANCHOR_REGRET_FIELDS, BRANCH_HOP_FIELDS, BRANCH_SCORE_FIELDS, CLIP_SELECTION_FIELDS,
    BranchHop, SCHEMA, build_anchor_regret, csv_bytes, read_frozen_selection, replay_one_time_interventions, score_branches,
    subject_block_bootstrap,
)
from adaptive_roi_rppg.evaluation.model_plan import load_frozen_model_plan
from adaptive_roi_rppg.evaluation.model_publication import artifact_map, validate_marker
from adaptive_roi_rppg.evaluation.source_subsets import audit_plan_body_sha256, validate_failure_audit_contract
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements


def _fail(message: str) -> None: raise ContractValidationError(message)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-per-hop", required=True, help="merged replay per-hop CSV")
    parser.add_argument("--source-manifest", required=True, help="replay source manifest JSON")
    parser.add_argument("--selection-dir", required=True, help="canonical selection frozen by freeze_mcd_long_term_regret_selection.py")
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), help="run exactly one frozen Advantage PPO checkpoint")
    parser.add_argument("--merge", action="store_true"); parser.add_argument("--shard-dir", action="append")
    # Frozen by the diagnostic config; command-line overrides would make the
    # reported interval a different estimand.
    return parser.parse_args()


def _load_config() -> Mapping[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs/evaluation/mcd_long_term_regret_v1.json"
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ContractValidationError("long-term regret config cannot be read") from exc
    expected = {"schema", "dataset_id", "source_metric", "family", "seeds", "expected_clip_count", "expected_subject_count", "selected_clip_count", "horizons", "minimum_remaining_hops", "bootstrap", "scope"}
    if set(value) != expected or value["schema"] != SCHEMA or (value["dataset_id"], value["source_metric"], value["family"], value["seeds"], value["expected_clip_count"], value["expected_subject_count"], value["selected_clip_count"], value["horizons"], value["minimum_remaining_hops"]) != ("mcd", "abs_error_bpm", "advantage_ppo", [0, 1, 2], 533, 89, 54, [1, 5, 15], 15): _fail("long-term regret config contract is invalid")
    if value["bootstrap"] != {"replicates": 10_000, "unit": "subject", "aggregation": "anchor_to_clip_mean_then_equal_clips_within_subject"}: _fail("long-term regret bootstrap contract is invalid")
    if "MMPD" not in value["scope"] or not isinstance(value["scope"], str): _fail("long-term regret scope statement is invalid")
    return value


def _source_manifest(per_hop: str | Path, path: str | Path) -> Mapping[str, Any]:
    source, manifest_path = Path(per_hop).resolve(), Path(path).resolve()
    if source.name not in {"per_hop.csv", "per_hop_counterfactual.csv"} or not source.is_file() or not manifest_path.is_file(): _fail("replay source files are missing or have unexpected names")
    if source.name == "per_hop_counterfactual.csv":
        if manifest_path != source.parent.parent / "audit_plan.json": _fail("counterfactual source plan must be run_root/audit_plan.json")
        value = read_json_object(manifest_path)
        complete_path = source.parent / "COMPLETE.json"
        value, complete, plan_hash, source_sha, source_bytes = validate_failure_audit_contract(source, manifest_path, complete_path)
        identities = value["checkpoint_identities"]
        advantage = {seed: item["checkpoint_sha256"] for seed, item in ((int(item["seed"]), item) for item in identities) if item["family"] == "advantage_ppo"}
        return {**value, "advantage_checkpoint_sha256": advantage, "audit_plan_sha256": plan_hash, "source_complete": complete, "source_sha256": source_sha, "source_bytes": source_bytes}
    root = source.parent; required = ("STARTED.json", "COMPLETE.json", "per_hop.csv", "per_clip.csv", "per_subject.csv", "report.json", "run_manifest.json")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in required): _fail("accepted replay directory is incomplete")
    value = read_json_object(manifest_path)
    if value.get("schema") != "gate8-model-run-manifest-v1" or value.get("status") != "complete": _fail("replay source manifest is not complete")
    cohort = value.get("cohort")
    if not isinstance(cohort, Mapping) or (cohort.get("clip_count"), cohort.get("subject_count")) != (533, 89): _fail("replay manifest cohort is not the accepted 533-clip/89-subject cohort")
    provenance = {key: value[key] for key in ("plan_id", "code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "job_id", "node", "command")}
    validate_marker(read_json_object(root / "STARTED.json"), expected_kind="run", expected_provenance=provenance)
    complete = read_json_object(root / "COMPLETE.json"); validate_marker(complete, expected_kind="run", expected_provenance=provenance)
    expected_outputs = ("per_hop.csv", "per_clip.csv", "per_subject.csv", "report.json", "run_manifest.json")
    if complete.get("outputs") != artifact_map(root, expected_outputs): _fail("replay completion artifact hashes/sizes do not match supplied source directory")
    identities = value.get("checkpoint_identities")
    if not isinstance(identities, list) or len(identities) != 9: _fail("replay manifest checkpoint identities are invalid")
    advantage = {item.get("seed"): item.get("sha256") for item in identities if item.get("family") == "advantage_ppo"}
    if set(advantage) != {0, 1, 2} or any(not isinstance(value, str) or len(value) != 64 for value in advantage.values()): _fail("replay Advantage PPO identities are invalid")
    return {**value, "advantage_checkpoint_sha256": advantage}


def _runtime_checkpoints(plan: Any, root_text: str, accepted_hashes: Mapping[int, str]) -> dict[int, Mapping[str, Any]]:
    root = Path(root_text)
    if root.is_symlink() or not root.is_dir(): _fail("checkpoint root must be a real directory")
    root = root.resolve(); result = {}
    for spec in plan.checkpoints:
        if spec["family"] != "advantage_ppo": continue
        candidate = (root / spec["locator"]).resolve()
        if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file(): _fail("checkpoint package path is invalid")
        if spec["sha256"] != accepted_hashes.get(spec["seed"]): _fail("runtime plan Advantage PPO hash differs from accepted replay identity")
        value = dict(spec); value["locator"] = str(candidate); result[spec["seed"]] = value
    if set(result) != {0, 1, 2}: _fail("replay plan lacks exactly the three Advantage PPO checkpoints")
    return result


def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink(): _fail("output directory must be a fresh path beneath a real existing parent")
    path.mkdir()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.write_bytes(csv_bytes(rows, fields))


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file(): _fail(f"missing regular shard artifact: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); rows = list(reader)
    if tuple(reader.fieldnames or ()) != fields or any(set(row) != set(fields) for row in rows): _fail(f"malformed shard artifact: {path.name}")
    return rows


def _validate_shard_rows(anchors: list[dict[str, Any]], branches: list[dict[str, Any]], scores: list[dict[str, Any]], *, expected_clip_ids: set[str], subject_by_clip: Mapping[str, str], seed: int) -> None:
    """Reject altered ownership, branch coverage, and factual anchor semantics."""
    anchor_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in anchors:
        key = (row["method_id"], row["seed"], row["clip_id"], int(row["anchor_hop_idx"]))
        if key in anchor_by_key or row["method_id"] != f"advantage_ppo_seed{row['seed']}" or row["seed"] != str(seed) or row["clip_id"] not in expected_clip_ids or row["subject_id"] != subject_by_clip.get(row["clip_id"]): _fail("duplicate or wrong-seed anchor row")
        anchor_by_key[key] = row
    branch_groups: dict[tuple[str, str, str, int, int], set[int]] = {}
    branch_keys: set[tuple[str, str, str, int, int, int]] = set()
    for row in branches:
        key = (row["method_id"], row["seed"], row["clip_id"], int(row["anchor_hop_idx"]), int(row["branch_action"]), int(row["offset"]))
        if key in branch_keys or not 0 <= key[4] < 12 or not 0 <= key[5] < 15: _fail("duplicate or invalid branch-hop key")
        branch_keys.add(key); anchor = anchor_by_key.get(key[:4])
        if anchor is None or anchor["eligible"] != "True": _fail("branch belongs to missing or ineligible anchor")
        if key[5] == 0 and (row["proposed_action"] != str(key[4]) or row["executed_action"] != str(key[4]) or row["legal"] != "True" or row["override_reason"]): _fail("forced anchor action does not satisfy free-choice control semantics")
        branch_groups.setdefault(key[:-1], set()).add(key[-1])
    for key, anchor in anchor_by_key.items():
        groups = [offsets for branch_key, offsets in branch_groups.items() if branch_key[:4] == key]
        if anchor["eligible"] == "True":
            if len(groups) != 12 or any(offsets != set(range(15)) for offsets in groups): _fail("eligible anchor does not have exact 12x15 branch coverage")
        elif groups: _fail("ineligible anchor has branch rows")
    expected_score = {(key[0], key[1], key[2], key[3], action, horizon) for key, anchor in anchor_by_key.items() if anchor["eligible"] == "True" for action in range(12) for horizon in (1, 5, 15)}
    actual_score = {(row["method_id"], row["seed"], row["clip_id"], int(row["anchor_hop_idx"]), int(row["branch_action"]), int(row["horizon"])) for row in scores}
    if actual_score != expected_score: _fail("branch score coverage does not match validated branches")


def _rebuild_scores_from_branches(args: argparse.Namespace, bundle: Any, branches: list[dict[str, Any]], *, seed: int, subject_by_clip: Mapping[str, str]) -> list[dict[str, Any]]:
    """Authenticate labels and regenerate every score from the lower branch trace."""
    by_clip = {clip.clip_id: clip for clip in bundle.clip_manifests}; grouped: dict[str, list[dict[str, Any]]] = {}
    for row in branches: grouped.setdefault(row["clip_id"], []).append(row)
    rebuilt = []
    for clip_id, values in sorted(grouped.items()):
        clip = by_clip.get(clip_id)
        if clip is None or clip.subject_id != subject_by_clip[clip_id]: _fail("branch clip ownership differs from authenticated MCD manifest")
        labels = read_mcd_eval_labels(bundle, args.gt_root, clip_id, clip)
        trace = [BranchHop(row["clip_id"], int(row["anchor_hop_idx"]), int(row["branch_action"]), int(row["offset"]), int(row["hop_idx"]), row["observation_sha256"], row["pre_control_state_sha256"], row["pre_predict_recurrent_sha256"], row["post_predict_recurrent_sha256"], int(row["proposed_action"]), int(row["executed_action"]), row["legal"] == "True", row["override_reason"] or None, float(row["post_belief_hr_bpm"])) for row in values]
        for row in score_branches(trace, labels): rebuilt.append({"method_id":f"advantage_ppo_seed{seed}", "seed":seed, "clip_id":clip_id, "subject_id":subject_by_clip[clip_id], **row})
    rebuilt.sort(key=lambda row: (row["seed"], row["clip_id"], row["anchor_hop_idx"], row["branch_action"], row["horizon"]))
    return rebuilt


def _select_shard(selection: tuple[dict[str, Any], ...], index: int, count: int) -> tuple[dict[str, Any], ...]:
    if count < 1 or not 0 <= index < count: _fail("invalid shard index/count")
    # Longest first work assignment is deterministic once source hop counts are
    # validated; clip-id ordering remains the canonical merged output order.
    return tuple(selection[index::count])


def _merge(args: argparse.Namespace, config: Mapping[str, Any], source_info: Mapping[str, Any], source: Path, selection: tuple[dict[str, Any], ...]) -> int:
    if args.seed is not None or not args.shard_dir or len(args.shard_dir) != args.shard_count * 3: _fail("merge requires exactly one shard for every Advantage seed and shard index")
    bundle = load_mcd_manifest_tree(args.manifest_tree); subject_by_clip = {row["clip_id"]: row["subject_id"] for row in selection}
    if len(subject_by_clip) != len(selection) or any((clip := next((item for item in bundle.clip_manifests if item.clip_id == clip_id), None)) is None or clip.subject_id != subject for clip_id, subject in subject_by_clip.items()): _fail("frozen selection does not match authenticated MCD manifest")
    roots = [Path(value) for value in args.shard_dir]; seen: set[tuple[int, int]] = set(); artifacts = {"anchor_states.csv": ANCHOR_FIELDS, "branch_hops.csv": BRANCH_HOP_FIELDS, "branch_scores.csv": BRANCH_SCORE_FIELDS, "anchor_regret.csv": ANCHOR_REGRET_FIELDS}
    gathered = {name: [] for name in artifacts}
    expected_selection = csv_bytes(selection, CLIP_SELECTION_FIELDS)
    for shard in roots:
        manifest = read_json_object(shard / "run_manifest.json"); shard_manifest = read_json_object(shard / "shard_manifest.json")
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "shard_complete" or manifest.get("source", {}).get("per_hop_sha256") != sha256_file(source): _fail("shard manifest differs from the declared replay source")
        shard_info = manifest.get("shard", {}); index = shard_info.get("index"); seeds = shard_info.get("seeds")
        if not isinstance(index, int) or not isinstance(seeds, list) or len(seeds) != 1 or seeds[0] not in (0, 1, 2) or (seeds[0], index) in seen: _fail("shard seed/index is invalid or duplicated")
        seen.add((seeds[0], index))
        expected_files = ("clip_selection.csv", *artifacts.keys(), "summary.json", "run_manifest.json")
        if shard_manifest != {"schema": SCHEMA, "seed": seeds[0], "shard_index": index, "shard_count": args.shard_count, "clip_ids": [row["clip_id"] for row in _select_shard(selection, index, args.shard_count)], "source_per_hop_sha256": sha256_file(source), "artifacts": artifact_map(shard, expected_files)}: _fail("shard manifest/artifact map or ownership differs")
        if (shard / "clip_selection.csv").read_bytes() != expected_selection: _fail("shard clip selection differs from canonical frozen selection")
        shard_rows = {name: _read_csv(shard / name, fields) for name, fields in artifacts.items()}
        expected_ids = {row["clip_id"] for row in _select_shard(selection, index, args.shard_count)}
        _validate_shard_rows(shard_rows["anchor_states.csv"], shard_rows["branch_hops.csv"], shard_rows["branch_scores.csv"], expected_clip_ids=expected_ids, subject_by_clip=subject_by_clip, seed=seeds[0])
        rebuilt_scores = _rebuild_scores_from_branches(args, bundle, shard_rows["branch_hops.csv"], seed=seeds[0], subject_by_clip=subject_by_clip)
        if csv_bytes(rebuilt_scores, BRANCH_SCORE_FIELDS) != csv_bytes(shard_rows["branch_scores.csv"], BRANCH_SCORE_FIELDS): _fail("branch score CSV differs from authenticated label-and-trace reconstruction")
        factual = {(row["method_id"], row["seed"], row["clip_id"], int(row["anchor_hop_idx"])): int(row["factual_proposed_action"]) for row in shard_rows["anchor_states.csv"] if row["eligible"] == "True"}
        derived_scores = [{**row, "anchor_hop_idx": int(row["anchor_hop_idx"]), "branch_action": int(row["branch_action"]), "horizon": int(row["horizon"]), "mean_abs_error_bpm": float(row["mean_abs_error_bpm"]), "is_factual": int(row["branch_action"]) == factual[(row["method_id"], row["seed"], row["clip_id"], int(row["anchor_hop_idx"]))]} for row in shard_rows["branch_scores.csv"]]
        # Rebuild the lower-level regret artifact rather than accepting it.
        by_clip: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in derived_scores: by_clip.setdefault((row["method_id"], row["seed"], row["clip_id"]), []).append(row)
        regenerated = []
        for (_method, _seed, _clip), values in by_clip.items():
            first = values[0]
            regenerated.extend({"method_id": first["method_id"], "seed": first["seed"], "clip_id": first["clip_id"], "subject_id": first["subject_id"], **regret} for regret in build_anchor_regret(values))
        if csv_bytes(regenerated, ANCHOR_REGRET_FIELDS) != csv_bytes(shard_rows["anchor_regret.csv"], ANCHOR_REGRET_FIELDS): _fail("anchor regret CSV differs from score-derived reconstruction")
        for name, rows in shard_rows.items(): gathered[name].extend(regenerated if name == "anchor_regret.csv" else rows)
    if seen != {(seed, index) for seed in (0, 1, 2) for index in range(args.shard_count)}: _fail("merge shard set is incomplete")
    root = Path(args.output_dir); _fresh(root)
    for name, fields in artifacts.items():
        numeric = {"seed", "anchor_hop_idx", "branch_action", "offset", "hop_idx", "horizon"}
        gathered[name].sort(key=lambda row: tuple(int(row[field]) if field in numeric else row[field] for field in fields if field in {"seed", "clip_id", "anchor_hop_idx", "branch_action", "offset", "horizon"}))
        _write_csv(root / name, gathered[name], fields)
    _write_csv(root / "clip_selection.csv", list(selection), CLIP_SELECTION_FIELDS)
    regrets = gathered["anchor_regret.csv"]
    summaries = {str(seed): {str(horizon): subject_block_bootstrap([row for row in regrets if int(row["seed"]) == seed and int(row["horizon"]) == horizon], replicates=10_000, seed=8101 + 100 * seed + horizon) for horizon in (1, 5, 15)} for seed in (0, 1, 2)}
    summary = {"schema": SCHEMA, "scope": "Conditional worst-54 MCD-only diagnostic. MMPD is not accessed; this is not a generalization, training, reward, or tuning result.", "seed_results": summaries, "seed_mean_and_range": {str(horizon): {"mean_point_estimate_bpm": sum(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2)) / 3.0, "range_point_estimate_bpm": [min(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2)), max(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2))]} for horizon in (1, 5, 15)}}
    manifest = {"schema": SCHEMA, "status": "complete", "config": config, "source": {"per_hop": str(source.resolve()), "per_hop_sha256": sha256_file(source), "manifest": str(Path(args.source_manifest).resolve()), "manifest_sha256": sha256_file(args.source_manifest), "plan_id": source_info.get("plan_id")}, "selection": {"clip_count": 54, "source_metric": "post-update belief abs_error_bpm", "selection_rule": "mean over Advantage PPO seeds 0,1,2; descending MAE then clip_id tie-break"}, "shard": {"merged_shard_count": args.shard_count * 3}}
    (root / "summary.json").write_bytes(canonical_json_bytes(summary)); (root / "run_manifest.json").write_bytes(canonical_json_bytes(manifest))
    return 0


def main() -> int:
    args = _args(); config = _load_config(); source_info = _source_manifest(args.source_per_hop, args.source_manifest)
    source = Path(args.source_per_hop)
    if source.is_symlink() or not source.is_file(): _fail("replay per-hop source must be a regular file")
    selection_manifest_sha = source_info.get("audit_plan_sha256", sha256_file(args.source_manifest))
    selection = read_frozen_selection(args.selection_dir, config_sha256=hashlib.sha256(canonical_json_bytes(config)).hexdigest(), source_sha256=sha256_file(source), source_manifest_sha256=selection_manifest_sha)
    if args.merge: return _merge(args, config, source_info, source, selection)
    selected = _select_shard(selection, args.shard_index, args.shard_count)
    plan = load_frozen_model_plan(args.plan); checkpoints = _runtime_checkpoints(plan, args.checkpoint_root, source_info["advantage_checkpoint_sha256"]); bundle = load_mcd_manifest_tree(args.manifest_tree)
    by_clip = {clip.clip_id: clip for clip in bundle.clip_manifests}
    if any(row["clip_id"] not in by_clip or by_clip[row["clip_id"]].subject_id != row["subject_id"] for row in selection): _fail("selected clip identities do not match the MCD manifest")
    root = Path(args.output_dir); _fresh(root)
    clip_rows = sorted(selected, key=lambda row: row["selection_rank"]); anchors_out: list[dict[str, Any]] = []; branch_out: list[dict[str, Any]] = []; score_out: list[dict[str, Any]] = []; regret_out: list[dict[str, Any]] = []
    run_seeds = (args.seed,) if args.seed is not None else (0, 1, 2)
    for seed in run_seeds:
        policy = load_frozen_recurrent_policy(dict(checkpoints[seed]))  # one checkpoint per worker/process in the launcher
        for selected_clip in clip_rows:
            clip_id, clip = selected_clip["clip_id"], by_clip[selected_clip["clip_id"]]
            frames = build_pos_measurements(read_mcd_canonical_frames(bundle, args.state_root, clip_id, "eval"))
            labels = read_mcd_eval_labels(bundle, args.gt_root, clip_id, clip)
            if len(labels) != len(frames): _fail("MCD label/frame coverage differs")
            anchors, branches = replay_one_time_interventions(frames, policy, clip_id=clip_id)
            # GT is deliberately joined only after this entire clip's unscored branch traces exist.
            scores = [dict(row) for row in score_branches(branches, labels)]
            factual_by_anchor = {row["anchor_hop_idx"]: row["factual_proposed_action"] for row in anchors if row["eligible"]}
            for row in scores: row["is_factual"] = row["branch_action"] == factual_by_anchor[row["anchor_hop_idx"]]
            regrets = build_anchor_regret(scores)
            for row in anchors: anchors_out.append({"method_id": policy.identity.method_id, "seed": seed, "clip_id": clip_id, "subject_id": clip.subject_id, **row})
            for row in branches: branch_out.append({"method_id": policy.identity.method_id, "seed": seed, "clip_id": clip_id, "subject_id": clip.subject_id, **{field: getattr(row, field) for field in row.__dataclass_fields__}})
            for row in scores: score_out.append({"method_id": policy.identity.method_id, "seed": seed, "clip_id": clip_id, "subject_id": clip.subject_id, **{key: value for key, value in row.items() if key != "is_factual"}})
            for row in regrets: regret_out.append({"method_id": policy.identity.method_id, "seed": seed, "clip_id": clip_id, "subject_id": clip.subject_id, **row})
    clip_selection = list(selection)
    anchors_out.sort(key=lambda r: (r["seed"], r["clip_id"], r["anchor_hop_idx"])); branch_out.sort(key=lambda r: (r["seed"], r["clip_id"], r["anchor_hop_idx"], r["branch_action"], r["offset"])); score_out.sort(key=lambda r: (r["seed"], r["clip_id"], r["anchor_hop_idx"], r["branch_action"], r["horizon"])); regret_out.sort(key=lambda r: (r["seed"], r["clip_id"], r["anchor_hop_idx"], r["horizon"]))
    _write_csv(root / "clip_selection.csv", clip_selection, CLIP_SELECTION_FIELDS); _write_csv(root / "anchor_states.csv", anchors_out, ANCHOR_FIELDS); _write_csv(root / "branch_hops.csv", branch_out, BRANCH_HOP_FIELDS); _write_csv(root / "branch_scores.csv", score_out, BRANCH_SCORE_FIELDS); _write_csv(root / "anchor_regret.csv", regret_out, ANCHOR_REGRET_FIELDS)
    summaries = {str(seed): {str(horizon): subject_block_bootstrap([row for row in regret_out if row["seed"] == seed and row["horizon"] == horizon], replicates=10_000, seed=8101 + 100 * seed + horizon) for horizon in (1, 5, 15)} for seed in run_seeds}
    complete = args.seed is None and args.shard_count == 1
    manifest = {"schema": SCHEMA, "status": "complete" if complete else "shard_complete", "config": config, "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(), "source": {"per_hop": str(source.resolve()), "per_hop_sha256": sha256_file(source), "manifest": str(Path(args.source_manifest).resolve()), "manifest_sha256": sha256_file(args.source_manifest), "plan_id": source_info.get("plan_id"), "advantage_checkpoint_sha256": source_info["advantage_checkpoint_sha256"]}, "selection": {"clip_count": 54, "source_metric": "post-update belief abs_error_bpm", "selection_rule": "mean over Advantage PPO seeds 0,1,2; descending MAE then clip_id tie-break"}, "shard": {"index": args.shard_index, "count": args.shard_count, "clip_count": len(selected), "seeds": list(run_seeds)}}
    seed_range = {str(horizon): {"mean_point_estimate_bpm": sum(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2)) / 3.0, "range_point_estimate_bpm": [min(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2)), max(summaries[str(seed)][str(horizon)]["point_estimate_bpm"] for seed in (0, 1, 2))]} for horizon in (1, 5, 15)} if complete else None
    summary = {"schema": SCHEMA, "scope": "Conditional worst-54 MCD-only diagnostic. MMPD is not accessed; this is not a generalization, training, reward, or tuning result.", "seed_results": summaries, "seed_mean_and_range": seed_range, "locked_hops_excluded": sum(not row["eligible"] and row["ineligible_reason"] == "locked_or_nonfree_action_set" for row in anchors_out), "insufficient_future_hops_excluded": sum(not row["eligible"] and row["ineligible_reason"] == "insufficient_future_hops" for row in anchors_out)}
    (root / "summary.json").write_bytes(canonical_json_bytes(summary)); (root / "run_manifest.json").write_bytes(canonical_json_bytes(manifest))
    if not complete:
        shard_manifest = {"schema": SCHEMA, "seed": run_seeds[0], "shard_index": args.shard_index, "shard_count": args.shard_count, "clip_ids": [row["clip_id"] for row in selected], "source_per_hop_sha256": sha256_file(source), "artifacts": artifact_map(root, ("clip_selection.csv", "anchor_states.csv", "branch_hops.csv", "branch_scores.csv", "anchor_regret.csv", "summary.json", "run_manifest.json"))}
        (root / "shard_manifest.json").write_bytes(canonical_json_bytes(shard_manifest))
    return 0


if __name__ == "__main__": raise SystemExit(main())
