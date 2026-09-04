#!/usr/bin/env python3
"""Publish one authenticated, deterministic MCD repeated-beam clip/seed task.

``--smoke`` restricts execution to the first seed-0 task.  The same verifier
also serves the explicit full-cohort task mapping; submission remains external
to this worker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.evaluation.long_term_regret import csv_bytes, read_frozen_selection, partition_clip_seed_tasks
from adaptive_roi_rppg.evaluation.model_plan import load_frozen_model_plan
from adaptive_roi_rppg.evaluation.repeated_correction_beam import SCHEMA, SENSITIVITY_WIDTHS, eligible_anchors, factual_snapshots, repeated_correction_beam
from adaptive_roi_rppg.evaluation.long_term_regret import replay_one_time_interventions, score_branches, validate_replay_parity
from adaptive_roi_rppg.evaluation.source_subsets import FULL_TASK_COUNT, TASK_COUNT, read_task_subset
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements
from verify_mcd_long_term_regret import _load_config, _runtime_checkpoints

HOP_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "width", "path_rank", "offset", "hop_idx", "source", "origin_sequence", "executed_sequence", "observation_sha256", "pre_control_state_sha256", "pre_predict_recurrent_sha256", "post_predict_recurrent_sha256", "ppo_action", "legal_actions", "ppo_legal", "rank1_action", "proposed_action", "executed_action", "legal", "override_reason", "immediate_abs_error_bpm", "cumulative_abs_error_bpm", "endpoint_abs_error_bpm")
STAT_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "width", "offset", "generated", "deduped", "retained", "pruned", "factual_cumulative_abs_error_bpm", "factual_endpoint_abs_error_bpm", "best_cumulative_abs_error_bpm", "best_endpoint_abs_error_bpm")
ONE_HOP_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "branch_action", "offset", "hop_idx", "proposed_action", "executed_action", "post_belief_hr_bpm")
ONE_SCORE_FIELDS = ("method_id", "seed", "clip_id", "subject_id", "anchor_hop_idx", "branch_action", "horizon", "mean_abs_error_bpm")


def _fail(message: str) -> None: raise ContractValidationError(message)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="restrict to the first seed-0 task")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--selection-dir")
    parser.add_argument("--source-subsets-dir", required=True)
    parser.add_argument("--beam-config", default=str(Path(__file__).resolve().parents[1] / "configs/evaluation/mcd_repeated_correction_beam_v1.json"))
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--task-index", type=int, default=0); parser.add_argument("--task-count", type=int, default=162)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", "local-smoke")); parser.add_argument("--node", default=socket.gethostname())
    return parser.parse_args()


def _config(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ContractValidationError("beam config cannot be read") from exc
    worst54 = {"schema", "dataset_id", "selection", "window_hops", "primary_width", "sensitivity_widths", "ranking", "retention", "scope"}
    full = {"schema", "dataset_id", "source_type", "clip_count", "subject_count", "seeds", "task_count", "task_mapping", "window_hops", "primary_width", "sensitivity_widths", "ranking", "retention", "scope"}
    if value.get("schema") == SCHEMA:
        valid = set(value) == worst54 and value.get("selection") == {"source":"frozen_worst54_mcd_replay_advantage_ppo", "clip_count":54, "seeds":[0,1,2]}
    elif value.get("schema") == "mcd-repeated-correction-beam-full-v1":
        valid = (set(value) == full and value.get("source_type") == "full_frozen_eval" and value.get("clip_count") == 533 and
                 value.get("subject_count") == 89 and value.get("seeds") == [0, 1, 2] and value.get("task_count") == FULL_TASK_COUNT and
                 value.get("task_mapping") == "task_index = 3 * source_plan_clip_position + seed")
    else:
        valid = False
    if not valid or value.get("dataset_id") != "mcd" or value.get("window_hops") != 15 or value.get("primary_width") != 8 or value.get("sensitivity_widths") != list(SENSITIVITY_WIDTHS):
        _fail("beam config contract is invalid")
    return value


def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink(): _fail("output must be a fresh path below a real parent")
    path.mkdir()


def _rows(result: Any, *, method_id: str, seed: int, clip_id: str, subject_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    for path in (result.factual, *result.retained):
        for hop in path:
            output.append({"method_id":method_id, "seed":seed, "clip_id":clip_id, "subject_id":subject_id,
                **{field: (json.dumps(list(value), separators=(",", ":")) if field in {"origin_sequence", "executed_sequence", "legal_actions"} else value) for field, value in ((field, getattr(hop, field)) for field in hop.__dataclass_fields__)}})
    stats = []
    for depth_info in result.depth_stats:
        depth = int(depth_info["offset"])
        stats.append({"method_id":method_id, "seed":seed, "clip_id":clip_id, "subject_id":subject_id,
            "anchor_hop_idx":result.anchor_hop_idx, "width":result.width, "offset":depth,
            "generated":depth_info["generated"], "deduped":depth_info["deduped"], "retained":depth_info["retained"], "pruned":depth_info["pruned"],
            "factual_cumulative_abs_error_bpm":depth_info["factual_cumulative_abs_error_bpm"], "factual_endpoint_abs_error_bpm":depth_info["factual_endpoint_abs_error_bpm"],
            "best_cumulative_abs_error_bpm":depth_info["best_cumulative_abs_error_bpm"], "best_endpoint_abs_error_bpm":depth_info["best_endpoint_abs_error_bpm"]})
    return output, stats


def _validate_rows(rows: list[dict[str, Any]], stats: list[dict[str, Any]], *, clip_id: str, subject_id: str, seed: int) -> None:
    keys = set()
    for row in rows:
        key = tuple(row[field] for field in ("method_id", "seed", "clip_id", "anchor_hop_idx", "width", "path_rank", "offset", "source"))
        if key in keys or row["method_id"] != f"advantage_ppo_seed{seed}" or row["seed"] != seed or row["clip_id"] != clip_id or row["subject_id"] != subject_id or row["source"] not in {"factual_ppo", "rank1_gt", "ppo_legal"}: _fail("beam row identity or source is invalid")
        keys.add(key)
        legal = tuple(json.loads(row["legal_actions"]))
        if not legal or any(not isinstance(action, int) or not 0 <= action < 12 for action in legal) or bool(row["ppo_legal"]) != (row["ppo_action"] in legal) or row["executed_action"] not in legal:
            _fail("beam action audit fields are inconsistent")
        if row["source"] != "factual_ppo" and (not row["legal"] or row["override_reason"] not in (None, "")):
            _fail("retained beam child is not a legal unoverridden action")
    for stat in stats:
        group = [row for row in rows if row["anchor_hop_idx"] == stat["anchor_hop_idx"] and row["width"] == stat["width"]]
        if int(stat["offset"]) not in range(15): _fail("beam pruning depth is invalid")
        factual = sorted((row for row in group if row["source"] == "factual_ppo"), key=lambda row: row["offset"])
        at_depth = [row for row in factual if int(row["offset"]) == int(stat["offset"])]
        if [row["offset"] for row in factual] != list(range(15)) or len(at_depth) != 1 or abs(float(at_depth[0]["cumulative_abs_error_bpm"]) - float(stat["factual_cumulative_abs_error_bpm"])) > 1e-12: _fail("beam factual running sum is not lower-row reproducible")
        if int(stat["generated"]) != int(stat["deduped"]) + int(stat["retained"]) + int(stat["pruned"]): _fail("beam depth accounting is inconsistent")
    expected_stats = {(int(row["anchor_hop_idx"]), int(row["width"])) for row in stats}
    if any({int(row["offset"]) for row in stats if (int(row["anchor_hop_idx"]), int(row["width"])) == key} != set(range(15)) or sum(1 for row in stats if (int(row["anchor_hop_idx"]), int(row["width"])) == key) != 15 for key in expected_stats):
        _fail("beam pruning statistics must cover offsets 0..14 exactly")


def _resolve_task(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[dict[str, str]], tuple[str, int], Mapping[str, Any]]:
    """Resolve one worker from its small subset before touching frozen selection."""
    task_entry, subset_rows = read_task_subset(args.source_subsets_dir, args.task_index, task_count=args.task_count)
    if config.get("schema", SCHEMA) == SCHEMA:
        if not args.selection_dir:
            _fail("worst-54 mode requires frozen selection")
        base_config = _load_config()
        selection = read_frozen_selection(args.selection_dir, config_sha256=hashlib.sha256(canonical_json_bytes(base_config)).hexdigest(),
                                          source_sha256=task_entry["_source_sha256"], source_manifest_sha256=task_entry["_plan_sha256"])
        pairs = partition_clip_seed_tasks(selection, task_index=args.task_index, task_count=args.task_count)
        if len(pairs) != 1:
            _fail("each task must own exactly one clip/seed pair")
        pair = pairs[0]
        chosen = next((row for row in selection if row["clip_id"] == pair[0]), None)
        if chosen is None:
            _fail("frozen selection missing assigned clip")
    else:
        pair = (str(task_entry["clip_id"]), int(task_entry["seed"]))
        if int(task_entry.get("clip_position", -1)) != args.task_index // 3 or pair[1] != args.task_index % 3:
            _fail("full task index mapping is invalid")
        chosen = {"clip_id": task_entry["clip_id"], "subject_id": task_entry["subject_id"]}
    if task_entry["clip_id"] != pair[0] or int(task_entry["seed"]) != pair[1]:
        _fail("task subset does not match clip/seed mapping")
    if args.smoke and (args.task_index != 0 or pair[1] != 0):
        _fail("smoke requires task 0, seed 0")
    return task_entry, subset_rows, pair, chosen


def main() -> int:
    args = _args()
    started = time.monotonic()
    config = _config(args.beam_config)
    expected_task_count = TASK_COUNT if config.get("schema", SCHEMA) == SCHEMA else FULL_TASK_COUNT
    if args.task_count != expected_task_count or not 0 <= args.task_index < expected_task_count: _fail("task count does not match beam config")
    task_entry, subset_rows, pair, chosen = _resolve_task(args, config)
    # Compatibility with the pre-full-mode resolver contract used by focused
    # runner tests; production resolution always returns the mapping directly.
    if not isinstance(chosen, Mapping):
        chosen = next((row for row in chosen if row.get("clip_id") == pair[0]), None)
        if chosen is None:
            _fail("assigned selection clip is missing")
    source_info = {"advantage_checkpoint_sha256": {int(seed): value for seed, value in json.loads((Path(args.source_subsets_dir) / "index.json").read_text(encoding="utf-8"))["checkpoint_sha256"].items()}}
    run_seed = pair[1]
    if task_entry["clip_id"] != pair[0] or int(task_entry["seed"]) != run_seed or task_entry["subject_id"] != chosen["subject_id"]: _fail("task subset is not the assigned frozen clip/seed")
    plan = load_frozen_model_plan(args.plan); checks = _runtime_checkpoints(plan, args.checkpoint_root, source_info["advantage_checkpoint_sha256"]); checkpoint = checks[run_seed]
    bundle = load_mcd_manifest_tree(args.manifest_tree); by_clip = {clip.clip_id: clip for clip in bundle.clip_manifests}; clip = by_clip.get(chosen["clip_id"])
    if clip is None or clip.subject_id != chosen["subject_id"]: _fail("frozen selection ownership does not match MCD manifest")
    root = Path(args.output_dir); _fresh(root)
    try:
        policy = load_frozen_recurrent_policy(dict(checkpoint)); frames = build_pos_measurements(read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "eval")); labels = read_mcd_eval_labels(bundle, args.gt_root, clip.clip_id, clip)
        snapshots = factual_snapshots(frames, labels, policy, clip_id=clip.clip_id); anchors = eligible_anchors(snapshots)
        if not anchors: _fail("smoke clip has no eligible anchors")
        validate_replay_parity(subset_rows, snapshots, clip_id=clip.clip_id, seed=run_seed)
        one_anchors, branches = replay_one_time_interventions(frames, policy, clip_id=clip.clip_id, snapshots=snapshots)
        if tuple(row["anchor_hop_idx"] for row in one_anchors if row["eligible"]) != anchors: _fail("one-time and beam anchors differ")
        one_hops = [{"method_id":policy.identity.method_id,"seed":run_seed,"clip_id":clip.clip_id,"subject_id":clip.subject_id,**{field:getattr(row,field) for field in ("anchor_hop_idx","branch_action","offset","hop_idx","proposed_action","executed_action","post_belief_hr_bpm")}} for row in branches]
        one_scores = [{"method_id":policy.identity.method_id,"seed":run_seed,"clip_id":clip.clip_id,"subject_id":clip.subject_id,**row} for row in score_branches(branches, labels)]
        hops: list[dict[str, Any]] = []; stats: list[dict[str, Any]] = []
        for anchor in anchors:
            for width in SENSITIVITY_WIDTHS:
                result = repeated_correction_beam(frames, labels, policy, clip_id=clip.clip_id, anchor_hop_idx=anchor, width=width, snapshots=snapshots)
                a, b = _rows(result, method_id=policy.identity.method_id, seed=run_seed, clip_id=clip.clip_id, subject_id=clip.subject_id); hops.extend(a); stats.extend(b)
        hops.sort(key=lambda row: (row["anchor_hop_idx"], row["width"], row["source"], row["path_rank"], row["offset"])); stats.sort(key=lambda row: (row["anchor_hop_idx"], row["width"])); _validate_rows(hops, stats, clip_id=clip.clip_id, subject_id=clip.subject_id, seed=run_seed)
        factual = [{"method_id":policy.identity.method_id,"seed":run_seed,"clip_id":clip.clip_id,"subject_id":clip.subject_id,"anchor_hop_idx":row.hop_idx,"branch_action":row.proposed_action,"offset":0,"hop_idx":row.hop_idx,"proposed_action":row.proposed_action,"executed_action":row.executed_action,"post_belief_hr_bpm":row.post_belief_hr_bpm} for row in snapshots]
        (root / "factual_hops.csv").write_bytes(csv_bytes(factual, ONE_HOP_FIELDS)); (root / "one_time_hops.csv").write_bytes(csv_bytes(one_hops, ONE_HOP_FIELDS)); (root / "one_time_scores.csv").write_bytes(csv_bytes(one_scores, ONE_SCORE_FIELDS)); (root / "beam_hops.csv").write_bytes(csv_bytes(hops, HOP_FIELDS)); (root / "beam_stats.csv").write_bytes(csv_bytes(stats, STAT_FIELDS))
        (root / "summary.json").write_bytes(canonical_json_bytes({"schema":SCHEMA, "beam_config_schema":config["schema"], "beam_config_sha256":hashlib.sha256(canonical_json_bytes(config)).hexdigest(), "scope":config["scope"], "task_index":args.task_index, "task_count":args.task_count, "clip_id":clip.clip_id, "subject_id":clip.subject_id, "seed":run_seed, "job_id":args.job_id, "node":args.node, "elapsed_seconds":round(time.monotonic() - started, 3), "eligible_anchor_count":len(anchors), "row_counts":{"factual_hops":len(factual), "one_time_hops":len(one_hops), "one_time_scores":len(one_scores), "beam_hops":len(hops), "beam_stats":len(stats)}, "widths":list(SENSITIVITY_WIDTHS), "parity_pass":True, "source_subset":{"sha256":task_entry["sha256"], "bytes":task_entry["bytes"]}}))
    except BaseException:
        raise
    return 0


if __name__ == "__main__": raise SystemExit(main())
