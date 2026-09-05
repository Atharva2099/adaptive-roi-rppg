#!/usr/bin/env python3
"""Analyze saved-row repeated-beam behavior for the full MCD cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.evaluation.source_subsets import FULL_TASK_COUNT, read_task_subset

WINDOW = 15
PRIMARY_WIDTH = 8
SENSITIVITY_WIDTHS = (1, 2, 4, 8, 16, 32)
SCHEMA = "mcd-repeated-correction-beam-v1"
SEEDS = (0, 1, 2)
FILES = ("factual_hops.csv", "one_time_hops.csv", "one_time_scores.csv", "beam_hops.csv", "beam_stats.csv")
COMPARISON_TOLERANCE_BPM = 1e-6

def _subject_block_bootstrap(anchor_rows: Sequence[Mapping[str, Any]], *, replicates: int = 10_000, seed: int = 8101) -> dict[str, Any]:
    """Clip mean then equal-clips-within-subject, with subject-block resampling."""
    if replicates < 1: _fail("bootstrap replicates must be positive")
    clips: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in anchor_rows:
        value = float(row["factual_regret_bpm"])
        if not math.isfinite(value) or not row.get("subject_id") or not row.get("clip_id"):
            _fail("anchor regret row is invalid")
        clips[(str(row["subject_id"]), str(row["clip_id"]))].append(value)
    by_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _clip), values in clips.items(): by_subject[subject].append(sum(values) / len(values))
    if not by_subject: _fail("no eligible anchors for bootstrap")
    import numpy as np
    values = np.asarray([sum(items) / len(items) for _, items in sorted(by_subject.items())], dtype=np.float64)
    point = float(np.mean(values)); rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), size=(replicates, len(values)))], axis=1); draws.sort()
    return {"point_estimate_bpm": point, "ci_95_percentile_bpm": [float(draws[int(.025 * (replicates - 1))]), float(draws[int(.975 * (replicates - 1))])], "replicates": replicates}
def _fail(message: str) -> None:
    raise ValueError(message)
def _int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}") from exc
def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}") from exc
    if not math.isfinite(value): _fail(f"non-finite {key}")
    return value
def _bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    _fail(f"invalid {key}")
def _seq(row: Mapping[str, Any], key: str) -> tuple[int, ...]:
    try:
        value = json.loads(str(row[key]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {key}") from exc
    if not isinstance(value, list) or any(type(item) is not int or item not in range(12) for item in value): _fail(f"invalid {key}")
    return tuple(value)
def _csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file(): _fail(f"missing regular file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames or not rows or any(None in row for row in rows): _fail(f"malformed or empty {path.name}")
    return rows
def _close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6): _fail(message)
def _source_identity(rows: Sequence[Mapping[str, Any]], seed: int, checkpoint: str) -> bool:
    return all((row.get("method_id"), row.get("family"), row.get("checkpoint_sha256")) == (f"advantage_ppo_seed{seed}", "advantage_ppo", checkpoint) for row in rows)
def _factual(rows: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    truth = {}
    for row in source:
        hop = _int(row, "hop_idx")
        belief, gt = _float(row, "selected_post_belief_hr_bpm"), _float(row, "gt_hr_bpm")
        if hop in truth: _fail("duplicate source hop")
        truth[hop] = {"gt": gt, "belief": belief, "error": abs(belief - gt),
                      "proposed": _int(row, "proposed_action"), "executed": _int(row, "executed_action")}
        _close(_float(row, "selected_abs_error_bpm"), truth[hop]["error"], "source GT error mismatch")
    if sorted(truth) != list(range(len(truth))) or len(rows) != len(truth): _fail("factual/source coverage mismatch")
    seen = set()
    for row in rows:
        hop, item = _int(row, "hop_idx"), truth.get(_int(row, "hop_idx"))
        if item is None or hop in seen or (_int(row, "anchor_hop_idx"), _int(row, "offset")) != (hop, 0): _fail("factual hop identity mismatch")
        if (_int(row, "branch_action"), _int(row, "proposed_action"), _int(row, "executed_action")) != (item["proposed"], item["proposed"], item["executed"]):
            _fail("factual action mismatch")
        _close(_float(row, "post_belief_hr_bpm"), item["belief"], "factual belief mismatch")
        seen.add(hop)
    return truth
def _branches(hops: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], factual: Mapping[int, Mapping[str, Any]]) -> dict[tuple[int, int], list[float]]:
    groups = defaultdict(list)
    for row in hops:
        groups[(_int(row, "anchor_hop_idx"), _int(row, "branch_action"))].append(row)
    output = {}
    for (anchor, action), trace in groups.items():
        trace.sort(key=lambda row: _int(row, "offset"))
        if action not in range(12) or [_int(row, "offset") for row in trace] != list(range(WINDOW)): _fail("one-time branch inventory mismatch")
        errors = []
        for offset, row in enumerate(trace):
            hop = _int(row, "hop_idx")
            if hop != anchor + offset or hop not in factual: _fail("one-time branch hop mismatch")
            if offset == 0 and (_int(row, "proposed_action"), _int(row, "executed_action")) != (action, action): _fail("one-time branch did not execute its named free action")
            errors.append(abs(_float(row, "post_belief_hr_bpm") - factual[hop]["gt"]))
        output[(anchor, action)] = errors
    anchors = {anchor for anchor, _action in output}
    if any({action for candidate, action in output if candidate == anchor} != set(range(12)) for anchor in anchors): _fail("one-time action inventory mismatch")
    score_keys = {(_int(row, "anchor_hop_idx"), _int(row, "branch_action"), _int(row, "horizon")) for row in scores}
    if len(score_keys) != len(scores) or score_keys != {(anchor, action, horizon) for anchor, action in output for horizon in (1, 5, 15)}: _fail("one-time score inventory mismatch")
    for row in scores:
        _float(row, "mean_abs_error_bpm")
    for anchor in anchors:
        action = int(factual[anchor]["executed"])
        _close(output[(anchor, action)][0], float(factual[anchor]["error"]), "factual branch mismatch")
    return output
def _path_key(path: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return (sum(_float(row, "immediate_abs_error_bpm") for row in path),
            _float(path[-1], "immediate_abs_error_bpm"), _seq(path[-1], "executed_sequence"),
            _seq(path[-1], "origin_sequence"), _int(path[-1], "path_rank"))
def _is_correction(row: Mapping[str, Any]) -> bool:
    return row.get("source") == "rank1_gt" and _bool(row, "ppo_legal") and _int(row, "proposed_action") != _int(row, "ppo_action")
def _beam(hops: Sequence[Mapping[str, Any]], stats: Sequence[Mapping[str, Any]], factual: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    paths = defaultdict(list)
    for row in hops:
        source = row.get("source")
        if source not in {"factual_ppo", "rank1_gt", "ppo_legal"}: _fail("invalid beam source")
        rank = -1 if source == "factual_ppo" else _int(row, "path_rank")
        paths[(_int(row, "anchor_hop_idx"), _int(row, "width"), rank)].append(row)
    anchors = {key[0] for key in paths}
    expected = {(anchor, width) for anchor in anchors for width in SENSITIVITY_WIDTHS}
    if {(anchor, width) for anchor, width, rank in paths if rank == -1} != expected or {(anchor, width) for anchor, width, rank in paths if rank >= 0} != expected: _fail("beam anchor/width inventory mismatch")
    by_width = {}
    for anchor, width in sorted(expected):
        factual_path = paths[(anchor, width, -1)]
        retained = sorted((rank, path) for (a, w, rank), path in paths.items() if (a, w) == (anchor, width) and rank >= 0)
        if not retained or [rank for rank, _path in retained] != list(range(len(retained))) or len(retained) > width: _fail("beam path-rank inventory mismatch")
        for rank, path in [(-1, factual_path), *retained]:
            path.sort(key=lambda row: _int(row, "offset"))
            if [_int(row, "offset") for row in path] != list(range(WINDOW)) or any(_int(row, "hop_idx") != anchor + offset for offset, row in enumerate(path)): _fail("beam path coverage mismatch")
            running = 0.0
            for offset, row in enumerate(path):
                running += _float(row, "immediate_abs_error_bpm")
                _close(_float(row, "cumulative_abs_error_bpm"), running, "beam cumulative mismatch")
                _close(_float(row, "endpoint_abs_error_bpm"), _float(row, "immediate_abs_error_bpm"), "beam endpoint mismatch")
                legal, ppo = _seq(row, "legal_actions"), _int(row, "ppo_action")
                if not legal or len(set(legal)) != len(legal) or _bool(row, "ppo_legal") != (ppo in legal): _fail("beam legal-action audit mismatch")
                if rank == -1:
                    item = factual[anchor + offset]
                    if (_int(row, "proposed_action"), _int(row, "executed_action")) != (item["proposed"], item["executed"]):
                        _fail("beam factual action mismatch")
                    _close(_float(row, "immediate_abs_error_bpm"), item["error"], "beam factual error mismatch")
                else:
                    proposed, executed, rank1 = (_int(row, key) for key in ("proposed_action", "executed_action", "rank1_action"))
                    if not _bool(row, "legal") or executed != proposed or proposed not in legal or row.get("override_reason") not in (None, ""):
                        _fail("retained action audit mismatch")
                    if (row["source"] == "rank1_gt" and proposed != rank1) or (row["source"] == "ppo_legal" and (proposed != ppo or proposed == rank1)):
                        _fail("retained source/action mismatch")
                    if _seq(row, "origin_sequence") != tuple(_int(item, "proposed_action") for item in path[:offset + 1]) or _seq(row, "executed_sequence") != tuple(_int(item, "executed_action") for item in path[:offset + 1]):
                        _fail("retained sequence mismatch")
        rank_keys = [_path_key(path)[:-1] for _rank, path in retained]
        if rank_keys != sorted(rank_keys):
            _fail("retained paths are not ordered by declared rank")
        by_width[(anchor, width)] = (factual_path, [path for _rank, path in retained])
    stat_groups = defaultdict(list)
    for row in stats:
        stat_groups[(_int(row, "anchor_hop_idx"), _int(row, "width"))].append(row)
    if set(stat_groups) != expected:
        _fail("beam stats inventory mismatch")
    output = {}
    for key, (factual_path, retained) in by_width.items():
        rows = sorted(stat_groups[key], key=lambda row: _int(row, "offset"))
        if [_int(row, "offset") for row in rows] != list(range(WINDOW)): _fail("beam stats coverage mismatch")
        for row in rows:
            generated, deduped, kept, pruned = (_int(row, name) for name in ("generated", "deduped", "retained", "pruned"))
            if min(generated, deduped, kept, pruned) < 0 or generated != deduped + kept + pruned: _fail("beam pruning accounting mismatch")
            _float(row, "best_cumulative_abs_error_bpm")
            _float(row, "best_endpoint_abs_error_bpm")
        best, final = retained[0], rows[-1]
        _close(_float(final, "factual_cumulative_abs_error_bpm"), sum(_float(row, "immediate_abs_error_bpm") for row in factual_path), "final factual stats mismatch")
        _close(_float(final, "factual_endpoint_abs_error_bpm"), _float(factual_path[-1], "immediate_abs_error_bpm"), "final factual endpoint mismatch")
        _close(_float(final, "best_cumulative_abs_error_bpm"), _path_key(best)[0], "final beam best mismatch")
        _close(_float(final, "best_endpoint_abs_error_bpm"), _float(best[-1], "immediate_abs_error_bpm"), "final beam endpoint mismatch")
        if _int(final, "retained") != len(retained): _fail("final survivor count mismatch")
        if key[1] == PRIMARY_WIDTH:
            repeated = [path for path in retained if sum(_is_correction(row) for row in path) >= 2]
            output[key[0]] = {"factual": factual_path, "repeated": min(repeated, key=_path_key) if repeated else None,
                              "repeated_path_count": len(repeated), "pruned": sum(_int(row, "pruned") for row in rows)}
    return output
def recovery_classification(gains: Sequence[float]) -> str:
    positive = [gain > COMPARISON_TOLERANCE_BPM for gain in gains]
    if positive and all(positive):
        return "better_throughout"
    if not any(positive):
        return "never_better"
    cut = next((index for index, value in enumerate(positive) if not value), len(positive))
    if cut > 0 and not any(positive[cut:]) and gains[-1] < -COMPARISON_TOLERANCE_BPM:
        return "briefly_better_then_ends_worse"
    cut = next(index for index, value in enumerate(positive) if value)
    if cut > 0 and all(positive[cut:]):
        return "only_ends_better_late"
    return "mixed_intermittent"
def _anchor(anchor: int, factual: Mapping[int, Mapping[str, Any]], branches: Mapping[tuple[int, int], Sequence[float]], beam: Mapping[str, Any]) -> dict[str, Any]:
    factual_errors = [float(factual[anchor + offset]["error"]) for offset in range(WINDOW)]
    row = {"anchor_hop_idx": anchor, "analysis_eligible": 0, "exclusion_reason": "",
           "factual_15hop_mae_bpm": mean(factual_errors), "factual_endpoint_abs_error_bpm": factual_errors[-1],
           "width8_pruned_count": beam["pruned"], "repeated_path_count": beam["repeated_path_count"]}
    singles = []
    for offset, beam_row in enumerate(beam["factual"]):
        legal = set(_seq(beam_row, "legal_actions"))
        if len(legal) == 1:
            continue
        if legal != set(range(12)):
            row["exclusion_reason"] = "unsupported_legal_action_inventory"
            return row
        if any((anchor + offset, action) not in branches for action in range(12)):
            row["exclusion_reason"] = "incomplete_single_coverage"
            return row
        for action in range(12):
            if action == factual[anchor + offset]["executed"]:
                continue
            errors = factual_errors[:offset] + list(branches[(anchor + offset, action)][:WINDOW - offset])
            singles.append((sum(errors), errors[-1], offset, action))
    if not singles:
        row["exclusion_reason"] = "no_genuine_single_correction"
        return row
    best = min(singles)
    same = min(item for item in singles if item[2] == 0)
    immediate = min((branches[(anchor, action)][0], action) for action in range(12))
    row.update({"analysis_eligible": 1, "immediate_rank1_action": immediate[1], "immediate_rank1_error_bpm": immediate[0],
                "ppo_immediate_regret_bpm": factual_errors[0] - immediate[0], "same_anchor_best_nonppo_action": same[3],
                "same_anchor_best_nonppo_immediate_error_bpm": branches[(anchor, same[3])][0],
                "same_anchor_best_nonppo_15hop_mae_bpm": same[0] / WINDOW, "same_anchor_best_nonppo_endpoint_abs_error_bpm": same[1],
                "same_step_myopic_reversal": int(branches[(anchor, same[3])][0] - factual_errors[0] > COMPARISON_TOLERANCE_BPM and (sum(factual_errors) - same[0]) / WINDOW > COMPARISON_TOLERANCE_BPM),
                "best_single_offset": best[2], "best_single_action": best[3], "best_single_15hop_mae_bpm": best[0] / WINDOW,
                "best_single_endpoint_abs_error_bpm": best[1], "best_single_gain_vs_factual_mae_bpm": (sum(factual_errors) - best[0]) / WINDOW,
                "best_single_beats_factual": int((sum(factual_errors) - best[0]) / WINDOW > COMPARISON_TOLERANCE_BPM),
                "positive_immediate_regret": int(factual_errors[0] - immediate[0] > COMPARISON_TOLERANCE_BPM)})
    repeated = beam["repeated"]
    if repeated is None:
        row.update(repeated_available=0, repeated_correction_count=0, repeated_correction_offsets="[]")
        return row
    repeated_errors = [_float(item, "immediate_abs_error_bpm") for item in repeated]
    offsets = [offset for offset, item in enumerate(repeated) if _is_correction(item)]
    decisions = WINDOW - offsets[0] - 1
    changed = sum(_int(repeated[offset], "ppo_action") != _int(beam["factual"][offset], "ppo_action") for offset in range(offsets[0] + 1, WINDOW))
    gains = [left - right for left, right in zip(factual_errors, repeated_errors)]
    total = sum(repeated_errors)
    row.update(repeated_available=1, repeated_correction_count=len(offsets), repeated_correction_offsets=json.dumps(offsets),
               repeated_15hop_mae_bpm=total / WINDOW, repeated_endpoint_abs_error_bpm=repeated_errors[-1],
               repeated_gain_vs_factual_mae_bpm=(sum(factual_errors) - total) / WINDOW,
               repeated_gain_vs_best_single_mae_bpm=(best[0] - total) / WINDOW, repeated_beats_best_single=int((best[0] - total) / WINDOW > COMPARISON_TOLERANCE_BPM),
               downstream_ppo_proposal_change_count=changed, downstream_ppo_decision_count=decisions,
               downstream_ppo_proposal_change_rate=changed / decisions, pointwise_hop_gains_bpm=json.dumps(gains),
               recovery_classification=recovery_classification(gains))
    return row

def _estimate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"point": None, "lower_95": None, "upper_95": None, "subject_count": 0, "replicates": 10_000}
    adapted = [dict(row, factual_regret_bpm=float(row[key])) for row in rows]
    result = _subject_block_bootstrap(adapted)
    return {"point": result["point_estimate_bpm"], "lower_95": result["ci_95_percentile_bpm"][0],
            "upper_95": result["ci_95_percentile_bpm"][1], "subject_count": len({row["subject_id"] for row in rows}), "replicates": result["replicates"]}

def _boolean(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return {"raw": {"numerator": sum(int(row[key]) for row in rows), "denominator": len(rows)},
            "subject_weighted": _estimate(rows, key)}

def _seed_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["analysis_eligible"]]
    repeated = [row for row in eligible if row.get("repeated_available")]
    result = {"raw_counts": {"anchors": len(rows), "analysis_eligible": len(eligible), "repeated_available": len(repeated)}}
    for key, subset in (("positive_immediate_regret", eligible), ("best_single_beats_factual", eligible), ("same_step_myopic_reversal", eligible), ("repeated_beats_best_single", repeated)):
        result[key] = _boolean(subset, key)
    flags = [dict(row, value=int(bool(row.get("repeated_available")))) for row in eligible]
    result["repeated_available"] = _boolean(flags, "value")
    result["recovery"] = {label: _boolean([dict(row, value=int(row["recovery_classification"] == label)) for row in repeated], "value") for label in
                          ("better_throughout", "briefly_better_then_ends_worse", "only_ends_better_late", "mixed_intermittent", "never_better")}
    result["numeric"] = {key: {"subject_weighted": _estimate(subset, key)} for key, subset in
                         (("factual_15hop_mae_bpm", eligible), ("ppo_immediate_regret_bpm", eligible),
                          ("best_single_gain_vs_factual_mae_bpm", eligible), ("repeated_gain_vs_factual_mae_bpm", repeated),
                          ("repeated_gain_vs_best_single_mae_bpm", repeated), ("downstream_ppo_proposal_change_rate", repeated))}
    return result

def _expected_task_indices(clip_count: int, seeds: Sequence[int]) -> tuple[int, ...]:
    """Return global source-plan task IDs for the requested seed subset."""
    if clip_count < 1 or not seeds or len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        _fail("requested seed task mapping is invalid")
    return tuple(3 * position + seed for position in range(clip_count) for seed in seeds)

def _full_subjects_from_task_entries(task_entries: Mapping[int, Mapping[str, Any]], clip_count: int = 533) -> dict[str, str]:
    """Validate source-plan triplets and return the clip-to-subject mapping."""
    subjects = {}
    for position in range(clip_count):
        triplet = [task_entries.get(3 * position + seed) for seed in SEEDS]
        if any(item is None for item in triplet):
            _fail("full subset task triplet is incomplete")
        if any((str(item["clip_id"]), str(item["subject_id"])) != (str(triplet[0]["clip_id"]), str(triplet[0]["subject_id"])) for item in triplet):
            _fail("full subset task triplet identity is inconsistent")
        subjects[str(triplet[0]["clip_id"])] = str(triplet[0]["subject_id"])
    return subjects

def _result_inventory(results: Path, run_prefix: str, job_ids: Sequence[str], expected_tasks: Sequence[int]) -> dict[int, Path]:
    """Require one declared-job output for every expected task and no extras."""
    prefixes = tuple(f"{run_prefix}-{job_id}-" for job_id in job_ids)
    observed = {path.name for path in results.iterdir() if any(path.name.startswith(prefix) for prefix in prefixes)}
    inventory = {}
    for task in expected_tasks:
        candidates = [results / f"{run_prefix}-{job_id}-{task}" for job_id in job_ids]
        runs = [candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()]
        if len(runs) != 1:
            _fail("each task must have exactly one result from the declared jobs")
        inventory[task] = runs[0]
    expected = {run.name for run in inventory.values()}
    if observed != expected:
        _fail("result inventory contains an undeclared or duplicate task output")
    return inventory

def _beam_config(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("beam config cannot be read") from exc
    expected_keys = {
        "schema", "dataset_id", "source_type", "clip_count", "subject_count", "seeds", "task_count",
        "task_mapping", "window_hops", "primary_width", "sensitivity_widths", "ranking", "retention", "scope",
    }
    if set(value) != expected_keys:
        _fail("beam config keys are invalid")
    if value.get("schema") != "mcd-repeated-correction-beam-full-v1":
        _fail("beam config schema is invalid")
    if (value.get("dataset_id") != "mcd" or value.get("source_type") != "full_frozen_eval" or
            value.get("clip_count") != 533 or value.get("subject_count") != 89 or value.get("seeds") != [0, 1, 2] or
            value.get("task_count") != FULL_TASK_COUNT or value.get("task_mapping") != "task_index = 3 * source_plan_clip_position + seed"):
        _fail("full beam config is invalid")
    if value.get("window_hops") != WINDOW or value.get("primary_width") != PRIMARY_WIDTH or value.get("sensitivity_widths") != list(SENSITIVITY_WIDTHS):
        _fail("beam config search parameters are invalid")
    if value.get("ranking") != "offline_gt_immediate_post_update_abs_error_then_action_index":
        _fail("beam config ranking is invalid")
    if value.get("retention") != "cumulative_abs_error_then_endpoint_abs_error_then_executed_sequence_then_origin_sequence":
        _fail("beam config retention is invalid")
    return value


def analyze_results(results_root: str | Path, output_dir: str | Path, *, job_id: str | int | Sequence[str | int], source_subsets_dir: str | Path,
                    beam_config: str | Path,
                    seeds: Sequence[int] | None = None) -> None:
    job_ids = tuple(str(value) for value in (job_id if isinstance(job_id, Sequence) and not isinstance(job_id, str) else (job_id,)))
    if not job_ids or len(set(job_ids)) != len(job_ids):
        _fail("job IDs must be nonempty and unique")
    config = _beam_config(beam_config)
    configured_seeds = tuple(int(seed) for seed in config.get("seeds", SEEDS))
    requested_seeds = configured_seeds if seeds is None else tuple(int(seed) for seed in seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds) or any(seed not in SEEDS for seed in requested_seeds):
        _fail("requested seeds must be a nonempty subset of [0, 1, 2]")
    if any(seed not in configured_seeds for seed in requested_seeds):
        _fail("requested seeds are not present in beam config")
    results, output, subsets = map(Path, (results_root, output_dir, source_subsets_dir))
    inputs = (results, subsets)
    if any(path.is_symlink() or not path.is_dir() for path in inputs): _fail("input roots must be real directories")
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.is_symlink(): _fail("output must be a fresh path below a real parent")
    index_path = subsets / "index.json"
    if index_path.is_symlink(): _fail("subset index must be a real file")
    index = read_json_object(index_path)
    entries = index.get("tasks", [])
    task_count = FULL_TASK_COUNT
    index_schema = "mcd-repeated-correction-beam-full-subsets-v1"
    if index.get("schema") != index_schema or index.get("task_count") != task_count or len(entries) != task_count or {item.get("task_index") for item in entries} != set(range(task_count)) or index.get("audit_plan_sha256") != index.get("plan_sha256"):
        _fail("subset index provenance/inventory mismatch")
    checkpoints = index.get("checkpoint_sha256")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"0", "1", "2"} or any(not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef") for value in checkpoints.values()): _fail("subset checkpoint inventory mismatch")
    beam_config_hash = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    if index.get("clip_count") != 533 or index.get("subject_count") != 89:
        _fail("full subset index cohort is invalid")
    task_entries = {int(item["task_index"]): item for item in entries}
    if any((int(item.get("clip_position", -1)), int(item.get("seed", -1))) != (task // 3, task % 3) for task, item in task_entries.items()):
        _fail("full subset task mapping is invalid")
    subjects = _full_subjects_from_task_entries(task_entries)
    if len(subjects) != 533 or len(set(subjects.values())) != 89:
        _fail("full subset clip/subject inventory is invalid")
    selection = tuple({"clip_id": clip, "subject_id": subject} for clip, subject in subjects.items())
    run_prefix = "mcd-repeated-beam-full"
    clip_order = [str(row["clip_id"]) for row in selection]
    expected_tasks = _expected_task_indices(len(clip_order), requested_seeds)
    inventory = _result_inventory(results, run_prefix, job_ids, expected_tasks)
    anchors, seen, digest = [], set(), hashlib.sha256()
    for task in expected_tasks:
        entry_expected = task_entries[task]
        clip, seed = str(entry_expected["clip_id"]), int(entry_expected["seed"])
        subject, run = subjects[clip], inventory[task]
        entry, source_rows = read_task_subset(subsets, task, task_count=task_count)
        if run.is_symlink() or not run.is_dir() or (entry["clip_id"], entry["subject_id"], entry["seed"]) != (clip, subject, seed): _fail("task mapping/output directory mismatch")
        if not _source_identity(source_rows, seed, checkpoints[str(seed)]): _fail("source row model identity mismatch")
        paths = [run / "summary.json", *(run / name for name in FILES)]
        if any(path.is_symlink() or not path.is_file() for path in paths): _fail("task output is missing or symlinked")
        summary = read_json_object(paths[0])
        identity = (summary.get("schema"), summary.get("job_id"), summary.get("task_index"), summary.get("task_count"),
                    summary.get("clip_id"), summary.get("subject_id"), summary.get("seed"), summary.get("parity_pass"), summary.get("widths"))
        if identity != (SCHEMA, run.name.rsplit("-", 2)[1], task, task_count, clip, subject, seed, True, list(SENSITIVITY_WIDTHS)) or summary.get("source_subset") != {"sha256": entry["sha256"], "bytes": entry["bytes"]}:
            _fail("task summary binding mismatch")
        if (summary.get("beam_config_schema"), summary.get("beam_config_sha256")) != (config["schema"], beam_config_hash):
            _fail("full task summary beam config binding mismatch")
        tables = {name[:-4]: _csv(run / name) for name in FILES}
        if summary.get("row_counts") != {name: len(rows) for name, rows in tables.items()}: _fail("task row-count mismatch")
        expected = (f"advantage_ppo_seed{seed}", str(seed), clip, subject)
        if any((row.get("method_id"), str(row.get("seed")), row.get("clip_id"), row.get("subject_id")) != expected for rows in tables.values() for row in rows): _fail("saved row identity mismatch")
        factual = _factual(tables["factual_hops"], source_rows)
        branches = _branches(tables["one_time_hops"], tables["one_time_scores"], factual)
        beams = _beam(tables["beam_hops"], tables["beam_stats"], factual)
        if summary.get("eligible_anchor_count") != len(beams) or {anchor for anchor, _action in branches} != set(beams): _fail("producer anchor inventories disagree")
        anchors.extend({"task_index": task, "clip_id": clip, "subject_id": subject, "seed": seed, **_anchor(anchor, factual, branches, beam)} for anchor, beam in sorted(beams.items()))
        for path in paths:
            digest.update(f"{task}:{path.name}:{sha256_file(path)}\n".encode())
        if (clip, seed) in seen: _fail("duplicate clip/seed task")
        seen.add((clip, seed))
    if seen != {(row["clip_id"], seed) for row in selection for seed in requested_seeds}: _fail("coverage does not equal configured clips by requested seeds")
    if not anchors: _fail("no anchors were saved")
    anchors.sort(key=lambda row: (row["seed"], row["clip_id"], row["anchor_hop_idx"]))
    output.mkdir()
    fields = list(anchors[0]) + sorted(set().union(*(set(row) for row in anchors)) - set(anchors[0]))
    with (output / "anchors.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(anchors)
    exclusions = defaultdict(int)
    for row in anchors:
        if not row["analysis_eligible"]:
            exclusions[row["exclusion_reason"]] += 1
    report = {"schema": "mcd-repeated-beam-behavior-v2", "job_ids": list(job_ids),
              "requested_seeds": list(requested_seeds), "source_index_schema": index_schema,
              "source_index_task_count": task_count, "analyzed_task_count": len(seen),
              "comparison_tolerance_bpm": COMPARISON_TOLERANCE_BPM,
              "comparison_scale": "reported BPM; immediate and pointwise differences compare directly, H15 categorical differences compare MAE differences",
              "cohort": {"dataset": "MCD", "clips": len(selection), "subjects": len(set(subjects.values())), "tasks": len(seen), "anchors": len(anchors), "seeds": list(requested_seeds)},
              "provenance": {"beam_config_sha256": beam_config_hash,
                             "subset_index_sha256": sha256_file(index_path), "source_sha256": index["source_sha256"],
                             "source_plan_sha256": index["plan_sha256"], "job_outputs_sha256": digest.hexdigest()},
              "exclusions": dict(sorted(exclusions.items())),
              "per_seed": {str(seed): _seed_summary([row for row in anchors if row["seed"] == seed]) for seed in requested_seeds},
              "aggregation": "anchor mean within clip, equal clips within subject, equal subjects within seed; 10,000 subject bootstrap draws",
              "limitations": ["Intermediate beam winners cannot be reconstructed from final survivors.",
                              "Incomplete fixed-horizon one-time coverage is excluded.",
                              "Width 8 is retained search, not exhaustive search.",
                              "MCD-only offline diagnostic; no MMPD access and no causal diagnosis of PPO training."]}
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--source-subsets-dir", required=True)
    parser.add_argument("--beam-config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="seed subset to aggregate; defaults to every seed in the beam config")
    args = parser.parse_args()
    analyze_results(results_root=args.results_root, output_dir=args.output_dir, job_id=args.job_id,
                    source_subsets_dir=args.source_subsets_dir, beam_config=args.beam_config,
                    seeds=args.seeds)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
