#!/usr/bin/env python3
"""Run the frozen MCD simple-selector comparison.

This deliberately has no MMPD imports or labels in the replay path.  The
existing Gate 8 bindings and recurrent-policy loader are reused, while the
diagnostic rows are a separate JSON/CSV contract.  This is a diagnostic, not
a publication artifact: there is no STARTED/COMPLETE/FAILED marker machinery
and no code-snapshot hash.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.model_plan import load_frozen_model_plan
from adaptive_roi_rppg.evaluation.model_publication import canonical_subject_shard, expected_shard_clip_ids
from adaptive_roi_rppg.evaluation.simple_selectors import DEFAULT_RANDOM_SEED, FixedFullFaceSelector, MaxPprSelector, rollout_diagnostic_policy, score_simple_rollout
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_gate8_mcd_frozen_models import _bindings, _runtime_checkpoints
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy

FULL_FACE = 12.631680651997897
ORACLE_B = 3.7328175200960665
PER_HOP_FIELDS = ("method_id", "family", "seed", "subject_id", "clip_id", "hop_idx", "hop_time_s",
                   "gt_hr_bpm", "abs_error_bpm", "requested_action", "proposed_action", "executed_action", "selected_valid")
OBS_COLS = tuple(f"obs_{i:03d}" for i in range(101))
ALT_ERR_COLS = tuple(f"alt_err_{i:02d}" for i in range(12))
OBS_FIELDS = ("clip_id", "hop_idx", *OBS_COLS, *ALT_ERR_COLS, "executed_action", "gt_hr_bpm")


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output_dir", "manifest_tree", "state_root", "gt_root", "plan", "checkpoint_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    parser.add_argument("--limit-clips", dest="limit_clips", type=int, default=None)
    parser.add_argument("--clip-offset", dest="clip_offset", type=int, default=0)
    parser.add_argument("--shard-count", dest="shard_count", type=int, default=1)
    parser.add_argument("--shard-index", dest="shard_index", type=int, default=0)
    parser.add_argument("--merge", dest="merge", action="store_true")
    parser.add_argument("--shard-dir", dest="shard_dir", action="append", default=[])
    return parser.parse_args()


def _validate_shard_args(args: argparse.Namespace) -> None:
    if not isinstance(args.shard_count, int) or not isinstance(args.shard_index, int) or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        _fail("invalid shard assignment")


def _run_policy(frames, labels, policy, clip, per_hop_writer, obs_writer, obs_method, clip_accum, behavior):
    rollout = rollout_diagnostic_policy(frames, policy, clip_id=clip.clip_id)
    scored = score_simple_rollout(rollout, labels, {"subject_id": clip.subject_id, "view": clip.view, "condition": clip.condition})
    method_id = scored[0]["method_id"]
    stats = behavior.setdefault(method_id, {"switch": 0, "switch_denom": 0, "override": 0, "invalid": 0, "hop_count": 0, "hist": [0] * 12})
    previous_executed = None
    for hop, row in zip(rollout, scored):
        per_hop_writer.writerow({"method_id": row["method_id"], "family": row["family"], "seed": row["seed"],
                                  "subject_id": row["subject_id"], "clip_id": row["clip_id"], "hop_idx": row["hop_idx"],
                                  "hop_time_s": row["hop_time_s"], "gt_hr_bpm": row["gt_hr_bpm"], "abs_error_bpm": row["abs_error_bpm"],
                                  "requested_action": row["requested_action"], "proposed_action": row["proposed_action"],
                                  "executed_action": row["executed_action"], "selected_valid": hop.audit.selected.selected_valid})
        key = (method_id, row["subject_id"], row["clip_id"])
        entry = clip_accum.setdefault(key, [0.0, 0])
        entry[0] += row["abs_error_bpm"]; entry[1] += 1
        stats["hop_count"] += 1
        stats["hist"][row["executed_action"]] += 1
        if row["proposed_action"] != row["executed_action"]:
            stats["override"] += 1
        if not hop.audit.selected.selected_valid:
            stats["invalid"] += 1
        if previous_executed is not None:
            stats["switch_denom"] += 1
            if row["executed_action"] != previous_executed:
                stats["switch"] += 1
        previous_executed = row["executed_action"]
        if obs_writer is not None and method_id == obs_method:
            obs_row = {"clip_id": row["clip_id"], "hop_idx": row["hop_idx"], "executed_action": row["executed_action"], "gt_hr_bpm": row["gt_hr_bpm"]}
            obs_row.update(zip(OBS_COLS, row["observation_values"]))
            obs_row.update(zip(ALT_ERR_COLS, row["alternative_abs_error_bpm"]))
            obs_writer.writerow(obs_row)


def _aggregate(clip_accum):
    clips = []
    for (method, subject, clip), (total, count) in sorted(clip_accum.items()):
        clips.append({"method_id": method, "subject_id": subject, "clip_id": clip, "mae_bpm": total / count, "hop_count": count})
    methods = {}
    for method in sorted({row["method_id"] for row in clips}):
        values = [row["mae_bpm"] for row in clips if row["method_id"] == method]
        methods[method] = {"method_id": method, "clip_count": len(values), "equal_clip_mae_bpm": sum(values) / len(values)}
        methods[method]["headroom_captured_percent"] = (FULL_FACE - methods[method]["equal_clip_mae_bpm"]) / (FULL_FACE - ORACLE_B) * 100
    return clips, methods


def bootstrap_equal_clip_mae(clips, subjects, methods, *, replicates: int = 10_000, seed: int = DEFAULT_RANDOM_SEED):
    """Subject-block paired bootstrap: draws resample subjects with replacement.

    Returns ``method -> float32[replicates]`` of the equal-clip MAE for each
    draw (a subject drawn twice contributes its clips twice).
    """
    subject_index = {subject: i for i, subject in enumerate(subjects)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    out = {}
    for method in methods:
        sums = np.zeros(len(subjects)); counts = np.zeros(len(subjects))
        for row in clips:
            if row["method_id"] != method: continue
            i = subject_index[row["subject_id"]]; sums[i] += row["mae_bpm"]; counts[i] += 1
        drawn_sums = sums[draws].sum(axis=1); drawn_counts = counts[draws].sum(axis=1)
        out[method] = drawn_sums / drawn_counts
    return out


def _percentile_ci(values) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_bootstrap_difference(draws, left_method: str, right_method: str):
    """Subtract two methods within each shared subject-bootstrap draw."""
    if left_method not in draws or right_method not in draws:
        _fail("paired bootstrap method is missing")
    left = np.asarray(draws[left_method])
    right = np.asarray(draws[right_method])
    if left.shape != right.shape or left.ndim != 1:
        _fail("paired bootstrap draws are not aligned")
    return left - right


def _behavioral_report(behavior):
    report = {}
    for method, stats in behavior.items():
        n = stats["hop_count"]
        report[method] = {
            "switch_rate": stats["switch"] / stats["switch_denom"] if stats["switch_denom"] else 0.0,
            "min_hold_override_rate": stats["override"] / n if n else 0.0,
            "invalid_selected_rate": stats["invalid"] / n if n else 0.0,
            "executed_action_histogram": stats["hist"],
        }
    return report


def _write_shard_accum(root, clip_accum, behavior) -> None:
    entries = [{"method_id": method, "subject_id": subject, "clip_id": clip, "total_abs_error": total, "hop_count": count}
               for (method, subject, clip), (total, count) in sorted(clip_accum.items())]
    (root / "shard_accum.json").write_bytes(canonical_json_bytes({"clip_accum": entries, "behavior": behavior}))


def _merge_shard_accums(shard_dirs):
    clip_accum: dict[tuple[str, str, str], list] = {}
    behavior: dict[str, dict] = {}
    for shard_dir in shard_dirs:
        path = Path(shard_dir) / "shard_accum.json"
        if not path.is_file(): _fail("missing shard_accum.json in a shard directory")
        payload = read_json_object(path)
        for entry in payload["clip_accum"]:
            key = (entry["method_id"], entry["subject_id"], entry["clip_id"])
            if key in clip_accum: _fail("duplicated (method_id, subject_id, clip_id) key across shards")
            clip_accum[key] = [entry["total_abs_error"], entry["hop_count"]]
        for method, stats in payload["behavior"].items():
            accum = behavior.setdefault(method, {"switch": 0, "switch_denom": 0, "override": 0, "invalid": 0, "hop_count": 0, "hist": [0] * 12})
            if len(stats["hist"]) != 12: _fail("shard behavior histogram is not 12 bins")
            for field in ("switch", "switch_denom", "override", "invalid", "hop_count"): accum[field] += stats[field]
            accum["hist"] = [a + b for a, b in zip(accum["hist"], stats["hist"])]
    return clip_accum, behavior


def _finalize(root, clips, adv, clip_accum, behavior, partial_cohort) -> None:
    clips_out, methods = _aggregate(clip_accum)
    if not partial_cohort and abs(methods["full_face"]["equal_clip_mae_bpm"] - FULL_FACE) > 1e-9:
        _fail("full-cohort full_face equal-clip MAE does not match the accepted E-031 constant")

    subjects = sorted({clip.subject_id for clip in clips})
    advantage_methods = tuple(spec["method_id"] for spec in adv)
    bootstrap_methods = ("full_face", "max_ppr", *advantage_methods)
    draws = bootstrap_equal_clip_mae(clips_out, subjects, bootstrap_methods)
    advantage_mean_draws = np.mean([draws[method] for method in advantage_methods], axis=0)
    diff_max_ppr = draws["max_ppr"] - advantage_mean_draws
    diff_max_ppr_full_face = paired_bootstrap_difference(draws, "max_ppr", "full_face")
    advantage_mean_point = sum(methods[method]["equal_clip_mae_bpm"] for method in advantage_methods) / len(advantage_methods)

    bootstrap = {method: {"point_bpm": methods[method]["equal_clip_mae_bpm"], "ci_95_bpm": _percentile_ci(draws[method])} for method in bootstrap_methods}
    paired = {
        "max_ppr_minus_advantage_mean": {"point_bpm": methods["max_ppr"]["equal_clip_mae_bpm"] - advantage_mean_point, "ci_95_bpm": _percentile_ci(diff_max_ppr)},
        "max_ppr_minus_full_face": {"point_bpm": methods["max_ppr"]["equal_clip_mae_bpm"] - methods["full_face"]["equal_clip_mae_bpm"], "ci_95_bpm": _percentile_ci(diff_max_ppr_full_face)},
    }

    report = {
        "schema": "mcd-simple-selectors-report-v1",
        "cohort": {"clip_count": len(clips), "subject_count": len(subjects), "hop_count": behavior.get("full_face", {}).get("hop_count", 0), "limited": partial_cohort},
        "methods": methods,
        "bootstrap": {"replicates": 10_000, "seed": DEFAULT_RANDOM_SEED, "per_method": bootstrap, "paired_vs_advantage_mean": paired},
        "behavioral": _behavioral_report(behavior),
        "accepted_context": {"full_face_mae_bpm": FULL_FACE, "oracle_b_mae_bpm": ORACLE_B},
    }
    (root / "report.json").write_bytes(canonical_json_bytes(report))


def main() -> int:
    args = _args(); _validate_shard_args(args); root = Path(args.output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink(): _fail("output must be a fresh path under an existing real parent")
    root.mkdir()

    bundle, clips = _bindings(args); plan = load_frozen_model_plan(args.plan); checkpoints = _runtime_checkpoints(args, plan)
    adv = tuple(sorted((spec for spec in checkpoints if spec["family"] == "advantage_ppo"), key=lambda spec: spec["seed"]))
    if len(adv) != 3: _fail("expected exactly three Advantage PPO checkpoints")
    obs_method = next(spec["method_id"] for spec in adv if spec["seed"] == 1)
    if args.clip_offset < 0 or args.clip_offset >= len(clips): _fail("clip offset is negative or beyond the cohort size")
    if args.clip_offset: clips = clips[args.clip_offset:]
    if args.limit_clips is not None: clips = clips[:args.limit_clips]
    partial_cohort = args.limit_clips is not None or args.clip_offset != 0

    if args.merge:
        if not args.shard_dir or len(args.shard_dir) != args.shard_count: _fail("merge requires every deterministic shard")
        clip_accum, behavior = _merge_shard_accums(args.shard_dir)
        clip_ids = {key[2] for key in clip_accum}
        if not partial_cohort and (len(clip_ids) != 533 or len(clip_accum) != 533 * 5): _fail("merged clip cohort is not the full 533-clip cohort across all five arms")
        _finalize(root, clips, adv, clip_accum, behavior, partial_cohort)
        return 0

    if args.shard_count == 1:
        shard_clips = clips
    else:
        subjects = sorted({clip.subject_id for clip in clips})
        expected_subjects = set(canonical_subject_shard(subjects, args.shard_index, args.shard_count))
        expected_clip_ids = expected_shard_clip_ids([{"subject_id": clip.subject_id, "clip_id": clip.clip_id} for clip in clips], args.shard_index, args.shard_count)
        shard_clips = tuple(clip for clip in clips if clip.subject_id in expected_subjects)
        if tuple(clip.clip_id for clip in shard_clips) != expected_clip_ids: _fail("canonical shard selection differs from frozen clip order")

    policies = [FixedFullFaceSelector(), MaxPprSelector(), *(load_frozen_recurrent_policy(spec) for spec in adv)]

    clip_accum: dict[tuple[str, str, str], list] = {}
    behavior: dict[str, dict] = {}
    with (root / "per_hop.csv").open("x", newline="") as per_hop_handle, (root / "observations_advantage_ppo_seed1.csv").open("x", newline="") as obs_handle:
        per_hop_writer = csv.DictWriter(per_hop_handle, fieldnames=PER_HOP_FIELDS); per_hop_writer.writeheader()
        obs_writer = csv.DictWriter(obs_handle, fieldnames=OBS_FIELDS); obs_writer.writeheader()
        for clip in shard_clips:
            frames = build_pos_measurements(read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "eval"))
            labels = read_mcd_eval_labels(bundle, args.gt_root, clip.clip_id, clip)
            if len(frames) != len(labels): _fail("canonical frame and label coverage differs")
            for policy in policies:
                _run_policy(frames, labels, policy, clip, per_hop_writer, obs_writer, obs_method, clip_accum, behavior)

    if args.shard_count == 1:
        _finalize(root, clips, adv, clip_accum, behavior, partial_cohort)
    else:
        _write_shard_accum(root, clip_accum, behavior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
