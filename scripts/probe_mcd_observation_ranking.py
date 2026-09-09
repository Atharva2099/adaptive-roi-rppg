#!/usr/bin/env python3
"""Test whether PPO's stored observation predicts the best immediate ROI.

This is an offline, one-hop diagnostic.  It never rolls an alternative action
forward, and therefore its regret must not be compared with sequential MAE.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np

OBS_COLS = tuple(f"obs_{i:03d}" for i in range(101))
ALT_ERR_COLS = tuple(f"alt_err_{i:02d}" for i in range(12))
OBS_FIELDS = ("clip_id", "hop_idx", *OBS_COLS, *ALT_ERR_COLS, "executed_action", "gt_hr_bpm")
HOP_FIELDS = ("method_id", "family", "seed", "subject_id", "clip_id", "hop_idx", "hop_time_s", "gt_hr_bpm", "abs_error_bpm", "requested_action", "proposed_action", "executed_action", "selected_valid")
VALIDITY_INDICES = tuple(7 * k + 6 for k in range(12))


def _lazy_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:
        raise RuntimeError("Install the observation-probe dependency with `pip install -e '.[observation-probe]'`.") from exc
    return HistGradientBoostingRegressor, GroupKFold


def free_choice_mask(observations: np.ndarray, hop_idx: np.ndarray) -> np.ndarray:
    """Return hops at which the canonical minimum-hold controller permits choice."""
    if observations.ndim != 2 or observations.shape[1] < 101:
        raise ValueError("observations must have at least 101 columns")
    if len(hop_idx) != len(observations):
        raise ValueError("hop_idx and observations have different lengths")
    return (hop_idx == 0) | (observations[:, 100] >= 0.2)


def valid_roi_mask(observations: np.ndarray) -> np.ndarray:
    array = np.asarray(observations)
    if array.ndim == 1:
        if array.shape[0] < 101:
            raise ValueError("observations must have at least 101 columns")
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 101:
        raise ValueError("observations must have at least 101 columns")
    return np.isfinite(array[:, VALIDITY_INDICES]) & (array[:, VALIDITY_INDICES] > 0.5)


def subject_folds(subjects: list[str], folds: int, seed: int = 8101):
    """Yield disjoint train/test subject sets using the same GroupKFold rule."""
    _, GroupKFold = _lazy_sklearn()
    subjects = np.asarray(sorted(set(subjects)))
    if len(subjects) < folds:
        raise ValueError("fold count exceeds subject count")
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(subjects, groups=subjects):
        yield tuple(subjects[train]), tuple(subjects[test])


def action_order(errors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sort candidate ROI indices by (error, index), putting invalids last."""
    if errors.shape != valid.shape or errors.ndim != 1:
        raise ValueError("errors and valid must be aligned one-dimensional arrays")
    return np.array(sorted(range(len(errors)), key=lambda i: (float(errors[i]) if valid[i] else float("inf"), i)), dtype=int)


def choose_valid(predicted: np.ndarray, valid: np.ndarray) -> int:
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("cannot choose from an all-invalid hop")
    return int(min(candidates, key=lambda i: (float(predicted[i]), int(i))))


def bootstrap_subject_metrics(rows: list[dict], subjects: list[str], *, replicates: int = 10_000, seed: int = 8101):
    """Paired subject-block bootstrap of equal-clip mean immediate regret."""
    methods = tuple(sorted({row["arm"] for row in rows}))
    by_method_subject_clip: dict[str, dict[str, dict[str, list[float]]]] = {
        m: defaultdict(lambda: defaultdict(list)) for m in methods
    }
    for row in rows:
        by_method_subject_clip[row["arm"]][row["subject_id"]][row["clip_id"]].append(float(row["regret_bpm"]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    out = {m: np.empty(replicates, dtype=float) for m in methods}
    for m in methods:
        subject_clips = [[np.mean(values) for values in by_method_subject_clip[m].get(s, {}).values()] for s in subjects]
        if any(not clips for clips in subject_clips):
            raise ValueError(f"arm {m} is missing subjects")
        for draw_index, draw in enumerate(draws):
            out[m][draw_index] = equal_clip_mean_for_subject_draw(subject_clips, draw)
    return out


def equal_clip_mean_for_subject_draw(subject_clips: list[list[float]], draw: np.ndarray) -> float:
    """Aggregate sampled subjects by their clips, preserving equal clip weight."""
    selected = [clip_mean for subject_index in draw for clip_mean in subject_clips[int(subject_index)]]
    if not selected:
        raise ValueError("subject draw has no clips")
    return float(np.mean(selected))


def _ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read probe config: {path}") from exc
    if config.get("schema") != "mcd-observation-probe-config-v1":
        raise ValueError("unexpected observation probe config schema")
    model = config.get("model", {})
    expected_model = {"name", "max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "random_state"}
    if set(model) != expected_model or model["name"] != "HistGradientBoostingRegressor":
        raise ValueError("observation probe model config keys/name are invalid")
    if (model["max_iter"], model["learning_rate"], model["max_leaf_nodes"], model["l2_regularization"]) != (100, 0.1, 31, 0.0):
        raise ValueError("observation probe model settings differ from the fixed protocol")
    for section, key in (("split", "full_folds"), ("bootstrap", "replicates"), ("bootstrap", "seed")):
        if not isinstance(config.get(section, {}).get(key), int) or config[section][key] < 1:
            raise ValueError(f"invalid configured {section}.{key}")
    return config


def load_probe_rows(root: str | Path, *, max_subjects: int | None = None):
    """Load, join, validate, and filter the shard observations."""
    base = Path(root)
    obs_paths = sorted(base.glob("shard-*/observations_advantage_ppo_seed1.csv"))
    hop_paths = sorted(base.glob("shard-*/per_hop.csv"))
    if not obs_paths or len(obs_paths) != len(hop_paths):
        raise ValueError("shard root must contain matching shard-*/ observations and per_hop.csv files")
    observations = {}
    source = []
    observation_source_by_key = {}
    for path in obs_paths:
        source.append({"path": str(path), "sha256": _sha256(path), "rows": 0})
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != OBS_FIELDS:
                raise ValueError(f"unexpected observation header in {path}")
            for raw in reader:
                key = (raw["clip_id"], int(raw["hop_idx"]))
                if key in observations:
                    raise ValueError(f"duplicate observation key {key}")
                values = np.array([float(raw[c]) for c in OBS_COLS + ALT_ERR_COLS], dtype=float)
                action = int(raw["executed_action"])
                if action not in range(12):
                    raise ValueError("executed_action outside 0..11")
                observations[key] = {"obs": values[:101], "errors": values[101:], "action": action}
                observation_source_by_key[key] = source[-1]
                source[-1]["rows"] += 1
    subjects_by_key = {}
    hop_source = []
    for path in hop_paths:
        hop_source.append({"path": str(path), "sha256": _sha256(path), "rows": 0})
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if fields != HOP_FIELDS:
                raise ValueError(f"unexpected per_hop header in {path}")
            for raw in reader:
                hop_source[-1]["rows"] += 1
                if raw["method_id"] != "advantage_ppo_seed1":
                    continue
                key = (raw["clip_id"], int(raw["hop_idx"]))
                if key in subjects_by_key and subjects_by_key[key] != raw["subject_id"]:
                    raise ValueError(f"subject join conflict at {key}")
                subjects_by_key[key] = raw["subject_id"]
    if set(observations) != set(subjects_by_key):
        raise ValueError("observation and advantage_ppo_seed1 per_hop keys do not join exactly")
    subjects = sorted(set(subjects_by_key.values()))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    keep_subjects = set(subjects)
    selected_keys = {key for key in observations if subjects_by_key[key] in keep_subjects}
    for key in selected_keys:
        observation_source_by_key[key]["selected_rows"] = observation_source_by_key[key].get("selected_rows", 0) + 1
    rows = []
    dropped_invalid = 0
    dropped_no_valid = 0
    locked_rows = 0
    nonfinite_rows = 0
    for (clip_id, hop_idx), item in sorted(observations.items()):
        subject = subjects_by_key[(clip_id, hop_idx)]
        if subject not in keep_subjects:
            continue
        if not np.all(np.isfinite(item["obs"])) or not np.all(np.isfinite(item["errors"])):
            nonfinite_rows += 1
            continue
        if not free_choice_mask(item["obs"][None, :], np.array([hop_idx]))[0]:
            locked_rows += 1
            continue
        valid = valid_roi_mask(item["obs"])[0]
        if not np.any(valid):
            dropped_no_valid += 1
            continue
        errors = item["errors"]
        if not np.all(np.isfinite(errors[valid])):
            dropped_invalid += 1
            continue
        rows.append({"subject_id": subject, "clip_id": clip_id, "hop_idx": hop_idx,
                     "features": item["obs"].copy(), "errors": errors.copy(), "valid": valid.copy(),
                     "ppo_action": item["action"]})
    return rows, subjects, {"source": source, "free_rows": len(rows), "joined_rows": len(selected_keys),
        "total_source_rows": len(observations),
        "locked_rows": locked_rows, "dropped_no_valid": dropped_no_valid, "dropped_nonfinite": nonfinite_rows,
        "dropped_invalid": dropped_invalid, "free_fraction_before_no_valid_drop":
        (len(selected_keys) - nonfinite_rows - locked_rows) / (len(selected_keys) - nonfinite_rows)
        if len(selected_keys) > nonfinite_rows else 0.0,
        "subject_count": len(subjects), "per_hop_source": hop_source}


def feature_vector(row: dict, roi: int, layout: str) -> np.ndarray:
    if layout == "full_101":
        base = row["features"]
    elif layout == "local_shared":
        base = np.concatenate((row["features"][7 * roi:7 * roi + 7], row["features"][84:101]))
    else:
        raise ValueError(f"unknown feature layout: {layout}")
    identity = np.eye(12, dtype=float)[roi]
    return np.concatenate((base, identity))


def _design(rows, layout="full_101"):
    candidates = [(j, roi) for j, row in enumerate(rows) for roi in range(12) if row["valid"][roi]]
    feature_count = 113 if layout == "full_101" else 36 if layout == "local_shared" else 0
    if not feature_count:
        raise ValueError(f"unknown feature layout: {layout}")
    x = np.empty((len(candidates), feature_count), dtype=np.float32)
    y = np.empty(len(candidates), dtype=np.float32)
    row_ids = []
    for idx, (j, roi) in enumerate(candidates):
        row = rows[j]
        x[idx] = feature_vector(row, roi, layout)
        y[idx] = row["errors"][roi]
        row_ids.append(j)
    return x, y, np.asarray(row_ids), np.asarray([roi for _, roi in candidates])


def fit_predict(rows, *, layout="full_101", folds: int = 5, seed: int = 8101, shuffle_targets: bool = False, model_params=None):
    HistGradientBoostingRegressor, GroupKFold = _lazy_sklearn()
    if len({r["subject_id"] for r in rows}) < folds:
        raise ValueError("fold count exceeds subject count")
    x, y, row_ids, roi_ids = _design(rows, layout)
    hop_groups = np.array([r["subject_id"] for r in rows])
    splitter = GroupKFold(n_splits=folds)
    predictions = np.full((len(rows), 12), np.nan, dtype=float)
    fold_membership = []
    params = dict(model_params or {})
    params.setdefault("random_state", seed)
    for train_hops, test_hops in splitter.split(np.zeros(len(rows)), groups=hop_groups):
        fold_membership.append({"train_subjects": sorted({rows[i]["subject_id"] for i in train_hops}),
                                "test_subjects": sorted({rows[i]["subject_id"] for i in test_hops})})
        train_ids = np.flatnonzero(np.isin(row_ids, train_hops))
        test_ids = np.flatnonzero(np.isin(row_ids, test_hops))
        train_y = y[train_ids].copy()
        if shuffle_targets:
            # Permute complete 12-error vectors among training hops only.
            rng = np.random.default_rng(seed + int(train_hops[0]))
            order = rng.permutation(train_hops)
            train_position = {candidate_id: local for local, candidate_id in enumerate(train_ids)}
            for dst, src in zip(train_hops, order):
                dst_ids = train_ids[row_ids[train_ids] == dst]
                for target_id in dst_ids:
                    roi = roi_ids[target_id]
                    train_y[train_position[target_id]] = rows[src]["errors"][roi]
        model = HistGradientBoostingRegressor(**params)
        model.fit(x[train_ids], train_y)
        predictions[row_ids[test_ids], roi_ids[test_ids]] = model.predict(x[test_ids])
    for i, row in enumerate(rows):
        if not np.all(np.isfinite(predictions[i, row["valid"]])):
            raise ValueError("cross-validation left rows without predictions")
    return predictions, fold_membership


def score_arms(rows, predictions: dict[str, np.ndarray], layout: str):
    result = []
    for index, row in enumerate(rows):
        valid = row["valid"]
        order = action_order(row["errors"], valid)
        oracle = int(order[0])
        choices = {f"{layout}_probe": choose_valid(predictions["real"][index], valid),
                   f"{layout}_shuffled": choose_valid(predictions["shuffled"][index], valid)}
        if layout == "full_101":
            choices.update({"majority": 0 if valid[0] else int(np.flatnonzero(valid)[0]),
                            "ppo_factual": int(row["ppo_action"]), "oracle": oracle})
        top3 = set(order[valid[order]][:3])
        for arm, choice in choices.items():
            regret = float(row["errors"][choice] - row["errors"][oracle])
            result.append({"arm": arm, "subject_id": row["subject_id"], "clip_id": row["clip_id"],
                           "hop_idx": row["hop_idx"], "chosen_roi": choice, "regret_bpm": regret,
                           "top1": int(choice == oracle), "top3": int(choice in top3)})
    return result


def summarize(rows, subjects, *, replicates=10_000, seed=8101):
    finite_rows = [r for r in rows if np.isfinite(r["regret_bpm"])]
    draws = bootstrap_subject_metrics(finite_rows, subjects, replicates=replicates, seed=seed)
    report = {}
    for arm, values in draws.items():
        arm_rows = [r for r in finite_rows if r["arm"] == arm]
        by_clip = defaultdict(list)
        for row in arm_rows:
            by_clip[(row["subject_id"], row["clip_id"])].append(row["regret_bpm"])
        clip_means = [np.mean(v) for v in by_clip.values()]
        report[arm] = {"mean_immediate_regret_bpm": float(np.mean(clip_means)),
                       "ci_95_bpm": _ci(values), "top1_rate": float(np.mean([r["top1"] for r in arm_rows])),
                       "top3_rate": float(np.mean([r["top3"] for r in arm_rows])),
                       "action_histogram": [sum(r["chosen_roi"] == i for r in arm_rows) for i in range(12)],
                       "hop_count": len(arm_rows), "clip_count": len(by_clip)}
    paired = {}
    for left, right in (("full_101_probe", "majority"), ("full_101_probe", "full_101_shuffled"),
                        ("full_101_probe", "ppo_factual"), ("local_shared_probe", "majority"),
                        ("local_shared_probe", "local_shared_shuffled"), ("local_shared_probe", "ppo_factual")):
        diff = draws[left] - draws[right]
        paired[f"{left}_minus_{right}"] = {"point_bpm": report[left]["mean_immediate_regret_bpm"] - report[right]["mean_immediate_regret_bpm"],
                                            "ci_95_bpm": _ci(diff), "hypothesis_holds": bool(_ci(diff)[1] < 0)}
    return {"arms": report, "paired": paired, "bootstrap": {"replicates": replicates, "seed": seed}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/evaluation/mcd_observation_probe_v1.json")
    parser.add_argument("--subjects", type=int, default=None, help="Optional subject subset for bounded local inspection")
    parser.add_argument("--folds", type=int, default=None, help="Explicit fold override")
    parser.add_argument("--bootstrap-replicates", type=int, default=None, help="Explicit bootstrap override")
    parser.add_argument("--seed", type=int, default=None, help="Explicit seed override")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = _read_config(config_path)
    model_name = config["model"]["name"]
    model_params = {key: value for key, value in config["model"].items() if key != "name"}
    configured_folds = int(config["split"]["full_folds"])
    configured_replicates = int(config["bootstrap"]["replicates"])
    configured_seed = int(config["bootstrap"]["seed"])
    overrides = {}
    args.folds = args.folds if args.folds is not None else configured_folds
    args.bootstrap_replicates = args.bootstrap_replicates if args.bootstrap_replicates is not None else configured_replicates
    args.seed = args.seed if args.seed is not None else configured_seed
    if args.seed != configured_seed:
        overrides["seed"] = args.seed
    if args.folds != configured_folds:
        overrides["folds"] = args.folds
    if args.bootstrap_replicates != configured_replicates:
        overrides["bootstrap_replicates"] = args.bootstrap_replicates
    rows, subjects, provenance = load_probe_rows(args.shard_root, max_subjects=args.subjects)
    if not rows:
        raise ValueError("no eligible free hops remain")
    scored = []
    fold_membership = {}
    for layout in ("full_101", "local_shared"):
        real, folds = fit_predict(rows, layout=layout, folds=args.folds, seed=args.seed, model_params=model_params)
        shuffled, shuffled_folds = fit_predict(rows, layout=layout, folds=args.folds, seed=args.seed, shuffle_targets=True, model_params=model_params)
        scored.extend(score_arms(rows, {"real": real, "shuffled": shuffled}, layout))
        fold_membership[layout] = folds
        fold_membership[f"{layout}_shuffled"] = shuffled_folds
    output = {"schema": "mcd-observation-probe-report-v1", "provenance": provenance,
              "subjects": subjects, "row_count": len(rows), "results": summarize(scored, subjects, replicates=args.bootstrap_replicates, seed=args.seed),
              "settings": {"folds": args.folds, "seed": args.seed, "features": 101, "roi_identity_features": 12,
                           "model": model_name, "model_parameters": model_params, "feature_layouts": {"full_101": 113, "local_shared": 36},
                           "shuffle_targets": "complete training-hop 12-vector", "fold_membership": fold_membership,
                           "config": {"path": str(config_path), "sha256": _sha256(config_path)},
                           "script_sha256": _sha256(Path(__file__)), "run_date_utc": datetime.now(timezone.utc).isoformat(),
                           "python_version": platform.python_version(), "numpy_version": np.__version__,
                           "sklearn_version": __import__("sklearn").__version__, "overrides": overrides},
              "hypotheses": config["hypotheses"]}
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("output must be a fresh path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
