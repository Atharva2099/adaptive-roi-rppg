#!/usr/bin/env python3
"""Do within-hop relative features reduce immediate regret below local_shared?

Offline, one-hop diagnostic over the PPO observation shards. Reuses the
free-hop rule, ROI-validity rule, feature layouts, and bootstrap machinery
from probe_mcd_observation_ranking.py. Subject identity here is derived from
the clip_id prefix (`temp/probe_data` has no per_hop.csv to join), which was
verified separately to reproduce the existing 89-subject / 530-clip cohort
exactly; see docs/diagnostics/mcd_observation_probe.md.

Motivation: the local_shared arm cannot see the other 11 ROIs at all, and the
full_101 arm could see them but did not use them effectively (see the
observation-ranking probe). Relative features supply the cross-ROI
comparison directly, as deterministic functions of the existing 101 values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mcd_observation_ranking import (OBS_COLS, ALT_ERR_COLS, OBS_FIELDS,
    action_order, choose_valid, free_choice_mask, valid_roi_mask,
    bootstrap_subject_metrics, feature_vector, _design, _lazy_sklearn, _ci)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "evaluation" / "mcd_observation_probe_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_params() -> dict:
    """Load the same fixed model settings used by the existing probe."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {key: value for key, value in config["model"].items() if key != "name"}


def load_rows(root: str | Path):
    """Load, validate, and filter every shard's observations; derive subject_id from the clip_id prefix."""
    paths = sorted(Path(root).glob("shard-*/observations_advantage_ppo_seed1.csv"))
    if not paths:
        raise ValueError("shard root has no shard-*/observations_advantage_ppo_seed1.csv files")
    seen = {}
    source = []
    counts = {"nonfinite_obs_or_errors": 0, "locked": 0, "no_valid": 0, "nonfinite_valid_errors": 0}
    rows = []
    for path in paths:
        source.append({"path": str(path), "sha256": _sha256(path), "rows": 0})
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != OBS_FIELDS:
                raise ValueError(f"unexpected observation header in {path}")
            for raw in reader:
                source[-1]["rows"] += 1
                key = (raw["clip_id"], int(raw["hop_idx"]))
                if key in seen:
                    raise ValueError(f"duplicate observation key {key}")
                seen[key] = path
                obs = np.array([float(raw[c]) for c in OBS_COLS], dtype=float)
                errors = np.array([float(raw[c]) for c in ALT_ERR_COLS], dtype=float)
                action = int(raw["executed_action"])
                if action not in range(12):
                    raise ValueError("executed_action outside 0..11")
                if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(errors)):
                    counts["nonfinite_obs_or_errors"] += 1
                    continue
                if not free_choice_mask(obs[None, :], np.array([key[1]]))[0]:
                    counts["locked"] += 1
                    continue
                valid = valid_roi_mask(obs)[0]
                if not np.any(valid):
                    counts["no_valid"] += 1
                    continue
                if not np.all(np.isfinite(errors[valid])):
                    counts["nonfinite_valid_errors"] += 1
                    continue
                subject_id = raw["clip_id"].split("_")[0]
                rows.append({"subject_id": subject_id, "clip_id": key[0], "hop_idx": key[1],
                             "features": obs, "errors": errors, "valid": valid, "ppo_action": action})
    rows.sort(key=lambda r: (r["clip_id"], r["hop_idx"]))
    return rows, source, counts


def relative_features(features: np.ndarray, roi: int, valid: np.ndarray) -> np.ndarray:
    """Seven within-hop relative features for candidate ROI k over the valid ROIs at this hop."""
    valid_indices = np.flatnonzero(valid)
    valid_count = len(valid_indices)
    k_pos = int(np.flatnonzero(valid_indices == roi)[0])
    hr = np.array([features[7 * i + 0] for i in valid_indices], dtype=float)
    ppr = np.array([features[7 * i + 3] for i in valid_indices], dtype=float)
    coverage = np.array([features[7 * i + 4] for i in valid_indices], dtype=float)

    def rank_desc(values: np.ndarray, pos: int) -> float:
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))
        rank = order.index(pos)
        return rank / (len(values) - 1) if len(values) > 1 else 0.0

    def zscore(values: np.ndarray, pos: int) -> float:
        std = float(np.std(values))
        if std < 1e-9:
            return 0.0
        return float((values[pos] - np.mean(values)) / std)

    out = np.array([
        rank_desc(ppr, k_pos),
        zscore(ppr, k_pos),
        zscore(hr, k_pos),
        abs(float(hr[k_pos]) - float(np.median(hr))),
        rank_desc(coverage, k_pos),
        valid_count / 12.0,
        float(np.std(hr)),
    ], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite relative feature")
    return out


def feature_vector_relative(row: dict, roi: int) -> np.ndarray:
    """local_shared (36) with the 7 relative features appended (43 total)."""
    base = feature_vector(row, roi, "local_shared")
    rel = relative_features(row["features"], roi, row["valid"])
    return np.concatenate((base, rel))


def _design_relative(rows: list[dict]):
    candidates = [(j, roi) for j, row in enumerate(rows) for roi in range(12) if row["valid"][roi]]
    x = np.empty((len(candidates), 43), dtype=np.float32)
    y = np.empty(len(candidates), dtype=np.float32)
    row_ids = []
    for idx, (j, roi) in enumerate(candidates):
        row = rows[j]
        x[idx] = feature_vector_relative(row, roi)
        y[idx] = row["errors"][roi]
        row_ids.append(j)
    return x, y, np.asarray(row_ids), np.asarray([roi for _, roi in candidates])


def fit_predict_design(rows, x, y, row_ids, roi_ids, *, folds: int, seed: int, shuffle_targets: bool, model_params: dict):
    """Subject-grouped cross-validated fit/predict over a precomputed design matrix."""
    HistGradientBoostingRegressor, GroupKFold = _lazy_sklearn()
    hop_groups = np.array([r["subject_id"] for r in rows])
    splitter = GroupKFold(n_splits=folds)
    predictions = np.full((len(rows), 12), np.nan, dtype=float)
    fold_membership = []
    params = dict(model_params)
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


def score_arms(rows, predictions: dict[str, np.ndarray]):
    result = []
    for index, row in enumerate(rows):
        valid = row["valid"]
        order = action_order(row["errors"], valid)
        oracle = int(order[0])
        choices = {
            "local_shared": choose_valid(predictions["local_shared"][index], valid),
            "local_shared_relative": choose_valid(predictions["local_shared_relative"][index], valid),
            "local_shared_relative_shuffled": choose_valid(predictions["local_shared_relative_shuffled"][index], valid),
            "ppo_factual": int(row["ppo_action"]),
            "majority": 0 if valid[0] else int(np.flatnonzero(valid)[0]),
            "oracle": oracle,
        }
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
    for left, right in (("local_shared_relative", "local_shared"),
                        ("local_shared_relative", "local_shared_relative_shuffled"),
                        ("local_shared_relative", "ppo_factual")):
        diff = draws[left] - draws[right]
        paired[f"{left}_minus_{right}"] = {"point_bpm": report[left]["mean_immediate_regret_bpm"] - report[right]["mean_immediate_regret_bpm"],
                                            "ci_95_bpm": _ci(diff), "hypothesis_holds": bool(_ci(diff)[1] < 0)}
    return {"arms": report, "paired": paired, "bootstrap": {"replicates": replicates, "seed": seed}}


CONTRACT = {
    "question": "Do within-hop relative features, derived from the existing 101 observation values, reduce mean "
                "immediate regret below the local_shared baseline of 1.7417 BPM?",
    "required_arms": ["local_shared", "local_shared_relative", "local_shared_relative_shuffled",
                       "ppo_factual", "majority", "oracle"],
    "negative_control": "training-target shuffle for the relative arm, matching the existing probe's control "
                         "(permute complete 12-error vectors among training hops only)",
    "interpretation_limits": [
        "these seven relative features did not demonstrate improved regret under this fixed model and cohort; "
        "other representations remain untested",
        "a null result cannot establish that the fixed model already extracts equivalent information or rule out "
        "a useful re-expression",
        "single-hop counterfactuals are never continued",
        "free hops are conditioned on PPO's own switching",
        "subject identity is derived from the clip-id prefix rather than joined from per_hop.csv",
    ],
    "stop_next_rule": "this screen alone does not justify observation regeneration or retraining. These seven "
                       "relative features did not demonstrate improved regret under this fixed model and cohort; "
                       "other representations remain untested. Do not infer from this screen that re-expression "
                       "cannot help.",
}
BASELINE_REFERENCE_BPM = 1.7417


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)

    rows, source, drop_counts = load_rows(args.shard_root)
    if not rows:
        raise ValueError("no retained hops remain")
    subjects = sorted({r["subject_id"] for r in rows})
    model_params = _model_params()

    x_local, y_local, row_ids_local, roi_ids_local = _design(rows, "local_shared")
    pred_local, folds_local = fit_predict_design(rows, x_local, y_local, row_ids_local, roi_ids_local,
                                                  folds=args.folds, seed=args.seed, shuffle_targets=False,
                                                  model_params=model_params)

    x_rel, y_rel, row_ids_rel, roi_ids_rel = _design_relative(rows)
    pred_rel, folds_rel = fit_predict_design(rows, x_rel, y_rel, row_ids_rel, roi_ids_rel,
                                              folds=args.folds, seed=args.seed, shuffle_targets=False,
                                              model_params=model_params)
    pred_rel_shuffled, folds_rel_shuffled = fit_predict_design(rows, x_rel, y_rel, row_ids_rel, roi_ids_rel,
                                                                folds=args.folds, seed=args.seed, shuffle_targets=True,
                                                                model_params=model_params)

    scored = score_arms(rows, {"local_shared": pred_local, "local_shared_relative": pred_rel,
                                "local_shared_relative_shuffled": pred_rel_shuffled})
    results = summarize(scored, subjects, replicates=args.bootstrap_replicates, seed=args.seed)

    reproduced = results["arms"]["local_shared"]["mean_immediate_regret_bpm"]
    baseline_reproduction = {"reference_bpm": BASELINE_REFERENCE_BPM, "reproduced_bpm": reproduced,
                             "difference_bpm": reproduced - BASELINE_REFERENCE_BPM}

    output = {"schema": "relative-feature-screen-report-v1", "contract": CONTRACT,
              "baseline_reproduction": baseline_reproduction,
              "cohort": {"retained_hop_count": len(rows),
                        "clip_count": len({r["clip_id"] for r in rows}),
                        "subject_count": len(subjects)},
              "drop_counts": drop_counts, "results": results, "subjects": subjects,
              "provenance": {"source": source, "seed": args.seed, "folds": args.folds,
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "model_parameters": model_params,
                            "config": {"path": str(CONFIG_PATH), "sha256": _sha256(CONFIG_PATH)},
                            "fold_membership": {"local_shared": folds_local,
                                                "local_shared_relative": folds_rel,
                                                "local_shared_relative_shuffled": folds_rel_shuffled},
                            "feature_layouts": {"local_shared": 36, "local_shared_relative": 43},
                            "subject_derivation": "clip_id prefix before the first underscore",
                            "script_sha256": _sha256(Path(__file__)),
                            "run_date_utc": datetime.now(timezone.utc).isoformat(),
                            "python_version": platform.python_version(), "numpy_version": np.__version__,
                            "sklearn_version": __import__("sklearn").__version__}}
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("output must be a fresh path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
