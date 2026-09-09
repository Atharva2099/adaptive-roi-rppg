#!/usr/bin/env python3
"""Is the lowest-error ROI persistent between adjacent hops, or a coin flip?

Offline diagnostic over the PPO observation shards.  Reuses the free-hop and
ROI-validity definitions from probe_mcd_observation_ranking.py.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mcd_observation_ranking import (OBS_COLS, ALT_ERR_COLS, OBS_FIELDS,
    action_order, free_choice_mask, valid_roi_mask)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_rows(root: str | Path):
    """Load, validate, and filter every shard's observations; return retained rows and drop counts."""
    paths = sorted(Path(root).glob("shard-*/observations_advantage_ppo_seed1.csv"))
    if not paths:
        raise ValueError("shard root has no shard-*/observations_advantage_ppo_seed1.csv files")
    seen = {}
    source = []
    counts = {"nonfinite_obs": 0, "locked": 0, "no_valid": 0, "nonfinite_errors": 0}
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
                if not np.all(np.isfinite(obs)):
                    counts["nonfinite_obs"] += 1
                    continue
                if not free_choice_mask(obs[None, :], np.array([key[1]]))[0]:
                    counts["locked"] += 1
                    continue
                valid = valid_roi_mask(obs)[0]
                if not np.any(valid):
                    counts["no_valid"] += 1
                    continue
                if not np.all(np.isfinite(errors[valid])):
                    counts["nonfinite_errors"] += 1
                    continue
                best = int(action_order(errors, valid)[0])
                rows.append({"clip_id": key[0], "hop_idx": key[1], "errors": errors, "valid": valid, "best": best})
    return rows, source, counts

def adjacent_true_pairs(rows: list[dict]):
    """Pair retained hops that share a clip and have consecutive hop_idx values."""
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    pairs = []
    for clip_rows in by_clip.values():
        ordered = sorted(clip_rows, key=lambda r: r["hop_idx"])
        for a, b in zip(ordered, ordered[1:]):
            if b["hop_idx"] - a["hop_idx"] == 1:
                pairs.append((a, b))
    return pairs

def statistics(pairs: list[tuple[dict, dict]]):
    """Compute agreement fraction A and mean Spearman correlation B over adjacent pairs."""
    if not pairs:
        return 0.0, 0.0, 0
    agree = sum(a["best"] == b["best"] for a, b in pairs) / len(pairs)
    correlations = []
    skipped = 0
    for a, b in pairs:
        shared = a["valid"] & b["valid"]
        if int(np.sum(shared)) < 3:
            skipped += 1
            continue
        rho, _ = spearmanr(a["errors"][shared], b["errors"][shared])
        if not np.isfinite(rho):
            skipped += 1
            continue
        correlations.append(rho)
    mean_b = float(np.mean(correlations)) if correlations else 0.0
    return agree, mean_b, skipped

def shuffled_statistics(rows: list[dict], repeats: int, seed: int):
    """Repeat the within-clip hop-order shuffle and score A/B each time."""
    by_clip = defaultdict(list)
    for row in rows:
        by_clip[row["clip_id"]].append(row)
    rng = np.random.default_rng(seed)
    a_vals, b_vals, skip_vals = [], [], []
    for _ in range(repeats):
        pairs = []
        for clip_rows in by_clip.values():
            order = rng.permutation(len(clip_rows))
            permuted = [clip_rows[i] for i in order]
            pairs.extend(zip(permuted, permuted[1:]))
        a, b, skipped = statistics(pairs)
        a_vals.append(a); b_vals.append(b); skip_vals.append(skipped)
    return by_clip, a_vals, b_vals, skip_vals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    args = parser.parse_args(argv)

    rows, source, drop_counts = load_rows(args.shard_root)
    if not rows:
        raise ValueError("no retained hops remain")

    true_pairs = adjacent_true_pairs(rows)
    obs_a, obs_b, obs_b_skipped = statistics(true_pairs)

    marginal = np.array([sum(r["best"] == k for r in rows) for k in range(12)], dtype=float)
    p_k = marginal / len(rows)
    chance_a = float(np.sum(p_k ** 2))

    by_clip, shuffled_a, shuffled_b, shuffled_b_skipped = shuffled_statistics(rows, args.shuffle_repeats, args.seed)

    contract = {"question": "Is the identity of the lowest-error ROI persistent between consecutive hops within a "
                "clip, or is it consistent with independent draws?",
                "required_arms": ["observed adjacent-hop best-ROI agreement", "mean Spearman correlation of adjacent 12-error vectors"],
                "negative_control": "hop order shuffled within each clip (preserves marginals, destroys adjacency), plus analytic chance level sum_k p_k^2"}
    results = {"observed_agreement_a": obs_a, "observed_mean_spearman_b": obs_b, "observed_b_pairs_skipped": obs_b_skipped,
               "chance_level_a": chance_a, "shuffled_agreement_a_mean": float(np.mean(shuffled_a)),
               "shuffled_agreement_a_std": float(np.std(shuffled_a)), "shuffled_mean_spearman_b_mean": float(np.mean(shuffled_b)),
               "shuffled_mean_spearman_b_std": float(np.std(shuffled_b)), "shuffled_b_pairs_skipped_mean": float(np.mean(shuffled_b_skipped)),
               "adjacent_pair_count": len(true_pairs), "best_roi_marginal_histogram": marginal.astype(int).tolist()}
    cohort = {"retained_hop_count": len(rows), "clip_count": len(by_clip), "subject_count": None,
              "subject_count_note": "not available: shard root has no per_hop.csv, only observation files"}
    provenance = {"source": source, "seed": args.seed, "shuffle_repeats": args.shuffle_repeats,
                  "run_date_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
                  "numpy_version": np.__version__, "scipy_version": __import__("scipy").__version__,
                  "script_sha256": _sha256(Path(__file__))}
    output = {"schema": "best-roi-persistence-probe-report-v1", "contract": contract, "results": results,
              "cohort": cohort, "drop_counts": drop_counts, "provenance": provenance}
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("output must be a fresh path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
