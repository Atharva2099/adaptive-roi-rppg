import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from probe_mcd_observation_ranking import OBS_FIELDS, feature_vector
from probe_relative_feature_screen import (feature_vector_relative, load_rows,
    relative_features)


class RelativeFeatureScreenTests(unittest.TestCase):
    @classmethod
    def _features(cls):
        # 3 valid ROIs (0, 1, 2) with distinct hr_norm/ppr/coverage, ROI 3 invalid.
        features = np.zeros(101)
        for roi, (hr, ppr, coverage) in enumerate(((0.2, 0.9, 0.5), (0.5, 0.1, 0.8), (0.8, 0.5, 0.2))):
            features[7 * roi + 0] = hr
            features[7 * roi + 3] = ppr
            features[7 * roi + 4] = coverage
            features[7 * roi + 6] = 1.0
        features[7 * 3 + 6] = 0.0  # ROI 3 invalid
        valid = np.zeros(12, dtype=bool)
        valid[[0, 1, 2]] = True
        return features, valid

    def test_ppr_rank_extremes_exclude_invalid_rois(self):
        features, valid = self._features()
        # ROI 0 has the highest ppr (0.9) -> rank 0; ROI 1 has the lowest (0.1) -> rank 1.0.
        self.assertEqual(relative_features(features, 0, valid)[0], 0.0)
        self.assertEqual(relative_features(features, 1, valid)[0], 1.0)

    def test_zscore_zero_when_degenerate_and_finite_otherwise(self):
        features = np.zeros(101)
        for roi in range(3):
            features[7 * roi + 0] = 0.5
            features[7 * roi + 3] = 0.5
            features[7 * roi + 6] = 1.0
        valid = np.zeros(12, dtype=bool)
        valid[[0, 1, 2]] = True
        out = relative_features(features, 0, valid)
        self.assertEqual(out[1], 0.0)  # ppr z-score
        self.assertEqual(out[2], 0.0)  # hr z-score
        self.assertTrue(np.all(np.isfinite(out)))

        varied_features, varied_valid = self._features()
        varied_out = relative_features(varied_features, 0, varied_valid)
        self.assertTrue(np.all(np.isfinite(varied_out)))
        self.assertNotEqual(varied_out[1], 0.0)
        self.assertNotEqual(varied_out[2], 0.0)

    def test_relative_vector_is_43_long_and_prefix_matches_local_shared(self):
        features, valid = self._features()
        row = {"subject_id": "s0", "clip_id": "c0", "hop_idx": 0, "features": features, "valid": valid,
               "errors": np.arange(12, dtype=float), "ppo_action": 0}
        local = feature_vector(row, 0, "local_shared")
        relative = feature_vector_relative(row, 0)
        self.assertEqual(len(relative), 43)
        self.assertTrue(np.array_equal(relative[:36], local))

    def test_relative_features_never_read_alt_err_or_gt_hr(self):
        features, valid = self._features()
        row = {"subject_id": "s0", "clip_id": "c0", "hop_idx": 0, "features": features, "valid": valid,
               "errors": np.arange(12, dtype=float), "ppo_action": 0}
        before = relative_features(row["features"], 0, row["valid"])
        row["errors"] = row["errors"] + 1000.0
        row["gt_hr_bpm"] = 9999.0
        after = relative_features(row["features"], 0, row["valid"])
        self.assertTrue(np.array_equal(before, after))

    def test_load_rows_orders_by_clip_id_then_hop_idx(self):
        features, _ = self._features()
        # Rows written to the shard CSV out of (clip_id, hop_idx) order.
        keys = [("subjB_clipY", 2), ("subjA_clipX", 0), ("subjA_clipX", 1), ("subjB_clipY", 0)]
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shard-0"
            shard_dir.mkdir()
            csv_path = shard_dir / "observations_advantage_ppo_seed1.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=OBS_FIELDS)
                writer.writeheader()
                for clip_id, hop_idx in keys:
                    record = {"clip_id": clip_id, "hop_idx": hop_idx, "executed_action": 0, "gt_hr_bpm": 70.0}
                    for i, value in enumerate(features):
                        record[f"obs_{i:03d}"] = value
                    record["obs_100"] = 1.0  # satisfies free_choice_mask at hop_idx > 0
                    for i in range(12):
                        record[f"alt_err_{i:02d}"] = 1.0
                    writer.writerow(record)
            rows, _, _ = load_rows(tmp)
        self.assertEqual([(r["clip_id"], r["hop_idx"]) for r in rows],
                          sorted((clip_id, hop_idx) for clip_id, hop_idx in keys))


if __name__ == "__main__":
    unittest.main()
