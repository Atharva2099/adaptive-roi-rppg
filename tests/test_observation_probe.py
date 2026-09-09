import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, "scripts")
import probe_mcd_observation_ranking as probe
from probe_mcd_observation_ranking import (OBS_COLS, OBS_FIELDS, ALT_ERR_COLS, HOP_FIELDS, action_order,
    bootstrap_subject_metrics, equal_clip_mean_for_subject_draw, free_choice_mask, load_probe_rows,
    subject_folds, valid_roi_mask, fit_predict)
from probe_mcd_observation_ranking import feature_vector, _design


class ObservationProbeTests(unittest.TestCase):
    class FakeEstimator:
        calls = []
        def __init__(self, **kwargs):
            self.__class__.calls.append(kwargs)
            self.value = 0.0
        def fit(self, x, y):
            self.value = float(np.mean(y))
            return self
        def predict(self, x):
            return np.full(len(x), self.value)

    class FakeGroupKFold:
        def __init__(self, n_splits):
            self.n_splits = n_splits
        def split(self, x, groups):
            groups = np.asarray(groups)
            unique = sorted(set(groups))
            for subject in unique[:self.n_splits]:
                test = np.flatnonzero(groups == subject)
                train = np.flatnonzero(groups != subject)
                yield train, test

    @classmethod
    def _rows(cls):
        rows = []
        for subject, clip, offset in (("s0", "c0", 0.0), ("s1", "c1", 1.0)):
            features = np.zeros(101); features[6::7] = 1.0
            rows.append({"subject_id": subject, "clip_id": clip, "hop_idx": 0,
                         "features": features, "errors": np.arange(12, dtype=float) + offset,
                         "valid": np.ones(12, dtype=bool), "ppo_action": 0})
        return rows
    def test_free_mask_uses_hop_zero_or_hold_threshold(self):
        obs = np.zeros((4, 101))
        obs[:, 100] = [0.0, 0.1, 0.2, 0.9]
        self.assertEqual(free_choice_mask(obs, np.array([0, 1, 1, 2])).tolist(), [True, False, True, True])

    def test_validity_positions_and_oracle_order(self):
        obs = np.zeros((1, 101)); obs[0, [6, 13, 20]] = [1, 1, 0]
        valid = valid_roi_mask(obs)[0]
        self.assertEqual(valid[:4].tolist(), [True, True, False, False])
        self.assertEqual(action_order(np.array([2., 1., 0., 1.]), valid[:4]).tolist(), [1, 0, 2, 3])

    def test_subject_bootstrap_is_deterministic(self):
        rows = [{"arm": a, "subject_id": s, "clip_id": f"{s}-c", "regret_bpm": float(i), "top1": 0, "top3": 0}
                for a in ("probe", "oracle") for i, s in enumerate(("s0", "s1"))]
        first = bootstrap_subject_metrics(rows, ["s0", "s1"], replicates=30, seed=8101)
        second = bootstrap_subject_metrics(rows, ["s0", "s1"], replicates=30, seed=8101)
        self.assertTrue(np.array_equal(first["probe"], second["probe"]))
        self.assertTrue(np.array_equal(first["oracle"], second["oracle"]))

    def test_equal_clip_weighting_with_unequal_subject_clip_counts(self):
        self.assertEqual(equal_clip_mean_for_subject_draw([[0.0], [9.0, 9.0]], np.array([0, 1])), 6.0)

    def test_invalid_factual_action_gets_no_top_three_credit(self):
        errors = np.arange(12, dtype=float)
        valid = np.ones(12, dtype=bool); valid[11] = False
        order = action_order(errors, valid)
        self.assertNotIn(11, set(order[valid[order]][:3]))

    def test_loader_joins_representative_observation_and_selector_row(self):
        import csv
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard-0"; shard.mkdir()
            values = ["0"] * 101
            values[6] = "1"; values[13] = "1"
            observation = {"clip_id": "c0", "hop_idx": "0", **dict(zip(OBS_COLS, values)),
                           **dict(zip(ALT_ERR_COLS, ["1"] * 12)), "executed_action": "0", "gt_hr_bpm": "70"}
            with (shard / "observations_advantage_ppo_seed1.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=OBS_FIELDS); writer.writeheader(); writer.writerow(observation)
            hop = dict(zip(HOP_FIELDS, ["advantage_ppo_seed1", "advantage_ppo", "1", "s0", "c0", "0", "0", "70", "1", "0", "0", "0", "True"]))
            with (shard / "per_hop.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HOP_FIELDS); writer.writeheader(); writer.writerow(hop)
            rows, subjects, provenance = load_probe_rows(tmp)
            self.assertEqual((len(rows), subjects, provenance["joined_rows"]), (1, ["s0"], 1))

    def test_config_parameters_exclude_model_name(self):
        self.FakeEstimator.calls.clear()
        config = probe._read_config(__import__("pathlib").Path("configs/evaluation/mcd_observation_probe_v1.json"))
        model_params = {key: value for key, value in config["model"].items() if key != "name"}
        with mock.patch.object(probe, "_lazy_sklearn", return_value=(self.FakeEstimator, self.FakeGroupKFold)):
            fit_predict(self._rows(), folds=2, seed=8101,
                        model_params=model_params)
        self.assertEqual(len(self.FakeEstimator.calls), 2)
        self.assertTrue(all("name" not in call for call in self.FakeEstimator.calls))
        self.assertEqual(self.FakeEstimator.calls[0]["learning_rate"], 0.1)

    def test_feature_layouts_have_expected_candidate_information(self):
        rows = self._rows()
        full = feature_vector(rows[0], 2, "full_101")
        local = feature_vector(rows[0], 2, "local_shared")
        self.assertEqual(len(full), 113)
        self.assertEqual(len(local), 36)
        self.assertTrue(np.array_equal(local[:7], rows[0]["features"][14:21]))
        self.assertTrue(np.array_equal(local[7:24], rows[0]["features"][84:101]))
        self.assertTrue(np.array_equal(local[24:], np.eye(12)[2]))


    @unittest.skipUnless(__import__("importlib.util").util.find_spec("sklearn"), "scikit-learn optional dependency not installed")
    def test_subject_folds_are_disjoint(self):
        folds = list(subject_folds([f"s{i}" for i in range(6)], 3))
        self.assertEqual(set().union(*(set(test) for _, test in folds)), {f"s{i}" for i in range(6)})
        for train, test in folds:
            self.assertFalse(set(train) & set(test))

    def test_oracle_regret_is_zero(self):
        errors = np.array([3., 1., 4.]); valid = np.array([True, True, False])
        order = action_order(errors, valid)
        self.assertEqual(errors[order[0]] - errors[order[0]], 0.0)


if __name__ == "__main__":
    unittest.main()
