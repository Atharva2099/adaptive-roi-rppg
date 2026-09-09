import sys
import unittest

import numpy as np

sys.path.insert(0, "scripts")
from probe_mcd_observation_ranking import action_order
from probe_best_roi_persistence import adjacent_true_pairs, shuffled_statistics


class BestRoiPersistenceTests(unittest.TestCase):
    def test_adjacency_skips_gaps_and_never_crosses_clips(self):
        rows = [
            {"clip_id": "c0", "hop_idx": 0}, {"clip_id": "c0", "hop_idx": 1},
            {"clip_id": "c0", "hop_idx": 3}, {"clip_id": "c1", "hop_idx": 1},
        ]
        pairs = adjacent_true_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0][0]["hop_idx"], pairs[0][1]["hop_idx"]), (0, 1))
        self.assertTrue(all(a["clip_id"] == b["clip_id"] for a, b in pairs))

    def test_best_roi_ties_break_low_and_skip_invalid(self):
        errors = np.array([1.0, 1.0, 0.5, 2.0])
        valid = np.array([True, True, False, True])
        best = int(action_order(errors, valid)[0])
        self.assertEqual(best, 0)  # tie between 0 and 1 -> lowest index; ROI 2 is invalid despite lower error

    def test_shuffled_control_reproducible_and_seed_sensitive(self):
        rows = [{"clip_id": "c0", "hop_idx": i, "best": i % 3,
                 "errors": np.arange(12, dtype=float) + i, "valid": np.ones(12, dtype=bool)}
                for i in range(6)]
        _, a1, b1, _ = shuffled_statistics(rows, repeats=5, seed=8101)
        _, a2, b2, _ = shuffled_statistics(rows, repeats=5, seed=8101)
        _, a3, b3, _ = shuffled_statistics(rows, repeats=5, seed=9)
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)
        self.assertNotEqual((a1, b1), (a3, b3))

    def test_analytic_chance_level_matches_sum_of_squared_marginals(self):
        marginal = np.array([4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        p_k = marginal / marginal.sum()
        chance_a = float(np.sum(p_k ** 2))
        self.assertAlmostEqual(chance_a, (0.5 ** 2) + (0.25 ** 2) + (0.25 ** 2))


if __name__ == "__main__":
    unittest.main()
