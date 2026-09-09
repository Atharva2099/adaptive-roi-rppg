import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import build_observation, control_step, initial_control_state
from adaptive_roi_rppg.evaluation.model_publication import canonical_subject_shard, expected_shard_clip_ids
from adaptive_roi_rppg.evaluation.simple_selectors import MaxPprSelector, RandomLegalSelector, rollout_simple_selector, score_simple_rollout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_mcd_simple_selectors import _aggregate, _behavioral_report, _merge_shard_accums, _write_shard_accum, bootstrap_equal_clip_mae, paired_bootstrap_difference


def measurement(i, ppr=0.5, confidence=0.5, valid=True):
    return ROIMeasurement(i, ROI_NAMES[i], 70.0 + i if valid else None, confidence if valid else None,
                          ppr if valid else None, 0.8 if valid else 0.0, valid, None if valid else "missing", (), None, 0, 1, "sig")


def frame(hop=0, values=None):
    values = tuple(values or (measurement(i) for i in range(12)))
    return MeasurementFrame("mcd", "clip", hop, float(hop), values, "sig", any(x.valid for x in values), None if any(x.valid for x in values) else "none", f"p-{hop}")


class SimpleSelectorTests(unittest.TestCase):
    def test_max_ppr_ranking_and_index_tie(self):
        values = tuple(measurement(i, ppr=0.9 if i in (3, 5) else 0.1) for i in range(12))
        self.assertEqual(MaxPprSelector().requested_action(frame(values=values), initial_control_state("mcd", "clip")), 3)

    def test_current_tied_roi_is_kept(self):
        values = tuple(measurement(i, ppr=0.9 if i in (3, 5) else 0.1) for i in range(12))
        state, _ = control_step(frame(values=values), initial_control_state("mcd", "clip"), 5)
        self.assertEqual(MaxPprSelector().requested_action(frame(1, values), state), 5)

    def test_invalid_and_all_invalid_fallback(self):
        values = tuple(measurement(i, valid=False) for i in range(12))
        selector = MaxPprSelector(); state = initial_control_state("mcd", "clip")
        self.assertEqual(selector.requested_action(frame(values=values), state), 0)
        state, _ = control_step(frame(values=tuple(measurement(i) for i in range(12))), state, 4)
        self.assertEqual(selector.requested_action(frame(1, values=values), state), 4)

    def test_ppr_ranking_diverges_from_confidence_ranking(self):
        # ROI 0: high confidence via high coverage, but low PPR. ROI 7: low
        # confidence, but the highest PPR. Regression guard for the bug where
        # confidence = ppr * coverage let full_face win on coverage alone.
        values = tuple(
            measurement(0, ppr=0.05, confidence=0.9) if i == 0 else
            measurement(7, ppr=0.95, confidence=0.1) if i == 7 else
            measurement(i, ppr=0.2, confidence=0.2)
            for i in range(12)
        )
        self.assertEqual(MaxPprSelector().requested_action(frame(values=values), initial_control_state("mcd", "clip")), 7)

    def test_rollout_has_exact_observation_and_alternatives(self):
        hop = rollout_simple_selector([frame()], MaxPprSelector(), clip_id="clip")[0]
        self.assertEqual(len(hop.observation_values), 101)
        self.assertEqual(hop.alternative_count, 12)

    def test_random_is_reproducible_and_legal_under_hold(self):
        policy = RandomLegalSelector(8101); state, _ = control_step(frame(), initial_control_state("mcd", "clip"), 4)
        self.assertEqual(policy.requested_action(frame(1), state), 4)
        self.assertEqual(policy.requested_action(frame(1), state), policy.requested_action(frame(1), state))

    def test_empty_and_nan_label_rejected(self):
        with self.assertRaises(ContractValidationError): rollout_simple_selector([], MaxPprSelector(), clip_id="clip")
        hop = rollout_simple_selector([frame()], MaxPprSelector(), clip_id="clip")
        class Label:
            dataset_id = "mcd"; clip_id = "clip"; hop_idx = 0; hop_time_s = 0.0; valid = True; gt_hr_bpm = float("nan")
        with self.assertRaises(ContractValidationError): score_simple_rollout(hop, [Label()], {"subject_id": "s", "view": "v", "condition": "c"})

    def test_observation_values_are_float32_not_float64(self):
        state = initial_control_state("mcd", "clip")
        hop = rollout_simple_selector([frame()], MaxPprSelector(), clip_id="clip")[0]
        expected = build_observation(frame(), state).array()
        self.assertTrue(np.array_equal(np.asarray(hop.observation_values, dtype=np.float32), expected))
        self.assertTrue(all(v == np.float32(v) for v in hop.observation_values))

    def test_bootstrap_is_seed_reproducible(self):
        clips = [{"method_id": "a", "subject_id": "s0", "clip_id": "c0", "mae_bpm": 1.0},
                  {"method_id": "a", "subject_id": "s1", "clip_id": "c1", "mae_bpm": 3.0},
                  {"method_id": "b", "subject_id": "s0", "clip_id": "c0", "mae_bpm": 2.0},
                  {"method_id": "b", "subject_id": "s1", "clip_id": "c1", "mae_bpm": 4.0}]
        subjects = ["s0", "s1"]
        first = bootstrap_equal_clip_mae(clips, subjects, ("a", "b"), replicates=200, seed=8101)
        second = bootstrap_equal_clip_mae(clips, subjects, ("a", "b"), replicates=200, seed=8101)
        third = bootstrap_equal_clip_mae(clips, subjects, ("a", "b"), replicates=200, seed=8102)
        self.assertTrue(np.array_equal(first["a"], second["a"]))
        self.assertTrue(np.array_equal(first["b"], second["b"]))
        self.assertFalse(np.array_equal(first["a"], third["a"]))

    def test_paired_bootstrap_subtracts_within_shared_draws(self):
        draws = {"max_ppr": np.array([3.0, 7.0]), "full_face": np.array([2.0, 10.0])}
        self.assertTrue(np.array_equal(
            paired_bootstrap_difference(draws, "max_ppr", "full_face"),
            np.array([1.0, -3.0]),
        ))

    def test_equal_clip_aggregation_weights_clips_not_hops(self):
        clip_accum = {("m", "s0", "c0"): [10.0, 1], ("m", "s0", "c1"): [0.0, 100]}
        _, methods = _aggregate(clip_accum)
        self.assertAlmostEqual(methods["m"]["equal_clip_mae_bpm"], 5.0)

    def test_shard_assignment_covers_every_clip_once_and_is_subject_disjoint(self):
        clips = [{"subject_id": f"s{i % 7}", "clip_id": f"c{i:03d}"} for i in range(23)]
        shard_count = 4
        shard_clip_ids = [expected_shard_clip_ids(clips, index, shard_count) for index in range(shard_count)]
        shard_subjects = [set(canonical_subject_shard(sorted({c["subject_id"] for c in clips}), index, shard_count)) for index in range(shard_count)]
        all_ids = [clip_id for shard in shard_clip_ids for clip_id in shard]
        self.assertEqual(sorted(all_ids), sorted(c["clip_id"] for c in clips))
        self.assertEqual(len(set(all_ids)), len(clips))
        for i in range(shard_count):
            for j in range(i + 1, shard_count):
                self.assertEqual(shard_subjects[i] & shard_subjects[j], set())

    def test_merge_equivalence_matches_unsharded_pass(self):
        clip_accum_a = {("full_face", "s0", "c0"): [4.0, 2], ("max_ppr", "s0", "c0"): [6.0, 2]}
        behavior_a = {"full_face": {"switch": 1, "switch_denom": 1, "override": 0, "invalid": 0, "hop_count": 2, "hist": [1, 1] + [0] * 10}}
        clip_accum_b = {("full_face", "s1", "c1"): [9.0, 3], ("max_ppr", "s1", "c1"): [3.0, 3]}
        behavior_b = {"full_face": {"switch": 0, "switch_denom": 2, "override": 1, "invalid": 1, "hop_count": 3, "hist": [0, 3] + [0] * 10}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a, shard_b = root / "shard-a", root / "shard-b"; shard_a.mkdir(); shard_b.mkdir()
            _write_shard_accum(shard_a, clip_accum_a, behavior_a)
            _write_shard_accum(shard_b, clip_accum_b, behavior_b)
            merged_clip_accum, merged_behavior = _merge_shard_accums([shard_a, shard_b])

        unsharded_clip_accum = {**clip_accum_a, **clip_accum_b}
        _, expected_methods = _aggregate(unsharded_clip_accum)
        _, merged_methods = _aggregate(merged_clip_accum)
        self.assertEqual(merged_methods, expected_methods)

        expected_full_face = {"switch": 1, "switch_denom": 3, "override": 1, "invalid": 1, "hop_count": 5, "hist": [1, 4] + [0] * 10}
        self.assertEqual(merged_behavior["full_face"], expected_full_face)
        self.assertEqual(_behavioral_report(merged_behavior)["full_face"], _behavioral_report({"full_face": expected_full_face})["full_face"])

    def test_merge_rejects_duplicated_key_across_shards(self):
        clip_accum = {("full_face", "s0", "c0"): [4.0, 2]}
        behavior = {"full_face": {"switch": 0, "switch_denom": 0, "override": 0, "invalid": 0, "hop_count": 2, "hist": [2] + [0] * 11}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a, shard_b = root / "shard-a", root / "shard-b"; shard_a.mkdir(); shard_b.mkdir()
            _write_shard_accum(shard_a, clip_accum, behavior)
            _write_shard_accum(shard_b, clip_accum, behavior)
            with self.assertRaises(ContractValidationError):
                _merge_shard_accums([shard_a, shard_b])


if __name__ == "__main__":
    unittest.main()
