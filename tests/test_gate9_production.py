import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.adapters.mmpd.plan import Gate9Plan, build_engineering_plan
from adaptive_roi_rppg.evaluation.adapters.mmpd.production import (
    MMPD_GT_RULE_ID,
    _FullFacePolicy,
    _metadata,
    _oracle_rows,
    build_mmpd_labels,
)
from adaptive_roi_rppg.evaluation.adapters.mmpd.publication import HOP_FIELDS, fail_gate9_evaluation, start_gate9_evaluation, validate_gate9_evaluation_tree
from adaptive_roi_rppg.evaluation.adapters.mmpd.replay import rollout_gate9_policy
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import _face_landmarks_from_result


def _plan() -> Gate9Plan:
    payload = build_engineering_plan()
    return Gate9Plan(payload, payload["plan_id"], "a" * 64)


def _frame(clip_id: str, hop: int, value: float = 70.0) -> MeasurementFrame:
    measurements = tuple(ROIMeasurement(i, ROI_NAMES[i], value + i, 1.0, 0.5, 1.0, True, None, (), None, hop, hop + 1, "synthetic-pos") for i in range(12))
    return MeasurementFrame("mmpd", clip_id, hop, 8.0 + hop, measurements, "synthetic-pos", True, None, f"synthetic-{clip_id}")


class Gate9LabelRuleTests(unittest.TestCase):
    def test_known_signal_selects_expected_band_peak(self):
        samples = np.arange(1800, dtype=np.float64) / 30.0
        labels = build_mmpd_labels(np.sin(2.0 * math.pi * 1.2 * samples), clip_id="p1_0")
        self.assertEqual(len(labels), 53)
        self.assertTrue(all(label.valid for label in labels))
        self.assertTrue(all(abs(label.gt_hr_bpm - 72.0) < 1.0 for label in labels))
        self.assertTrue(all(label.gt_rule_id == MMPD_GT_RULE_ID for label in labels))

    def test_invalid_signal_is_explicitly_invalid(self):
        labels = build_mmpd_labels(np.zeros(1800), clip_id="p1_0")
        self.assertEqual(len(labels), 53)
        self.assertTrue(all(not label.valid and label.gt_hr_bpm is None for label in labels))


class Gate9LocalReplayTests(unittest.TestCase):
    def test_full_face_identity_and_clip_reset(self):
        policy = _FullFacePolicy()
        first = rollout_gate9_policy((_frame("p1_0", 0),), policy, clip_id="p1_0")[0]
        second = rollout_gate9_policy((_frame("p1_1", 0),), policy, clip_id="p1_1")[0]
        self.assertEqual(policy.identity.family, "fixed_full_face")
        self.assertIsNone(policy.identity.checkpoint_sha256)
        self.assertEqual(first["executed_action"], 0)
        self.assertEqual(second["pre_hold_count"], 0)
        self.assertTrue(first["causal_reset"] and second["causal_reset"])
        self.assertEqual(first["gt_observation_count"], 0)

    def test_mmpd_oracle_rows_match_publication_shape(self):
        frames = tuple(_frame("p1_0", hop) for hop in range(53))
        labels = tuple(LabelFrame("mmpd", "p1_0", hop, 8.0 + hop, 70.0, MMPD_GT_RULE_ID, True, None) for hop in range(53))
        clip = {"clip_id": "p1_0", "subject_id": "1", "view": "frontal", "condition": "rest"}
        plan = _plan()
        rows = _oracle_rows(frames, labels, clip, plan, "oracle_b_greedy")
        self.assertEqual(len(rows), 53)
        self.assertTrue(all(set(row) == set(HOP_FIELDS) for row in rows))
        self.assertTrue(all(row["dataset_id"] == "mmpd" and row["gt_observation_count"] == 0 for row in rows))


class Gate9ProductionInputTests(unittest.TestCase):
    def test_mediapipe_tasks_result_uses_face_landmarks(self):
        self.assertEqual(_face_landmarks_from_result(SimpleNamespace(face_landmarks=["face"])), ["face"])
        with self.assertRaises(ContractValidationError):
            _face_landmarks_from_result(SimpleNamespace(multi_face_landmarks=["legacy-field"]))

    def test_launcher_requires_model_hash_and_path(self):
        launcher = Path("slurm/gate9_mmpd_engineering.slurm").read_text(encoding="utf-8")
        cli = Path("scripts/evaluate_mmpd_gate9.py").read_text(encoding="utf-8")
        self.assertIn("FACE_LANDMARKER_MODEL", launcher)
        self.assertIn("FACE_LANDMARKER_MODEL_SHA256", launcher)
        self.assertIn("--face-landmarker-model-sha256", cli)

    def test_callable_metadata_rejects_missing_authenticated_input(self):
        with self.assertRaises(ContractValidationError):
            _metadata({}, _plan())

    def test_callable_metadata_rejects_incomplete_inventory(self):
        with self.assertRaises(ContractValidationError):
            _metadata({"rule_root": "/tmp", "metadata_coding_locator": "missing.json"}, _plan())

    def test_failed_partial_publication_is_quarantined_to_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = start_gate9_evaluation(Path(directory) / "run", {"plan_id": "synthetic"})
            (root / "per_hop.csv").write_bytes(b"partial")
            fail_gate9_evaluation(root, RuntimeError("synthetic"))
            validate_gate9_evaluation_tree(root, require_complete=False)
            self.assertEqual({item.name for item in root.iterdir()}, {"STARTED.json", "FAILED.json"})


if __name__ == "__main__":
    unittest.main()
