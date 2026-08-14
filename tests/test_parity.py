import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.parity import (PARITY_CLIP_FIELDS, PARITY_HOP_FIELDS,
    HistoricalFullFaceHop, classify_discrepancy, compare_full_face_trajectories,
    publish_gate6_report, replay_historical_full_face,
    verify_gate6_publication)


class Gate6ReplayTests(unittest.TestCase):
    def test_known_answer_predict_only_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1107_FullHDwebcam_before.npz"
            shape = (2, 12)
            np.savez(path, stem=np.array("1107_FullHDwebcam_before"), view=np.array("Frontal"), condition=np.array("before"), fs=np.array(30.0), hr_meas=np.full(shape, np.nan), conf=np.zeros(shape), ppr=np.zeros(shape), cov=np.ones(shape), gt_hr=np.full(2, 72.0), cache_schema_version=np.array(3, dtype=np.int16), gt_subharmonic_labels=np.array(False))
            rows = replay_historical_full_face(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].hr_bpm, None)
            self.assertEqual(rows[0].post_belief_hr_bpm, 70.0)
            self.assertEqual(rows[1].post_belief_hr_bpm, 70.0)

    def test_nan_hr_with_positive_confidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"
            shape = (1, 12)
            np.savez(path, stem=np.array("x"), view=np.array("Frontal"), condition=np.array("before"), fs=np.array(30.0), hr_meas=np.full(shape, np.nan), conf=np.ones(shape), ppr=np.zeros(shape), cov=np.ones(shape), gt_hr=np.full(1, 72.0), cache_schema_version=np.array(3, dtype=np.int16), gt_subharmonic_labels=np.array(False))
            with self.assertRaises(ContractValidationError): replay_historical_full_face(path)

    def test_exact_npz_envelope_rejects_extra_key_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"
            values = dict(stem=np.array("x"), view=np.array("Frontal"), condition=np.array("before"), fs=np.array(30.0), hr_meas=np.full((1, 12), np.nan), conf=np.zeros((1, 12)), ppr=np.zeros((1, 12)), cov=np.ones((1, 12)), gt_hr=np.full(1, 72.0), cache_schema_version=np.array(3, dtype=np.int16), gt_subharmonic_labels=np.array(False))
            np.savez(path, **values, extra=np.array(1))
            with self.assertRaises(ContractValidationError): replay_historical_full_face(path)
            np.savez(path, **values)
            with self.assertRaises(ContractValidationError): replay_historical_full_face(path, "0" * 64)

    def test_variable_hops_are_explicitly_reported(self):
        historical = (HistoricalFullFaceHop(0, 8.0, 70.0, 1.0, 70.0, 70.0), HistoricalFullFaceHop(1, 9.0, 70.0, 1.0, 70.0, 70.0))
        current = ({"dataset_id": "mcd", "clip_id": "x", "subject_id": "s", "view": "Frontal", "condition": "before", "hop_idx": 0, "hop_time_s": 8.0, "hr": 70.0, "confidence": 1.0, "gt": 70.0, "valid": True, "reason": None, "belief": 70.0, "source_window_complete": True},)
        rows = compare_full_face_trajectories(historical, current)
        self.assertEqual([row["hop_presence"] for row in rows], ["joined", "historical_only"])
        self.assertEqual(rows[1]["discrepancy_category"], "hop_historical_only")

    def test_category_precedence(self):
        self.assertEqual(classify_discrepancy(None, {"valid": False}, identity_ok=False, gt_equal=False), "hop_current_only")
        self.assertEqual(classify_discrepancy(None, {"valid": False}, gt_equal=False), "hop_current_only")
        self.assertEqual(classify_discrepancy(None, {"valid": False}, historical_fill_exposure=True), "hop_current_only")
        self.assertEqual(classify_discrepancy(None, {"valid": False}), "hop_current_only")

    def test_exact_match_and_unavailable_signal_categories(self):
        h = type("H", (), {"hr_bpm": 70.0, "confidence": 0.5, "post_belief_hr_bpm": 70.0, "gt_hr_bpm": 70.0})()
        c = {"valid": True, "hr": 70.0, "confidence": 0.5, "belief": 70.0, "gt": 70.0}
        self.assertEqual(classify_discrepancy(h, c, pos_final_subwindow_equal=True, peak_bin_equal=True), "exact_observed_match")
        self.assertEqual(classify_discrepancy(h, {**c, "belief": 71.0}), "belief_value_difference")
        self.assertEqual(classify_discrepancy(h, c), "exact_observed_match")

    def test_publication_verification_and_tamper(self):
        hop = {field: 0 for field in PARITY_HOP_FIELDS}
        hop.update({"run_id": "r", "plan_id": "p", "dataset_id": "mcd", "clip_id": "c", "subject_id": "s", "view": "Frontal", "condition": "before", "hop_time_s": 1.0, "historical_full_face_hr_bpm": 70.0, "current_full_face_hr_bpm": 70.0, "full_face_hr_delta_bpm": 0.0, "historical_confidence": 1.0, "current_confidence": 1.0, "confidence_delta": 0.0, "historical_gt_hr_bpm": 70.0, "current_gt_hr_bpm": 70.0, "gt_delta_bpm": 0.0, "historical_post_belief_hr_bpm": 70.0, "current_post_belief_hr_bpm": 70.0, "belief_delta_bpm": 0.0, "historical_abs_error_bpm": 0.0, "current_abs_error_bpm": 0.0, "abs_error_delta_bpm": 0.0, "current_selected_valid": True, "current_invalid_reason": "", "source_window_complete": True, "diagnostic_status": "not_supported_by_frozen_inputs", "discrepancy_category": "exact_observed_match", "signal_config_id": "s", "control_config_id": "c", "gt_rule_id": "g", "hop_idx": 0, "historical_hop_count": 1, "current_hop_count": 1})
        clip = {field: 0 for field in PARITY_CLIP_FIELDS}; clip.update({"run_id": "r", "plan_id": "p", "dataset_id": "mcd", "clip_id": "c", "subject_id": "s", "view": "Frontal", "condition": "before", "expected_hops": 1, "joined_hops": 1, "fixture_member": True, "clip_status": "complete"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            publish_gate6_report({"run_id": "r", "plan_id": "p", "dataset_id": "mcd", "phase": "fixture", "hop_rows": [hop], "clip_rows": [clip], "subject_rows": [], "category_rows": [], "plan_payload": {"plan_id": "p"}, "source_inventory": {"files": ["synthetic"]}, "diagnostics": {"causal_decomposition_status": "not_supported_by_frozen_inputs"}}, destination)
            verify_gate6_publication(destination)
            (destination / "parity_per_hop.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ContractValidationError): verify_gate6_publication(destination)


if __name__ == "__main__": unittest.main()
