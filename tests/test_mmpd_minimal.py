import tempfile
import unittest
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.evaluation.adapters.mmpd.minimal import (
    EXPECTED_LEARNED_METHOD_IDS, EXPECTED_METHOD_IDS, FullFacePolicy, load_minimal_source,
    replay_shared_measurements, validate_checkpoint_manifest, write_clip_csv,
)
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import load_mmpd_mat
from scripts.audit_mmpd_extraction import _inventory_record, _write_rows, audit_clip
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from scripts.evaluate_mmpd_clip import clip_for_array_index


class MinimalSourceTests(unittest.TestCase):
    def _write(self, video, gt):
        handle = tempfile.NamedTemporaryFile(suffix=".mat", delete=False)
        handle.close()
        savemat(handle.name, {"video": video, "GT_ppg": gt})
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_only_terminal_zero_suffix_is_removed(self):
        video = np.ones((4, 2, 2, 3), dtype=np.float64)
        video[-2:] = 0
        source = load_minimal_source(self._write(video, np.arange(4.0)))
        self.assertEqual(source.video.shape[0], 2)
        self.assertEqual(source.trimmed_tail_frames, 2)
        np.testing.assert_array_equal(source.gt_ppg, [0.0, 1.0])

    def test_interior_zero_and_nonfinite_inputs_are_rejected(self):
        video = np.ones((4, 2, 2, 3), dtype=np.float64)
        video[1] = 0
        with self.assertRaises(Exception):
            load_minimal_source(self._write(video, np.arange(4.0)))
        video[1, 0, 0, 0] = np.nan
        with self.assertRaises(Exception):
            load_minimal_source(self._write(video, np.arange(4.0)))

    def test_production_and_minimal_loaders_trim_identically(self):
        video = np.ones((4, 2, 2, 3), dtype=np.float64)
        video[-2:] = 0
        path = self._write(video, np.arange(4.0))
        minimal = load_minimal_source(path)
        production = load_mmpd_mat(path)
        np.testing.assert_array_equal(minimal.video, production["video"])
        np.testing.assert_array_equal(minimal.gt_ppg, production["gt_ppg"])
        self.assertEqual(minimal.original_frame_count, production["original_frame_count"])
        self.assertEqual(minimal.processed_frame_count, production["processed_frame_count"])
        self.assertEqual(minimal.trimmed_tail_frames, production["trimmed_tail_frames"])

    def test_production_rejects_nonfinite_gt(self):
        video = np.ones((2, 2, 2, 3), dtype=np.float64)
        with self.assertRaises(Exception):
            load_mmpd_mat(self._write(video, [0.0, np.inf]))

    def test_both_loaders_reject_all_zero_and_length_mismatch(self):
        with self.assertRaises(Exception):
            load_minimal_source(self._write(np.zeros((2, 2, 2, 3)), [0.0, 1.0]))
        with self.assertRaises(Exception):
            load_mmpd_mat(self._write(np.zeros((2, 2, 2, 3)), [0.0, 1.0]))
        video = np.ones((2, 2, 2, 3))
        with self.assertRaises(Exception):
            load_minimal_source(self._write(video, [0.0]))
        with self.assertRaises(Exception):
            load_mmpd_mat(self._write(video, [0.0]))


class ExtractionAuditSummaryTests(unittest.TestCase):
    def test_per_frame_rows_serialize_boxes_and_std_as_json(self):
        from adaptive_roi_rppg.contracts import CanonicalFrame, ROIFrameValue
        frame = CanonicalFrame("mmpd", "p1_0", 4, 4 / 30, 30, None, None, None, tuple(
            ROIFrameValue(i, ROI_NAMES[i], 1.0, 2.0, 3.0, 4.0, 0.5, True, None, None, None) for i in range(12)
        ), "source")
        diagnostic = {"frame_idx": 4, "face_box": (1.0, 2.0, 10.0, 11.0), "roi_boxes": [(1, 2, 3, 4)] * 12,
                      "geometry_source": "video", "geometry_source_frame_idx": 4, "geometry_bridged": False,
                      "current_frame_evidence": True, "recovery_attempts": ["video"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            _write_rows(path, (frame,), [diagnostic])
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["face_box"]), [1.0, 2.0, 10.0, 11.0])
        self.assertEqual(len(json.loads(row["roi_boxes"])), 12)
        self.assertEqual(row["full_face_std"], "4.0")
        self.assertEqual(row["current_frame_evidence"], "True")
        self.assertEqual(json.loads(row["recovery_attempts"]), ["video"])
        self.assertNotIn("face_detector_scope", row)
        self.assertNotIn("search_box", row)

    def test_inventory_rejects_duplicate_and_swapped_subject_records(self):
        with self.assertRaises(ValueError):
            _inventory_record([{"clip_id": "p1_0", "subject_id": "1", "locator": "a"},
                               {"clip_id": "p1_0", "subject_id": "1", "locator": "b"}], clip_id="p1_0", subject_id="1")
        with self.assertRaises(ValueError):
            _inventory_record([{"clip_id": "p1_0", "subject_id": "2", "locator": "a"}], clip_id="p1_0", subject_id="1")

    def test_summary_classifies_trimmed_source_as_repaired(self):
        frame = type("Frame", (), {"frame_idx": 0, "roi_values": ()})()
        diagnostic = {"frame_idx": 0, "face_box": (1.0, 2.0, 10.0, 11.0), "roi_boxes": [(1, 2, 3, 4)] * 12,
                      "geometry_source": "video", "geometry_source_frame_idx": 0, "geometry_bridged": False,
                      "current_frame_evidence": True, "recovery_attempts": ["video"]}
        def extract(*args, **kwargs):
            kwargs["emitted_frames"].append(frame)
            kwargs["diagnostics"].append(diagnostic)
            return (frame,)
        source = {"video": object(), "source_sha256": "sha", "original_frame_count": 3,
                  "processed_frame_count": 2, "trimmed_tail_frames": 1}
        with tempfile.TemporaryDirectory() as directory, patch("scripts.audit_mmpd_extraction.load_mmpd_mat", return_value=source), patch(
            "scripts.audit_mmpd_extraction.extract_face_relative_canonical_frames", side_effect=extract
        ):
            summary = audit_clip(raw_path=Path("unused"), clip_id="p1_0", subject_id="1", source_locator="raw/p1_0.mat", model_path="model", output_dir=Path(directory))
        self.assertEqual(summary["classification"], "repaired")
        self.assertEqual(summary["emitted_frame_count"], 1)
        self.assertEqual(summary["source_locator"], "raw/p1_0.mat")

    def test_audit_writes_failed_frame_json_and_counts_image_recovery(self):
        frame = type("Frame", (), {"frame_idx": 0, "roi_values": ()})()
        diagnostic = {"frame_idx": 0, "face_box": (1.0, 2.0, 10.0, 11.0), "roi_boxes": [(1, 2, 3, 4)] * 12,
                      "geometry_source": "image", "geometry_source_frame_idx": 0, "geometry_bridged": False,
                      "current_frame_evidence": True, "recovery_attempts": ["video", "image"]}
        def extract(*args, **kwargs):
            kwargs["emitted_frames"].append(frame)
            kwargs["diagnostics"].append(diagnostic)
            return (frame,)
        source = {"video": object(), "source_sha256": "sha", "original_frame_count": 1,
                  "processed_frame_count": 1, "trimmed_tail_frames": 0}
        with tempfile.TemporaryDirectory() as directory, patch("scripts.audit_mmpd_extraction.load_mmpd_mat", return_value=source), patch(
            "scripts.audit_mmpd_extraction.extract_face_relative_canonical_frames", side_effect=extract
        ):
            root = Path(directory)
            summary = audit_clip(raw_path=Path("unused"), clip_id="p1_0", subject_id="1", source_locator="raw/p1_0.mat", model_path="model", output_dir=root)
            failure = json.loads((root / "p1_0_failed_frames.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["image_recoveries"], 1)
        self.assertEqual(summary["classification"], "repaired")
        self.assertEqual(failure, [])

    def test_audit_rejects_stale_bridge_diagnostics(self):
        frame = type("Frame", (), {"frame_idx": 0, "roi_values": ()})()
        diagnostic = {"frame_idx": 0, "face_box": (1.0, 2.0, 10.0, 11.0), "roi_boxes": [(1, 2, 3, 4)] * 12,
                      "geometry_source": "video", "geometry_source_frame_idx": 0, "geometry_bridged": True,
                      "current_frame_evidence": True, "recovery_attempts": ["video"]}
        def extract(*args, **kwargs):
            kwargs["emitted_frames"].append(frame); kwargs["diagnostics"].append(diagnostic); return (frame,)
        source = {"video": object(), "source_sha256": "sha", "original_frame_count": 1, "processed_frame_count": 1, "trimmed_tail_frames": 0}
        with tempfile.TemporaryDirectory() as directory, patch("scripts.audit_mmpd_extraction.load_mmpd_mat", return_value=source), patch("scripts.audit_mmpd_extraction.extract_face_relative_canonical_frames", side_effect=extract):
            with self.assertRaisesRegex(ValueError, "forbidden stale geometry bridges"):
                audit_clip(raw_path=Path("unused"), clip_id="p1_0", subject_id="1", source_locator="raw/p1_0.mat", model_path="model", output_dir=Path(directory))


class MinimalInventoryTests(unittest.TestCase):
    def test_array_order_excludes_p29_3(self):
        self.assertEqual(clip_for_array_index(0)["clip_id"], "p1_0")
        self.assertEqual(clip_for_array_index(298)["clip_id"], "p29_19")
        with self.assertRaises(IndexError):
            clip_for_array_index(299)


class MinimalManifestTests(unittest.TestCase):
    def _records(self, methods=EXPECTED_LEARNED_METHOD_IDS):
        return [{"method_id": method, "locator": f"{method}.zip"} for method in methods]

    def test_manifest_requires_exact_nine_learned_methods(self):
        for methods in (
            EXPECTED_LEARNED_METHOD_IDS[:-1],
            EXPECTED_LEARNED_METHOD_IDS + ("renamed_arm",),
            EXPECTED_LEARNED_METHOD_IDS[:1] + (EXPECTED_LEARNED_METHOD_IDS[0],) + EXPECTED_LEARNED_METHOD_IDS[2:],
        ):
            with self.assertRaises(ContractValidationError):
                validate_checkpoint_manifest(self._records(methods))
        self.assertEqual(len(validate_checkpoint_manifest(self._records())), 9)


class MinimalCoverageTests(unittest.TestCase):
    def _rows(self, methods=EXPECTED_METHOD_IDS, hops=(0, 1)):
        return [{"method_id": method, "hop_idx": hop} for hop in hops for method in methods]

    def test_clip_csv_requires_exact_arm_hop_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.csv"
            write_clip_csv(self._rows(), path)
            self.assertTrue(path.exists())
            for rows in (self._rows()[:-1], self._rows() + [self._rows()[0]],
                         self._rows(methods=EXPECTED_METHOD_IDS[:-1] + ("extra_arm",))):
                with self.assertRaises(ContractValidationError):
                    write_clip_csv(rows, path)


class SharedReplayTests(unittest.TestCase):
    def test_each_arm_reuses_measurement_tuple(self):
        measurements = []
        for hop in range(1):
            items = tuple(ROIMeasurement(i, ROI_NAMES[i], 70.0, 1.0, .5, 1.0, True, None, (), None, 0, 1, "pos") for i in range(12))
            measurements.append(MeasurementFrame("mmpd", "p1_0", hop, 8.0, items, "pos", True, None, "source"))
        class Policy(FullFacePolicy):
            def __init__(self): self.seen = []
            def predict(self, observation, recurrent_state, *, episode_start):
                self.seen.append(observation.copy())
                return 0, None
        first, second = Policy(), Policy()
        policies = {method: FullFacePolicy() for method in EXPECTED_METHOD_IDS}
        policies["dagger_seed0"], policies["dagger_seed1"] = first, second
        rows = replay_shared_measurements(tuple(measurements), policies, clip_id="p1_0")
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(first.seen), 1)
        self.assertEqual(len(second.seen), 1)
        np.testing.assert_array_equal(first.seen[0], second.seen[0])


if __name__ == "__main__":
    unittest.main()
