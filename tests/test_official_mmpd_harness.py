import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.inspect_official_mmpd_cache import _clip_key, _inventory_record, _selected_inputs
from scripts.run_official_mmpd_extractor import (
    _pair_summary, apply_scipy_compatibility, clip_key, input_to_label,
    read_clip_list, select_raw_entries, subject_key,
)


class OfficialMMPDHarnessTests(unittest.TestCase):
    def test_cache_summary_reports_pairs_shapes_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "subject1_L1_MO1_E2_S1_GE1_GL1_H1_MA1_input0.npy"
            label_path = root / "subject1_L1_MO1_E2_S1_GE1_GL1_H1_MA1_label0.npy"
            np.save(input_path, np.zeros((3, 72, 72, 3), dtype=np.uint8))
            np.save(label_path, np.asarray([1.0, np.nan, 3.0], dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                _pair_summary([str(input_path)], [str(label_path)], 1, 1)

    def test_pair_summary_accepts_finite_pair_and_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "subject1_clip_input0.npy"
            label_path = input_to_label(input_path)
            np.save(input_path, np.zeros((3, 72, 72, 3), dtype=np.uint8))
            np.save(label_path, np.ones(3, dtype=np.float32))
            summary = _pair_summary([str(input_path)], [str(label_path)], 1, 1)
            self.assertEqual(summary["pair_count"], 1)
            self.assertEqual(summary["subject_count"], 1)
            self.assertTrue(summary["all_values_finite"])
            np.save(input_path, np.zeros((3, 72, 72, 2), dtype=np.uint8))
            with self.assertRaisesRegex(RuntimeError, "shape"):
                _pair_summary([str(input_path)], [str(label_path)], 1, 1)

    def test_filename_helpers_do_not_modify_parent_directory(self):
        path = Path("/input/archive/subject1_clip_input0.npy")
        self.assertEqual(input_to_label(path), Path("/input/archive/subject1_clip_label0.npy"))
        self.assertEqual(clip_key(path), "subject1_clip")
        self.assertEqual(subject_key(path), "subject1")

    def test_scipy_shim_is_narrow_and_inert(self):
        scipy_module = type("SciPy", (), {"__version__": "test"})()
        scipy_config = type("Config", (), {})()
        info = apply_scipy_compatibility(scipy_module, scipy_config)
        self.assertTrue(info["shim_applied"])
        self.assertEqual(scipy_config.get_info("unused"), {})
        scipy_config.get_info = lambda name: {"name": name}
        info = apply_scipy_compatibility(scipy_module, scipy_config)
        self.assertFalse(info["shim_applied"])

    def test_selection_groups_input_chunks_by_clip_key(self):
        paths = [
            "/cache/subject2_z_input1.npy",
            "/cache/subject1_a_input0.npy",
            "/cache/subject2_z_input0.npy",
            "/cache/subject1_b_input0.npy",
        ]
        self.assertEqual(_clip_key(paths[0]), "subject2_z")
        self.assertEqual(_selected_inputs(paths, 2), [
            ("subject1_a", "/cache/subject1_a_input0.npy"),
            ("subject2_z", "/cache/subject2_z_input0.npy"),
        ])

    def test_clip_list_is_sorted_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clips.txt"
            path.write_text("p10_15\np1_9\n", encoding="utf-8")
            self.assertEqual(read_clip_list(path), ["p1_9", "p10_15"])
            path.write_text("p1_9\np1_9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_clip_list(path)

    def test_raw_selection_rejects_missing_duplicate_and_unexpected_ids(self):
        entries = [
            {"subject": 1, "index": "9", "path": "p1_9.mat"},
            {"subject": 10, "index": "15", "path": "p10_15.mat"},
        ]
        self.assertEqual(
            [entry["path"] for entry in select_raw_entries(entries, ["p1_9"])], ["p1_9.mat"])
        with self.assertRaisesRegex(ValueError, "missing"):
            select_raw_entries(entries, ["p1_9", "p2_1"])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            select_raw_entries(entries + [entries[0].copy()], ["p1_9"])
        with self.assertRaisesRegex(ValueError, "missing"):
            select_raw_entries(entries, ["p99_99"])

    def test_clip_list_rejects_unexpected_id_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clips.txt"
            path.write_text("subject1_9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid"):
                read_clip_list(path)

    def test_inspection_records_do_not_claim_source_id_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "subject1_L1_MO1_E2_S1_GE1_GL1_H1_MA1_input0.npy"
            np.save(input_path, np.zeros((3, 72, 72, 3), dtype=np.uint8))
            np.save(input_to_label(input_path), np.ones(3, dtype=np.float32))
            record = _inventory_record(str(input_path))
            self.assertIsNone(record["source_clip_id"])
            self.assertEqual(record["source_mapping"], "unmapped_cache_collection")

    def test_failure20_launcher_uses_bash_and_matching_cache_mount(self):
        launcher = Path("slurm/mmpd_official_extractor.slurm").read_text(encoding="utf-8")
        failure20 = Path("slurm/mmpd_official_extractor_failure20.slurm").read_text(encoding="utf-8")
        self.assertIn('CACHE_MOUNT_PATH="${CACHE_MOUNT_PATH:-/data/rPPG_dataset/processed_dataset}"', launcher)
        self.assertIn('--bind "${CACHE_ROOT}" "${CACHE_MOUNT_PATH}"', launcher)
        self.assertIn('export CACHE_MOUNT_PATH="/data/rPPG_dataset/processed_dataset_failure20"', failure20)
        self.assertIn('exec /bin/bash "${HARNESS_ROOT}/slurm/mmpd_official_extractor.slurm"', failure20)

if __name__ == "__main__":
    unittest.main()
