import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import (MCD_GT_COLUMNS, MCD_GT_SUFFIX,
    MCD_STATE_COLUMNS, MCD_STATE_SUFFIX, build_mcd_manifest_bundle,
    write_mcd_manifest_bundle)
from adaptive_roi_rppg.evaluation import (RunProvenance,
    build_train_full_face_plan, evaluate, evaluate_and_publish, publish_evaluation,
    verify_publication_against_sources, verify_publication_structure)
from adaptive_roi_rppg.labels.mcd import _label
from adaptive_roi_rppg.labels import mcd as labels_mcd


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gate5-real-fixture-"))
        self.state = self.root / "state"; self.gt = self.root / "gt"
        self.state.mkdir(); self.gt.mkdir()
        cameras = (("FullHDwebcam", 30.0), ("IriunWebcam", 24.0), ("USBVideo", 30.0))
        for subject in ("S1", "S2", "S3"):
            for camera, fps in cameras:
                for condition in ("before", "after"):
                    stem = f"{subject}_{camera}_{condition}"
                    n = round(9 * fps)
                    with (self.state / f"{stem}{MCD_STATE_SUFFIX}").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(MCD_STATE_COLUMNS)
                        for i in range(n):
                            row = [i, 0.0, 0.0, 0.0]
                            for roi in range(12):
                                phase = 2 * math.pi * 1.2 * i / fps + roi / 10
                                row.extend((100 + 2 * math.sin(phase), 100 + 3 * math.sin(phase + .4), 100 + 4 * math.sin(phase + .8), 1.0, 1.0))
                            writer.writerow(row)
                    signal = np.sin(2 * np.pi * 1.2 * np.arange(n) / fps)
                    with (self.gt / f"{stem}{MCD_GT_SUFFIX}").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(MCD_GT_COLUMNS); writer.writerows((i, value) for i, value in enumerate(signal))
        split = self.root / "split.csv"; split.write_text("subject_id,split\nS1,train\nS2,train\nS3,eval\n", encoding="utf-8")
        bundle = build_mcd_manifest_bundle(self.state, self.gt, split, created_at_utc="2026-08-11T00:00:00Z", producer_command="test")
        self.manifest = self.root / "manifest"; write_mcd_manifest_bundle(bundle, self.manifest)
        self.provenance = RunProvenance("gate5-test", "c" * 64, "fixed-test-command", {"python": "test"})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_known_answer_and_plan_normalization(self):
        for fps in (24.0, 30.0):
            signal = np.sin(2 * np.pi * 1.2 * np.arange(round(10 * fps)) / fps)
            self.assertEqual(_label(signal, fps, round(8 * fps), 0)[0], 72.0703125)
        plan = build_train_full_face_plan(self.manifest)
        self.assertEqual(len(plan.clip_bindings), 12)
        self.assertEqual(plan.expected_hops, tuple((key[1], 2) for key in plan.exact_clip_keys))
        self.assertEqual(plan, build_train_full_face_plan(self.manifest))

    def test_evaluator_has_no_numerical_injection_api_and_recomputes(self):
        parameters = tuple(inspect.signature(evaluate).parameters)
        self.assertEqual(parameters, ("manifest_tree", "state_root", "gt_root", "provenance", "plan"))
        self.assertFalse(set(parameters) & {"frames", "measurements", "transitions", "labels", "clips", "expected_hops", "bundle"})
        plan = build_train_full_face_plan(self.manifest)
        result_a = evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        result_b = evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        self.assertEqual(result_a["hop_rows"], result_b["hop_rows"])
        self.assertEqual(result_a["run_summary"]["clip_count"], 12)
        self.assertEqual(result_a["run_summary"]["subject_count"], 2)
        self.assertAlmostEqual(result_a["run_summary"]["equal_clip_mean_mae"], sum(r["mae_clip"] for r in result_a["clip_rows"]) / 12)
        with self.assertRaises((TypeError, ContractValidationError)): evaluate(plan, self.manifest, self.gt, (), self.provenance)

    def test_source_owned_missing_prediction_label_and_sources_fail(self):
        plan = build_train_full_face_plan(self.manifest)
        from adaptive_roi_rppg import evaluation as evaluation_api
        original_measurements = evaluation_api.__dict__.get("build_pos_measurements")
        self.assertIsNone(original_measurements)
        with mock.patch("adaptive_roi_rppg.evaluation.core.build_pos_measurements", return_value=()):
            with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        with mock.patch("adaptive_roi_rppg.evaluation.core.read_mcd_labels", side_effect=lambda *args, **kwargs: ()):
            with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        state_file = next(self.state.glob(f"*{MCD_STATE_SUFFIX}")); state_bytes = state_file.read_bytes(); state_file.unlink()
        with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        state_file.write_bytes(state_bytes)
        gt_file = next(self.gt.glob(f"*{MCD_GT_SUFFIX}")); gt_file.unlink()
        with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)

    def test_gt_reader_single_capture_failures_and_close_once(self):
        path = next(self.gt.glob(f"*{MCD_GT_SUFFIX}"))
        with path.open(encoding="utf-8") as handle: clip_count = sum(1 for _ in handle) - 1
        values, digest = labels_mcd._read_descriptor(path, clip_count)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest()); self.assertEqual(len(values), clip_count)
        original = path.read_bytes()
        for bad in (original[:10] + b"\xff" + original[11:], original.replace(b"frame_idx,gt_ppg", b"frame_idx;gt_ppg")):
            path.write_bytes(bad)
            with self.assertRaises(ContractValidationError): labels_mcd._read_descriptor(path, clip_count)
        path.write_bytes(original)
        close = os.close
        with mock.patch("adaptive_roi_rppg.labels.mcd.os.close", wraps=close) as close_mock:
            labels_mcd._read_descriptor(path, clip_count)
            self.assertEqual(close_mock.call_count, 1)
        real_read = labels_mcd.os.read
        def short_read(fd, size):
            chunk = real_read(fd, max(0, size - 1))
            return chunk
        with mock.patch("adaptive_roi_rppg.labels.mcd.os.read", side_effect=short_read):
            with self.assertRaises(ContractValidationError): labels_mcd._read_descriptor(path, clip_count)
        def close_then_error(fd):
            close(fd)
            raise OSError("close")
        with mock.patch("adaptive_roi_rppg.labels.mcd.os.close", side_effect=close_then_error):
            with self.assertRaises(ContractValidationError): labels_mcd._read_descriptor(path, clip_count)
        replacement = path.with_name(path.name + ".replacement")
        replacement.write_bytes(original)
        def read_then_replace(fd, size):
            chunk = real_read(fd, size)
            path.unlink(); replacement.rename(path)
            return chunk
        with mock.patch("adaptive_roi_rppg.labels.mcd.os.read", side_effect=read_then_replace):
            with self.assertRaises(ContractValidationError): labels_mcd._read_descriptor(path, clip_count)

    def test_complete_tree_and_source_tamper_fail(self):
        plan = build_train_full_face_plan(self.manifest)
        dataset = self.manifest / "dataset_manifest.json"; original = dataset.read_bytes()
        dataset.write_bytes(original.replace(b"mcd", b"xxx", 1))
        with self.assertRaises(ContractValidationError): build_train_full_face_plan(self.manifest)
        dataset.write_bytes(original)
        state_file = next(self.state.glob(f"*{MCD_STATE_SUFFIX}")); original = state_file.read_bytes(); state_file.write_bytes(original.replace(b",100.0", b",101.0", 1))
        with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)
        state_file.write_bytes(original)
        gt_file = next(self.gt.glob(f"*{MCD_GT_SUFFIX}")); original = gt_file.read_bytes(); gt_file.write_bytes(original.replace(b"0.0", b"0.1", 1))
        with self.assertRaises(ContractValidationError): evaluate(self.manifest, self.state, self.gt, self.provenance, plan)

    def test_publication_markers_verifiers_tamper_and_existing_destination(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        destination = self.root / "published"; publish_evaluation(result, destination)
        self.assertEqual({p.name for p in destination.iterdir()}, {"per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256", "STARTED.json", "COMPLETE.json"})
        verify_publication_structure(destination)
        verify_publication_against_sources(destination, self.manifest, self.state, self.gt)
        complete = json.loads((destination / "COMPLETE.json").read_text())
        self.assertEqual(set(complete["outputs"]), {"per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256"})
        hop = destination / "per_hop.csv"; hop.write_text(hop.read_text().replace(",70.0,0.0,", ",71.0,1.0,", 1))
        with self.assertRaises(ContractValidationError): verify_publication_against_sources(destination, self.manifest, self.state, self.gt)
        existing = self.root / "existing"; existing.mkdir(); (existing / "foreign").write_text("keep")
        with self.assertRaises(ContractValidationError): publish_evaluation(result, existing)
        self.assertEqual((existing / "foreign").read_text(), "keep")

    def test_publication_requires_existing_real_parent_without_side_effects(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        missing_parent = self.root / "missing" / "published"
        file_parent = self.root / "file-parent"
        file_parent.write_text("not a directory")
        symlink_parent = self.root / "symlink-parent"
        symlink_parent.symlink_to(self.root, target_is_directory=True)
        for destination, parent in ((missing_parent, missing_parent.parent), (file_parent / "published", file_parent), (symlink_parent / "published", symlink_parent)):
            with self.subTest(parent=parent):
                with self.assertRaises(ContractValidationError): publish_evaluation(result, destination)
                self.assertFalse(destination.exists())
        self.assertFalse(missing_parent.parent.exists())
        self.assertTrue(file_parent.is_file())
        self.assertTrue(symlink_parent.is_symlink())

    def test_evaluation_failure_publishes_failed_marker(self):
        destination = self.root / "failed"
        state_file = next(self.state.glob(f"*{MCD_STATE_SUFFIX}")); original = state_file.read_bytes(); state_file.write_bytes(original.replace(b",100.0", b",101.0", 1))
        with self.assertRaises(ContractValidationError): evaluate_and_publish(self.manifest, self.state, self.gt, self.provenance, destination)
        self.assertEqual({p.name for p in destination.iterdir()}, {"STARTED.json", "FAILED.json"})
        self.assertEqual(json.loads((destination / "FAILED.json").read_text())["error_code"], "EVALUATION_FAILED")

    def test_publication_failure_leaves_started_and_original_error(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        with mock.patch("adaptive_roi_rppg.evaluation.core._verify_internal_rows", side_effect=ContractValidationError("forced")):
            with self.assertRaisesRegex(ContractValidationError, "forced"): publish_evaluation(result, self.root / "failed")
        self.assertTrue((self.root / "failed" / "STARTED.json").exists())
        self.assertTrue((self.root / "failed" / "FAILED.json").exists())

    def test_failed_marker_write_preserves_original_error_and_started(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        with mock.patch("adaptive_roi_rppg.evaluation.core._verify_internal_rows", side_effect=ContractValidationError("original")), mock.patch("adaptive_roi_rppg.evaluation.core._publish_failed", side_effect=OSError("marker write")):
            with self.assertRaisesRegex(ContractValidationError, "original"): publish_evaluation(result, self.root / "marker-failure")
        self.assertTrue((self.root / "marker-failure" / "STARTED.json").exists())

    def test_started_marker_precedes_substantive_writes(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        destination = self.root / "ordering"
        from adaptive_roi_rppg.evaluation import core
        original = core._write_substantive_files
        def check_started(value, root):
            self.assertTrue((root / "STARTED.json").is_file())
            return original(value, root)
        with mock.patch("adaptive_roi_rppg.evaluation.core._write_substantive_files", side_effect=check_started): publish_evaluation(result, destination)

    def test_gate5_smoke_main_end_to_end(self):
        smoke = importlib.import_module("scripts.verify_gate5_mcd_smoke")
        output = self.root / "smoke-output"
        argv = [
            "verify_gate5_mcd_smoke.py",
            "--manifest-tree", str(self.manifest),
            "--state-root", str(self.state),
            "--gt-root", str(self.gt),
            "--output-dir", str(output),
            "--code-snapshot-sha256", "c" * 64,
        ]
        with mock.patch.object(smoke.sys, "argv", argv), mock.patch.dict(
            smoke.os.environ, {"SLURM_JOB_ID": "test-job", "SLURMD_NODENAME": "test-node"}, clear=False
        ):
            self.assertEqual(smoke.main(), 0)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256", "STARTED.json", "COMPLETE.json"},
        )
        verify_publication_structure(output)
        verify_publication_against_sources(output, self.manifest, self.state, self.gt)

    def test_gate5_smoke_rejects_unsafe_output_parents_before_scratch(self):
        smoke = importlib.import_module("scripts.verify_gate5_mcd_smoke")
        file_parent = self.root / "file-parent"
        file_parent.write_text("not a directory")
        symlink_parent = self.root / "symlink-parent"
        symlink_parent.symlink_to(self.root, target_is_directory=True)
        cases = (
            (self.root / "missing" / "output", self.root / "missing"),
            (file_parent / "output", file_parent),
            (symlink_parent / "output", symlink_parent),
        )
        before = set(self.root.iterdir())
        for output, parent in cases:
            argv = [
                "verify_gate5_mcd_smoke.py",
                "--manifest-tree", str(self.manifest),
                "--state-root", str(self.state),
                "--gt-root", str(self.gt),
                "--output-dir", str(output),
                "--code-snapshot-sha256", "c" * 64,
            ]
            with self.subTest(parent=parent):
                with mock.patch.object(smoke.sys, "argv", argv), mock.patch.dict(
                    smoke.os.environ, {"SLURM_JOB_ID": "test-job", "SLURMD_NODENAME": "test-node"}, clear=False
                ), mock.patch.object(smoke.tempfile, "mkdtemp", side_effect=AssertionError("scratch must not be created")), mock.patch.object(
                    smoke, "build_train_full_face_plan", side_effect=AssertionError("launcher must fail before planning")
                ):
                    with self.assertRaisesRegex(SystemExit, "output-dir parent"):
                        smoke.main()
                self.assertFalse(output.exists())
        self.assertFalse((self.root / "missing").exists())
        self.assertTrue(file_parent.is_file())
        self.assertTrue(symlink_parent.is_symlink())
        self.assertEqual(set(self.root.iterdir()), before)

    def test_mixed_marker_set_is_rejected(self):
        result = evaluate(self.manifest, self.state, self.gt, self.provenance)
        destination = self.root / "mixed"; publish_evaluation(result, destination)
        (destination / "FAILED.json").write_text("{}")
        with self.assertRaises(ContractValidationError): verify_publication_structure(destination)


if __name__ == "__main__": unittest.main()
