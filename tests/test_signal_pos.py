import ast
import csv
import hashlib
import math
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import periodogram

import adaptive_roi_rppg.signal.pos as pos
import adaptive_roi_rppg.data.mcd.frames as frame_reader
from adaptive_roi_rppg.contracts import (CanonicalFrame, ClipManifest, DatasetManifest, ManifestStatus, OverlapResult, ROIFrameValue, ROI_NAMES, SplitManifest, canonical_json_bytes, sha256_file)
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import MCDManifestBundle, MCD_SCHEMA_ID, MCD_STATE_COLUMNS, read_mcd_canonical_frames
from adaptive_roi_rppg.signal import POS_CONFIG_ID, POS_CONFIG_PAYLOAD, build_pos_measurements, wang_pos


def make_frames(n, fps=24.0, *, tone=False, missing=(), imputation=(), clip_id="clip", provenance="source", coverage=1.0):
    result = []
    for i in range(n):
        phase = 2 * math.pi * 1.2 * i / fps
        values = []
        for j in range(12):
            absent = (i, j) in missing
            if tone:
                rgb = (1 + .02 * math.sin(phase), 1 + .04 * math.sin(phase), 1 + .01 * math.cos(phase))
            else:
                rgb = (1 + .01 * math.sin(i / 3), 1 + .02 * math.sin(i / 4), 1 + .03 * math.sin(i / 5))
            if absent: rgb = (None, None, None)
            ages = {"r_mean": 0} if (i, j) in imputation else None
            values.append(ROIFrameValue(j, ROI_NAMES[j], *rgb, 1.0 if not absent else None, coverage if not absent else 0.0, not absent, None if not absent else "source_missing", ages, ages))
        result.append(CanonicalFrame("mcd", clip_id, i, i / fps, fps, None, None, None, tuple(values), provenance))
    return tuple(result)


def state_row(index, *, absent=False, bad=""):
    row = [str(index), "", "", ""]
    for _ in ROI_NAMES:
        row.extend(["", "", "", "", "0.0"] if absent else ["1.0", "1.0", "1.0", "1.0", "1.0"])
    if bad: row[1] = bad
    return row


class Gate3Tests(unittest.TestCase):
    def test_literal_wang_plus_sign_and_final_legal_window(self):
        rgb = np.array([[1, 2, 3], [2, 4, 6], [3, 2, 1], [4, 5, 2]], dtype=float)
        expected = np.array([0.01978209784325878, 0.03956419568651763, -0.43772536911724674, 0.37837907558747036])
        np.testing.assert_allclose(wang_pos(rgb, 2), expected, rtol=0, atol=1e-15)
        self.assertNotEqual(wang_pos(np.vstack([rgb, [5, 3, 4]]), 2)[-1], 0.0)

    def test_wang_malformed_short_and_degenerate_boundaries(self):
        for value, fps in (([], 24), ([[1, 2]], 24), ([[1, 2, np.nan]], 24), ([[1, 2, np.inf]], 24), ([[1, 2, 3]], 0), ([[1, 2, 3]] * 38, 24)):
            with self.assertRaises(ContractValidationError): wang_pos(value, fps)
        with self.assertRaises(ContractValidationError): wang_pos(np.ones((40, 3)), 24)
        channel = np.ones((39, 3)); channel[:, 0] = 1e-8
        with self.assertRaises(ContractValidationError): wang_pos(channel, 24)
        projection = np.ones((39, 3)); projection[:, 0] += np.arange(39) * 1e-12
        with self.assertRaises(ContractValidationError): wang_pos(projection, 24)

    def test_synthetic_tone_measurement_at_both_production_fps(self):
        for fps in (24.0, 30.0):
            frame = build_pos_measurements(make_frames(round(8 * fps), fps, tone=True))[0].measurements[0]
            self.assertTrue(frame.valid)
            self.assertLessEqual(abs(frame.hr_bpm - 72.0703125), .3515625 if fps == 24 else .439453125)

    def test_natural_ppr_and_confidence_use_natural_peak(self):
        freqs = np.array([0.5, 1.0, 2.0, 3.0]); padded = np.array([2., 9., 4., 1.]); natural = np.array([3., 5., 8., 2.])
        with patch.object(pos, "periodogram", side_effect=[(freqs, padded), (freqs, natural)]), patch.object(pos, "lfilter", return_value=np.arange(4, dtype=float)):
            hr, ppr = pos._spectrum(np.ones(4), 24.0)
        self.assertEqual(hr, 60.0); self.assertEqual(ppr, 8 / 18)
        measurement = build_pos_measurements(make_frames(192, missing={(0, 0)}))[0].measurements[1]
        self.assertAlmostEqual(measurement.confidence, measurement.peak_power_ratio * measurement.coverage, delta=1e-12)

    def test_all_invalid_reasons_and_no_string_error_parsing(self):
        base = make_frames(192)
        with patch.object(pos, "_postprocess", side_effect=pos._SignalInvalid("degenerate_pos_postprocess")):
            self.assertEqual(build_pos_measurements(base)[0].measurements[0].invalid_reason, "degenerate_pos_postprocess")
        with patch.object(pos, "_spectrum", side_effect=pos._SignalInvalid("degenerate_spectrum")):
            self.assertEqual(build_pos_measurements(base)[0].measurements[0].invalid_reason, "degenerate_spectrum")
        missing = make_frames(192, missing={(0, 0)})
        self.assertEqual(build_pos_measurements(missing)[0].measurements[0].invalid_reason, "missing_required_rgb")
        with patch.object(pos, "_wang_pos", side_effect=pos._SignalInvalid("degenerate_channel_mean")):
            self.assertEqual(build_pos_measurements(base)[0].measurements[0].invalid_reason, "degenerate_channel_mean")
        with patch.object(pos, "_wang_pos", side_effect=pos._SignalInvalid("degenerate_projection")):
            self.assertEqual(build_pos_measurements(base)[0].measurements[0].invalid_reason, "degenerate_projection")

    def test_missing_imputation_coverage_and_twelve_records(self):
        result = build_pos_measurements(make_frames(192, missing={(0, 0)}))[0]
        self.assertEqual(len(result.measurements), 12); self.assertEqual(result.measurements[0].coverage, 191 / 192); self.assertEqual(result.measurements[0].imputed_channels, ())
        self.assertEqual(result.measurements[0].max_imputation_age_frames, None)
        self.assertEqual(build_pos_measurements(make_frames(192, imputation={(0, 0)}))[0].measurements[0].invalid_reason, "missing_required_rgb")
        bad = list(make_frames(192)); roi_values = list(bad[0].roi_values); roi_values[0] = ROIFrameValue(0, ROI_NAMES[0], None, None, None, None, None, False, "source_missing", None, None); bad[0] = CanonicalFrame("mcd", "clip", 0, 0, 24, None, None, None, tuple(roi_values), "source")
        with self.assertRaises(ContractValidationError): build_pos_measurements(tuple(bad))

    def test_hops_sources_provenance_clip_reset_and_prefix_invariance(self):
        short = make_frames(191); self.assertEqual(build_pos_measurements(short), ())
        frames = make_frames(216); output = build_pos_measurements(frames); self.assertEqual(len(output), 2)
        for idx, frame in enumerate(output):
            self.assertEqual((frame.hop_idx, frame.hop_time_s), (idx, (192 + idx * 24) / 24))
            self.assertEqual((frame.measurements[0].source_frame_start, frame.measurements[0].source_frame_end), (idx * 24, idx * 24 + 191))
            payload = {"dataset_id": "mcd", "clip_id": "clip", "source_provenance_id": "source", "source_frame_start": idx * 24, "source_frame_end": idx * 24 + 191, "signal_config_id": POS_CONFIG_ID}
            self.assertEqual(frame.provenance_id, "pos-measurement-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest())
        prefix = make_frames(192)
        longer = make_frames(216)
        self.assertEqual(build_pos_measurements(prefix)[0], build_pos_measurements(longer)[0])
        reset = make_frames(192, clip_id="new-clip", provenance="new-source")
        self.assertNotEqual(build_pos_measurements(prefix)[0].provenance_id, build_pos_measurements(reset)[0].provenance_id)
        gap = list(frames); gap[1] = CanonicalFrame("mcd", "clip", 3, 3 / 24, 24, None, None, None, gap[1].roi_values, "source")
        with self.assertRaises(ContractValidationError): build_pos_measurements(tuple(gap))
        with self.assertRaises(ContractValidationError): build_pos_measurements(make_frames(192, fps=25))

    def test_config_is_deep_immutable_and_id_recomputes(self):
        with self.assertRaises(TypeError): POS_CONFIG_PAYLOAD["profile"] = "x"
        with self.assertRaises(TypeError): POS_CONFIG_PAYLOAD["projection"][0][0] = 9
        self.assertEqual(POS_CONFIG_ID, "pos-v1-" + hashlib.sha256(canonical_json_bytes(POS_CONFIG_PAYLOAD)).hexdigest())
        self.assertEqual(tuple(POS_CONFIG_PAYLOAD["invalid_reasons"]), pos._REASONS)

    def test_reader_state_only_success_and_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); state = root / "state"; state.mkdir(); path = state / "S1_FullHDwebcam_before_semantic_state_vectors.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(MCD_STATE_COLUMNS); writer.writerows([state_row(0), state_row(1, absent=True)])
            digest = sha256_file(path); clip_id = "S1_FullHDwebcam_before"; clip = ClipManifest("mcd", clip_id, "clip-manifest", "S1", "Frontal", "before", "FullHDwebcam", 30.0, f"state/{path.name}", digest, 2, None, None, None, "split", MCD_SCHEMA_ID, ManifestStatus.complete)
            split = SplitManifest("split", "mcd", ("S1",), (), (clip_id,), (), 1, 0, 1, 0, OverlapResult.zero, ("0" * 64,), ManifestStatus.complete)
            dataset = DatasetManifest("dataset", "mcd", MCD_SCHEMA_ID, (clip.clip_manifest_id,), ("0" * 64,), {"source_inventory_id": "inventory"}, 1, 1, "0" * 64, (), "2026-08-11T00:00:00Z", "test", ManifestStatus.complete)
            bundle = MCDManifestBundle({"inventory_id": "inventory"}, split, (clip,), dataset)
            original_open = Path.open
            def guard_ground_truth(path, *args, **kwargs):
                if path.name.endswith("_ground_truth.csv"):
                    raise AssertionError("reader opened a GT CSV")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guard_ground_truth):
                frames = read_mcd_canonical_frames(bundle, state, clip_id, "train")
            self.assertEqual(len(frames), 2); self.assertEqual(frames[0].head_yaw_deg, None); self.assertEqual(frames[1].roi_values[0].invalid_reason, "source_missing")
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, dataset_manifest=replace(dataset, dataset_id="other")), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, dataset_manifest=replace(dataset, clip_manifest_refs=())), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, clip_manifests=(replace(clip, state_row_count=0),)), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id, "eval")
            snapshot = frame_reader._snapshot(path)
            with patch.object(frame_reader, "_snapshot", side_effect=[snapshot, (snapshot[0], snapshot[1], snapshot[2] + 1, snapshot[3])]):
                with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)
            path.write_text(path.read_text().replace("1.0,1.0,1.0,1.0,1.0", "nan,1.0,1.0,1.0,1.0", 1), encoding="utf-8")
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)

    def test_reader_changed_read_and_no_gt_path(self):
        source = Path(__file__).parents[1] / "src" / "adaptive_roi_rppg" / "data" / "mcd" / "frames.py"
        tree = ast.parse(source.read_text(encoding="utf-8")); imports = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertNotIn("adaptive_roi_rppg.labels", imports); self.assertNotIn("adaptive_roi_rppg.evaluation", imports); self.assertNotIn("adaptive_roi_rppg.control", imports)
        text = source.read_text(encoding="utf-8"); self.assertNotIn("ground_truth.csv", text); self.assertNotIn("gt_root", text)

    def test_signal_import_closure(self):
        source = Path(__file__).parents[1] / "src" / "adaptive_roi_rppg" / "signal" / "pos.py"; tree = ast.parse(source.read_text(encoding="utf-8"))
        forbidden = ("data", "labels", "legacy", "mmpd", "control", "training", "evaluation", "ground_truth", "gt_")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = ".".join(alias.name for alias in node.names) if isinstance(node, ast.Import) else node.module or ""
                self.assertFalse(any(token in name.lower() for token in forbidden), name)

    def test_smoke_requires_snapshot_and_slurm_metadata(self):
        script = Path(__file__).parents[1] / "scripts" / "verify_gate3_mcd_smoke.py"
        env = dict(os.environ); env.pop("SLURM_JOB_ID", None); env.pop("SLURMD_NODENAME", None)
        result = subprocess.run(["python3", str(script), "--manifest-tree", "missing", "--state-root", "missing", "--output-dir", tempfile.gettempdir(), "--code-snapshot-sha256", "0" * 64], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertIn("SLURM_JOB_ID", result.stderr + result.stdout)


if __name__ == "__main__": unittest.main()
