import ast
import csv
import hashlib
import io
import math
import os
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import periodogram

import adaptive_roi_rppg.signal.pos as pos
import adaptive_roi_rppg.data.mcd.frames as frame_reader
from adaptive_roi_rppg.contracts import (CanonicalFrame, ClipManifest, DatasetManifest, ManifestStatus, OverlapResult, ROIFrameValue, ROI_NAMES, SplitManifest, canonical_json_bytes)
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


def reader_fixture(root, payload=None, row_count=2):
    state = root / "state"; state.mkdir()
    clip_id = "S1_FullHDwebcam_before"; path = state / f"{clip_id}_semantic_state_vectors.csv"
    if payload is None:
        text = io.StringIO(newline=""); writer = csv.writer(text); writer.writerow(MCD_STATE_COLUMNS); writer.writerows([state_row(0), state_row(1, absent=True)]); payload = text.getvalue().encode("utf-8")
    path.write_bytes(payload)
    clip = ClipManifest("mcd", clip_id, "clip-manifest", "S1", "Frontal", "before", "FullHDwebcam", 30.0, f"state/{path.name}", hashlib.sha256(payload).hexdigest(), row_count, None, None, None, "split", MCD_SCHEMA_ID, ManifestStatus.complete)
    split = SplitManifest("split", "mcd", ("S1",), (), (clip_id,), (), 1, 0, 1, 0, OverlapResult.zero, ("0" * 64,), ManifestStatus.complete)
    dataset = DatasetManifest("dataset", "mcd", MCD_SCHEMA_ID, (clip.clip_manifest_id,), ("0" * 64,), {"source_inventory_id": "inventory"}, 1, 1, "0" * 64, (), "2026-08-11T00:00:00Z", "test", ManifestStatus.complete)
    return state, path, MCDManifestBundle({"inventory_id": "inventory"}, split, (clip,), dataset), clip_id


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
            stable = make_frames(round(8 * fps), fps, tone=True)
            first = build_pos_measurements(stable)
            second = build_pos_measurements(stable)
            self.assertEqual(first, second)
            self.assertEqual(canonical_json_bytes([frame.to_dict() for frame in first]), canonical_json_bytes([frame.to_dict() for frame in second]))
            frame = first[0].measurements[0]
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
            state, path, bundle, clip_id = reader_fixture(Path(temp)); dataset = bundle.dataset_manifest; clip = bundle.clip_manifests[0]
            original_open = Path.open
            def guard_ground_truth(path, *args, **kwargs):
                if path.name.endswith("_ground_truth.csv"):
                    raise AssertionError("reader opened a GT CSV")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guard_ground_truth):
                frames = read_mcd_canonical_frames(bundle, state, clip_id, "train")
            self.assertEqual(len(frames), 2); self.assertEqual(frames[0].head_yaw_deg, None); self.assertEqual(frames[1].roi_values[0].invalid_reason, "source_missing")
            frames_again = read_mcd_canonical_frames(bundle, state, clip_id, "train")
            self.assertEqual(frames, frames_again)
            self.assertEqual(canonical_json_bytes([frame.to_dict() for frame in frames]), canonical_json_bytes([frame.to_dict() for frame in frames_again]))
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, dataset_manifest=replace(dataset, dataset_id="other")), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, dataset_manifest=replace(dataset, clip_manifest_refs=())), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(replace(bundle, clip_manifests=(replace(clip, state_row_count=0),)), state, clip_id)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id, "eval")
            snapshot = frame_reader._snapshot(path)
            with patch.object(frame_reader, "_snapshot", side_effect=[snapshot, (snapshot[0], snapshot[1], snapshot[2] + 1, snapshot[3])]):
                with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)
            original_open = frame_reader.os.open
            original_mtime = path.stat().st_mtime_ns
            original_bytes = path.read_bytes()
            altered_bytes = original_bytes.replace(b"1.0,1.0,1.0,1.0,1.0", b"9.0,1.0,1.0,1.0,1.0", 1)
            self.assertEqual(len(original_bytes), len(altered_bytes))
            def alter_after_open(open_path, flags, mode=0o777):
                descriptor = original_open(open_path, flags, mode)
                if open_path == path:
                    writer = original_open(path, os.O_WRONLY)
                    try:
                        os.write(writer, altered_bytes)
                    finally:
                        os.close(writer)
                    os.utime(path, ns=(original_mtime, original_mtime))
                return descriptor
            with patch.object(frame_reader.os, "open", side_effect=alter_after_open):
                with self.assertRaisesRegex(ContractValidationError, "SHA-256 mismatch"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
            path.write_bytes(original_bytes); os.utime(path, ns=(original_mtime, original_mtime))
            with patch.object(frame_reader.os, "read", return_value=b""):
                with self.assertRaisesRegex(ContractValidationError, "short read"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
            descriptor_snapshot = frame_reader._descriptor_snapshot
            descriptor_calls = 0
            def change_descriptor_metadata(fd):
                nonlocal descriptor_calls
                value = descriptor_snapshot(fd)
                descriptor_calls += 1
                return value if descriptor_calls == 1 else (value[0], value[1], value[2] + 1, value[3])
            with patch.object(frame_reader, "_descriptor_snapshot", side_effect=change_descriptor_metadata):
                with self.assertRaisesRegex(ContractValidationError, "metadata changed"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
            replacement = state / "replacement.csv"
            replacement.write_bytes(original_bytes)
            path.unlink()
            path.symlink_to(replacement)
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)
            path.unlink()
            path.write_bytes(original_bytes)
            path.write_text(path.read_text().replace("1.0,1.0,1.0,1.0,1.0", "nan,1.0,1.0,1.0,1.0", 1), encoding="utf-8")
            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)

    def test_reader_single_frozen_bytes_and_authenticated_parse_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            state, _, bundle, clip_id = reader_fixture(Path(temp))
            real_sha256 = hashlib.sha256; real_string_io = io.StringIO; hashed = []; decoded = []
            def checked_hash(value):
                self.assertIs(type(value), bytes); hashed.append(value); return real_sha256(value)
            def checked_string_io(value, *args, **kwargs):
                decoded.append(value); return real_string_io(value, *args, **kwargs)
            with patch.object(frame_reader.hashlib, "sha256", side_effect=checked_hash), patch.object(frame_reader.io, "StringIO", side_effect=checked_string_io):
                read_mcd_canonical_frames(bundle, state, clip_id)
            self.assertEqual(decoded, [hashed[0].decode("utf-8", errors="strict")])
        for payload in (b"\xff", (",".join(MCD_STATE_COLUMNS) + '\n"unterminated\n').encode("utf-8")):
            with self.subTest(payload=payload[:12]), tempfile.TemporaryDirectory() as temp:
                state, _, bundle, clip_id = reader_fixture(Path(temp), payload, 1)
                with self.assertRaisesRegex(ContractValidationError, "cannot parse captured bytes"):
                    read_mcd_canonical_frames(bundle, state, clip_id)

    def test_reader_descriptor_path_binding_and_primary_error_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            state, path, bundle, clip_id = reader_fixture(Path(temp)); original = path.read_bytes(); real_open = os.open
            held = state / "held.csv"; replacement = state / "replacement.csv"; replacement.write_bytes(original)
            def open_equal_content_replacement(open_path, flags, mode=0o777):
                os.replace(path, held); os.replace(replacement, path)
                descriptor = real_open(open_path, flags, mode)
                os.replace(path, replacement); os.replace(held, path)
                return descriptor
            with patch.object(frame_reader.os, "open", side_effect=open_equal_content_replacement):
                with self.assertRaisesRegex(ContractValidationError, "opened descriptor does not match initial path"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
        with tempfile.TemporaryDirectory() as temp:
            state, path, bundle, clip_id = reader_fixture(Path(temp)); replacement = state / "replacement.csv"; replacement.write_bytes(path.read_bytes()); real_sha256 = hashlib.sha256
            def replace_path_after_capture(value): os.replace(replacement, path); return real_sha256(value)
            with patch.object(frame_reader.hashlib, "sha256", side_effect=replace_path_after_capture):
                with self.assertRaisesRegex(ContractValidationError, "final path does not match opened descriptor"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
        with tempfile.TemporaryDirectory() as temp:
            state, path, bundle, clip_id = reader_fixture(Path(temp)); real_close = os.close
            def close_then_fail(fd): real_close(fd); raise OSError("close failed")
            with patch.object(frame_reader.os, "read", return_value=b""), patch.object(frame_reader.os, "close", side_effect=close_then_fail):
                with self.assertRaisesRegex(ContractValidationError, "short read"):
                    read_mcd_canonical_frames(bundle, state, clip_id)
        with tempfile.TemporaryDirectory() as temp:
            state, path, bundle, clip_id = reader_fixture(Path(temp)); real_sha256 = hashlib.sha256
            def unlink_then_hash(value): path.unlink(); return real_sha256(b"wrong")
            with patch.object(frame_reader.hashlib, "sha256", side_effect=unlink_then_hash):
                with self.assertRaisesRegex(ContractValidationError, "SHA-256 mismatch"):
                    read_mcd_canonical_frames(bundle, state, clip_id)

    def test_reader_real_size_changes_and_close_once(self):
        for operation in ("grow", "truncate"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temp:
                state, path, bundle, clip_id = reader_fixture(Path(temp)); real_read = os.read; changed = False
                def change_size(fd, count):
                    nonlocal changed
                    if not changed:
                        changed = True
                        if operation == "grow":
                            with path.open("ab") as writer: writer.write(b"x")
                        else:
                            with path.open("r+b") as writer: writer.truncate(max(0, path.stat().st_size - 1))
                    return real_read(fd, count)
                with patch.object(frame_reader.os, "read", side_effect=change_size):
                    with self.assertRaisesRegex(ContractValidationError, "short read|metadata changed"):
                        read_mcd_canonical_frames(bundle, state, clip_id)
        scenarios = ("success", "hash", "utf8", "csv", "short", "stat")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                payload = b"\xff" if scenario == "utf8" else ((",".join(MCD_STATE_COLUMNS) + '\n"unterminated\n').encode() if scenario == "csv" else None)
                state, _, bundle, clip_id = reader_fixture(Path(temp), payload, 1 if payload else 2); real_close = os.close; closes = []
                def counted_close(fd): closes.append(fd); real_close(fd)
                patches = [patch.object(frame_reader.os, "close", side_effect=counted_close)]
                if scenario == "hash": bundle = replace(bundle, clip_manifests=(replace(bundle.clip_manifests[0], state_sha256="0" * 64),))
                if scenario == "short": patches.append(patch.object(frame_reader.os, "read", return_value=b""))
                if scenario == "stat": patches.append(patch.object(frame_reader, "_descriptor_snapshot", side_effect=[frame_reader._snapshot(state / f"{clip_id}_semantic_state_vectors.csv"), ContractValidationError("state file: cannot fstat descriptor")]))
                with patches[0]:
                    with patches[1] if len(patches) > 1 else nullcontext():
                        if scenario == "success": read_mcd_canonical_frames(bundle, state, clip_id)
                        else:
                            with self.assertRaises(ContractValidationError): read_mcd_canonical_frames(bundle, state, clip_id)
                self.assertEqual(len(closes), 1)

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


if __name__ == "__main__": unittest.main()
