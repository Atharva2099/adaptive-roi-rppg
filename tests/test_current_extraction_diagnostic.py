import unittest
import sys
import tempfile
import csv
import io
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from unittest.mock import patch

from adaptive_roi_rppg.contracts import CanonicalFrame, ROIFrameValue, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd.schema import MCD_STATE_COLUMNS
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import _rectangles, _roi_pixel_bounds, extract_face_relative_canonical_frames, extract_mmpd_canonical_frames
import scripts.diagnose_current_extraction as diagnostic_module
from scripts.diagnose_current_extraction import _contact_sheet, _display_rgb, _read_mcd_state, _write_outputs, invalid_diagnostic_failure


def _frame(valid=True):
    values = tuple(ROIFrameValue(i, ROI_NAMES[i], 1.0, 1.1, 1.2, 0.1, 1.0, valid, None if valid else "no_face", None, None) for i in range(12))
    return CanonicalFrame("mcd", "clip", 0, 0.0, 30.0, None, None, None, values, "source")


def _measurement(valid=True, confidence=1.0):
    return ROIMeasurement(0, ROI_NAMES[0], 72.0 if valid else None, confidence if valid else None, 0.5 if valid else None, 1.0, valid, None if valid else "degenerate_spectrum", (), None, 0, 239, "pos")


def _diagnostic(*, source="video", source_frame_idx=0):
    return {"face_valid": True, "face_box": (6.0, 6.0, 24.0, 24.0), "roi_boxes": [(6, 6, 24, 24)] * 12, "geometry_source": source, "geometry_source_frame_idx": source_frame_idx, "geometry_bridged": False, "current_frame_evidence": True, "recovery_attempts": [source]}


def _points(x0=0.21, y0=0.21, x1=0.79, y1=0.79):
    return [SimpleNamespace(x=x0, y=y0), SimpleNamespace(x=x1, y=y1)]


def _fake_mediapipe(video_faces, image_faces, calls, images):
    state = {"video_idx": 0, "image_idx": 0}

    class Detector:
        def __init__(self, mode): self.mode = mode
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def detect_for_video(self, image, timestamp):
            index = state["video_idx"]; state["video_idx"] += 1
            calls.append((self.mode, "video", index, id(image)))
            faces = video_faces[index] if index < len(video_faces) else None
            return SimpleNamespace(face_landmarks=[] if faces is None else [faces])
        def detect(self, image):
            index = state["image_idx"]; state["image_idx"] += 1
            calls.append((self.mode, "image", index, id(image)))
            faces = image_faces[index] if index < len(image_faces) else None
            return SimpleNamespace(face_landmarks=[] if faces is None else [faces])

    class FakeLandmarker:
        @staticmethod
        def create_from_options(options): return Detector(options["running_mode"])

    class FakeBase:
        class Delegate: CPU = "CPU"
        def __init__(self, **kwargs): pass

    class FakeVision:
        FaceLandmarkerOptions = lambda **kwargs: kwargs
        FaceLandmarker = FakeLandmarker
        class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"

    def fake_image(**kwargs):
        images.append(kwargs)
        return kwargs

    return SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=FakeVision, BaseOptions=FakeBase), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=fake_image)


def _video(colors):
    return np.asarray([np.full((30, 30, 3), color, dtype=np.float32) for color in colors])


def _state_csv(clip_id, *, invalid=False, missing=False):
    row = ["0", "", "", ""]
    for index in range(12):
        if invalid and index == 0:
            row.extend(["1.0", "1.0", "1.0", "1.0", "0.0"])
        elif missing and index == 0:
            row.extend(["", "", "", "", "0.0"])
        else:
            row.extend(["1.0", "1.0", "1.0", "1.0", "1.0"])
    text = io.StringIO(newline="")
    writer = csv.writer(text); writer.writerow(MCD_STATE_COLUMNS); writer.writerow(row)
    return clip_id, text.getvalue()


class CurrentExtractionDiagnosticTests(unittest.TestCase):
    def test_rectangles_are_canonical_and_face_relative(self):
        boxes = _rectangles(10, 20, 110, 220)
        self.assertEqual(len(boxes), 12)
        self.assertEqual(boxes[0], (10, 20, 110, 220))
        self.assertTrue(all(x0 < x1 and y0 < y1 for x0, y0, x1, y1 in boxes))

    def test_floor_ceil_bounds_preserve_a_thin_valid_roi(self):
        self.assertEqual(
            _roi_pixel_bounds((3.1, 4.1, 3.2, 4.2), width=10, height=10, roi_name="thin", clip_id="clip", frame_idx=0),
            (3, 4, 4, 5),
        )

    def test_degenerate_or_outside_roi_still_hard_fails(self):
        for box in ((3.0, 4.0, 3.0, 5.0), (11.0, 4.0, 12.0, 5.0)):
            with self.assertRaisesRegex(ContractValidationError, r"empty ROI thin for clip_id=clip, frame_idx=0"):
                _roi_pixel_bounds(box, width=10, height=10, roi_name="thin", clip_id="clip", frame_idx=0)

    def test_fractional_nonpositive_roi_geometry_hard_fails_before_rounding(self):
        for box in ((3.1, 4.1, 3.1, 4.2), (3.1, 4.1, 3.2, 4.1), (3.2, 4.1, 3.1, 4.2)):
            with self.assertRaisesRegex(ContractValidationError, r"empty ROI thin for clip_id=clip, frame_idx=0"):
                _roi_pixel_bounds(box, width=10, height=10, roi_name="thin", clip_id="clip", frame_idx=0)

    def test_no_face_source_is_reported_at_first_roi(self):
        failure = invalid_diagnostic_failure((_frame(False),), (), ())
        self.assertEqual(failure, {"kind": "source_roi", "frame": 0, "roi": "full_face", "reason": "no_face"})

    def test_mcd_state_parser_requires_exact_filename_and_marks_invalid_roi(self):
        clip_id, text = _state_csv("clip", invalid=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{clip_id}_semantic_state_vectors.csv"; path.write_text(text, encoding="utf-8")
            rows = _read_mcd_state(path, clip_id)
            self.assertFalse(rows[0]["_state_valid"])
            self.assertEqual(rows[0]["_state_invalid_roi"], "full_face")
            with self.assertRaises(RuntimeError): _read_mcd_state(Path(directory) / "wrong.csv", clip_id)
            bad_header = path.with_name("bad_header_semantic_state_vectors.csv"); bad_header.write_text(text.replace("frame_idx", "wrong", 1), encoding="utf-8")
            with self.assertRaises(RuntimeError): _read_mcd_state(bad_header, "bad_header")
            bad_width = path.with_name("bad_width_semantic_state_vectors.csv"); bad_width.write_text(text + "extra\n", encoding="utf-8")
            with self.assertRaises(RuntimeError): _read_mcd_state(bad_width, "bad_width")
            bad_frame = path.with_name("bad_frame_semantic_state_vectors.csv"); bad_frame.write_text(text.replace("\n0,", "\n2,", 1), encoding="utf-8")
            with self.assertRaises(RuntimeError): _read_mcd_state(bad_frame, "bad_frame")

    def test_invalid_mcd_state_is_delayed_until_after_output_validation(self):
        _, text = _state_csv("clip", invalid=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip_semantic_state_vectors.csv"; path.write_text(text, encoding="utf-8")
            rows = _read_mcd_state(path, "clip")
            failure = invalid_diagnostic_failure((_frame(True),), (), (), rows)
            self.assertEqual(failure["kind"], "state")
            self.assertEqual(failure["frame"], 0)

    def test_zero_coverage_blank_roi_is_source_missing_and_delayed(self):
        _, text = _state_csv("clip", missing=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "clip_semantic_state_vectors.csv"; path.write_text(text, encoding="utf-8")
            rows = _read_mcd_state(path, "clip")
            roi = rows[0]["_state_rois"][0]
            self.assertFalse(roi["valid"])
            self.assertEqual(roi["reason"], "source_missing")
            self.assertFalse(rows[0]["_state_valid"])
            _write_outputs(root, "mcd", (_frame(True),), [_diagnostic()], rows, (), ())
            with (root / "mcd_per_frame.csv").open(newline="", encoding="utf-8") as handle:
                output = next(csv.DictReader(handle))
            self.assertEqual(output["full_face_state_valid"], "False")
            self.assertEqual(output["full_face_state_reason"], "source_missing")
            failure = invalid_diagnostic_failure((_frame(True),), (), (), rows)
            self.assertEqual(failure, {"kind": "state", "frame": 0, "roi": "full_face", "reason": "full_face: source_missing"})

    def test_per_hop_output_is_long_form_for_all_rois(self):
        measurements = tuple(ROIMeasurement(i, ROI_NAMES[i], 70.0 + i, 0.5, 0.4, 1.0, True, None, (), None, 0, 239, "pos") for i in range(12))
        hop = SimpleNamespace(hop_idx=0, hop_time_s=8.0, measurements=measurements)
        transition = SimpleNamespace(selected_measurement=measurements[0], pre_belief=SimpleNamespace(mean_hr=70.0), post_belief=SimpleNamespace(mean_hr=71.0))
        with tempfile.TemporaryDirectory() as directory:
            _write_outputs(Path(directory), "mmpd", (_frame(True),), [_diagnostic()], None, (hop,), (transition,))
            with (Path(directory) / "mmpd_per_hop.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["roi_name"], "full_face")
            self.assertIn("selected_hr_bpm", rows[0])
            self.assertIn("roi_ppr", rows[0])

    def test_per_frame_csv_preserves_geometry_source_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_outputs(Path(directory), "mmpd", (_frame(True),), [_diagnostic(source="image", source_frame_idx=0)], None, (), ())
            with (Path(directory) / "mmpd_per_frame.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["geometry_source"], "image")
        self.assertEqual(row["geometry_source_frame_idx"], "0")
        self.assertEqual(row["geometry_bridged"], "False")
        self.assertEqual(row["current_frame_evidence"], "True")
        self.assertEqual(json.loads(row["recovery_attempts"]), ["image"])
        self.assertNotIn("face_detector_scope", row)
        self.assertNotIn("search_box", row)

    def test_float_rgb_display_conversion_is_uint8_without_mutating_input(self):
        video = np.full((3, 8, 8, 3), 0.5, dtype=np.float32)
        original = video.copy()
        display = _display_rgb(video[0])
        self.assertEqual(display.dtype, np.uint8)
        self.assertTrue(np.all(display == 127))
        np.testing.assert_array_equal(video, original)

    def test_synthetic_video_records_face_and_no_face_diagnostics(self):
        points = [SimpleNamespace(x=0.21, y=0.21), SimpleNamespace(x=0.79, y=0.79)]
        images = []
        calls = []
        class Detector:
            def __init__(self, mode): self.mode = mode
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def detect_for_video(self, image, timestamp):
                calls.append((self.mode, "video", timestamp))
                return SimpleNamespace(face_landmarks=[points] if timestamp == 0 else [])
            def detect(self, image):
                calls.append((self.mode, "image", None))
                return SimpleNamespace(face_landmarks=[points])
        class FakeLandmarker:
            @staticmethod
            def create_from_options(options): return Detector(options["running_mode"])
        class FakeBase:
            class Delegate: CPU = "CPU"
            def __init__(self, **kwargs): pass
        class FakeVision:
            FaceLandmarkerOptions = lambda **kwargs: kwargs
            FaceLandmarker = FakeLandmarker
            class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"
        def fake_image(**kwargs):
            images.append(kwargs)
            return kwargs
        fake = SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=FakeVision, BaseOptions=FakeBase), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=fake_image)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            diagnostics = []
            video = np.zeros((2, 30, 30, 3), dtype=np.float32)
            video[0, ..., 1] = 0.5
            video[0, ..., 2] = 1.0
            video[1, ..., 0] = 0.25
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(video, dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics)
        self.assertEqual(len(images), 2)
        received = images[0]["data"]
        self.assertEqual(images[0]["image_format"], "SRGB")
        self.assertEqual(received.dtype, np.uint8)
        self.assertTrue(received.flags.c_contiguous)
        np.testing.assert_array_equal(received[0, 0], np.array([0, 127, 255], dtype=np.uint8))
        self.assertTrue(diagnostics[0]["face_valid"])
        self.assertEqual(len(diagnostics[0]["roi_boxes"]), 12)
        self.assertTrue(frames[0].roi_values[0].valid)
        self.assertEqual(frames[0].roi_values[0].coverage, 1.0)
        self.assertLessEqual(frames[0].roi_values[0].coverage, 1.0)
        self.assertTrue(diagnostics[1]["face_valid"])
        self.assertTrue(all(roi.valid for roi in frames[1].roi_values))
        self.assertEqual(calls, [("VIDEO", "video", 0), ("VIDEO", "video", 33), ("IMAGE", "image", None)])

    def test_image_fallback_recovers_a_video_miss_on_the_same_frame(self):
        points = [SimpleNamespace(x=0.21, y=0.21), SimpleNamespace(x=0.79, y=0.79)]
        calls = []
        class Detector:
            def __init__(self, mode): self.mode = mode
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def detect_for_video(self, image, timestamp):
                calls.append((self.mode, "video", timestamp, id(image)))
                return SimpleNamespace(face_landmarks=[])
            def detect(self, image):
                calls.append((self.mode, "image", None, id(image)))
                return SimpleNamespace(face_landmarks=[points])
        class FakeLandmarker:
            @staticmethod
            def create_from_options(options): return Detector(options["running_mode"])
        class FakeBase:
            class Delegate: CPU = "CPU"
            def __init__(self, **kwargs): pass
        class FakeVision:
            FaceLandmarkerOptions = lambda **kwargs: kwargs
            FaceLandmarker = FakeLandmarker
            class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"
        fake = SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=FakeVision, BaseOptions=FakeBase), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            video = np.full((1, 30, 30, 3), 0.25, dtype=np.float32)
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(video, dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertTrue(frames[0].roi_values[0].valid)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:3], ("VIDEO", "video", 0))
        self.assertEqual(calls[1][:3], ("IMAGE", "image", None))
        self.assertEqual(calls[0][3], calls[1][3])

    def test_video_and_image_miss_fails_before_no_face_can_reach_replay(self):
        calls = []
        class Detector:
            def __init__(self, mode): self.mode = mode
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def detect_for_video(self, image, timestamp):
                calls.append((self.mode, "video"))
                return SimpleNamespace(face_landmarks=[])
            def detect(self, image):
                calls.append((self.mode, "image"))
                return SimpleNamespace(face_landmarks=[])
        class FakeLandmarker:
            @staticmethod
            def create_from_options(options): return Detector(options["running_mode"])
        class FakeBase:
            class Delegate: CPU = "CPU"
            def __init__(self, **kwargs): pass
        class FakeVision:
            FaceLandmarkerOptions = lambda **kwargs: kwargs
            FaceLandmarker = FakeLandmarker
            class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"
        fake = SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=FakeVision, BaseOptions=FakeBase), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            video = np.full((1, 30, 30, 3), 0.25, dtype=np.float32)
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"clip_id=clip, frame_idx=0"):
                    extract_face_relative_canonical_frames(video, dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual(calls, [("VIDEO", "video"), ("IMAGE", "image")])

    def test_video_source_is_recorded_without_image_fallback(self):
        calls, images, diagnostics = [], [], []
        fake = _fake_mediapipe([_points()], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(_video([0.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics)
        self.assertTrue(frames[0].roi_values[0].valid)
        self.assertEqual(diagnostics[0]["geometry_source"], "video")
        self.assertEqual(diagnostics[0]["geometry_source_frame_idx"], 0)
        self.assertFalse(diagnostics[0]["geometry_bridged"])
        self.assertEqual([call[1] for call in calls], ["video"])

    def test_image_source_is_recorded_and_uses_same_mp_image(self):
        calls, images, diagnostics = [], [], []
        fake = _fake_mediapipe([None], [_points()], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                extract_face_relative_canonical_frames(_video([0.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics)
        self.assertEqual(diagnostics[0]["geometry_source"], "image")
        self.assertEqual(calls[0][3], calls[1][3])
        self.assertEqual(len(images), 1)

    def test_missing_face_landmarks_field_hard_fails_without_image_fallback(self):
        calls = []
        class Landmarker:
            def __init__(self, mode): self.mode = mode
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def detect_for_video(self, image, timestamp): calls.append("video"); return SimpleNamespace()
            def detect(self, image): calls.append("image"); return SimpleNamespace(face_landmarks=[])
        class Base:
            class Delegate: CPU = "CPU"
            def __init__(self, **kwargs): pass
        class Vision:
            FaceLandmarker = SimpleNamespace(create_from_options=lambda options: Landmarker(options["running_mode"]))
            FaceLandmarkerOptions = lambda **kwargs: kwargs
            class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"
        fake = SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=Vision, BaseOptions=Base), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, "lacks face_landmarks"):
                    extract_face_relative_canonical_frames(_video([.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual(calls, ["video"])

    def test_emitted_geometry_is_full_current_frame_only(self):
        calls, images, diagnostics = [], [], []
        fake = _fake_mediapipe([_points(), None], [_points()], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(_video([(.25, .5, .75), (.75, .25, .125)]), dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics)
        self.assertEqual([row["geometry_source"] for row in diagnostics], ["video", "image"])
        self.assertTrue(all(row["geometry_source_frame_idx"] == row["frame_idx"] and row["current_frame_evidence"] and not row["geometry_bridged"] for row in diagnostics))
        self.assertTrue(all(set(row).isdisjoint({"last_accepted_box", "face_detector", "previous_frame", "crop", "search_box"}) for row in diagnostics))
        self.assertEqual([item["data"].shape for item in images], [(30, 30, 3), (30, 30, 3)])
        self.assertEqual([call[:2] for call in calls], [("VIDEO", "video"), ("VIDEO", "video"), ("IMAGE", "image")])
        self.assertGreater(frames[1].roi_values[0].r_mean, frames[1].roi_values[0].g_mean)

    def test_exit_stack_closes_the_video_landmarker_when_image_initialization_fails(self):
        closed, created = [], {"count": 0}
        class Resource:
            def __init__(self, name): self.name = name
            def __enter__(self): return self
            def __exit__(self, *args): closed.append(self.name); return False
            def detect_for_video(self, *args): return SimpleNamespace(face_landmarks=[_points()])
            def detect(self, *args): return SimpleNamespace(face_landmarks=[])
        class Landmarker:
            @staticmethod
            def create_from_options(options):
                created["count"] += 1
                if created["count"] == 2: raise RuntimeError("init failure")
                return Resource(options["running_mode"])
        class Base:
            class Delegate: CPU = "CPU"
            def __init__(self, **kwargs): pass
        class Vision:
            FaceLandmarker = Landmarker
            FaceLandmarkerOptions = lambda **kwargs: kwargs
            class RunningMode: VIDEO = "VIDEO"; IMAGE = "IMAGE"
        fake = SimpleNamespace(__version__="0.10.20", tasks=SimpleNamespace(vision=Vision, BaseOptions=Base), ImageFormat=SimpleNamespace(SRGB="SRGB"), Image=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, "could not initialize"):
                    extract_face_relative_canonical_frames(_video([.25]), dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual(closed, ["VIDEO"])

    def test_failed_no_face_and_empty_roi_diagnostics_survive(self):
        calls, images, failed = [], [], []
        fake = _fake_mediapipe([_points(), None], [None, None], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaises(ContractValidationError):
                    extract_face_relative_canonical_frames(
                        _video([.25, .5]), dataset_id="mmpd", clip_id="clip", source_provenance_id="source",
                        model_asset_path=str(model), failed_diagnostics=failed,
                    )
        self.assertEqual(failed[0]["frame_idx"], 1)
        self.assertEqual(failed[-1]["reason"], "no_current_frame_landmarks")
        self.assertEqual(failed[-1]["recovery_attempts"], ["video", "image"])

        failed = []
        fake = _fake_mediapipe([_points()], [], [], [])
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}), patch(
                "adaptive_roi_rppg.evaluation.adapters.mmpd.extraction._rectangles",
                return_value=((31.0, 1.0, 32.0, 2.0),),
            ):
                with self.assertRaisesRegex(ContractValidationError, r"empty ROI full_face"):
                    extract_face_relative_canonical_frames(
                        _video([.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source",
                        model_asset_path=str(model), failed_diagnostics=failed,
                    )
        self.assertEqual(failed[0]["failed_roi_name"], "full_face")
        self.assertEqual(failed[0]["face_box"], (6.3, 6.3, 23.700000000000003, 23.700000000000003))
        self.assertEqual(failed[0]["attempted_roi_boxes"], [(31.0, 1.0, 32.0, 2.0)])

    def test_shared_extractor_hard_fails_mcd_without_current_landmarks(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points(), None], [None], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"clip_id=clip, frame_idx=1"):
                    extract_face_relative_canonical_frames(_video([0.25, 0.5]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))

    def test_partial_hard_failure_preserves_only_completed_frame_pairs(self):
        calls, images, diagnostics, emitted = [], [], [], []
        fake = _fake_mediapipe([_points(), _points(), None], [None], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"clip_id=clip, frame_idx=2"):
                    extract_face_relative_canonical_frames(_video([0.25, 0.5, 0.75]), dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics, emitted_frames=emitted)
        self.assertEqual(len(emitted), len(diagnostics))
        self.assertEqual([frame.frame_idx for frame in emitted], [row["frame_idx"] for row in diagnostics])
    def test_mmpd_exact_raw_zero_hard_fails_before_recovery(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points(), None], [], calls, images)
        video = _video([0.25, 0.25]); video[1] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"exact all-zero raw frame for clip_id=clip, frame_idx=1"):
                    extract_face_relative_canonical_frames(video, dataset_id="mmpd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), reject_exact_zero_raw_frames=True)
        self.assertEqual([call[1:3] for call in calls], [("video", 0)])

    def test_nonfinite_raw_rejects_before_either_detector(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points()], [], calls, images)
        video = _video([0.25]); video[0, 0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"non-finite raw frame for clip_id=clip, frame_idx=0"):
                    extract_face_relative_canonical_frames(video, dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual(calls, [])

    def test_tiny_nonzero_raw_is_not_rejected_after_uint8_quantization(self):
        calls, images, diagnostics = [], [], []
        fake = _fake_mediapipe([_points()], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(np.full((1, 30, 30, 3), 1e-8, dtype=np.float32), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model), diagnostics=diagnostics)
        self.assertTrue(frames[0].roi_values[0].valid)
        self.assertEqual(images[0]["data"].max(), 0)
        self.assertEqual(diagnostics[0]["geometry_source"], "video")

    def test_video_malformed_geometry_hard_fails_without_image_fallback(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points(0.5, 0.2, 0.5, 0.8)], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"invalid face geometry for clip_id=clip, frame_idx=0"):
                    extract_face_relative_canonical_frames(_video([0.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual([call[:2] for call in calls], [("VIDEO", "video")])

    def test_direct_geometry_failures_preserve_serializable_diagnostics(self):
        malformed = [SimpleNamespace(y=.2), SimpleNamespace(y=.8)]
        calls, images, failed = [], [], []
        fake = _fake_mediapipe([malformed], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"invalid face geometry"):
                    extract_face_relative_canonical_frames(
                        _video([.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source",
                        model_asset_path=str(model), failed_diagnostics=failed,
                    )
        self.assertEqual(failed[0]["geometry_source"], "video")
        self.assertEqual(failed[0]["landmark_summary"], {"landmark_count": 2, "normalized_bounds": None})
        json.dumps(failed)

        nonfinite = [SimpleNamespace(x=np.nan, y=.2), SimpleNamespace(x=.8, y=.8)]
        calls, images, failed = [], [], []
        fake = _fake_mediapipe([None], [nonfinite], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"invalid face geometry"):
                    extract_face_relative_canonical_frames(
                        _video([.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source",
                        model_asset_path=str(model), failed_diagnostics=failed,
                    )
        self.assertEqual(failed[0]["geometry_source"], "image")
        self.assertTrue(failed[0]["landmark_summary"]["contains_nonfinite"])
        json.dumps(failed)

    def test_image_malformed_geometry_hard_fails_with_image_diagnostic(self):
        malformed = [SimpleNamespace(y=.2), SimpleNamespace(y=.8)]
        calls, images, failed = [], [], []
        fake = _fake_mediapipe([None], [malformed], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                with self.assertRaisesRegex(ContractValidationError, r"invalid face geometry"):
                    extract_face_relative_canonical_frames(
                        _video([.25]), dataset_id="mmpd", clip_id="clip", source_provenance_id="source",
                        model_asset_path=str(model), failed_diagnostics=failed,
                    )
        self.assertEqual(failed[0]["geometry_source"], "image")
        json.dumps(failed)

    def test_thin_face_box_produces_valid_rois(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points(0.10, 0.10, 0.11, 0.11)], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_face_relative_canonical_frames(_video([0.25]), dataset_id="mcd", clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertTrue(all(roi.valid for roi in frames[0].roi_values))

    def test_mmpd_wrapper_uses_full_current_frame_geometry_only(self):
        calls, images = [], []
        fake = _fake_mediapipe([_points(), _points()], [], calls, images)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"; model.write_bytes(b"model")
            with patch.dict(sys.modules, {"mediapipe": fake}):
                frames = extract_mmpd_canonical_frames(_video([0.25, 0.5]), clip_id="clip", source_provenance_id="source", model_asset_path=str(model))
        self.assertEqual(len(frames), 2)

    def test_contact_sheet_uses_explicit_frame_indices(self):
        displayed, labels = [], []

        class FakeImage:
            width = 30
            height = 30
            @classmethod
            def fromarray(cls, array):
                displayed.append(float(array[0, 0, 0])); return cls()
            @classmethod
            def new(cls, *args): return cls()
            def convert(self, *args): return self
            def paste(self, *args): pass
            def save(self, *args): pass

        class FakeDraw:
            def rectangle(self, *args, **kwargs): pass
            def text(self, *args, **kwargs): labels.append(args[1])

        fake_pil = SimpleNamespace(
            Image=SimpleNamespace(fromarray=FakeImage.fromarray, new=FakeImage.new),
            ImageDraw=SimpleNamespace(Draw=lambda image: FakeDraw()),
            ImageFont=SimpleNamespace(load_default=lambda: object()),
        )
        with tempfile.TemporaryDirectory() as directory:
            video = _video([0.25, 0.5, 0.75])
            diagnostics = [_diagnostic(source_frame_idx=index) for index in range(3)]
            frames = tuple(_frame(True) for _ in range(3))
            with patch.dict(sys.modules, {"PIL": fake_pil}):
                _contact_sheet(Path(directory), "mmpd", video, diagnostics, frames, frame_indices=[2])
        self.assertEqual(displayed, [191.0])
        self.assertTrue(any("geometry=video source_frame=2" in label for label in labels))

    def test_diagnostic_caller_uses_only_the_mmpd_raw_frame_policy(self):
        with (
            patch.object(diagnostic_module, "extract_face_relative_canonical_frames", return_value=()) as extractor,
            patch.object(diagnostic_module, "build_pos_measurements", return_value=()),
            patch.object(diagnostic_module, "initial_control_state", return_value=object()),
            patch.object(diagnostic_module, "_write_outputs"),
            patch.object(diagnostic_module, "_contact_sheet") as contact_sheet,
            patch.object(diagnostic_module, "invalid_diagnostic_failure", return_value=None),
        ):
            diagnostic_module._run_one("mmpd", _video([0.25]), "clip", "source", "model", 30.0, Path("out"), contact_sheet_frame_indices=[1310, 1311, 1312])
            diagnostic_module._run_one("mcd", _video([0.25]), "clip", "source", "model", 30.0, Path("out"))
        mmpd_kwargs = extractor.call_args_list[0].kwargs
        mcd_kwargs = extractor.call_args_list[1].kwargs
        self.assertTrue(mmpd_kwargs["reject_exact_zero_raw_frames"])
        self.assertFalse(mcd_kwargs["reject_exact_zero_raw_frames"])
        self.assertNotIn("allow_current_frame_detector_recovery", mmpd_kwargs)
        self.assertNotIn("allow_current_frame_detector_recovery", mcd_kwargs)
        self.assertEqual(contact_sheet.call_args_list[0].kwargs["frame_indices"], [1310, 1311, 1312])
        self.assertIsNone(contact_sheet.call_args_list[1].kwargs["frame_indices"])

    def test_main_forwards_explicit_mmpd_contact_indices(self):
        with (
            patch.object(diagnostic_module, "_read_avi", return_value=_video([0.25])),
            patch.object(diagnostic_module, "_read_mcd_state", return_value=[object()]),
            patch.object(diagnostic_module, "load_mmpd_mat", return_value={"video": _video([0.25]), "source_sha256": "source"}),
            patch.object(diagnostic_module, "_run_one", return_value=None) as run_one,
        ):
            exit_code = diagnostic_module.main([
                "--mcd-video", "mcd.avi", "--mcd-state-csv", "mcd.csv", "--mcd-clip-id", "mcd_clip",
                "--mmpd-mat", "mmpd.mat", "--mmpd-clip-id", "P1_1", "--face-landmarker-model", "model.task", "--output-dir", "out",
                "--mmpd-contact-frame-indices", "1310", "1311", "1312",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(run_one.call_args_list[1].kwargs["contact_sheet_frame_indices"], [1310, 1311, 1312])

    def test_invalid_selected_pos_is_reported(self):
        transition = SimpleNamespace(selected_measurement=_measurement(False), pre_belief=object(), post_belief=object(), hop_idx=0)
        failure = invalid_diagnostic_failure((_frame(True),), (SimpleNamespace(measurements=(_measurement(False),), hop_idx=0),), (transition,))
        self.assertEqual(failure["kind"], "selected_pos")


if __name__ == "__main__":
    unittest.main()
