"""Explicit injectable extraction contract for Gate 9.

The production runner accepts a callable, never a precomputed per-hop file.
An implementation may use MediaPipe/MAT or a test double, but it must return
primary rows plus independent Oracle B/C rows before publication can begin.
"""
from __future__ import annotations
import importlib
import hashlib
import io
import math
from typing import Any, Callable, Mapping
import numpy as np
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts import CanonicalFrame, ROIFrameValue, ROI_NAMES
from adaptive_roi_rppg.signal import build_pos_measurements

def load_extractor(spec: str) -> Callable[..., Mapping[str, Any]]:
    if ":" not in spec: raise ContractValidationError("Gate 9 extractor must be module:function")
    module_name, function_name = spec.split(":", 1)
    try: function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc: raise ContractValidationError("Gate 9 extractor cannot be loaded") from exc
    if not callable(function): raise ContractValidationError("Gate 9 extractor is not callable")
    return function

def execute_extractor(extractor: Callable[..., Mapping[str, Any]], plan: Any, provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    try: result = extractor(plan=plan, provenance=dict(provenance))
    except TypeError as exc: raise ContractValidationError("Gate 9 extractor must accept plan= and provenance=") from exc
    if not isinstance(result, Mapping) or set(result) != {"per_hop", "oracle_b", "oracle_c"}: raise ContractValidationError("Gate 9 extractor must return per_hop, oracle_b, and oracle_c")
    for name in ("per_hop", "oracle_b", "oracle_c"):
        if not isinstance(result[name], (bytes, str)): raise ContractValidationError(f"Gate 9 extractor output {name} must be CSV bytes or text")
    return result

def load_mmpd_mat(path: str, *, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    """Authenticate and load one MMPD MAT source without changing its bytes."""
    from .raw_source import capture_source_bytes
    captured = capture_source_bytes(path, expected_sha256, expected_bytes)
    try:
        from scipy.io import loadmat
        values = loadmat(io.BytesIO(captured.data), squeeze_me=False, struct_as_record=False)
    except Exception as exc:
        raise ContractValidationError("Gate 9 extraction: source is not a readable MAT file") from exc
    if "video" not in values or "GT_ppg" not in values:
        raise ContractValidationError("Gate 9 extraction: MAT must contain video and GT_ppg")
    video = np.asarray(values["video"])
    gt = np.asarray(values["GT_ppg"], dtype=np.float64).reshape(-1)
    if video.ndim != 4 or video.shape[-1] < 3 or video.shape[0] != gt.size or video.shape[0] == 0:
        raise ContractValidationError("Gate 9 extraction: video/GT shapes are inconsistent")
    if not np.isfinite(video).all() or not np.isfinite(gt).all():
        raise ContractValidationError("Gate 9 extraction: video and GT must be finite")
    return {"video": video[..., :3], "gt_ppg": gt, "source_sha256": captured.sha256, "source_bytes": captured.byte_size}

def _rectangles(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float, float, float], ...]:
    """Return twelve deterministic face-relative ROI boxes in canonical order."""
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2.0
    return (
        (x0, y0, x1, y1),
        (x0 + .12*w, y0 + .05*h, cx - .04*w, y0 + .36*h),
        (cx + .04*w, y0 + .05*h, x1 - .12*w, y0 + .36*h),
        (x0 + .02*w, y0 + .25*h, x0 + .23*w, y0 + .50*h),
        (x1 - .23*w, y0 + .25*h, x1 - .02*w, y0 + .50*h),
        (cx - .12*w, y0 + .28*h, cx + .12*w, y0 + .70*h),
        (x0 + .10*w, y0 + .38*h, cx - .06*w, y0 + .62*h),
        (cx + .06*w, y0 + .38*h, x1 - .10*w, y0 + .62*h),
        (x0 + .12*w, y0 + .58*h, cx - .08*w, y0 + .84*h),
        (cx + .08*w, y0 + .58*h, x1 - .12*w, y0 + .84*h),
        (cx - .16*w, y0 + .66*h, cx + .16*w, y0 + .78*h),
        (cx - .18*w, y0 + .78*h, cx + .18*w, y1 - .03*h),
    )

def extract_mmpd_canonical_frames(video: np.ndarray, *, clip_id: str, source_provenance_id: str, model_asset_path: str, fps: float = 30.0) -> tuple[CanonicalFrame, ...]:
    """Run MediaPipe FaceMesh and produce canonical RGB traces for one MAT video."""
    if fps not in (24.0, 30.0):
        raise ContractValidationError("Gate 9 extraction: FPS must be 24 or 30")
    try:
        import mediapipe as mp
        from pathlib import Path
        model = Path(model_asset_path)
        if model.is_symlink() or not model.is_file():
            raise ContractValidationError("Gate 9 extraction: face-landmarker model must be a regular file")
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU), running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except ImportError as exc:
        raise ContractValidationError("Gate 9 extraction: mediapipe is required for MAT/video execution") from exc
    except (AttributeError, RuntimeError) as exc:
        raise ContractValidationError("Gate 9 extraction: MediaPipe CPU face-landmarker could not initialize") from exc
    if video.ndim != 4 or video.shape[-1] != 3 or not np.isfinite(video).all():
        raise ContractValidationError("Gate 9 extraction: video must be finite (frames,height,width,3)")
    scale = 255.0 if float(np.nanmax(video)) <= 1.0 else 1.0
    frames: list[CanonicalFrame] = []
    with detector:
        for frame_idx, raw in enumerate(video):
            image = np.clip(raw * scale, 0.0, 255.0).astype(np.uint8)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=image), int(frame_idx * 1000.0 / fps))
            values: list[ROIFrameValue] = []
            if not result.multi_face_landmarks:
                values = [ROIFrameValue(i, ROI_NAMES[i], None, None, None, None, 0.0, False, "no_face", None, None) for i in range(len(ROI_NAMES))]
            else:
                landmarks = result.multi_face_landmarks[0].landmark
                height, width = image.shape[:2]
                xs = np.asarray([point.x for point in landmarks], dtype=np.float64) * width
                ys = np.asarray([point.y for point in landmarks], dtype=np.float64) * height
                x0, x1 = float(np.clip(xs.min(), 0, width)), float(np.clip(xs.max(), 0, width))
                y0, y1 = float(np.clip(ys.min(), 0, height)), float(np.clip(ys.max(), 0, height))
                face_area = max(1.0, (x1 - x0) * (y1 - y0))
                for roi_idx, box in enumerate(_rectangles(x0, y0, x1, y1)):
                    xa, ya, xb, yb = (int(max(0, min(width, round(value)))) for value in box)
                    pixels = image[ya:yb, xa:xb, :3].astype(np.float64).reshape(-1, 3) if xb > xa and yb > ya else np.empty((0, 3))
                    if pixels.size == 0 or not np.isfinite(pixels).all():
                        values.append(ROIFrameValue(roi_idx, ROI_NAMES[roi_idx], None, None, None, None, 0.0, False, "empty_roi", None, None)); continue
                    coverage = float(pixels.shape[0] / face_area)
                    values.append(ROIFrameValue(roi_idx, ROI_NAMES[roi_idx], float(pixels[:, 0].mean()), float(pixels[:, 1].mean()), float(pixels[:, 2].mean()), float(pixels[:, 1].std()), coverage, True, None, None, None))
            frames.append(CanonicalFrame("mmpd", clip_id, frame_idx, frame_idx / fps, fps, None, None, None, tuple(values), source_provenance_id))
    return tuple(frames)

def extract_mmpd_measurements(path: str, *, clip_id: str, expected_sha256: str, expected_bytes: int, model_asset_path: str, fps: float = 30.0) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
    source = load_mmpd_mat(path, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    frames = extract_mmpd_canonical_frames(source["video"], clip_id=clip_id, source_provenance_id=source["source_sha256"], model_asset_path=model_asset_path, fps=fps)
    return build_pos_measurements(frames), source["gt_ppg"], source
