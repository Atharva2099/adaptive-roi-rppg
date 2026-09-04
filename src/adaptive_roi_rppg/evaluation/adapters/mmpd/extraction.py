"""Explicit injectable extraction contract for Gate 9.

The production runner accepts a callable, never a precomputed per-hop file.
An implementation may use MediaPipe/MAT or a test double, but it must return
primary rows plus independent Oracle B/C rows before publication can begin.
"""
from __future__ import annotations
import importlib
import io
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts import CanonicalFrame, ROIFrameValue, ROI_NAMES
from adaptive_roi_rppg.signal import build_pos_measurements


def trim_mmpd_terminal_zero_suffix(video: np.ndarray, gt_ppg: np.ndarray, *, context: str = "MMPD source") -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate an MMPD source and remove only its exact-zero terminal suffix."""
    video = np.asarray(video)
    gt_ppg = np.asarray(gt_ppg, dtype=np.float64).reshape(-1)
    if video.ndim != 4 or video.shape[-1] < 3 or video.shape[0] == 0 or video.shape[0] != gt_ppg.size:
        raise ContractValidationError(f"{context}: video/GT shapes are inconsistent")
    rgb = np.asarray(video[..., :3], dtype=np.float64)
    if not np.isfinite(video).all() or not np.isfinite(gt_ppg).all():
        raise ContractValidationError(f"{context}: video and GT must be finite")
    zero = np.all(rgb == 0, axis=(1, 2, 3))
    last_nonzero = int(np.flatnonzero(~zero)[-1]) if np.any(~zero) else -1
    if last_nonzero < 0:
        raise ContractValidationError(f"{context}: video is all zero")
    if np.any(zero[:last_nonzero]):
        raise ContractValidationError(f"{context}: zero frame occurs before terminal suffix")
    original = int(video.shape[0])
    trimmed = original - last_nonzero - 1
    return video[:last_nonzero + 1, ..., :3], gt_ppg[:last_nonzero + 1], original, trimmed

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
    video, gt, original, trimmed = trim_mmpd_terminal_zero_suffix(video, gt, context="Gate 9 extraction")
    return {"video": video, "gt_ppg": gt, "source_sha256": captured.sha256, "source_bytes": captured.byte_size,
            "original_frame_count": original, "processed_frame_count": int(video.shape[0]), "trimmed_tail_frames": trimmed}

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


def _face_landmarks_from_result(result: Any) -> Any:
    """Read the MediaPipe Tasks result field used by the current runner."""
    value = getattr(result, "face_landmarks", None)
    if value is None:
        raise ContractValidationError("Gate 9 extraction: MediaPipe result lacks face_landmarks")
    return value


def _direct_face_box(face_landmarks: Any, *, width: int, height: int, clip_id: str, frame_idx: int) -> tuple[float, float, float, float]:
    """Convert one direct MediaPipe result to a validated scalar face box."""
    try:
        landmarks = face_landmarks[0]
        xs = np.asarray([point.x for point in landmarks], dtype=np.float64) * width
        ys = np.asarray([point.y for point in landmarks], dtype=np.float64) * height
    except (IndexError, TypeError, AttributeError, ValueError) as exc:
        raise ContractValidationError(
            f"Gate 9 extraction: invalid face geometry for clip_id={clip_id}, frame_idx={frame_idx}"
        ) from exc
    if xs.size == 0 or ys.size == 0 or not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ContractValidationError(
            f"Gate 9 extraction: invalid face geometry for clip_id={clip_id}, frame_idx={frame_idx}"
        )
    x0, x1 = float(np.clip(xs.min(), 0, width)), float(np.clip(xs.max(), 0, width))
    y0, y1 = float(np.clip(ys.min(), 0, height)), float(np.clip(ys.max(), 0, height))
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 <= x0 or y1 <= y0:
        raise ContractValidationError(
            f"Gate 9 extraction: invalid face geometry for clip_id={clip_id}, frame_idx={frame_idx}"
        )
    return (x0, y0, x1, y1)


def _landmark_summary(face_landmarks: Any) -> dict[str, Any]:
    """Return a compact JSON-safe description of attempted landmark geometry."""
    try:
        landmarks = face_landmarks[0]
        points = list(landmarks)
    except (IndexError, TypeError):
        return {"landmark_count": None, "normalized_bounds": None}
    values: list[tuple[float, float]] = []
    try:
        for point in points:
            values.append((float(point.x), float(point.y)))
    except (AttributeError, TypeError, ValueError):
        return {"landmark_count": len(points), "normalized_bounds": None}
    if not values:
        return {"landmark_count": 0, "normalized_bounds": None}
    coordinates = np.asarray(values, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        return {"landmark_count": len(points), "normalized_bounds": None, "contains_nonfinite": True}
    return {
        "landmark_count": len(points),
        "normalized_bounds": [
            float(coordinates[:, 0].min()), float(coordinates[:, 1].min()),
            float(coordinates[:, 0].max()), float(coordinates[:, 1].max()),
        ],
    }


def _record_geometry_failure(
    failed_diagnostics: list[dict[str, Any]] | None,
    *,
    frame_idx: int,
    geometry_source: str,
    reason: str,
    face_landmarks: Any,
    pixel_bounds: tuple[int, int, int, int] | None = None,
) -> None:
    if failed_diagnostics is not None:
        record: dict[str, Any] = {
            "frame_idx": frame_idx,
            "geometry_source": geometry_source,
            "reason": reason,
            "landmark_summary": _landmark_summary(face_landmarks),
        }
        if pixel_bounds is not None:
            record["attempted_pixel_bounds"] = list(pixel_bounds)
        failed_diagnostics.append(record)


def _roi_pixel_bounds(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    roi_name: str,
    clip_id: str,
    frame_idx: int,
) -> tuple[int, int, int, int]:
    """Convert a float ROI to an in-image pixel crop without inventing pixels."""
    x0, y0, x1, y1 = box
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 <= x0 or y1 <= y0:
        raise ContractValidationError(
            f"Gate 9 extraction: empty ROI {roi_name} for clip_id={clip_id}, frame_idx={frame_idx}"
        )
    xa, ya = max(0, min(width, math.floor(x0))), max(0, min(height, math.floor(y0)))
    xb, yb = max(0, min(width, math.ceil(x1))), max(0, min(height, math.ceil(y1)))
    if xb <= xa or yb <= ya:
        raise ContractValidationError(
            f"Gate 9 extraction: empty ROI {roi_name} for clip_id={clip_id}, frame_idx={frame_idx}"
        )
    return xa, ya, xb, yb


def extract_face_relative_canonical_frames(
    video: np.ndarray,
    *,
    dataset_id: str,
    clip_id: str,
    source_provenance_id: str,
    model_asset_path: str,
    fps: float = 30.0,
    diagnostics: list[dict[str, Any]] | None = None,
    failed_diagnostics: list[dict[str, Any]] | None = None,
    emitted_frames: list[CanonicalFrame] | None = None,
    reject_exact_zero_raw_frames: bool = False,
) -> tuple[CanonicalFrame, ...]:
    """Extract canonical face-relative RGB traces from an in-memory RGB video.

    ``diagnostics`` contains completed frame/value pairs.  Failures are kept
    separately so callers can report them without misaligning completed rows.
    """
    if fps not in (24.0, 30.0):
        raise ContractValidationError("Gate 9 extraction: FPS must be 24 or 30")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ContractValidationError("Gate 9 extraction: video must have shape (frames,height,width,3)")
    if video.shape[0] == 0:
        raise ContractValidationError("Gate 9 extraction: video must contain at least one frame")
    try:
        import mediapipe as mp
        version = getattr(mp, "__version__", "")
        if not isinstance(version, str) or not version.startswith("0.10."):
            raise ContractValidationError("Gate 9 extraction: tested MediaPipe Tasks 0.10.x is required")
        model = Path(model_asset_path)
        if model.is_symlink() or not model.is_file():
            raise ContractValidationError("Gate 9 extraction: face-landmarker model must be a regular file")
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU)
        video_options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        image_options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.5)
    except ImportError as exc:
        raise ContractValidationError("Gate 9 extraction: mediapipe is required for MAT/video execution") from exc
    except (AttributeError, RuntimeError) as exc:
        raise ContractValidationError("Gate 9 extraction: MediaPipe CPU face-landmarker could not initialize") from exc
    scale = 255.0 if float(np.nanmax(video)) <= 1.0 else 1.0
    frames: list[CanonicalFrame] = []
    with ExitStack() as stack:
        try:
            video_landmarker = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(video_options))
            image_landmarker = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(image_options))
        except ContractValidationError:
            raise
        except (AttributeError, RuntimeError, OSError) as exc:
            raise ContractValidationError("Gate 9 extraction: MediaPipe face-landmarker could not initialize") from exc
        for frame_idx, raw in enumerate(video):
            if not np.isfinite(raw).all():
                raise ContractValidationError(
                    f"Gate 9 extraction: non-finite raw frame for clip_id={clip_id}, frame_idx={frame_idx}"
                )
            if reject_exact_zero_raw_frames and np.all(raw == 0):
                raise ContractValidationError(
                    f"Gate 9 extraction: exact all-zero raw frame for clip_id={clip_id}, frame_idx={frame_idx}"
                )
            rgb_u8 = np.clip(raw * scale, 0.0, 255.0).astype(np.uint8)
            image = np.ascontiguousarray(rgb_u8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            height, width = image.shape[:2]
            recovery_attempts = ["video"]
            video_landmarks = _face_landmarks_from_result(
                video_landmarker.detect_for_video(mp_image, int(frame_idx * 1000.0 / fps))
            )
            if not isinstance(video_landmarks, list):
                raise ContractValidationError("Gate 9 extraction: face_landmarks must be a list")
            if video_landmarks:
                try:
                    face_box = _direct_face_box(video_landmarks, width=width, height=height, clip_id=clip_id, frame_idx=frame_idx)
                except ContractValidationError as exc:
                    _record_geometry_failure(failed_diagnostics, frame_idx=frame_idx, geometry_source="video", reason=str(exc), face_landmarks=video_landmarks)
                    raise
                geometry_source = "video"
            else:
                recovery_attempts.append("image")
                image_landmarks = _face_landmarks_from_result(image_landmarker.detect(mp_image))
                if not isinstance(image_landmarks, list):
                    raise ContractValidationError("Gate 9 extraction: face_landmarks must be a list")
                if image_landmarks:
                    try:
                        face_box = _direct_face_box(image_landmarks, width=width, height=height, clip_id=clip_id, frame_idx=frame_idx)
                    except ContractValidationError as exc:
                        _record_geometry_failure(failed_diagnostics, frame_idx=frame_idx, geometry_source="image", reason=str(exc), face_landmarks=image_landmarks)
                        raise
                    geometry_source = "image"
                else:
                    if failed_diagnostics is not None:
                        failed_diagnostics.append({"frame_idx": frame_idx, "reason": "no_current_frame_landmarks", "recovery_attempts": recovery_attempts})
                    raise ContractValidationError(f"Gate 9 extraction: no current-frame landmarks for clip_id={clip_id}, frame_idx={frame_idx}")
            geometry_source_frame_idx = frame_idx
            values: list[ROIFrameValue] = []
            diagnostic: dict[str, Any] = {
                "frame_idx": frame_idx,
                "face_box": face_box,
                "roi_boxes": [],
                "face_valid": True,
                "geometry_source": geometry_source,
                "geometry_source_frame_idx": geometry_source_frame_idx,
                "geometry_bridged": False,
                "current_frame_evidence": True,
                "recovery_attempts": recovery_attempts,
            }
            x0, y0, x1, y1 = face_box
            face_area = max(1.0, (x1 - x0) * (y1 - y0))
            for roi_idx, box in enumerate(_rectangles(x0, y0, x1, y1)):
                try:
                    xa, ya, xb, yb = _roi_pixel_bounds(
                        box, width=width, height=height, roi_name=ROI_NAMES[roi_idx].value,
                        clip_id=clip_id, frame_idx=frame_idx,
                    )
                except ContractValidationError as exc:
                    diagnostic["roi_boxes"].append(None)
                    if failed_diagnostics is not None:
                        failed_diagnostics.append({
                            "frame_idx": frame_idx,
                            "reason": str(exc),
                            "face_box": face_box,
                            "attempted_roi_boxes": diagnostic["roi_boxes"][:-1] + [box],
                            "failed_roi_name": ROI_NAMES[roi_idx].value,
                        })
                    raise
                diagnostic["roi_boxes"].append((xa, ya, xb, yb))
                pixels = image[ya:yb, xa:xb, :3].astype(np.float64).reshape(-1, 3)
                if pixels.size == 0 or not np.isfinite(pixels).all():
                    reason = f"Gate 9 extraction: empty ROI {ROI_NAMES[roi_idx].value} for clip_id={clip_id}, frame_idx={frame_idx}"
                    if failed_diagnostics is not None:
                        failed_diagnostics.append({
                            "frame_idx": frame_idx,
                            "reason": reason,
                            "face_box": face_box,
                            "attempted_roi_boxes": diagnostic["roi_boxes"],
                            "failed_roi_name": ROI_NAMES[roi_idx].value,
                        })
                    raise ContractValidationError(
                        reason
                    )
                coverage = min(1.0, float(pixels.shape[0] / face_area))
                values.append(ROIFrameValue(roi_idx, ROI_NAMES[roi_idx], float(pixels[:, 0].mean()), float(pixels[:, 1].mean()), float(pixels[:, 2].mean()), float(pixels[:, 1].std()), coverage, True, None, None, None))
            canonical = CanonicalFrame(dataset_id, clip_id, frame_idx, frame_idx / fps, fps, None, None, None, tuple(values), source_provenance_id)
            frames.append(canonical)
            if emitted_frames is not None:
                emitted_frames.append(canonical)
            if diagnostics is not None:
                diagnostics.append(diagnostic)
    return tuple(frames)


def extract_mmpd_canonical_frames(video: np.ndarray, *, clip_id: str, source_provenance_id: str, model_asset_path: str, fps: float = 30.0) -> tuple[CanonicalFrame, ...]:
    """Run the shared face-relative extractor for one MAT video."""
    return extract_face_relative_canonical_frames(video, dataset_id="mmpd", clip_id=clip_id, source_provenance_id=source_provenance_id, model_asset_path=model_asset_path, fps=fps, reject_exact_zero_raw_frames=True)

def extract_mmpd_measurements(path: str, *, clip_id: str, expected_sha256: str, expected_bytes: int, model_asset_path: str, fps: float = 30.0) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
    source = load_mmpd_mat(path, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    frames = extract_mmpd_canonical_frames(source["video"], clip_id=clip_id, source_provenance_id=source["source_sha256"], model_asset_path=model_asset_path, fps=fps)
    return build_pos_measurements(frames), source["gt_ppg"], source
