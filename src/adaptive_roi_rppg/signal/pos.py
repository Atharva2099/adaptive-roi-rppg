from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from typing import Sequence

import numpy as np
from scipy.signal import butter, lfilter, periodogram

from adaptive_roi_rppg.contracts import CanonicalFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement, canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError

_TOL = 1e-8
_TIME_TOL = 1e-9
_PRODUCTION_FPS = (24.0, 30.0)
_REASONS = ("missing_required_rgb", "degenerate_channel_mean", "degenerate_projection", "degenerate_pos_postprocess", "degenerate_spectrum")


class _SignalInvalid(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason

_PAYLOAD = {
    "profile": "causal-wang-pos",
    "version": 1,
    "signal_window_seconds": 8.0,
    "hop_seconds": 1.0,
    "production_fps": [24.0, 30.0],
    "wang_window_seconds": 1.6,
    "wang_window_rule": "ceil",
    "projection": [[0, 1, -1], [-2, 1, 1]],
    "population_ddof": 0,
    "pos_butter": {"ba_order": 1, "band_hz": [0.75, 3.0]},
    "hr_butter": {"ba_order": 3, "band_hz": [0.5, 3.0]},
    "filter": "scipy.signal.lfilter independent zero initial state each ROI/hop",
    "pos_combination": "S0+std(S0)/std(S1)*S1",
    "local_mean_removal": True,
    "subwindow_starts": "0..n-L inclusive",
    "overlap_add": "sum without count normalization",
    "bare_fps": "finite positive",
    "periodogram": {"window": "hann", "detrend": "constant", "return_onesided": True, "scaling": "density"},
    "hr_nfft": "max(4096,nextpow2(n))",
    "ppr_nfft": "n",
    "frequency_mask_hz": {"low_inclusive": 0.5, "high_inclusive": 3.0},
    "peak_tie": "first ascending argmax",
    "subharmonic": False,
    "confidence": "PPR*arithmetic_mean_coverage",
    "missing_data": "no fill or clipping",
    "degeneracy_threshold": 1e-8,
    "degeneracy_threshold_applications": [{"name": "channel_mean_abs", "threshold": 1e-8}, {"name": "projection_denominator_population_std", "threshold": 1e-8}, {"name": "pos_postfilter_population_std", "threshold": 1e-8}, {"name": "spectral_filter_population_std", "threshold": 1e-8}, {"name": "natural_inband_total_power", "threshold": 1e-8}],
    "timestamp_abs_tolerance": 1e-9,
    "invalid_reasons": list(_REASONS),
}


def _freeze(value: object) -> object:
    if isinstance(value, dict): return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list): return tuple(_freeze(item) for item in value)
    return value


POS_CONFIG_PAYLOAD = _freeze(_PAYLOAD)
POS_CONFIG_ID = "pos-v1-" + hashlib.sha256(canonical_json_bytes(_PAYLOAD)).hexdigest()


def _fail(message: str) -> None:
    if message in _REASONS:
        raise _SignalInvalid(message)
    raise ContractValidationError(message)


def _array(rgb: object, fps: object | None = None) -> tuple[np.ndarray, float]:
    if fps is None or isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        _fail("fps: must be a finite positive number")
    try:
        values = np.asarray(rgb, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("rgb: must be a finite numeric array") from exc
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0 or not np.isfinite(values).all():
        _fail("rgb: must have finite float64 shape (n, 3) with n > 0")
    return values, float(fps)


def _population_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0))


# Wang et al., 2017, Algorithmic Principles of Remote-PPG, IEEE Transactions on Biomedical Engineering 64(7):1479-1491, Algorithm 1, DOI 10.1109/TBME.2016.2609282.
def _wang_pos(rgb: object, fps: object) -> np.ndarray:
    values, rate = _array(rgb, fps)
    length = int(math.ceil(1.6 * rate))
    if len(values) < length:
        _fail("rgb: shorter than the Wang window")
    output = np.zeros(len(values), dtype=np.float64)
    projection = np.asarray(((0.0, 1.0, -1.0), (-2.0, 1.0, 1.0)), dtype=np.float64)
    for start in range(len(values) - length + 1):
        window = values[start:start + length]
        means = window.mean(axis=0)
        if np.any(np.abs(means) <= _TOL):
            _fail("degenerate_channel_mean")
        normalized = window / means
        signals = normalized @ projection.T
        denominator = _population_std(signals[:, 1])
        if not math.isfinite(denominator) or denominator <= _TOL:
            _fail("degenerate_projection")
        alpha = _population_std(signals[:, 0]) / denominator
        local = signals[:, 0] + alpha * signals[:, 1]
        output[start:start + length] += local - local.mean()
    return output


def wang_pos(rgb: object, fps: object) -> np.ndarray:
    try:
        return _wang_pos(rgb, fps)
    except _SignalInvalid as exc:
        raise ContractValidationError(exc.reason) from exc


def _postprocess(signal: np.ndarray, fps: float) -> np.ndarray:
    b, a = butter(1, [0.75, 3.0], btype="bandpass", fs=fps, output="ba")
    filtered = lfilter(b, a, signal)
    filtered = filtered - filtered.mean()
    scale = _population_std(filtered)
    if not np.isfinite(filtered).all() or not math.isfinite(scale) or scale <= _TOL:
        _fail("degenerate_pos_postprocess")
    return filtered / scale


def _spectrum(signal: np.ndarray, fps: float) -> tuple[float, float]:
    centered = signal - signal.mean()
    b, a = butter(3, [0.5, 3.0], btype="bandpass", fs=fps, output="ba")
    filtered = lfilter(b, a, centered)
    scale = _population_std(filtered)
    if not np.isfinite(filtered).all() or not math.isfinite(scale) or scale <= _TOL:
        _fail("degenerate_spectrum")
    n = len(filtered)
    frequencies, psd = periodogram(filtered, fs=fps, window="hann", detrend="constant", return_onesided=True, scaling="density", nfft=max(4096, 1 << (n - 1).bit_length()))
    natural_frequencies, natural_psd = periodogram(filtered, fs=fps, window="hann", detrend="constant", return_onesided=True, scaling="density", nfft=n)
    mask = (frequencies >= 0.5) & (frequencies <= 3.0)
    natural_mask = (natural_frequencies >= 0.5) & (natural_frequencies <= 3.0)
    if not mask.any() or not natural_mask.any() or not np.isfinite(psd).all() or not np.isfinite(natural_psd).all():
        _fail("degenerate_spectrum")
    total = float(natural_psd[natural_mask].sum())
    if not math.isfinite(total) or total <= _TOL:
        _fail("degenerate_spectrum")
    peak = int(np.flatnonzero(mask)[int(np.argmax(psd[mask]))])
    natural_peak = int(np.flatnonzero(natural_mask)[int(np.argmax(natural_psd[natural_mask]))])
    ppr = float(natural_psd[natural_peak]) / total
    return float(frequencies[peak] * 60.0), ppr


def _invalid(index: int, start: int, end: int, coverage: float, reason: str) -> ROIMeasurement:
    return ROIMeasurement(index, ROI_NAMES[index], None, None, None, coverage, False, reason, (), None, start, end, POS_CONFIG_ID)


def _measurement_provenance(dataset_id: str, clip_id: str, source_provenance_id: str, start: int, end: int) -> str:
    payload = {"dataset_id": dataset_id, "clip_id": clip_id, "source_provenance_id": source_provenance_id, "source_frame_start": start, "source_frame_end": end, "signal_config_id": POS_CONFIG_ID}
    return "pos-measurement-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_pos_measurements(frames: Sequence[CanonicalFrame]) -> tuple[MeasurementFrame, ...]:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        _fail("frames: must be a nonempty sequence")
    first = frames[0]
    if not isinstance(first, CanonicalFrame) or first.camera_fps not in _PRODUCTION_FPS:
        _fail("frames: production FPS must be exactly 24 or 30")
    fps = float(first.camera_fps)
    for index, frame in enumerate(frames):
        if not isinstance(frame, CanonicalFrame) or frame.dataset_id != first.dataset_id or frame.clip_id != first.clip_id or frame.provenance_id != first.provenance_id or frame.camera_fps != fps or frame.frame_idx != index or abs(frame.timestamp_s - index / fps) > _TIME_TOL:
            _fail("frames: identity, FPS, frame indices, or timestamps do not match")
    window = round(8.0 * fps)
    hop = round(1.0 * fps)
    if len(frames) < window:
        return ()
    output: list[MeasurementFrame] = []
    for end_exclusive in range(window, len(frames) + 1, hop):
        start, end = end_exclusive - window, end_exclusive - 1
        measurements: list[ROIMeasurement] = []
        for roi_index in range(len(ROI_NAMES)):
            values = [frames[i].roi_values[roi_index] for i in range(start, end_exclusive)]
            if any(value.coverage is None or not math.isfinite(value.coverage) for value in values):
                _fail("frames: coverage must be finite")
            coverage = float(np.mean([value.coverage for value in values]))
            has_imputation = any(value.imputation_age_frames or value.imputation_origin_frame_idx for value in values)
            rgb = np.asarray([[value.r_mean, value.g_mean, value.b_mean] for value in values], dtype=np.float64) if not has_imputation and all(value.valid and value.r_mean is not None and value.g_mean is not None and value.b_mean is not None for value in values) else None
            if rgb is None:
                measurements.append(_invalid(roi_index, start, end, coverage, "missing_required_rgb")); continue
            try:
                pos = _postprocess(_wang_pos(rgb, fps), fps)
                hr, ppr = _spectrum(pos, fps)
                confidence = ppr * coverage
                measurements.append(ROIMeasurement(roi_index, ROI_NAMES[roi_index], hr, confidence, ppr, coverage, True, None, (), None, start, end, POS_CONFIG_ID))
            except _SignalInvalid as exc:
                measurements.append(_invalid(roi_index, start, end, coverage, exc.reason))
        output.append(MeasurementFrame(first.dataset_id, first.clip_id, len(output), end_exclusive / fps, tuple(measurements), POS_CONFIG_ID, any(item.valid for item in measurements), None if any(item.valid for item in measurements) else "no_valid_roi_measurement", _measurement_provenance(first.dataset_id, first.clip_id, first.provenance_id, start, end)))
    return tuple(output)
