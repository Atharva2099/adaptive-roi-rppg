"""Repository-owned, frozen MMPD Gate 9 corrected-GT rule."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.signal import butter, lfilter, periodogram

from adaptive_roi_rppg.contracts import LabelFrame
from adaptive_roi_rppg.contracts.errors import ContractValidationError

MMPD_GT_RULE_ID = "mmpd_causal_periodogram_peak_v1"
MMPD_FPS = 30.0
WINDOW_SECONDS = 8.0
HOP_SECONDS = 1.0


def build_mmpd_labels(gt_ppg: Sequence[float], *, clip_id: str, fps: float = MMPD_FPS,
                      hop_count: int = 53) -> tuple[LabelFrame, ...]:
    """Apply the fixed 30 Hz, 8-second causal-endpoint periodogram ruler."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != MMPD_FPS:
        raise ContractValidationError("Gate 9 MMPD rule requires 30 Hz")
    if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count != 53:
        raise ContractValidationError("Gate 9 MMPD rule requires the exact 53-hop lattice")
    values = np.asarray(gt_ppg, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return tuple(LabelFrame("mmpd", clip_id, i, WINDOW_SECONDS + i, None, MMPD_GT_RULE_ID, False, "nonfinite_or_empty_gt") for i in range(hop_count))
    labels: list[LabelFrame] = []
    half = round(4.0 * MMPD_FPS)
    window = round(WINDOW_SECONDS * MMPD_FPS)
    for hop in range(hop_count):
        end = window + hop * round(HOP_SECONDS * MMPD_FPS)
        segment = values[max(0, end - half):min(values.size, end + half)]
        reason: str | None = None
        hr: float | None = None
        if segment.size < 8:
            reason = "insufficient_finite_segment"
        else:
            centered = segment - float(segment.mean())
            if not math.isfinite(float(centered.std())) or float(centered.std()) <= 0.0:
                reason = "degenerate_variation"
            else:
                try:
                    b, a = butter(3, [0.5, 3.0], btype="bandpass", fs=MMPD_FPS, output="ba")
                    filtered = lfilter(b, a, centered)
                    nfft = max(4096, 1 << (len(filtered) - 1).bit_length())
                    frequencies, power = periodogram(filtered, fs=MMPD_FPS, window="hann", detrend="constant", return_onesided=True, scaling="density", nfft=nfft)
                    mask = (frequencies >= 0.5) & (frequencies <= 3.0)
                    if not mask.any() or not np.isfinite(power).all() or not np.any(power[mask] > 0):
                        reason = "degenerate_spectrum"
                    else:
                        peak = np.flatnonzero(mask)[int(np.argmax(power[mask]))]
                        hr = float(frequencies[peak] * 60.0)
                        if not math.isfinite(hr):
                            hr, reason = None, "degenerate_spectrum"
                except (ValueError, FloatingPointError):
                    reason = "degenerate_spectrum"
        labels.append(LabelFrame("mmpd", clip_id, hop, WINDOW_SECONDS + hop, hr, MMPD_GT_RULE_ID, hr is not None, reason))
    return tuple(labels)


__all__ = ["MMPD_FPS", "MMPD_GT_RULE_ID", "build_mmpd_labels"]
