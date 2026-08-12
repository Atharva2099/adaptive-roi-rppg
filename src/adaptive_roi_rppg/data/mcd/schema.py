"""Exact raw CSV columns for the MCD Gate 2 adapter."""

from adaptive_roi_rppg.contracts.constants import ROI_NAMES

MCD_STATE_COLUMNS = ("frame_idx", "head_yaw", "head_pitch", "head_roll", *(f"{roi.value}_{field}" for roi in ROI_NAMES for field in ("r_mean", "g_mean", "b_mean", "std", "coverage")))
MCD_GT_COLUMNS = ("frame_idx", "gt_ppg")
