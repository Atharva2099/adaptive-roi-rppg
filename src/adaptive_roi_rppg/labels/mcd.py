"""Authenticated, historical schema-v3 MCD GT labels."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import stat
from types import MappingProxyType
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, periodogram

from adaptive_roi_rppg.contracts import ClipManifest, LabelFrame, canonical_json_bytes, require_structurally_complete, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd.adapter import MCDManifestBundle, MCD_DATASET_ID, MCD_GT_SUFFIX, MCD_SCHEMA_ID
from adaptive_roi_rppg.data.mcd.schema import MCD_GT_COLUMNS

GT_RULE_ID = "causal_periodogram_peak_no_subharmonic_v3"
_GT_RULE_PAYLOAD = {
    "rule_id": GT_RULE_ID, "window_seconds": 8.0, "hop_seconds": 1.0,
    "segment_half_seconds": 4.0, "minimum_finite_samples": 8,
    "filter": {"kind": "butter", "order": 3, "band_hz": [0.5, 3.0], "method": "lfilter", "initial_state": "zero", "output": "ba"},
    "periodogram": {"window": "hann", "detrend": "constant", "return_onesided": True, "scaling": "density", "nfft": "max(4096,nextpow2(n))", "peak_band_hz_inclusive": True, "tie_break": "first_ascending", "subharmonic_correction": False},
}


def _freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list): return tuple(_freeze(v) for v in value)
    return value


GT_RULE_PAYLOAD = _freeze(_GT_RULE_PAYLOAD)
GT_RULE_PAYLOAD_SHA256 = hashlib.sha256(canonical_json_bytes(_GT_RULE_PAYLOAD)).hexdigest()


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _read_descriptor(path: Path, expected_count: int) -> tuple[np.ndarray, str]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if os.name != "posix" or nofollow is None: _fail("GT file: safe descriptor open is unsupported")

    def snapshot(target: Path) -> tuple[int, int, int, int]:
        try:
            value = os.lstat(target)
            if not stat.S_ISREG(value.st_mode): _fail("GT file must be a direct regular non-symlink file")
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        except OSError as exc:
            raise ContractValidationError("GT file: cannot stat") from exc

    def descriptor_snapshot(descriptor: int) -> tuple[int, int, int, int]:
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode): _fail("GT file: descriptor is not a regular file")
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        except OSError as exc:
            raise ContractValidationError("GT file: cannot fstat descriptor") from exc

    initial_path = snapshot(path)
    fd = None
    descriptor_before = None
    primary_error: BaseException | None = None
    values: list[float] = []
    try:
        try:
            fd = os.open(path, os.O_RDONLY | nofollow)
            descriptor_before = descriptor_snapshot(fd)
            if initial_path != descriptor_before: _fail("GT file: opened descriptor does not match initial path")
            captured = bytearray()
            while len(captured) < descriptor_before[2]:
                chunk = os.read(fd, descriptor_before[2] - len(captured))
                if not chunk: break
                captured.extend(chunk)
            descriptor_after = descriptor_snapshot(fd)
            if len(captured) != descriptor_before[2]: _fail("GT file: short read")
            if descriptor_after != descriptor_before: _fail("GT file: descriptor changed during capture")
            captured_bytes = bytes(captured)
            digest = hashlib.sha256(captured_bytes).hexdigest()
            try:
                text = captured_bytes.decode("utf-8", errors="strict")
                reader = csv.reader(io.StringIO(text, newline=""), strict=True)
                if tuple(next(reader, ())) != MCD_GT_COLUMNS: _fail("GT header must be exactly frame_idx,gt_ppg")
                indices = []
                for row in reader:
                    if len(row) != 2 or not row[0].isdigit() or (row[0] != "0" and row[0].startswith("0")): _fail("GT frame indices must be canonical")
                    index = int(row[0])
                    if index != len(indices): _fail("GT frame indices must equal range(row_count)")
                    try: value = float(row[1])
                    except ValueError as exc: raise ContractValidationError("GT values must be numeric") from exc
                    if not math.isfinite(value): _fail("GT values must be finite")
                    indices.append(index); values.append(value)
            except (UnicodeError, csv.Error) as exc:
                raise ContractValidationError("GT CSV cannot be parsed") from exc
        except ContractValidationError as exc:
            primary_error = exc
        except OSError as exc:
            primary_error = ContractValidationError("GT file: cannot capture through descriptor")
            primary_error.__cause__ = exc
        except BaseException as exc:
            primary_error = exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                close_error = ContractValidationError("GT file: cannot close descriptor")
                close_error.__cause__ = exc
                if primary_error is None: primary_error = close_error
                else: primary_error.add_note(str(close_error))
        if descriptor_before is not None:
            try:
                if snapshot(path) != descriptor_before: _fail("GT file: final path does not match opened descriptor")
            except ContractValidationError as exc:
                if primary_error is None: primary_error = exc
                else: primary_error.add_note(str(exc))
    if primary_error is not None: raise primary_error
    if len(values) != expected_count: _fail("GT file: row count does not match authenticated state")
    return np.asarray(values, dtype=np.float64), digest


def _gt_path(bundle: MCDManifestBundle, gt_root: str | os.PathLike[str], clip_id: str, required_split: str = "train"):
    if not isinstance(bundle, MCDManifestBundle): _fail("bundle must be an MCDManifestBundle")
    require_structurally_complete(bundle.split_manifest); require_structurally_complete(bundle.dataset_manifest)
    canonical = {c.clip_id: c for c in bundle.clip_manifests}
    if len(canonical) != len(bundle.clip_manifests): _fail("bundle clip IDs must be unique")
    clips = [canonical[clip_id]] if clip_id in canonical else []
    if len(clips) != 1: _fail("clip_id must select exactly one clip")
    clip = clips[0]; require_structurally_complete(clip)
    references = dict(zip(bundle.dataset_manifest.clip_manifest_refs, bundle.dataset_manifest.clip_manifest_hashes))
    expected_hash = hashlib.sha256(canonical_json_bytes(clip.to_dict())).hexdigest()
    if references.get(clip.clip_manifest_id) != expected_hash: _fail("clip is not an exact dataset-manifest reference")
    if clip.dataset_id != MCD_DATASET_ID or clip.schema_id != MCD_SCHEMA_ID or clip.gt_locator != f"gt/{clip_id}{MCD_GT_SUFFIX}" or clip.gt_sha256 is None or clip.gt_row_count is None or clip.gt_row_count != clip.state_row_count:
        _fail("clip GT manifest identity is invalid")
    if required_split not in ("train", "eval"):
        _fail("required_split must be train or eval")
    allowed = bundle.split_manifest.train_clip_ids if required_split == "train" else bundle.split_manifest.eval_clip_ids
    if clip.split_id != bundle.split_manifest.split_id or clip_id not in allowed:
        _fail(f"clip must belong to the {required_split} split")
    root = Path(gt_root)
    if root.is_symlink() or not root.is_dir(): _fail("gt_root must be a real directory")
    path = root / f"{clip_id}{MCD_GT_SUFFIX}"
    if path.is_symlink() or not path.is_file() or path.parent.resolve() != root.resolve(): _fail("GT file must be direct under gt_root")
    return clip, path


def _read_gt(path: Path, expected_count: int) -> tuple[np.ndarray, str]:
    return _read_descriptor(path, expected_count)


# Historical schema-v3 provenance: cache manifest -> estimate_clip -> gt_hr_centered(center_frame=end); this is ruler compatibility, not a paper-algorithm claim.
def _label(signal: np.ndarray, fps: float, end: int, hop: int) -> tuple[float | None, bool, str | None]:
    half = round(4 * fps); segment = signal[max(0, end - half):min(len(signal), end + half)]
    if len(segment) < 8 or not np.isfinite(segment).all(): return None, False, "insufficient_finite_segment"
    centered = segment - segment.mean()
    if not math.isfinite(float(np.std(centered))) or float(np.std(centered)) <= 0.0: return None, False, "degenerate_variation"
    try:
        b, a = butter(3, [0.5, 3.0], btype="bandpass", fs=fps, output="ba")
        filtered = lfilter(b, a, centered)
        if not np.isfinite(filtered).all() or float(np.std(filtered)) <= 0.0: return None, False, "degenerate_spectrum"
        n = len(filtered); nfft = max(4096, 1 << (n - 1).bit_length())
        frequencies, power = periodogram(filtered, fs=fps, window="hann", detrend="constant", return_onesided=True, scaling="density", nfft=nfft)
        mask = (frequencies >= 0.5) & (frequencies <= 3.0)
        if not mask.any() or not np.isfinite(power).all() or not np.any(power[mask] > 0): return None, False, "degenerate_spectrum"
        if float(np.sum(power[mask])) <= 0.0: return None, False, "degenerate_spectrum"
        selected = np.flatnonzero(mask); peak = selected[int(np.argmax(power[mask]))]
        value = float(frequencies[peak] * 60.0)
        if not math.isfinite(value): return None, False, "degenerate_spectrum"
        return value, True, None
    except (ValueError, FloatingPointError):
        return None, False, "degenerate_spectrum"


def read_mcd_labels(bundle: MCDManifestBundle, gt_root: str | os.PathLike[str], clip_id: str, required_split: str = "train", clip_manifest: ClipManifest | None = None) -> tuple[LabelFrame, ...]:
    if required_split != "train": _fail("required_split must be train for Gate 5")
    clip, path = _gt_path(bundle, gt_root, clip_id, "train")
    if clip_manifest is not None and (not isinstance(clip_manifest, ClipManifest) or clip_manifest != clip): _fail("supplied clip is not the exact canonical bundle clip")
    values, digest = _read_gt(path, clip.gt_row_count)
    if digest != clip.gt_sha256: _fail("GT SHA-256 mismatch")
    if clip.camera_fps not in (24.0, 30.0): _fail("camera FPS must be exactly 24 or 30")
    fps = float(clip.camera_fps); window = round(8 * fps); hop = round(1 * fps)
    frames = []
    for hop_idx, end in enumerate(range(window, len(values) + 1, hop)):
        hr, valid, reason = _label(values, fps, end, hop_idx)
        frames.append(LabelFrame(clip.dataset_id, clip.clip_id, hop_idx, end / fps, hr, GT_RULE_ID, valid, reason))
    return tuple(frames)


def read_mcd_eval_labels(bundle: MCDManifestBundle, gt_root: str | os.PathLike[str], clip_id: str, clip_manifest: ClipManifest | None = None) -> tuple[LabelFrame, ...]:
    """Read the authenticated MCD test90 labels for the separate Gate 6 consumer."""
    clip, path = _gt_path(bundle, gt_root, clip_id, "eval")
    if clip_manifest is not None and (not isinstance(clip_manifest, ClipManifest) or clip_manifest != clip): _fail("supplied clip is not the exact canonical bundle clip")
    values, digest = _read_gt(path, clip.gt_row_count)
    if digest != clip.gt_sha256: _fail("GT SHA-256 mismatch")
    if clip.camera_fps not in (24.0, 30.0): _fail("camera FPS must be exactly 24 or 30")
    fps = float(clip.camera_fps); window = round(8 * fps); hop = round(1 * fps)
    frames = []
    for hop_idx, end in enumerate(range(window, len(values) + 1, hop)):
        hr, valid, reason = _label(values, fps, end, hop_idx)
        frames.append(LabelFrame(clip.dataset_id, clip.clip_id, hop_idx, end / fps, hr, GT_RULE_ID, valid, reason))
    return tuple(frames)


__all__ = ["GT_RULE_ID", "GT_RULE_PAYLOAD", "GT_RULE_PAYLOAD_SHA256", "read_mcd_labels", "read_mcd_eval_labels"]
