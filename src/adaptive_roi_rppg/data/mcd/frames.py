from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from pathlib import Path

from adaptive_roi_rppg.contracts import CanonicalFrame, ROIFrameValue, ROI_NAMES, require_structurally_complete
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from .adapter import MCD_DATASET_ID, MCDManifestBundle, MCD_SCHEMA_ID, MCD_STATE_SUFFIX
from .schema import MCD_STATE_COLUMNS

_FRAME = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _number(raw: str, field: str) -> float:
    try: value = float(raw)
    except (TypeError, ValueError) as exc: raise ContractValidationError(f"{field}: not numeric") from exc
    if not math.isfinite(value): _fail(f"{field}: must be finite")
    return value


def _path(locator: str, state_root: Path, expected: str) -> Path:
    if not isinstance(locator, str) or locator != f"state/{expected}" or locator.count("/") != 1 or "\\" in locator or ".." in locator or "/" in expected:
        _fail("state_locator: must be the direct expected state filename")
    if state_root.is_symlink() or not state_root.is_dir(): _fail("state_root: must be a real directory")
    candidate = state_root / expected
    if candidate.is_symlink() or not candidate.is_file() or candidate.parent.resolve() != state_root.resolve(): _fail("state file: must be a direct regular non-symlink file")
    return candidate


def _snapshot(path: Path) -> tuple[int, int, int, int]:
    try:
        value = os.lstat(path)
        if not stat.S_ISREG(value.st_mode): _fail("state file: must be a direct regular non-symlink file")
    except OSError as exc: raise ContractValidationError("state file: cannot stat") from exc
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _descriptor_snapshot(fd: int) -> tuple[int, int, int, int]:
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode): _fail("state file: descriptor is not a regular file")
    except OSError as exc: raise ContractValidationError("state file: cannot fstat descriptor") from exc
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def read_mcd_canonical_frames(bundle: MCDManifestBundle, state_root: str | os.PathLike[str], clip_id: str, required_split: str | None = None) -> tuple[CanonicalFrame, ...]:
    if not isinstance(bundle, MCDManifestBundle): _fail("bundle: must be an MCDManifestBundle")
    if required_split not in (None, "train", "eval"): _fail("required_split: must be train, eval, or null")
    require_structurally_complete(bundle.split_manifest); require_structurally_complete(bundle.dataset_manifest)
    clips = [clip for clip in bundle.clip_manifests if clip.clip_id == clip_id]
    if len(clips) != 1: _fail("clip_id: must select exactly one clip manifest")
    clip = clips[0]
    require_structurally_complete(clip)
    if bundle.split_manifest.dataset_id != MCD_DATASET_ID or bundle.dataset_manifest.dataset_id != MCD_DATASET_ID or bundle.dataset_manifest.schema_id != MCD_SCHEMA_ID or clip.dataset_id != MCD_DATASET_ID or clip.schema_id != MCD_SCHEMA_ID or clip.split_id != bundle.split_manifest.split_id or clip.clip_manifest_id not in bundle.dataset_manifest.clip_manifest_refs or clip.state_row_count <= 0:
        _fail("clip: does not match bundle dataset, schema, or split")
    if required_split == "train" and clip.clip_id not in bundle.split_manifest.train_clip_ids: _fail("clip_id: not in train split")
    if required_split == "eval" and clip.clip_id not in bundle.split_manifest.eval_clip_ids: _fail("clip_id: not in eval split")
    expected = f"{clip.clip_id}{MCD_STATE_SUFFIX}"
    path = _path(clip.state_locator, Path(state_root), expected)
    before = _snapshot(path)
    rows: list[list[str]] = []
    fd = None
    descriptor_before = None
    primary_error: BaseException | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if os.name != "posix" or nofollow is None: _fail("state file: safe descriptor open is unsupported")
        try:
            fd = os.open(path, os.O_RDONLY | nofollow)
            descriptor_before = _descriptor_snapshot(fd)
            if before != descriptor_before: _fail("state file: opened descriptor does not match initial path")
            captured = bytearray()
            while len(captured) < descriptor_before[2]:
                chunk = os.read(fd, descriptor_before[2] - len(captured))
                if not chunk: break
                captured.extend(chunk)
            descriptor_after = _descriptor_snapshot(fd)
        except OSError as exc:
            raise ContractValidationError("state file: cannot capture through descriptor") from exc
        if len(captured) != descriptor_before[2]: _fail("state file: short read")
        if descriptor_before != descriptor_after: _fail("state file: descriptor metadata changed during capture")
        captured_bytes = bytes(captured)
        digest = hashlib.sha256(captured_bytes).hexdigest()
        if digest != clip.state_sha256: _fail("state file: SHA-256 mismatch")
        try:
            handle = io.StringIO(captured_bytes.decode("utf-8", errors="strict"), newline="")
            reader = csv.reader(handle, strict=True)
            if tuple(next(reader, ())) != MCD_STATE_COLUMNS: _fail("state CSV: wrong header")
            for row in reader:
                if len(row) != len(MCD_STATE_COLUMNS): _fail("state CSV: wrong width")
                rows.append(row)
        except (UnicodeError, csv.Error) as exc: raise ContractValidationError("state CSV: cannot parse captured bytes") from exc
    except ContractValidationError as exc:
        primary_error = exc
    except OSError as exc:
        primary_error = ContractValidationError("state CSV: cannot parse captured bytes")
        primary_error.__cause__ = exc
    except BaseException as exc:
        primary_error = exc
    if fd is not None:
        try:
            os.close(fd)
        except OSError as exc:
            close_error = ContractValidationError("state file: cannot close descriptor")
            close_error.__cause__ = exc
            if primary_error is None: primary_error = close_error
            else: primary_error.add_note(str(close_error))
    if descriptor_before is not None:
        try:
            after = _snapshot(path)
            if after != descriptor_before: _fail("state file: final path does not match opened descriptor")
        except ContractValidationError as exc:
            if primary_error is None: primary_error = exc
            else: primary_error.add_note(str(exc))
    if primary_error is not None: raise primary_error
    if len(rows) != clip.state_row_count: _fail("state CSV: row count mismatch")
    result = []
    for expected_idx, row in enumerate(rows):
        if not _FRAME.fullmatch(row[0]) or int(row[0]) != expected_idx: _fail("state CSV: frame index is not canonical")
        pose_raw = row[1:4]; pose_missing = [raw == "" for raw in pose_raw]
        if any(pose_missing) and not all(pose_missing): _fail("state CSV: partial pose")
        pose = (None, None, None) if all(pose_missing) else tuple(_number(raw, "pose") for raw in pose_raw)
        values = []
        for roi_index in range(12):
            offset = 4 + roi_index * 5; fields = row[offset:offset + 4]; coverage = _number(row[offset + 4], "coverage")
            if not 0 <= coverage <= 1: _fail("coverage: outside [0,1]")
            missing = [raw == "" for raw in fields]
            if coverage == 0.0:
                if not all(missing): _fail("state CSV: zero coverage must have blank ROI values")
                values.append(ROIFrameValue(roi_index, ROI_NAMES[roi_index], None, None, None, None, 0.0, False, "source_missing", None, None))
            else:
                if any(missing): _fail("state CSV: positive coverage requires all ROI values")
                numbers = tuple(_number(raw, "ROI value") for raw in fields)
                values.append(ROIFrameValue(roi_index, ROI_NAMES[roi_index], *numbers[:3], numbers[3], coverage, True, None, None, None))
        result.append(CanonicalFrame(clip.dataset_id, clip.clip_id, expected_idx, expected_idx / clip.camera_fps, clip.camera_fps, *pose, tuple(values), clip.clip_manifest_id))
    return tuple(result)

__all__ = ["read_mcd_canonical_frames"]
