"""Authenticated raw-source boundary.  It never follows a source symlink."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any
from adaptive_roi_rppg.contracts.errors import ContractValidationError

@dataclass(frozen=True, slots=True)
class MMPDSourceCapture:
    path: Path
    data: bytes
    sha256: str
    byte_size: int

def _fail(message: str) -> None: raise ContractValidationError("MMPD source: " + message)

def _safe_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file(): _fail("source must be a regular non-symlink file")
    current = path
    while current != current.parent:
        # macOS commonly exposes /tmp and /var through the /private mount;
        # those fixed system aliases are safe and are not user-controlled.
        if current not in {Path("/tmp"), Path("/var"), Path("/private")} and current.is_symlink(): _fail("every existing path component must be non-symlink")
        current = current.parent

def capture_source_bytes(path: str | os.PathLike[str], expected_sha256: str | None = None, expected_bytes: int | None = None) -> MMPDSourceCapture:
    target = Path(path)
    _safe_file(target)
    before = target.stat()
    try: data = target.read_bytes()
    except OSError as exc: raise ContractValidationError("MMPD source: source cannot be read") from exc
    after = target.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino): _fail("source changed during capture")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256: _fail("SHA-256 mismatch")
    if expected_bytes is not None and len(data) != expected_bytes: _fail("byte-size mismatch")
    return MMPDSourceCapture(target, data, digest, len(data))

def validate_authenticated_file(path: str | os.PathLike[str], expected_sha256: str, expected_bytes: int) -> None:
    capture_source_bytes(path, expected_sha256, expected_bytes)

def load_raw_mat_contract(path: str | os.PathLike[str], expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    capture = capture_source_bytes(path, expected_sha256, expected_bytes)
    try:
        from scipy.io import loadmat
        import io
        values = loadmat(io.BytesIO(capture.data), squeeze_me=False, struct_as_record=False)
    except Exception as exc: raise ContractValidationError("MMPD source: MAT file cannot be loaded") from exc
    approved = {"video", "GT_ppg", "light", "motion", "exercise", "skin_color"}
    if not approved.intersection(values): _fail("MAT has no approved variables")
    return {"source_sha256": capture.sha256, "source_bytes": capture.byte_size, **{key: values[key] for key in approved if key in values}}
