from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .errors import ContractValidationError


def _json_value(value: Any, field: str = "payload") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not __import__("math").isfinite(value):
            raise ContractValidationError(f"{field}: NaN and infinity are not allowed")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value, field)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{field}: mapping keys must be strings")
            result[key] = _json_value(item, f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field) for item in value]
    raise ContractValidationError(f"{field}: unsupported JSON value")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractValidationError(f"file: cannot hash {path}") from exc
    return digest.hexdigest()


def verify_file_sha256(path: str | os.PathLike[str], expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ContractValidationError("expected: must be 64 lowercase hexadecimal characters")
    candidate = Path(path)
    if not candidate.is_file():
        raise ContractValidationError(f"file: missing or not a regular file: {path}")
    actual = sha256_file(candidate)
    if actual != expected:
        raise ContractValidationError(f"file: SHA-256 mismatch for {path}")


def canonical_json_bytes(payload: Any) -> bytes:
    safe = _json_value(payload)
    return (json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ContractValidationError(f"json: invalid numeric constant {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"json: duplicate key {key}")
        result[key] = value
    return result


def read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"json: cannot read {path}") from exc
    if not isinstance(value, dict):
        raise ContractValidationError("json: top-level value must be an object")
    return value


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: str | os.PathLike[str], payload: Any, *, overwrite: bool = False) -> None:
    target = Path(path)
    if not target.parent.is_dir():
        raise ContractValidationError("path: parent directory must already exist")
    data = canonical_json_bytes(payload)
    temp_name: str | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temp_name)
        if overwrite:
            os.replace(temporary, target)
            temp_name = None
        else:
            os.link(temporary, target)
            temporary.unlink()
            temp_name = None
        _fsync_parent(target)
    except FileExistsError as exc:
        raise ContractValidationError(f"path: refusing to overwrite {target}") from exc
    except OSError as exc:
        raise ContractValidationError(f"path: atomic write failed for {target}") from exc
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
