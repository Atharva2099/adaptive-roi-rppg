from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .constants import ManifestStatus, OverlapResult
from .errors import ContractValidationError
from .records import _error, _freeze, _optional_int, _required_string, _strict, _thaw, _int, _number, _tuple_of_strings

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _hash(value: Any, field: str) -> str:
    value = _required_string(value, field)
    if not _SHA256.fullmatch(value):
        raise _error(field, "must be 64 lowercase hexadecimal characters")
    return value


def _timestamp(value: Any, field: str) -> str:
    value = _required_string(value, field)
    if not _RFC3339_UTC.fullmatch(value):
        raise _error(field, "must be an RFC 3339 UTC string ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _error(field, "must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise _error(field, "must be timezone-aware UTC")
    return value


def _status(value: Any, field: str = "status") -> ManifestStatus:
    try:
        return value if isinstance(value, ManifestStatus) else ManifestStatus(value)
    except (TypeError, ValueError) as exc:
        raise _error(field, "must be planned, building, complete, or failed") from exc


def _overlap(value: Any) -> OverlapResult:
    try:
        return value if isinstance(value, OverlapResult) else OverlapResult(value)
    except (TypeError, ValueError) as exc:
        raise _error("overlap_result", "must be zero or overlap") from exc


def _optional_hash(value: Any, field: str) -> str | None:
    return None if value is None else _hash(value, field)


def _optional_string(value: Any, field: str) -> str | None:
    return None if value is None else _required_string(value, field)


def _hash_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _error(field, "must be a list or tuple of SHA-256 strings")
    return tuple(_hash(item, field) for item in value)


def _exact_clip_keys(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise _error("exact_clip_keys", "must be a list or tuple")
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise _error(f"exact_clip_keys[{index}]", "must be a two-element pair")
        result.append((_required_string(item[0], "exact_clip_keys.dataset_id"), _required_string(item[1], "exact_clip_keys.clip_id")))
    if len(result) != len(set(result)):
        raise _error("exact_clip_keys", "must contain unique keys")
    if not result:
        raise _error("exact_clip_keys", "must not be empty")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ClipManifest:
    dataset_id: str
    clip_id: str
    clip_manifest_id: str
    subject_id: str
    view: str
    condition: str
    camera_id: str
    camera_fps: float
    state_locator: str
    state_sha256: str
    state_row_count: int
    gt_locator: str | None
    gt_sha256: str | None
    gt_row_count: int | None
    split_id: str
    schema_id: str
    status: ManifestStatus

    def __post_init__(self) -> None:
        for field in ("dataset_id", "clip_id", "clip_manifest_id", "subject_id", "view", "condition", "camera_id", "state_locator", "split_id", "schema_id"):
            object.__setattr__(self, field, _required_string(getattr(self, field), field))
        fps = _number(self.camera_fps, "camera_fps")
        if fps <= 0:
            raise _error("camera_fps", "must be > 0")
        object.__setattr__(self, "camera_fps", fps)
        object.__setattr__(self, "state_sha256", _hash(self.state_sha256, "state_sha256"))
        object.__setattr__(self, "state_row_count", _int(self.state_row_count, "state_row_count", minimum=0))
        gt_values = (self.gt_locator, self.gt_sha256, self.gt_row_count)
        if all(value is None for value in gt_values):
            pass
        elif any(value is None for value in gt_values):
            raise _error("gt_locator", "gt_locator, gt_sha256, and gt_row_count must be all present or all null")
        else:
            object.__setattr__(self, "gt_locator", _required_string(self.gt_locator, "gt_locator"))
            object.__setattr__(self, "gt_sha256", _hash(self.gt_sha256, "gt_sha256"))
            object.__setattr__(self, "gt_row_count", _int(self.gt_row_count, "gt_row_count", minimum=0))
        object.__setattr__(self, "status", _status(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "clip_id": self.clip_id, "clip_manifest_id": self.clip_manifest_id, "subject_id": self.subject_id, "view": self.view, "condition": self.condition, "camera_id": self.camera_id, "camera_fps": self.camera_fps, "state_locator": self.state_locator, "state_sha256": self.state_sha256, "state_row_count": self.state_row_count, "gt_locator": self.gt_locator, "gt_sha256": self.gt_sha256, "gt_row_count": self.gt_row_count, "split_id": self.split_id, "schema_id": self.schema_id, "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClipManifest":
        fields = {"dataset_id", "clip_id", "clip_manifest_id", "subject_id", "view", "condition", "camera_id", "camera_fps", "state_locator", "state_sha256", "state_row_count", "gt_locator", "gt_sha256", "gt_row_count", "split_id", "schema_id", "status"}
        return cls(**_strict(payload, fields, "ClipManifest"))


@dataclass(frozen=True, slots=True)
class SplitManifest:
    split_id: str
    dataset_id: str
    train_subject_ids: tuple[str, ...]
    eval_subject_ids: tuple[str, ...]
    train_clip_ids: tuple[str, ...]
    eval_clip_ids: tuple[str, ...]
    train_subject_count: int
    eval_subject_count: int
    train_clip_count: int
    eval_clip_count: int
    overlap_result: OverlapResult
    source_hashes: tuple[str, ...]
    status: ManifestStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_id", _required_string(self.split_id, "split_id"))
        object.__setattr__(self, "dataset_id", _required_string(self.dataset_id, "dataset_id"))
        fields = ("train_subject_ids", "eval_subject_ids", "train_clip_ids", "eval_clip_ids")
        for field in fields:
            values = _tuple_of_strings(getattr(self, field), field, unique=True)
            object.__setattr__(self, field, values)
        if set(self.train_subject_ids) & set(self.eval_subject_ids):
            raise _error("train_subject_ids", "overlap eval_subject_ids")
        if set(self.train_clip_ids) & set(self.eval_clip_ids):
            raise _error("train_clip_ids", "overlap eval_clip_ids")
        counts = (("train_subject_count", self.train_subject_ids), ("eval_subject_count", self.eval_subject_ids), ("train_clip_count", self.train_clip_ids), ("eval_clip_count", self.eval_clip_ids))
        for field, values in counts:
            count = _int(getattr(self, field), field, minimum=0)
            if count != len(values):
                raise _error(field, "must equal the corresponding tuple length")
            object.__setattr__(self, field, count)
        overlap = _overlap(self.overlap_result)
        expected = bool(set(self.train_subject_ids) & set(self.eval_subject_ids) or set(self.train_clip_ids) & set(self.eval_clip_ids))
        if (overlap is OverlapResult.overlap) != expected:
            raise _error("overlap_result", "does not match declared ID overlap")
        object.__setattr__(self, "overlap_result", overlap)
        object.__setattr__(self, "source_hashes", _hash_tuple(self.source_hashes, "source_hashes"))
        object.__setattr__(self, "status", _status(self.status))

    def require_usable(self) -> "SplitManifest":
        if self.status is not ManifestStatus.complete or self.overlap_result is not OverlapResult.zero:
            raise _error("split", "must be complete with zero overlap")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"split_id": self.split_id, "dataset_id": self.dataset_id, "train_subject_ids": list(self.train_subject_ids), "eval_subject_ids": list(self.eval_subject_ids), "train_clip_ids": list(self.train_clip_ids), "eval_clip_ids": list(self.eval_clip_ids), "train_subject_count": self.train_subject_count, "eval_subject_count": self.eval_subject_count, "train_clip_count": self.train_clip_count, "eval_clip_count": self.eval_clip_count, "overlap_result": self.overlap_result.value, "source_hashes": list(self.source_hashes), "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitManifest":
        fields = {"split_id", "dataset_id", "train_subject_ids", "eval_subject_ids", "train_clip_ids", "eval_clip_ids", "train_subject_count", "eval_subject_count", "train_clip_count", "eval_clip_count", "overlap_result", "source_hashes", "status"}
        return cls(**_strict(payload, fields, "SplitManifest"))


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    manifest_id: str
    dataset_id: str
    schema_id: str
    clip_manifest_refs: tuple[str, ...]
    clip_manifest_hashes: tuple[str, ...]
    cohort: Mapping[str, Any]
    subject_count: int
    clip_count: int
    source_inventory_sha256: str
    transform_ids: tuple[str, ...]
    created_at_utc: str
    producer_command: str
    status: ManifestStatus

    def __post_init__(self) -> None:
        for field in ("manifest_id", "dataset_id", "schema_id", "source_inventory_sha256", "created_at_utc", "producer_command"):
            value = getattr(self, field)
            if field == "source_inventory_sha256":
                value = _hash(value, field)
            elif field == "created_at_utc":
                value = _timestamp(value, field)
            else:
                value = _required_string(value, field)
            object.__setattr__(self, field, value)
        refs = _tuple_of_strings(self.clip_manifest_refs, "clip_manifest_refs", unique=True)
        hashes = _hash_tuple(self.clip_manifest_hashes, "clip_manifest_hashes")
        if len(refs) != len(hashes):
            raise _error("clip_manifest_hashes", "must have the same length as clip_manifest_refs")
        clip_count = _int(self.clip_count, "clip_count", minimum=0)
        if clip_count != len(refs):
            raise _error("clip_count", "must equal clip_manifest_refs length")
        object.__setattr__(self, "clip_manifest_refs", refs)
        object.__setattr__(self, "clip_manifest_hashes", hashes)
        if not isinstance(self.cohort, Mapping):
            raise _error("cohort", "must be a mapping")
        object.__setattr__(self, "cohort", _freeze(self.cohort, "cohort"))
        object.__setattr__(self, "subject_count", _int(self.subject_count, "subject_count", minimum=0))
        object.__setattr__(self, "clip_count", clip_count)
        object.__setattr__(self, "transform_ids", _tuple_of_strings(self.transform_ids, "transform_ids", unique=True))
        object.__setattr__(self, "status", _status(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {"manifest_id": self.manifest_id, "dataset_id": self.dataset_id, "schema_id": self.schema_id, "clip_manifest_refs": list(self.clip_manifest_refs), "clip_manifest_hashes": list(self.clip_manifest_hashes), "cohort": _thaw(self.cohort), "subject_count": self.subject_count, "clip_count": self.clip_count, "source_inventory_sha256": self.source_inventory_sha256, "transform_ids": list(self.transform_ids), "created_at_utc": self.created_at_utc, "producer_command": self.producer_command, "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        fields = {"manifest_id", "dataset_id", "schema_id", "clip_manifest_refs", "clip_manifest_hashes", "cohort", "subject_count", "clip_count", "source_inventory_sha256", "transform_ids", "created_at_utc", "producer_command", "status"}
        return cls(**_strict(payload, fields, "DatasetManifest"))


@dataclass(frozen=True, slots=True)
class RunInputManifest:
    run_input_id: str
    dataset_manifest_id: str
    dataset_manifest_sha256: str
    split_manifest_id: str
    split_manifest_sha256: str
    exact_clip_keys: tuple[tuple[str, str], ...]
    signal_config_id: str
    observation_schema_id: str
    gt_rule_id: str | None
    transform_ids: tuple[str, ...]
    seed: int | None
    code_commit: str
    environment_lock_sha256: str
    created_at_utc: str
    status: ManifestStatus

    def __post_init__(self) -> None:
        for field in ("run_input_id", "dataset_manifest_id", "split_manifest_id", "signal_config_id", "observation_schema_id", "code_commit"):
            object.__setattr__(self, field, _required_string(getattr(self, field), field))
        object.__setattr__(self, "dataset_manifest_sha256", _hash(self.dataset_manifest_sha256, "dataset_manifest_sha256"))
        object.__setattr__(self, "split_manifest_sha256", _hash(self.split_manifest_sha256, "split_manifest_sha256"))
        object.__setattr__(self, "environment_lock_sha256", _hash(self.environment_lock_sha256, "environment_lock_sha256"))
        object.__setattr__(self, "exact_clip_keys", _exact_clip_keys(self.exact_clip_keys))
        object.__setattr__(self, "gt_rule_id", _optional_string(self.gt_rule_id, "gt_rule_id"))
        object.__setattr__(self, "transform_ids", _tuple_of_strings(self.transform_ids, "transform_ids", unique=True))
        object.__setattr__(self, "seed", _optional_int(self.seed, "seed"))
        object.__setattr__(self, "created_at_utc", _timestamp(self.created_at_utc, "created_at_utc"))
        object.__setattr__(self, "status", _status(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {"run_input_id": self.run_input_id, "dataset_manifest_id": self.dataset_manifest_id, "dataset_manifest_sha256": self.dataset_manifest_sha256, "split_manifest_id": self.split_manifest_id, "split_manifest_sha256": self.split_manifest_sha256, "exact_clip_keys": [list(key) for key in self.exact_clip_keys], "signal_config_id": self.signal_config_id, "observation_schema_id": self.observation_schema_id, "gt_rule_id": self.gt_rule_id, "transform_ids": list(self.transform_ids), "seed": self.seed, "code_commit": self.code_commit, "environment_lock_sha256": self.environment_lock_sha256, "created_at_utc": self.created_at_utc, "status": self.status.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunInputManifest":
        fields = {"run_input_id", "dataset_manifest_id", "dataset_manifest_sha256", "split_manifest_id", "split_manifest_sha256", "exact_clip_keys", "signal_config_id", "observation_schema_id", "gt_rule_id", "transform_ids", "seed", "code_commit", "environment_lock_sha256", "created_at_utc", "status"}
        return cls(**_strict(payload, fields, "RunInputManifest"))


_ALLOWED_TRANSITIONS = {
    ManifestStatus.planned: {ManifestStatus.planned, ManifestStatus.building, ManifestStatus.failed},
    ManifestStatus.building: {ManifestStatus.building, ManifestStatus.complete, ManifestStatus.failed},
    ManifestStatus.complete: {ManifestStatus.complete},
    ManifestStatus.failed: {ManifestStatus.failed},
}


def validate_status_transition(old: ManifestStatus | str, new: ManifestStatus | str) -> None:
    old_status = _status(old, "old_status")
    new_status = _status(new, "new_status")
    if new_status not in _ALLOWED_TRANSITIONS[old_status]:
        raise ContractValidationError(f"status transition: {old_status.value} -> {new_status.value} is not allowed")


def validate_manifest_transition(old_id: str, old_status: ManifestStatus | str, new_id: str, new_status: ManifestStatus | str) -> None:
    if _required_string(old_id, "old_id") != _required_string(new_id, "new_id"):
        raise _error("manifest_id", "must remain unchanged across a lifecycle")
    validate_status_transition(old_status, new_status)


def require_structurally_complete(manifest: Any) -> Any:
    if not hasattr(manifest, "status") or _status(manifest.status) is not ManifestStatus.complete:
        raise _error("status", "must be complete")
    if isinstance(manifest, SplitManifest):
        return manifest.require_usable()
    if isinstance(manifest, DatasetManifest) and manifest.clip_count != len(manifest.clip_manifest_refs):
        raise _error("clip_count", "does not match clip references")
    return manifest
