from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .constants import FORBIDDEN_OBSERVATION_FIELDS, ROI_NAMES, ROIName
from .errors import ContractValidationError


def _error(field: str, message: str) -> ContractValidationError:
    return ContractValidationError(f"{field}: {message}")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(field, "must be a nonempty string")
    return value


def _number(value: Any, field: str, *, finite: bool = True) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(field, "must be an int or float")
    if finite and not math.isfinite(value):
        raise _error(field, "must be finite")
    return value


def _int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(field, "must be an int, not bool")
    if minimum is not None and value < minimum:
        raise _error(field, f"must be >= {minimum}")
    return value


def _optional_number(value: Any, field: str) -> int | float | None:
    return None if value is None else _number(value, field)


def _optional_int(value: Any, field: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _int(value, field, minimum=minimum)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise _error(field, "must be a bool")
    return value


def _optional_reason(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _strict(payload: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise _error(name, "must be an object")
    actual = set(payload)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise _error(name, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise _error(name, f"unknown fields: {', '.join(sorted(unknown))}")
    return dict(payload)


def _freeze(value: Any, field: str = "value") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(field, "must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(field, "mapping keys must be strings")
            frozen[key] = _freeze(item, f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, field) for item in value)
    raise _error(field, "contains an unsupported value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _tuple_of_strings(value: Any, field: str, *, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _error(field, "must be a list or tuple")
    result = tuple(_required_string(item, field) for item in value)
    if unique and len(result) != len(set(result)):
        raise _error(field, "must contain unique values")
    return result


@dataclass(frozen=True, slots=True)
class ROIFrameValue:
    roi_index: int
    roi_name: ROIName
    r_mean: float | None
    g_mean: float | None
    b_mean: float | None
    std: float | None
    coverage: float | None
    valid: bool
    invalid_reason: str | None
    imputation_age_frames: Mapping[str, int] | None
    imputation_origin_frame_idx: Mapping[str, int] | None

    def __post_init__(self) -> None:
        index = _int(self.roi_index, "roi_index", minimum=0)
        if index >= len(ROI_NAMES):
            raise _error("roi_index", "must be in the canonical ROI range")
        try:
            name = self.roi_name if isinstance(self.roi_name, ROIName) else ROIName(self.roi_name)
        except (TypeError, ValueError) as exc:
            raise _error("roi_name", "is not a canonical ROI name") from exc
        if name is not ROI_NAMES[index]:
            raise _error("roi_name", "does not match roi_index")
        object.__setattr__(self, "roi_index", index)
        object.__setattr__(self, "roi_name", name)
        for field in ("r_mean", "g_mean", "b_mean", "std", "coverage"):
            value = _optional_number(getattr(self, field), field)
            if field == "coverage" and value is not None and not 0 <= value <= 1:
                raise _error(field, "must be in [0, 1]")
            object.__setattr__(self, field, value)
        valid = _bool(self.valid, "valid")
        reason = _optional_reason(self.invalid_reason, "invalid_reason")
        if valid:
            if any(getattr(self, field) is None for field in ("r_mean", "g_mean", "b_mean", "std", "coverage")):
                raise _error("valid", "requires all numeric fields")
            if reason is not None:
                raise _error("invalid_reason", "must be null when valid")
        elif reason is None:
            raise _error("invalid_reason", "is required when invalid")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid_reason", reason)
        age = self._freeze_imputation(self.imputation_age_frames, "imputation_age_frames")
        origin = self._freeze_imputation(self.imputation_origin_frame_idx, "imputation_origin_frame_idx")
        if (age is None) != (origin is None):
            raise _error("imputation_age_frames", "and imputation_origin_frame_idx must both be null or mappings")
        if age is not None and set(age) != set(origin or {}):
            raise _error("imputation_age_frames", "and imputation_origin_frame_idx keys must match")
        object.__setattr__(self, "imputation_age_frames", age)
        object.__setattr__(self, "imputation_origin_frame_idx", origin)

    @staticmethod
    def _freeze_imputation(value: Any, field: str) -> Mapping[str, int] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise _error(field, "must be a mapping or null")
        allowed = {"r_mean", "g_mean", "b_mean"}
        if set(value) - allowed:
            raise _error(field, "contains an unsupported channel")
        frozen: dict[str, int] = {}
        for key, item in value.items():
            frozen[key] = _int(item, f"{field}.{key}", minimum=0)
        return MappingProxyType(frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_name": self.roi_name.value,
            "r_mean": self.r_mean,
            "g_mean": self.g_mean,
            "b_mean": self.b_mean,
            "std": self.std,
            "coverage": self.coverage,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "imputation_age_frames": _thaw(self.imputation_age_frames),
            "imputation_origin_frame_idx": _thaw(self.imputation_origin_frame_idx),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ROIFrameValue":
        fields = {"roi_index", "roi_name", "r_mean", "g_mean", "b_mean", "std", "coverage", "valid", "invalid_reason", "imputation_age_frames", "imputation_origin_frame_idx"}
        return cls(**_strict(payload, fields, "ROIFrameValue"))


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    dataset_id: str
    clip_id: str
    frame_idx: int
    timestamp_s: float
    camera_fps: float
    head_yaw_deg: float | None
    head_pitch_deg: float | None
    head_roll_deg: float | None
    roi_values: tuple[ROIFrameValue, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        for field in ("dataset_id", "clip_id", "provenance_id"):
            object.__setattr__(self, field, _required_string(getattr(self, field), field))
        object.__setattr__(self, "frame_idx", _int(self.frame_idx, "frame_idx", minimum=0))
        for field in ("timestamp_s", "camera_fps"):
            value = _number(getattr(self, field), field)
            if field == "camera_fps" and value <= 0:
                raise _error(field, "must be > 0")
            object.__setattr__(self, field, value)
        pose = tuple(getattr(self, field) for field in ("head_yaw_deg", "head_pitch_deg", "head_roll_deg"))
        if all(value is None for value in pose):
            pass
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in pose):
            object.__setattr__(self, "head_yaw_deg", float(pose[0]))
            object.__setattr__(self, "head_pitch_deg", float(pose[1]))
            object.__setattr__(self, "head_roll_deg", float(pose[2]))
        else:
            raise _error("pose", "must contain three finite values or three nulls")
        if not isinstance(self.roi_values, (list, tuple)) or len(self.roi_values) != len(ROI_NAMES):
            raise _error("roi_values", "must contain exactly 12 records")
        values = tuple(value if isinstance(value, ROIFrameValue) else ROIFrameValue.from_dict(value) for value in self.roi_values)
        for index, value in enumerate(values):
            if value.roi_index != index:
                raise _error("roi_values", "must use canonical order")
            if value.imputation_origin_frame_idx is not None:
                for channel, origin in value.imputation_origin_frame_idx.items():
                    if origin > self.frame_idx:
                        raise _error(f"roi_values[{index}].imputation_origin_frame_idx.{channel}", "cannot be after frame_idx")
                    age = value.imputation_age_frames[channel]
                    if age != self.frame_idx - origin:
                        raise _error(f"roi_values[{index}].imputation_age_frames.{channel}", "does not match frame index and origin")
        object.__setattr__(self, "roi_values", values)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in ("dataset_id", "clip_id", "frame_idx", "timestamp_s", "camera_fps", "head_yaw_deg", "head_pitch_deg", "head_roll_deg", "provenance_id")} | {"roi_values": [value.to_dict() for value in self.roi_values]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalFrame":
        fields = {"dataset_id", "clip_id", "frame_idx", "timestamp_s", "camera_fps", "head_yaw_deg", "head_pitch_deg", "head_roll_deg", "roi_values", "provenance_id"}
        return cls(**_strict(payload, fields, "CanonicalFrame"))


@dataclass(frozen=True, slots=True)
class ROIMeasurement:
    roi_index: int
    roi_name: ROIName
    hr_bpm: float | None
    confidence: float | None
    peak_power_ratio: float | None
    coverage: float | None
    valid: bool
    invalid_reason: str | None
    imputed_channels: tuple[str, ...]
    max_imputation_age_frames: int | None
    source_frame_start: int
    source_frame_end: int
    signal_config_id: str

    def __post_init__(self) -> None:
        index = _int(self.roi_index, "roi_index", minimum=0)
        if index >= len(ROI_NAMES):
            raise _error("roi_index", "must be in the canonical ROI range")
        try:
            name = self.roi_name if isinstance(self.roi_name, ROIName) else ROIName(self.roi_name)
        except (TypeError, ValueError) as exc:
            raise _error("roi_name", "is not a canonical ROI name") from exc
        if name is not ROI_NAMES[index]:
            raise _error("roi_name", "does not match roi_index")
        object.__setattr__(self, "roi_index", index)
        object.__setattr__(self, "roi_name", name)
        for field in ("hr_bpm", "confidence", "peak_power_ratio", "coverage"):
            value = _optional_number(getattr(self, field), field)
            if field == "coverage" and value is not None and not 0 <= value <= 1:
                raise _error(field, "must be in [0, 1]")
            object.__setattr__(self, field, value)
        valid = _bool(self.valid, "valid")
        reason = _optional_reason(self.invalid_reason, "invalid_reason")
        if valid:
            if any(getattr(self, field) is None for field in ("hr_bpm", "confidence", "peak_power_ratio", "coverage")):
                raise _error("valid", "requires all numeric fields")
            if reason is not None:
                raise _error("invalid_reason", "must be null when valid")
        elif reason is None:
            raise _error("invalid_reason", "is required when invalid")
        channels = _tuple_of_strings(self.imputed_channels, "imputed_channels", unique=True)
        allowed = {"r_mean", "g_mean", "b_mean"}
        if set(channels) - allowed:
            raise _error("imputed_channels", "contains an unsupported channel")
        max_age = _optional_int(self.max_imputation_age_frames, "max_imputation_age_frames")
        if channels and max_age is None:
            raise _error("max_imputation_age_frames", "is required when channels are imputed")
        if not channels and max_age is not None:
            raise _error("max_imputation_age_frames", "must be null when no channels are imputed")
        start = _int(self.source_frame_start, "source_frame_start", minimum=0)
        end = _int(self.source_frame_end, "source_frame_end", minimum=0)
        if start > end:
            raise _error("source_frame_start", "must be <= source_frame_end")
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid_reason", reason)
        object.__setattr__(self, "imputed_channels", channels)
        object.__setattr__(self, "max_imputation_age_frames", max_age)
        object.__setattr__(self, "source_frame_start", start)
        object.__setattr__(self, "source_frame_end", end)
        object.__setattr__(self, "signal_config_id", _required_string(self.signal_config_id, "signal_config_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_name": self.roi_name.value,
            "hr_bpm": self.hr_bpm,
            "confidence": self.confidence,
            "peak_power_ratio": self.peak_power_ratio,
            "coverage": self.coverage,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "imputed_channels": list(self.imputed_channels),
            "max_imputation_age_frames": self.max_imputation_age_frames,
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
            "signal_config_id": self.signal_config_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ROIMeasurement":
        fields = {"roi_index", "roi_name", "hr_bpm", "confidence", "peak_power_ratio", "coverage", "valid", "invalid_reason", "imputed_channels", "max_imputation_age_frames", "source_frame_start", "source_frame_end", "signal_config_id"}
        return cls(**_strict(payload, fields, "ROIMeasurement"))


@dataclass(frozen=True, slots=True)
class MeasurementFrame:
    dataset_id: str
    clip_id: str
    hop_idx: int
    hop_time_s: float
    measurements: tuple[ROIMeasurement, ...]
    signal_config_id: str
    valid: bool
    invalid_reason: str | None
    provenance_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required_string(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "clip_id", _required_string(self.clip_id, "clip_id"))
        object.__setattr__(self, "hop_idx", _int(self.hop_idx, "hop_idx", minimum=0))
        hop_time = _number(self.hop_time_s, "hop_time_s")
        if hop_time < 0:
            raise _error("hop_time_s", "must be >= 0")
        object.__setattr__(self, "hop_time_s", hop_time)
        config = _required_string(self.signal_config_id, "signal_config_id")
        provenance = _required_string(self.provenance_id, "provenance_id")
        if not isinstance(self.measurements, (list, tuple)) or len(self.measurements) != len(ROI_NAMES):
            raise _error("measurements", "must contain exactly 12 records")
        measurements = tuple(item if isinstance(item, ROIMeasurement) else ROIMeasurement.from_dict(item) for item in self.measurements)
        for index, item in enumerate(measurements):
            if item.roi_index != index:
                raise _error("measurements", "must use canonical order")
            if item.signal_config_id != config:
                raise _error("measurements", "signal_config_id must match frame")
        valid = _bool(self.valid, "valid")
        reason = _optional_reason(self.invalid_reason, "invalid_reason")
        if valid != any(item.valid for item in measurements):
            raise _error("valid", "must equal whether any ROI measurement is valid")
        if valid and reason is not None:
            raise _error("invalid_reason", "must be null when valid")
        if not valid and reason is None:
            raise _error("invalid_reason", "is required when invalid")
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "signal_config_id", config)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid_reason", reason)
        object.__setattr__(self, "provenance_id", provenance)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "clip_id": self.clip_id, "hop_idx": self.hop_idx, "hop_time_s": self.hop_time_s, "measurements": [item.to_dict() for item in self.measurements], "signal_config_id": self.signal_config_id, "valid": self.valid, "invalid_reason": self.invalid_reason, "provenance_id": self.provenance_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementFrame":
        fields = {"dataset_id", "clip_id", "hop_idx", "hop_time_s", "measurements", "signal_config_id", "valid", "invalid_reason", "provenance_id"}
        return cls(**_strict(payload, fields, "MeasurementFrame"))


@dataclass(frozen=True, slots=True)
class LabelFrame:
    dataset_id: str
    clip_id: str
    hop_idx: int
    hop_time_s: float
    gt_hr_bpm: float | None
    gt_rule_id: str
    valid: bool
    invalid_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required_string(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "clip_id", _required_string(self.clip_id, "clip_id"))
        object.__setattr__(self, "hop_idx", _int(self.hop_idx, "hop_idx", minimum=0))
        hop_time = _number(self.hop_time_s, "hop_time_s")
        if hop_time < 0:
            raise _error("hop_time_s", "must be >= 0")
        object.__setattr__(self, "hop_time_s", hop_time)
        object.__setattr__(self, "gt_rule_id", _required_string(self.gt_rule_id, "gt_rule_id"))
        valid = _bool(self.valid, "valid")
        reason = _optional_reason(self.invalid_reason, "invalid_reason")
        hr = _optional_number(self.gt_hr_bpm, "gt_hr_bpm")
        if valid and (hr is None or reason is not None):
            raise _error("valid", "requires finite gt_hr_bpm and no invalid_reason")
        if not valid and (hr is not None or reason is None):
            raise _error("invalid_reason", "invalid labels require gt_hr_bpm=null and a reason")
        object.__setattr__(self, "gt_hr_bpm", hr)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid_reason", reason)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "clip_id": self.clip_id, "hop_idx": self.hop_idx, "hop_time_s": self.hop_time_s, "gt_hr_bpm": self.gt_hr_bpm, "gt_rule_id": self.gt_rule_id, "valid": self.valid, "invalid_reason": self.invalid_reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelFrame":
        fields = {"dataset_id", "clip_id", "hop_idx", "hop_time_s", "gt_hr_bpm", "gt_rule_id", "valid", "invalid_reason"}
        return cls(**_strict(payload, fields, "LabelFrame"))


def validate_observation_field_names(names: Any) -> tuple[str, ...]:
    if not isinstance(names, (list, tuple)):
        raise _error("observation_fields", "must be a list or tuple")
    result = tuple(_required_string(name, "observation_fields") for name in names)
    if len(result) != len(set(result)):
        raise _error("observation_fields", "must contain unique names")
    if not result:
        raise _error("observation_fields", "must not be empty")
    for name in result:
        if name in FORBIDDEN_OBSERVATION_FIELDS or name.startswith("gt_"):
            raise _error("observation_fields", f"forbidden field: {name}")
    return result
