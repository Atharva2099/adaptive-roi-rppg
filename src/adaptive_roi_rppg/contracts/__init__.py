"""Immutable, dataset-neutral contract records and manifest validators."""

from .constants import (
    FORBIDDEN_OBSERVATION_FIELDS,
    ManifestStatus,
    OverlapResult,
    ROIName,
    ROI_NAMES,
)
from .errors import ContractValidationError
from .io import canonical_json_bytes, read_json_object, sha256_file, verify_file_sha256, write_json_atomic
from .manifests import (
    ClipManifest,
    DatasetManifest,
    RunInputManifest,
    SplitManifest,
    require_structurally_complete,
    validate_manifest_transition,
    validate_status_transition,
)
from .records import (
    CanonicalFrame,
    LabelFrame,
    MeasurementFrame,
    ROIFrameValue,
    ROIMeasurement,
    validate_observation_field_names,
)

__all__ = [
    "CanonicalFrame",
    "ClipManifest",
    "ContractValidationError",
    "DatasetManifest",
    "FORBIDDEN_OBSERVATION_FIELDS",
    "LabelFrame",
    "ManifestStatus",
    "MeasurementFrame",
    "OverlapResult",
    "ROIFrameValue",
    "ROIName",
    "ROI_NAMES",
    "ROIMeasurement",
    "RunInputManifest",
    "SplitManifest",
    "canonical_json_bytes",
    "read_json_object",
    "require_structurally_complete",
    "sha256_file",
    "validate_manifest_transition",
    "validate_observation_field_names",
    "validate_status_transition",
    "verify_file_sha256",
    "write_json_atomic",
]
