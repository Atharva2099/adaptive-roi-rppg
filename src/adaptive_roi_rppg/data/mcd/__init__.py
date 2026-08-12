"""MCD source discovery, validation, and manifest construction."""

from .adapter import (
    MCD_CAMERA_FPS, MCD_CAMERA_VIEWS, MCD_DATASET_ID, MCD_GT_SUFFIX, MCD_SCHEMA_ID,
    MCD_STATE_SUFFIX, MCDClipMetadata, MCDManifestBundle, MCDSourcePair,
    MCDSplitAssignment, ValidatedMCDSource, build_mcd_manifest_bundle,
    discover_mcd_sources, load_mcd_manifest_tree, load_mcd_split_assignment,
    parse_mcd_stem, resolve_mcd_locator, validate_mcd_source_pair,
    write_mcd_manifest_bundle,
)
from .schema import MCD_GT_COLUMNS, MCD_STATE_COLUMNS
from .frames import read_mcd_canonical_frames

__all__ = [
    "MCD_CAMERA_FPS", "MCD_CAMERA_VIEWS", "MCD_DATASET_ID", "MCD_GT_SUFFIX", "MCD_SCHEMA_ID",
    "MCD_STATE_SUFFIX", "MCD_GT_COLUMNS", "MCD_STATE_COLUMNS", "MCDClipMetadata",
    "MCDManifestBundle", "MCDSourcePair", "MCDSplitAssignment", "ValidatedMCDSource",
    "build_mcd_manifest_bundle", "discover_mcd_sources", "load_mcd_manifest_tree",
    "load_mcd_split_assignment", "parse_mcd_stem", "resolve_mcd_locator",
    "read_mcd_canonical_frames",
    "validate_mcd_source_pair", "write_mcd_manifest_bundle",
]
