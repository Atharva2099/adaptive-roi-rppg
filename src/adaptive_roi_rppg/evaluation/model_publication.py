"""Strict, filesystem-safe Gate 8 model-evaluation publication primitives.

This module is deliberately independent of Slurm and MCD readers.  It is the
small boundary that turns replay rows into auditable artifacts and rejects
anything that is not exactly the artifact set Gate 8 specifies.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from adaptive_roi_rppg.contracts import canonical_json_bytes, read_json_object, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError

MARKER_SCHEMA = "gate8-publication-state-v1"
SHARD_SCHEMA = "gate8-model-shard-v1"
REPORT_SCHEMA = "gate8-model-report-v1"
RUN_MANIFEST_SCHEMA = "gate8-model-run-manifest-v1"
MARKER_NAMES = {"STARTED": "STARTED.json", "FAILED": "FAILED.json", "COMPLETE": "COMPLETE.json"}
SHA256 = set("0123456789abcdef")

HOP_FIELDS = ("subject_id", "view", "condition", "dataset_id", "clip_id", "hop_idx", "hop_time_s", "method_id", "family", "seed", "checkpoint_sha256", "observation_schema_id", "observation_sha256", "frame_provenance_id", "signal_config_id", "control_config_id", "gt_rule_id", "gt_hr_bpm", "abs_error_bpm", "proposed_action", "executed_action", "previous_action", "legal", "override_reason", "pre_hold_count", "post_hold_count", "selected_valid", "selected_invalid_reason", "selected_hr_bpm", "selected_confidence", "selected_ppr", "selected_coverage", "pre_belief_hr_bpm", "post_belief_hr_bpm", "post_belief_velocity", "post_belief_std_bpm")
CLIP_FIELDS = ("method_id", "family", "seed", "checkpoint_sha256", "clip_id", "subject_id", "view", "condition", "hop_count", "mae_bpm", "proposed_action_count", "executed_action_count", "override_count", "invalid_selected_count", "hold_mean", "hold_max", "switch_count", "switches_per_hop", "roi_index_jump_mean_abs", "selected_hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm") + tuple(f"action_count_{i:02d}" for i in range(12))
SUBJECT_FIELDS = ("method_id", "family", "seed", "checkpoint_sha256", "subject_id", "clip_count", "equal_clip_mae_bpm", "proposed_action_count", "executed_action_count", "override_count", "invalid_selected_count", "hold_mean", "hold_max", "switches_per_hop", "roi_index_jump_mean_abs", "selected_hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm")

def _fail(message: str) -> None:
    raise ContractValidationError(message)

def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256

def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file(): _fail(f"Gate 8 requires a regular non-symlink file: {path}")

def canonical_subject_shard(subject_ids: Sequence[str], shard_index: int, shard_count: int) -> tuple[str, ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count: _fail("invalid Gate 8 shard index/count")
    ordered = tuple(sorted(subject_ids))
    if len(set(ordered)) != len(ordered) or any(not isinstance(x, str) or not x for x in ordered): _fail("canonical subject IDs are invalid")
    return ordered[shard_index::shard_count]

def expected_shard_clip_ids(clips: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int) -> tuple[str, ...]:
    # A subject normally has several clips.  Deduplicate before applying the
    # stride; duplicate clip IDs are still forbidden below.
    subjects = canonical_subject_shard(sorted({str(c["subject_id"]) for c in clips}), shard_index, shard_count)
    chosen = [c for c in clips if str(c["subject_id"]) in set(subjects)]
    ids = tuple(str(c["clip_id"]) for c in sorted(chosen, key=lambda c: str(c["clip_id"])))
    if len(set(ids)) != len(ids): _fail("canonical clip IDs are duplicated")
    return ids

def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    if not rows: _fail("Gate 8 CSV cannot be empty")
    if any(tuple(row.keys()) != tuple(fields) for row in rows): _fail("Gate 8 CSV row schema differs from its frozen schema")
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode("utf-8")

def write_csv_exclusive(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    target = Path(path)
    with target.open("xb") as handle: handle.write(csv_bytes(rows, fields))

_INTS = {"hop_idx", "seed", "proposed_action", "executed_action", "previous_action", "pre_hold_count", "post_hold_count", "hop_count", "clip_count", "proposed_action_count", "executed_action_count", "override_count", "invalid_selected_count", "hold_max", "switch_count"} | {f"action_count_{i:02d}" for i in range(12)}
_FLOATS = {"hop_time_s", "gt_hr_bpm", "abs_error_bpm", "selected_hr_bpm", "selected_confidence", "selected_ppr", "selected_coverage", "pre_belief_hr_bpm", "post_belief_hr_bpm", "post_belief_velocity", "post_belief_std_bpm", "mae_bpm", "equal_clip_mae_bpm", "hold_mean", "switches_per_hop", "roi_index_jump_mean_abs", "selected_hr_jump_mean_abs_bpm", "belief_jump_mean_abs_bpm"}
_OPTIONAL = {"seed", "checkpoint_sha256", "previous_action", "override_reason", "selected_invalid_reason", "selected_hr_bpm", "selected_coverage"}
_BOOLEANS = {"legal", "selected_valid"}

def read_strict_csv(path: str | Path, fields: Sequence[str]) -> list[dict[str, Any]]:
    target = Path(path); _regular(target)
    try:
        with target.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields): _fail("Gate 8 CSV columns are missing, unknown, or reordered")
            raw = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc: raise ContractValidationError("Gate 8 CSV cannot be read") from exc
    if not raw: _fail("Gate 8 CSV cannot be empty")
    result: list[dict[str, Any]] = []
    for row in raw:
        value: dict[str, Any] = {}
        for key, item in row.items():
            if key is None or item is None: _fail("Gate 8 CSV has malformed row")
            if key in _OPTIONAL and item == "": value[key] = None; continue
            try:
                if key in _INTS:
                    if item.strip() != item or item in {"", "+0", "-0"}: _fail("Gate 8 integer encoding is invalid")
                    value[key] = int(item)
                elif key in _FLOATS:
                    value[key] = float(item)
                    if not math.isfinite(value[key]): _fail("Gate 8 numeric value is non-finite")
                elif key in _BOOLEANS:
                    if item not in {"True", "False"}: _fail("Gate 8 boolean encoding is invalid")
                    value[key] = item == "True"
                else: value[key] = item
            except ValueError as exc: raise ContractValidationError("Gate 8 CSV value is malformed") from exc
        _validate_row(value); result.append(value)
    return result

def _validate_row(row: Mapping[str, Any]) -> None:
    if "hop_idx" in row and (row["hop_idx"] < 0 or row.get("proposed_action", 0) not in range(12) or row.get("executed_action", 0) not in range(12)): _fail("Gate 8 action or hop is invalid")
    if row.get("previous_action") is not None and row["previous_action"] not in range(12): _fail("Gate 8 previous action is invalid")
    method = row.get("method_id")
    if method == "full_face":
        if row.get("family") != "fixed_full_face" or row.get("seed") is not None or row.get("checkpoint_sha256") is not None: _fail("full_face identity must use null seed and checkpoint hash")
    elif "method_id" in row and (not isinstance(row.get("seed"), int) or not _sha(row.get("checkpoint_sha256"))): _fail("learned checkpoint identity is incomplete")

def validate_hop_rows(rows: Sequence[Mapping[str, Any]], expected_clip_ids: Sequence[str] | None = None, expected_hops: Mapping[str, int] | None = None) -> None:
    keys: set[tuple[Any, ...]] = set(); by_clip: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if tuple(row.keys()) != HOP_FIELDS: _fail("hop row schema is invalid")
        _validate_row(row); key = (row["method_id"], row["clip_id"], row["hop_idx"])
        if key in keys: _fail("duplicate Gate 8 hop key")
        keys.add(key); by_clip.setdefault((row["method_id"], row["clip_id"]), []).append(row)
    if expected_clip_ids is not None and {clip for _, clip in by_clip} != set(expected_clip_ids): _fail("Gate 8 shard clip coverage differs")
    for (method, clip), values in by_clip.items():
        values.sort(key=lambda r: r["hop_idx"])
        if [x["hop_idx"] for x in values] != list(range(len(values))): _fail("Gate 8 hops are not contiguous")
        if expected_hops is not None and len(values) != expected_hops[clip]: _fail("Gate 8 hop count differs from canonical source")

def artifact_map(root: str | Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    base = Path(root); result = {}
    for name in names:
        path = base / name; _regular(path)
        result[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result

def marker_payload(state: str, kind: str, provenance: Mapping[str, Any], *, shard_index: int | None = None, shard_count: int | None = None, outputs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    payload = {"schema": MARKER_SCHEMA, "state": state, "kind": kind, **dict(provenance)}
    if kind == "shard": payload.update({"shard_index": shard_index, "shard_count": shard_count})
    if state == "COMPLETE": payload["outputs"] = dict(outputs or {})
    validate_marker(payload, expected_kind=kind, expected_provenance=provenance)
    return payload

def validate_marker(payload: Mapping[str, Any], *, expected_kind: str, expected_provenance: Mapping[str, Any] | None = None) -> None:
    state = payload.get("state"); base = {"schema", "state", "kind", "plan_id", "code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "job_id", "node", "command"}
    if payload.get("schema") != MARKER_SCHEMA or state not in MARKER_NAMES or payload.get("kind") != expected_kind: _fail("Gate 8 marker schema/state/kind is invalid")
    required = base | ({"shard_index", "shard_count"} if expected_kind == "shard" else set()) | ({"outputs"} if state == "COMPLETE" else set())
    if set(payload) != required: _fail("Gate 8 marker fields are not exact")
    if not _sha(payload["code_snapshot_sha256"]) or not _sha(payload["source_inventory_sha256"]) or not _sha(payload["environment_sha256"]) or not all(isinstance(payload[k], str) and payload[k] for k in ("plan_id", "job_id", "node", "command")): _fail("Gate 8 marker provenance is malformed")
    if expected_kind == "shard" and (not isinstance(payload["shard_index"], int) or not isinstance(payload["shard_count"], int) or not 0 <= payload["shard_index"] < payload["shard_count"]): _fail("Gate 8 marker shard fields are invalid")
    if state == "COMPLETE":
        if not isinstance(payload["outputs"], Mapping) or not payload["outputs"]: _fail("Gate 8 COMPLETE needs outputs")
        for name, info in payload["outputs"].items():
            if not isinstance(name, str) or not isinstance(info, Mapping) or set(info) != {"sha256", "bytes"} or not _sha(info["sha256"]) or not isinstance(info["bytes"], int) or info["bytes"] < 0: _fail("Gate 8 COMPLETE output map is invalid")
    if expected_provenance is not None and any(payload.get(k) != expected_provenance.get(k) for k in expected_provenance): _fail("Gate 8 marker provenance mismatch")

def write_marker(root: str | Path, payload: Mapping[str, Any]) -> None:
    base = Path(root); state = str(payload.get("state")); validate_marker(payload, expected_kind=str(payload.get("kind")))
    target = base / MARKER_NAMES[state]
    if target.exists() or target.is_symlink(): _fail("Gate 8 marker already exists")
    with target.open("xb") as handle: handle.write(canonical_json_bytes(payload))

def read_marker(root: str | Path, state: str, *, kind: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    marker = read_json_object(Path(root) / MARKER_NAMES[state]); validate_marker(marker, expected_kind=kind, expected_provenance=provenance); return marker

def validate_directory(root: str | Path, expected: set[str], *, kind: str, provenance: Mapping[str, Any], complete_outputs: Sequence[str] | None = None) -> None:
    base = Path(root)
    if base.is_symlink() or not base.is_dir(): _fail("Gate 8 publication directory is invalid")
    names = {p.name for p in base.iterdir()}
    if names != expected: _fail("Gate 8 publication has missing or extra files")
    if "FAILED.json" in names and "COMPLETE.json" in names: _fail("Gate 8 publication cannot be both FAILED and COMPLETE")
    read_marker(base, "STARTED", kind=kind, provenance=provenance)
    if "COMPLETE.json" in names:
        complete = read_marker(base, "COMPLETE", kind=kind, provenance=provenance)
        outputs = artifact_map(base, complete_outputs or ())
        if complete["outputs"] != outputs: _fail("Gate 8 COMPLETE output hashes or sizes disagree")
    elif "FAILED.json" in names: read_marker(base, "FAILED", kind=kind, provenance=provenance)
    else: _fail("Gate 8 STARTED-only directory is not acceptable for merge")
    for name in names: _regular(base / name)

def validate_precomplete_directory(root: str | Path, outputs: Sequence[str], *, kind: str, provenance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Perform every final fallible publication check before COMPLETE exists."""
    base = Path(root); expected = {MARKER_NAMES["STARTED"], *outputs}
    if base.is_symlink() or not base.is_dir() or {path.name for path in base.iterdir()} != expected:
        _fail("Gate 8 pre-COMPLETE publication has missing or extra files")
    read_marker(base, "STARTED", kind=kind, provenance=provenance)
    for name in expected: _regular(base / name)
    return artifact_map(base, outputs)

def validate_shard_manifest(payload: Mapping[str, Any], *, provenance: Mapping[str, Any], expected_subject_ids: Sequence[str], expected_clip_ids: Sequence[str], shard_index: int, shard_count: int, checkpoint_identities: Sequence[Mapping[str, Any]]) -> None:
    fields = {"schema", "plan_id", "code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "job_id", "node", "command", "shard_index", "shard_count", "subject_ids", "clip_ids", "checkpoint_identities", "expected_hops"}
    if set(payload) != fields or payload.get("schema") != SHARD_SCHEMA: _fail("Gate 8 shard manifest schema is invalid")
    if any(payload.get(k) != provenance.get(k) for k in provenance) or payload["shard_index"] != shard_index or payload["shard_count"] != shard_count: _fail("Gate 8 shard manifest provenance is invalid")
    if payload["subject_ids"] != list(expected_subject_ids) or payload["clip_ids"] != list(expected_clip_ids) or payload["checkpoint_identities"] != list(checkpoint_identities): _fail("Gate 8 shard manifest ownership or identities differ")
    if not isinstance(payload["expected_hops"], Mapping) or set(payload["expected_hops"]) != set(expected_clip_ids) or any(not isinstance(n, int) or n < 1 for n in payload["expected_hops"].values()): _fail("Gate 8 shard expected hops are invalid")

def validate_exact_json(payload: Mapping[str, Any], *, schema: str, expected: Mapping[str, Any]) -> None:
    """Validate a final Gate 8 JSON artifact against a rebuilt canonical value."""
    if not isinstance(payload, Mapping) or payload.get("schema") != schema or set(payload) != set(expected):
        _fail("Gate 8 JSON schema or fields are invalid")
    if dict(payload) != dict(expected): _fail("Gate 8 JSON contents differ from the rebuilt artifact")

__all__ = ["MARKER_SCHEMA", "SHARD_SCHEMA", "REPORT_SCHEMA", "RUN_MANIFEST_SCHEMA", "MARKER_NAMES", "HOP_FIELDS", "CLIP_FIELDS", "SUBJECT_FIELDS", "canonical_subject_shard", "expected_shard_clip_ids", "csv_bytes", "write_csv_exclusive", "read_strict_csv", "validate_hop_rows", "artifact_map", "marker_payload", "validate_marker", "write_marker", "read_marker", "validate_directory", "validate_precomplete_directory", "validate_shard_manifest", "validate_exact_json"]
