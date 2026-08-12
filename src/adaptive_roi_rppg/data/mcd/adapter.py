"""Small, strict adapter from raw MCD CSVs to Gate 1 manifest contracts."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from adaptive_roi_rppg.contracts import (
    ClipManifest, DatasetManifest, ManifestStatus, OverlapResult, SplitManifest,
    canonical_json_bytes, read_json_object, sha256_file, write_json_atomic,
)
from adaptive_roi_rppg.contracts.constants import ROI_NAMES
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from .schema import MCD_GT_COLUMNS, MCD_STATE_COLUMNS

MCD_DATASET_ID = "mcd"
MCD_SCHEMA_ID = "mcd-semantic-state-64-structured-missing-paired-gt-v2"
MCD_STATE_SUFFIX = "_semantic_state_vectors.csv"
MCD_GT_SUFFIX = "_ground_truth.csv"
MCD_CAMERA_VIEWS = {"FullHDwebcam": "Frontal", "USBVideo": "SideA", "IriunWebcam": "SideB"}
MCD_CAMERA_FPS = {"FullHDwebcam": 30.0, "USBVideo": 30.0, "IriunWebcam": 24.0}
_SUBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
_FRAME = re.compile(r"^(?:0|[1-9][0-9]*)$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_CLIP_FILE = re.compile(r"^mcd-clip-[0-9a-f]{64}\.json$")


def _fail(field: str, message: str) -> None:
    raise ContractValidationError(f"{field}: {message}")


def _require(condition: bool, field: str, message: str) -> None:
    if not condition: _fail(field, message)


def _id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _without(value: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in names}


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _positive_ints(*values: Any) -> bool:
    return all(type(value) is int and value > 0 for value in values)


@dataclass(frozen=True, slots=True)
class MCDClipMetadata:
    clip_id: str
    subject_id: str
    camera_id: str
    view: str
    camera_fps: float
    condition: str

    def to_dict(self) -> dict[str, Any]:
        return {"clip_id": self.clip_id, "subject_id": self.subject_id, "camera_id": self.camera_id,
                "view": self.view, "camera_fps": self.camera_fps, "condition": self.condition}


@dataclass(frozen=True, slots=True)
class MCDSourcePair:
    metadata: MCDClipMetadata
    state_path: Path
    gt_path: Path
    state_locator: str
    gt_locator: str


@dataclass(frozen=True, slots=True)
class ValidatedMCDSource:
    pair: MCDSourcePair
    state_sha256: str
    gt_sha256: str
    state_size: int
    gt_size: int
    state_row_count: int
    gt_row_count: int


@dataclass(frozen=True, slots=True)
class MCDSplitAssignment:
    train_subject_ids: tuple[str, ...]
    eval_subject_ids: tuple[str, ...]
    source_sha256: str
    source_locator: str
    split_id: str


@dataclass(frozen=True, slots=True)
class MCDManifestBundle:
    source_inventory: dict[str, Any]
    split_manifest: SplitManifest
    clip_manifests: tuple[ClipManifest, ...]
    dataset_manifest: DatasetManifest


def parse_mcd_stem(stem: str) -> MCDClipMetadata:
    if not isinstance(stem, str) or stem.count("_") != 2:
        _fail("stem", "must match <subject>_<camera>_<before|after>")
    subject, camera, condition = stem.split("_")
    if not _SUBJECT.fullmatch(subject) or camera not in MCD_CAMERA_VIEWS or condition not in {"before", "after"}:
        _fail("stem", "must match <subject>_<camera>_<before|after>")
    return MCDClipMetadata(stem, subject, camera, MCD_CAMERA_VIEWS[camera], MCD_CAMERA_FPS[camera], condition)


def _root(value: str | os.PathLike[str], field: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_dir() or not any(path.iterdir()):
        _fail(field, "must be a real, nonempty directory")
    return path


def _enumerate(root: Path, suffix: str, label: str) -> tuple[dict[str, Path], tuple[str, ...]]:
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ContractValidationError(f"{label}: cannot enumerate root") from exc
    names = tuple(path.name for path in entries)
    result: dict[str, Path] = {}
    for path in entries:
        if path.is_symlink() or not path.is_file() or not path.name.endswith(suffix):
            _fail(label, "every direct child must be an expected regular non-symlink file")
        stem = path.name[:-len(suffix)]
        metadata = parse_mcd_stem(stem)
        if stem in result or metadata.clip_id in result:
            _fail(label, "duplicate clip stem")
        result[stem] = path
    if not result:
        _fail(label, "must contain at least one source file")
    return result, names


def _source_roots(state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    state, gt = _root(state_root, "state_root"), _root(gt_root, "gt_root")
    if state.resolve() == gt.resolve():
        _fail("roots", "state and GT roots must be distinct")
    if state.resolve() in {gt.resolve()} or gt.resolve() in {state.resolve()}:
        _fail("roots", "state and GT roots must be distinct")
    return state, gt


def discover_mcd_sources(state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str]) -> tuple[MCDSourcePair, ...]:
    state, gt = _source_roots(state_root, gt_root)
    states, _ = _enumerate(state, MCD_STATE_SUFFIX, "state_root")
    gts, _ = _enumerate(gt, MCD_GT_SUFFIX, "gt_root")
    if set(states) != set(gts):
        _fail("sources", "state and GT stems must match exactly")
    return tuple(MCDSourcePair(parse_mcd_stem(stem), states[stem], gts[stem],
                               f"state/{states[stem].name}", f"gt/{gts[stem].name}") for stem in sorted(states))


def resolve_mcd_locator(locator: str, *, state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str]) -> Path:
    state, gt = _source_roots(state_root, gt_root)
    if not isinstance(locator, str) or locator.count("/") != 1 or "\\" in locator:
        _fail("locator", "must be a relative POSIX state/gt locator")
    prefix, filename = locator.split("/")
    root, suffix = (state, MCD_STATE_SUFFIX) if prefix == "state" else (gt, MCD_GT_SUFFIX) if prefix == "gt" else (None, None)
    if root is None or not filename.endswith(suffix) or not filename[:-len(suffix)]:
        _fail("locator", "has an invalid source filename")
    path = root / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root.resolve():
        _fail("locator", "must resolve to a direct regular file")
    parse_mcd_stem(filename[:-len(suffix)])
    return path


def _snapshot(path: Path) -> tuple[int, int, int, str]:
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise ContractValidationError(f"source: cannot inspect {path}") from exc
    left = (before.st_size, before.st_mtime_ns, before.st_ino)
    right = (after.st_size, after.st_mtime_ns, after.st_ino)
    if left != right:
        _fail("source", "changed while being hashed")
    return (*left, digest)


def _numeric(raw: str, field: str, path: Path) -> float:
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"csv: {path.name} {field} is not numeric") from exc
    if not math.isfinite(number):
        _fail("csv", f"{path.name} {field} must be finite")
    return number


def _read_frame_rows(path: Path, columns: tuple[str, ...]) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader, ())) != columns:
                _fail("csv", f"{path.name} has the wrong header")
            for row in reader:
                if len(row) != len(columns):
                    _fail("csv", f"{path.name} has a wrong-width row")
                if not _FRAME.fullmatch(row[0]):
                    _fail("frame_idx", f"{path.name} has a non-canonical frame index")
                index = int(row[0])
                if rows and index <= rows[-1][0]:
                    _fail("frame_idx", f"{path.name} is not strictly increasing")
                rows.append((index, row))
    except (UnicodeError, csv.Error) as exc:
        raise ContractValidationError(f"csv: cannot parse {path.name} as UTF-8 CSV") from exc
    if not rows:
        _fail("csv", f"{path.name} must contain at least one data row")
    return rows


def _read_gt_csv(path: Path) -> tuple[int, ...]:
    rows = _read_frame_rows(path, MCD_GT_COLUMNS)
    for _, row in rows:
        _numeric(row[1], "gt_ppg", path)
    return tuple(index for index, _ in rows)


def _read_state_csv(path: Path) -> tuple[int, ...]:
    rows = _read_frame_rows(path, MCD_STATE_COLUMNS)
    pose_fields = ("head_yaw", "head_pitch", "head_roll")
    for _, row in rows:
        pose = row[1:4]
        pose_missing = [raw == "" for raw in pose]
        if any(pose_missing) and not all(pose_missing):
            _fail("state", f"{path.name} pose fields must be all absent or all present")
        if not all(pose_missing):
            for field, raw in zip(pose_fields, pose):
                _numeric(raw, field, path)
        for roi_index, roi in enumerate(ROI_NAMES):
            offset = 4 + roi_index * 5
            fields = row[offset:offset + 4]
            coverage = _numeric(row[offset + 4], f"{roi.value}_coverage", path)
            if not 0 <= coverage <= 1:
                _fail("csv", f"{path.name} {roi.value}_coverage must be within [0,1]")
            missing = [raw == "" for raw in fields]
            if coverage == 0.0:
                if not all(missing):
                    _fail("state", f"{path.name} {roi.value} values must be absent at zero coverage")
            elif any(missing):
                _fail("state", f"{path.name} {roi.value} values must be present at positive coverage")
            else:
                for field, raw in zip(("r_mean", "g_mean", "b_mean", "std"), fields):
                    _numeric(raw, f"{roi.value}_{field}", path)
    return tuple(index for index, _ in rows)


def validate_mcd_source_pair(pair: MCDSourcePair) -> ValidatedMCDSource:
    state_before, gt_before = _snapshot(pair.state_path), _snapshot(pair.gt_path)
    state_indices = _read_state_csv(pair.state_path)
    gt_indices = _read_gt_csv(pair.gt_path)
    if state_indices != gt_indices:
        _fail("frame_idx", f"{pair.metadata.clip_id} state and GT do not join exactly")
    row_count = len(state_indices)
    state_after, gt_after = _snapshot(pair.state_path), _snapshot(pair.gt_path)
    if state_before != state_after or gt_before != gt_after:
        _fail("source", f"{pair.metadata.clip_id} changed during validation")
    return ValidatedMCDSource(pair, state_before[3], gt_before[3], state_before[0], gt_before[0],
                              row_count, row_count)


def load_mcd_split_assignment(split_csv: str | os.PathLike[str], subjects: Iterable[str]) -> MCDSplitAssignment:
    path = Path(split_csv)
    if path.is_symlink() or not path.is_file():
        _fail("split_csv", "must be a regular non-symlink file")
    before = _snapshot(path)
    assignments: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader, ())) != ("subject_id", "split"):
                _fail("split", "header must be exactly subject_id,split")
            for row in reader:
                if len(row) != 2 or not row[0] or row[0] in assignments or row[1] not in {"train", "eval"}:
                    _fail("split", "rows must be unique subject_id,train|eval entries")
                assignments[row[0]] = row[1]
    except (UnicodeError, csv.Error) as exc:
        raise ContractValidationError("split: cannot parse as UTF-8 CSV") from exc
    if before != _snapshot(path):
        _fail("split", "changed during parsing")
    expected = set(subjects)
    if set(assignments) != expected:
        _fail("split", "must exhaustively list discovered subjects")
    train = tuple(sorted(s for s, arm in assignments.items() if arm == "train"))
    evaluate = tuple(sorted(s for s, arm in assignments.items() if arm == "eval"))
    if not train or not evaluate:
        _fail("split", "train and eval must both be nonempty")
    identity = {"dataset_id": MCD_DATASET_ID, "memberships": [[s, assignments[s]] for s in sorted(assignments)],
                "source_sha256": before[3]}
    return MCDSplitAssignment(train, evaluate, before[3], f"split/{path.name}", _id("mcd-split", identity))


def _inventory(sources: tuple[ValidatedMCDSource, ...], split: MCDSplitAssignment) -> dict[str, Any]:
    clips = [{"metadata": source.pair.metadata.to_dict(), "state_locator": source.pair.state_locator,
              "gt_locator": source.pair.gt_locator, "state_sha256": source.state_sha256,
              "gt_sha256": source.gt_sha256, "state_byte_size": source.state_size,
              "gt_byte_size": source.gt_size, "state_row_count": source.state_row_count,
              "gt_row_count": source.gt_row_count} for source in sources]
    identity = {"dataset_id": MCD_DATASET_ID, "schema_id": MCD_SCHEMA_ID, "clips": clips}
    return {"inventory_id": _id("mcd-inventory", identity), "dataset_id": MCD_DATASET_ID,
            "schema_id": MCD_SCHEMA_ID, "locator_policy": "state/<filename> and gt/<filename>",
            "split_source_locator": split.source_locator, "split_source_sha256": split.source_sha256,
            "clips": clips}


def _new_clip(source: ValidatedMCDSource, split_id: str) -> ClipManifest:
    pair = source.pair
    provisional = ClipManifest(MCD_DATASET_ID, pair.metadata.clip_id, "pending", pair.metadata.subject_id,
        pair.metadata.view, pair.metadata.condition, pair.metadata.camera_id, pair.metadata.camera_fps,
        pair.state_locator, source.state_sha256, source.state_row_count, pair.gt_locator, source.gt_sha256,
        source.gt_row_count, split_id, MCD_SCHEMA_ID, ManifestStatus.complete)
    return ClipManifest(**{**provisional.to_dict(), "clip_manifest_id": _id("mcd-clip", _without(provisional.to_dict(), "clip_manifest_id", "status"))})


def _validate_bundle(bundle: MCDManifestBundle) -> MCDManifestBundle:
    inventory = bundle.source_inventory
    required = {"inventory_id", "dataset_id", "schema_id", "locator_policy", "split_source_locator", "split_source_sha256", "clips"}
    if (not isinstance(inventory, dict) or set(inventory) != required or inventory["dataset_id"] != MCD_DATASET_ID or
            inventory["schema_id"] != MCD_SCHEMA_ID or inventory["locator_policy"] != "state/<filename> and gt/<filename>" or
            not isinstance(inventory["split_source_locator"], str) or inventory["split_source_locator"].count("/") != 1 or
            not inventory["split_source_locator"].startswith("split/")):
        _fail("source_inventory", "has the wrong fields or identity")
    _require(isinstance(inventory["clips"], list) and bool(inventory["clips"]), "source_inventory", "clips must be a nonempty list")
    expected_inventory_id = _id("mcd-inventory", _without(inventory, "inventory_id", "locator_policy", "split_source_locator", "split_source_sha256"))
    if inventory["inventory_id"] != expected_inventory_id or not isinstance(inventory["split_source_sha256"], str) or not _HEX.fullmatch(inventory["split_source_sha256"]):
        _fail("source_inventory", "inventory ID or split hash is invalid")
    clips = tuple(sorted(bundle.clip_manifests, key=lambda clip: clip.clip_id))
    _require(clips == bundle.clip_manifests and bool(clips), "clips", "must be sorted and nonempty")
    records = inventory["clips"]
    fields = {"metadata", "state_locator", "gt_locator", "state_sha256", "gt_sha256", "state_byte_size", "gt_byte_size", "state_row_count", "gt_row_count"}
    _require(not any(set(record) != fields for record in records), "source_inventory", "clip records have wrong fields")
    _require([record["metadata"]["clip_id"] for record in records] == [clip.clip_id for clip in clips], "source_inventory", "clip records do not match manifests")
    for record, clip in zip(records, clips):
        metadata = parse_mcd_stem(clip.clip_id)
        state_locator = f"state/{clip.clip_id}{MCD_STATE_SUFFIX}"
        gt_locator = f"gt/{clip.clip_id}{MCD_GT_SUFFIX}"
        _require((clip.subject_id, clip.camera_id, clip.view, clip.camera_fps, clip.condition) == (metadata.subject_id, metadata.camera_id, metadata.view, metadata.camera_fps, metadata.condition), "clips", "clip metadata does not match its stem")
        _require(clip.state_locator == state_locator and clip.gt_locator == gt_locator and clip.gt_locator is not None and clip.gt_sha256 is not None and clip.gt_row_count is not None, "clips", "source locators and GT provenance are not exact")
        _require(_hash_ok(clip.state_sha256) and _hash_ok(clip.gt_sha256), "clips", "source hashes must be lowercase SHA-256")
        _require(_positive_ints(clip.state_row_count, clip.gt_row_count) and clip.state_row_count == clip.gt_row_count, "clips", "source row counts must be positive, equal integers")
        expected_metadata = {"clip_id": clip.clip_id, "subject_id": metadata.subject_id, "camera_id": metadata.camera_id, "view": metadata.view, "camera_fps": metadata.camera_fps, "condition": metadata.condition}
        _require(record["metadata"] == expected_metadata, "source_inventory", "metadata does not match clip")
        _require(record["state_locator"] == state_locator and record["gt_locator"] == gt_locator, "source_inventory", "locators do not match clip")
        if (not _hash_ok(record["state_sha256"]) or not _hash_ok(record["gt_sha256"]) or
                (record["state_sha256"], record["gt_sha256"]) != (clip.state_sha256, clip.gt_sha256) or
                not _positive_ints(record["state_row_count"], record["gt_row_count"]) or
                (record["state_row_count"], record["gt_row_count"]) != (clip.state_row_count, clip.gt_row_count)):
            _fail("source_inventory", "clip source fields do not match manifest")
        _require(_positive_ints(record["state_byte_size"], record["gt_byte_size"]), "source_inventory", "byte sizes must be positive integers")
    split = bundle.split_manifest
    try: split.require_usable()
    except ContractValidationError: _fail("split", "must be complete with zero overlap")
    memberships = [[subject, "train"] for subject in split.train_subject_ids] + [[subject, "eval"] for subject in split.eval_subject_ids]
    split_identity = {"dataset_id": MCD_DATASET_ID,
                      "memberships": sorted(memberships),
                      "source_sha256": split.source_hashes[0] if split.source_hashes else ""}
    _require(split.dataset_id == MCD_DATASET_ID and split.split_id == _id("mcd-split", split_identity), "split", "identity is invalid")
    subjects = {clip.subject_id for clip in clips}
    if (set(split.train_subject_ids) | set(split.eval_subject_ids) != subjects or set(split.train_subject_ids) & set(split.eval_subject_ids) or
            split.overlap_result is not OverlapResult.zero or split.train_subject_count != len(split.train_subject_ids) or split.eval_subject_count != len(split.eval_subject_ids)):
        _fail("split", "subjects are not exhaustive and disjoint")
    train_clips = tuple(clip.clip_id for clip in clips if clip.subject_id in split.train_subject_ids)
    eval_clips = tuple(clip.clip_id for clip in clips if clip.subject_id in split.eval_subject_ids)
    _require(split.train_clip_ids == train_clips and split.eval_clip_ids == eval_clips, "split", "clip membership is invalid")
    inventory_hash = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    _require(split.source_hashes == (inventory["split_source_sha256"], inventory_hash), "split", "source hashes are invalid")
    dataset = bundle.dataset_manifest
    refs = tuple(clip.clip_manifest_id for clip in clips)
    hashes = tuple(hashlib.sha256(canonical_json_bytes(clip.to_dict())).hexdigest() for clip in clips)
    for clip in clips:
        _require(clip.status is ManifestStatus.complete and clip.dataset_id == MCD_DATASET_ID and clip.schema_id == MCD_SCHEMA_ID, "clips", "all clips must be complete MCD manifests")
        expected_clip_id = _id("mcd-clip", _without(clip.to_dict(), "clip_manifest_id", "status"))
        _require(clip.clip_manifest_id == expected_clip_id, "clips", "clip manifest ID is invalid")
    expected_cohort = {"split_id": split.split_id, "source_inventory_id": inventory["inventory_id"], "gt_field": "gt_ppg",
                       "train_subject_count": split.train_subject_count, "eval_subject_count": split.eval_subject_count}
    if (dataset.dataset_id != MCD_DATASET_ID or dataset.schema_id != MCD_SCHEMA_ID or dataset.clip_manifest_refs != refs or
            dataset.clip_manifest_hashes != hashes or dataset.source_inventory_sha256 != inventory_hash or
            dataset.subject_count != len(subjects) or dataset.clip_count != len(clips) or dataset.cohort != expected_cohort or
            dataset.transform_ids != () or dataset.status is not ManifestStatus.complete):
        _fail("dataset", "references, counts, hash, or status are invalid")
    _require(dataset.manifest_id == _id("mcd-dataset", _without(dataset.to_dict(), "manifest_id", "created_at_utc", "producer_command", "status")), "dataset", "manifest ID is invalid")
    return MCDManifestBundle(dict(inventory), split, clips, dataset)


def build_mcd_manifest_bundle(state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str], split_csv: str | os.PathLike[str], *, created_at_utc: str, producer_command: str = "build_mcd_manifests.py") -> MCDManifestBundle:
    state, gt = _source_roots(state_root, gt_root)
    state_map, state_names = _enumerate(state, MCD_STATE_SUFFIX, "state_root")
    gt_map, gt_names = _enumerate(gt, MCD_GT_SUFFIX, "gt_root")
    if set(state_map) != set(gt_map):
        _fail("sources", "state and GT stems must match exactly")
    pairs = tuple(MCDSourcePair(parse_mcd_stem(stem), state_map[stem], gt_map[stem], f"state/{state_map[stem].name}", f"gt/{gt_map[stem].name}") for stem in sorted(state_map))
    validated = tuple(validate_mcd_source_pair(pair) for pair in pairs)
    split = load_mcd_split_assignment(split_csv, {source.pair.metadata.subject_id for source in validated})
    inventory = _inventory(validated, split)
    clips = tuple(sorted((_new_clip(source, split.split_id) for source in validated), key=lambda clip: clip.clip_id))
    train_ids = tuple(clip.clip_id for clip in clips if clip.subject_id in split.train_subject_ids)
    eval_ids = tuple(clip.clip_id for clip in clips if clip.subject_id in split.eval_subject_ids)
    split_manifest = SplitManifest(split.split_id, MCD_DATASET_ID, split.train_subject_ids, split.eval_subject_ids,
        train_ids, eval_ids, len(split.train_subject_ids), len(split.eval_subject_ids), len(train_ids), len(eval_ids),
        OverlapResult.zero, (split.source_sha256, hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()), ManifestStatus.complete)
    cohort = {"split_id": split.split_id, "source_inventory_id": inventory["inventory_id"], "gt_field": "gt_ppg", "train_subject_count": len(split.train_subject_ids), "eval_subject_count": len(split.eval_subject_ids)}
    provisional = DatasetManifest("pending", MCD_DATASET_ID, MCD_SCHEMA_ID, tuple(c.clip_manifest_id for c in clips),
        tuple(hashlib.sha256(canonical_json_bytes(c.to_dict())).hexdigest() for c in clips), cohort, len({c.subject_id for c in clips}), len(clips), hashlib.sha256(canonical_json_bytes(inventory)).hexdigest(), (), created_at_utc, producer_command, ManifestStatus.complete)
    dataset = DatasetManifest(**{**provisional.to_dict(), "manifest_id": _id("mcd-dataset", _without(provisional.to_dict(), "manifest_id", "created_at_utc", "producer_command", "status"))})
    result = _validate_bundle(MCDManifestBundle(inventory, split_manifest, clips, dataset))
    _, final_state_names = _enumerate(state, MCD_STATE_SUFFIX, "state_root")
    _, final_gt_names = _enumerate(gt, MCD_GT_SUFFIX, "gt_root")
    if (state_names, gt_names) != (final_state_names, final_gt_names):
        _fail("roots", "source membership changed during build")
    return result


def _load_tree(root: Path) -> MCDManifestBundle:
    if root.is_symlink() or not root.is_dir():
        _fail("output", "must be a real directory")
    expected = {"source_inventory.json", "split_manifest.json", "clips", "dataset_manifest.json"}
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ContractValidationError("output: cannot enumerate tree") from exc
    if {p.name for p in entries} != expected:
        _fail("output", "tree has missing or extra artifacts")
    for name in ("source_inventory.json", "split_manifest.json", "dataset_manifest.json"):
        try:
            metadata = (root / name).lstat()
        except OSError as exc:
            raise ContractValidationError(f"output: cannot inspect {name}") from exc
        if not stat.S_ISREG(metadata.st_mode) or (root / name).is_symlink():
            _fail("output", f"{name} must be a regular non-symlink file")
    clips_dir = root / "clips"
    if clips_dir.is_symlink() or not clips_dir.is_dir():
        _fail("output", "clips must be a real directory")
    try:
        paths = sorted(clips_dir.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise ContractValidationError("output: cannot enumerate clips") from exc
    if not paths or any(p.is_symlink() or not p.is_file() or not _CLIP_FILE.fullmatch(p.name) for p in paths):
        _fail("output", "clips contains invalid artifacts")
    try:
        inventory = read_json_object(root / "source_inventory.json")
        split = SplitManifest.from_dict(read_json_object(root / "split_manifest.json"))
        clips = tuple(sorted((ClipManifest.from_dict(read_json_object(path)) for path in paths), key=lambda c: c.clip_id))
        dataset = DatasetManifest.from_dict(read_json_object(root / "dataset_manifest.json"))
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ContractValidationError("output: invalid manifest object") from exc
    if {path.name for path in paths} != {f"{clip.clip_manifest_id}.json" for clip in clips}:
        _fail("output", "clip filenames do not match manifest IDs")
    return _validate_bundle(MCDManifestBundle(inventory, split, clips, dataset))


def load_mcd_manifest_tree(root: str | os.PathLike[str]) -> MCDManifestBundle:
    return _load_tree(Path(root))


def write_mcd_manifest_bundle(bundle: MCDManifestBundle, output_dir: str | os.PathLike[str]) -> Path:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        _fail("output_dir", "must not exist and its parent must exist")
    validated = _validate_bundle(bundle)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (staging / "clips").mkdir()
        write_json_atomic(staging / "source_inventory.json", validated.source_inventory)
        write_json_atomic(staging / "split_manifest.json", validated.split_manifest.to_dict())
        for clip in validated.clip_manifests:
            write_json_atomic(staging / "clips" / f"{clip.clip_manifest_id}.json", clip.to_dict())
        write_json_atomic(staging / "dataset_manifest.json", validated.dataset_manifest.to_dict())
        staged = _load_tree(staging)
        if staged != validated:
            _fail("publication", "staged bundle differs from validated bundle")
        if destination.exists():
            _fail("output_dir", "destination appeared during publication")
        os.rename(staging, destination)
        staging = None
        published = _load_tree(destination)
        if published != validated:
            _fail("publication", "published bundle differs from validated bundle")
        return destination / "dataset_manifest.json"
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
