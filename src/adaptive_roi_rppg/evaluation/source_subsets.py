"""Small streamed input split for the MCD failure-audit replay CSV."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file

FULL_TASK_COUNT = 1599
SOURCE_FIELDS = (
    "subject_id", "clip_id", "hop_idx", "method_id", "family", "seed",
    "checkpoint_sha256", "proposed_action", "executed_action",
    "selected_post_belief_hr_bpm", "gt_hr_bpm", "selected_abs_error_bpm",
)


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _json(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot read {Path(path).name}") from exc
    if not isinstance(value, Mapping): _fail(f"{Path(path).name} must be an object")
    return value


def _hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value): _fail(f"{name} is not a SHA-256")
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def audit_plan_body_sha256(plan: Mapping[str, Any]) -> str:
    """Hash the plan definition, excluding its self-referential hash field."""
    body = dict(plan)
    body.pop("audit_plan_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


class _HashingReader(io.RawIOBase):
    """Binary source reader that hashes exactly the bytes consumed by CSV parsing."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        data = self._handle.read(len(buffer))
        if not data:
            return 0
        buffer[: len(data)] = data
        self._digest.update(data)
        self.bytes += len(data)
        return len(data)

    def digest(self) -> str:
        return self._digest.hexdigest()


def _validate_plan_contract(plan: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    """Validate the exact 48121 plan shape before reading replay rows."""
    if plan.get("schema") != "mcd-frozen-failure-audit-plan-v1" or plan.get("split") != "eval":
        _fail("source plan schema or split is invalid")
    cohort = plan.get("cohort")
    if not isinstance(cohort, Mapping): _fail("source plan cohort is invalid")
    clips, subjects, expected_hops = cohort.get("clip_ids"), cohort.get("subject_ids"), cohort.get("expected_hops")
    if (not isinstance(clips, list) or len(clips) != 533 or len(set(clips)) != 533 or
            any(not isinstance(item, str) or not item for item in clips)):
        _fail("source plan must contain exactly 533 unique clip IDs")
    if (not isinstance(subjects, list) or len(subjects) != 89 or len(set(subjects)) != 89 or
            any(not isinstance(item, str) or not item for item in subjects)):
        _fail("source plan must contain exactly 89 unique subject IDs")
    if not isinstance(expected_hops, Mapping) or set(expected_hops) != set(clips):
        _fail("source plan expected_hops keys must equal the clip IDs")
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in expected_hops.values()) or
            sum(expected_hops.values()) != 91227):
        _fail("source plan expected_hops must total exactly 91,227")
    identities = plan.get("checkpoint_identities")
    if not isinstance(identities, list) or len(identities) != 9:
        _fail("source plan must contain exactly nine checkpoint identities")
    if any(not isinstance(item, Mapping) for item in identities) or len({item.get("method_id") for item in identities}) != 9:
        _fail("source plan checkpoint identities must be unique")
    for item in identities:
        if (not isinstance(item.get("method_id"), str) or not isinstance(item.get("family"), str) or
                isinstance(item.get("seed"), bool) or not isinstance(item.get("seed"), int) or
                not isinstance(item.get("checkpoint_sha256"), str) or
                len(item["checkpoint_sha256"]) != 64 or set(item["checkpoint_sha256"]) - set("0123456789abcdef")):
            _fail("source plan checkpoint identity is invalid")
    advantage = {item["seed"]: item for item in identities if item["family"] == "advantage_ppo"}
    if set(advantage) != {0, 1, 2} or any(advantage[seed]["method_id"] != f"advantage_ppo_seed{seed}" for seed in (0, 1, 2)):
        _fail("source plan must contain exactly Advantage PPO seeds 0, 1, and 2")
    return advantage


def validate_failure_audit_contract(source: str | Path, plan_path: str | Path, complete_path: str | Path) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str, int]:
    """Validate the 48121 plan/completion binding and return source metadata."""
    source, plan_path, complete_path = Path(source), Path(plan_path), Path(complete_path)
    if plan_path.resolve() != source.parent.parent.resolve() / "audit_plan.json" or complete_path.resolve() != source.parent.resolve() / "COMPLETE.json":
        _fail("source plan and completion files do not match the 48121 run layout")
    plan, complete = _json(plan_path), _json(complete_path)
    advantage = _validate_plan_contract(plan)
    if (complete.get("schema") != "mcd-frozen-failure-audit-v2" or complete.get("state") != "COMPLETE" or
            not isinstance(complete.get("run_id"), str) or complete.get("run_id") != plan.get("run_id")):
        _fail("source completion record is invalid or not bound to the plan run")
    plan_hash = audit_plan_body_sha256(plan)
    if plan.get("audit_plan_sha256") != plan_hash: _fail("audit plan self-hash does not match its canonical body")
    if complete.get("audit_plan_sha256") != plan_hash: _fail("completion record uses a different audit plan")
    output = complete.get("outputs", {}).get(source.name)
    if (not isinstance(output, Mapping) or _hex(output.get("sha256"), "source CSV sha256") is None or
            isinstance(output.get("bytes"), bool) or not isinstance(output.get("bytes"), int) or output["bytes"] < 0):
        _fail("completion record does not authenticate source CSV")
    return plan, complete, plan_hash, output["sha256"], output["bytes"]


def prepare_full_task_subsets(source_csv: str | Path, plan_json: str | Path, complete_json: str | Path,
                              output_dir: str | Path) -> Mapping[str, Any]:
    """Make the fixed 533-clip x 3-seed task index from the authenticated source.

    This intentionally does not consult the worst-54 selection.  The plan's
    clip order is the full-cohort order and task ``3*clip_position + seed`` is
    the public, stable mapping.
    """
    source = Path(source_csv)
    if source.is_symlink() or not source.is_file() or source.name != "per_hop_counterfactual.csv":
        _fail("source CSV must be per_hop_counterfactual.csv")
    plan, _complete, plan_hash, expected_source_sha, expected_source_bytes = validate_failure_audit_contract(source, plan_json, complete_json)
    root = Path(output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink():
        _fail("subset output must be a fresh directory")
    clips = tuple(plan["cohort"]["clip_ids"])
    subjects = tuple(plan["cohort"]["subject_ids"])
    expected_hops = plan["cohort"]["expected_hops"]
    # The source rows authenticate each clip's subject identity.  We retain
    # only what each worker needs, so the full source is read once here.
    identities = {int(item["seed"]): item for item in plan["checkpoint_identities"] if item.get("family") == "advantage_ppo"}
    if set(identities) != {0, 1, 2}:
        _fail("source plan Advantage PPO identities are invalid")
    rows_by_pair: dict[tuple[str, int], list[dict[str, str]]] = {(clip, seed): [] for clip in clips for seed in (0, 1, 2)}
    subjects_by_clip: dict[str, str] = {}
    hops_seen: dict[tuple[str, int], list[int]] = {key: [] for key in rows_by_pair}
    with source.open("rb") as binary_handle:
        hashing_reader = _HashingReader(binary_handle)
        with io.TextIOWrapper(hashing_reader, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(SOURCE_FIELDS).issubset(tuple(reader.fieldnames or ())):
                _fail("source CSV is missing required failure-audit fields")
            for row in reader:
                if row.get("family") != "advantage_ppo":
                    continue
                try:
                    key = (row["clip_id"], int(row["seed"]))
                    hop = int(row["hop_idx"])
                except (KeyError, ValueError) as exc:
                    raise ContractValidationError("source row has invalid clip, seed, or hop") from exc
                if key not in rows_by_pair:
                    _fail("source advantage row is outside authenticated clip inventory")
                expected = identities[key[1]]
                if (row.get("method_id"), row.get("checkpoint_sha256")) != (f"advantage_ppo_seed{key[1]}", expected["checkpoint_sha256"]):
                    _fail("source row model identity differs from source plan")
                subject = row.get("subject_id", "")
                if not subject or (key[0] in subjects_by_clip and subjects_by_clip[key[0]] != subject):
                    _fail("source clip subject identity is inconsistent")
                subjects_by_clip[key[0]] = subject
                rows_by_pair[key].append({field: row.get(field, "") for field in SOURCE_FIELDS})
                hops_seen[key].append(hop)
        source_sha, source_bytes = hashing_reader.digest(), hashing_reader.bytes
    if source_sha != expected_source_sha or source_bytes != expected_source_bytes:
        _fail("source CSV bytes differ from COMPLETE metadata")
    _validate_full_subject_inventory(clips, subjects, subjects_by_clip)
    for key, values in hops_seen.items():
        expected = int(expected_hops[key[0]])
        if values != list(range(expected)):
            _fail("full clip/seed does not have exact contiguous source hops")
    root.mkdir()
    tasks = []
    for clip_position, clip in enumerate(clips):
        for seed in (0, 1, 2):
            task_index = 3 * clip_position + seed
            path = root / f"task-{task_index:04d}.csv"
            path.write_bytes(_csv_bytes(rows_by_pair[(clip, seed)], SOURCE_FIELDS))
            tasks.append({"task_index": task_index, "clip_position": clip_position, "clip_id": clip,
                          "subject_id": subjects_by_clip[clip], "seed": seed,
                          "row_count": len(rows_by_pair[(clip, seed)]), "sha256": sha256_file(path),
                          "bytes": path.stat().st_size})
    index = {"schema": "mcd-repeated-correction-beam-full-subsets-v1", "task_count": FULL_TASK_COUNT,
             "clip_count": len(clips), "subject_count": len(subjects), "source_sha256": source_sha,
             "source_bytes": source_bytes, "plan_sha256": plan_hash, "audit_plan_sha256": plan_hash,
             "complete_sha256": sha256_file(complete_json),
             "checkpoint_sha256": {str(seed): identities[seed]["checkpoint_sha256"] for seed in (0, 1, 2)},
             "tasks": tasks}
    (root / "index.json").write_bytes(canonical_json_bytes(index))
    return index


def _validate_full_subject_inventory(clips: Sequence[str], plan_subjects: Sequence[str],
                                     source_subject_by_clip: Mapping[str, str]) -> None:
    """Require exact source-plan subject membership, not merely a matching count."""
    if set(source_subject_by_clip) != set(clips):
        _fail("source does not contain exactly the authenticated clip inventory")
    if set(source_subject_by_clip.values()) != set(plan_subjects):
        _fail("source subject identities differ from the authenticated source plan")


def read_task_subset(root: str | Path, task_index: int, *, task_count: int = FULL_TASK_COUNT) -> tuple[Mapping[str, Any], list[dict[str, str]]]:
    if task_count != FULL_TASK_COUNT or not 0 <= task_index < task_count: _fail("invalid task index/count")
    directory = Path(root); index = _json(directory / "index.json")
    if index.get("schema") != "mcd-repeated-correction-beam-full-subsets-v1" or index.get("task_count") != task_count: _fail("subset index schema is invalid")
    entries = index.get("tasks", []); entry = next((item for item in entries if item.get("task_index") == task_index), None)
    path = directory / f"task-{task_index:04d}.csv"
    if entry is None or path.is_symlink() or not path.is_file() or sha256_file(path) != entry.get("sha256") or path.stat().st_size != entry.get("bytes"): _fail("task subset identity is invalid")
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != entry.get("row_count") or any(row.get("clip_id") != entry.get("clip_id") or row.get("subject_id") != entry.get("subject_id") or row.get("seed") != str(entry.get("seed")) for row in rows): _fail("task subset rows do not match its index")
    entry = dict(entry); entry["_source_sha256"] = index["source_sha256"]; entry["_plan_sha256"] = index["plan_sha256"]; entry["_complete_sha256"] = index["complete_sha256"]
    return entry, rows


__all__ = ["FULL_TASK_COUNT", "SOURCE_FIELDS", "audit_plan_body_sha256", "validate_failure_audit_contract", "prepare_full_task_subsets", "read_task_subset"]
