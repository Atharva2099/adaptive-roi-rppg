"""Canonical Gate 9 row validation, aggregation, and exclusive publication."""
from __future__ import annotations
import csv, hashlib, io, json, math, os, random
from pathlib import Path
from typing import Any, Mapping, Sequence
from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError

HOP_FIELDS = ("plan_id", "plan_sha256", "dataset_id", "clip_id", "subject_id", "view", "condition", "method_id", "family", "seed", "checkpoint_sha256", "hop_idx", "hop_time_s", "gt_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm", "selected_valid", "selected_invalid_reason", "proposed_action", "executed_action", "legal", "override_reason", "pre_hold_count", "post_hold_count", "causal_reset", "gt_observation_count", "validity_reason")
ARTIFACTS = ("per_hop.csv", "oracle_b.csv", "oracle_c.csv", "per_clip.csv", "per_subject.csv", "report.json", "run_manifest.json", "artifacts.sha256")
PRIMARY_METHODS = ("full_face_pos", "dagger_seed0", "dagger_seed1", "dagger_seed2", "standard_ppo_seed0", "standard_ppo_seed1", "standard_ppo_seed2", "advantage_ppo_seed0", "advantage_ppo_seed1", "advantage_ppo_seed2")
ORACLE_METHODS = ("oracle_b_greedy", "oracle_c_beam")

def _fail(message: str) -> None: raise ContractValidationError("Gate 9 publication: " + message)
def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n", extrasaction="raise"); writer.writeheader()
    boolean_fields = {"selected_valid", "legal", "causal_reset"}
    writer.writerows([{k: ("true" if row[k] is True else "false" if row[k] is False else row[k]) if k in boolean_fields else row[k] for k in fields} for row in rows]); return stream.getvalue().encode()
def _read(data: bytes | str) -> list[dict[str, str]]:
    try: rows = list(csv.DictReader(io.StringIO(data.decode() if isinstance(data, bytes) else data)))
    except (UnicodeDecodeError, csv.Error) as exc: raise ContractValidationError("Gate 9 publication: per-hop is not strict UTF-8 CSV") from exc
    return rows

def validate_hop_rows(rows: Sequence[Mapping[str, Any]], plan: Any, *, primary_only: bool = True) -> None:
    clips = {x["clip_id"]: x for x in plan.payload["clips"]}; arms = set(plan.payload["arms"])
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        if set(row) != set(HOP_FIELDS): _fail("per-hop schema is not exact")
        try: hop = int(row["hop_idx"])
        except (TypeError, ValueError) as exc: raise ContractValidationError("Gate 9 publication: hop index is not an integer") from exc
        key = (str(row["method_id"]), str(row["clip_id"]), hop)
        if key in seen: _fail("duplicate method/clip/hop row")
        seen.add(key); clip = clips.get(key[1])
        if clip is None or row["plan_id"] != plan.plan_id or row["plan_sha256"] != plan.plan_sha256 or row["dataset_id"] != "mmpd" or row["subject_id"] != clip["subject_id"]: _fail("row is outside sealed cohort or plan binding")
        allowed = arms if primary_only else set(ORACLE_METHODS)
        if row["method_id"] not in allowed: _fail("unknown method arm")
        if not 0 <= key[2] < 53: _fail("hop is outside the 53-hop lattice")
        if not isinstance(row["causal_reset"], bool): _fail("causal_reset must be a strict boolean")
        causal = row["causal_reset"]
        if int(row["gt_observation_count"]) != 0 or causal != (hop == 0): _fail("causal reset or GT boundary is not explicit")
        if not isinstance(row["selected_valid"], bool) or not isinstance(row["legal"], bool): _fail("validity and legal fields must be strict booleans")
        selected_valid = row["selected_valid"]
        if selected_valid and row["selected_invalid_reason"] != "": _fail("valid rows cannot carry an invalid reason")
        if not selected_valid and not row["validity_reason"]: _fail("invalid rows need a validity reason")
    expected = len(clips) * (len(arms) if primary_only else 1) * 53
    if len(rows) != expected: _fail(f"method lattice has {len(rows)} rows, expected {expected}")

def _group(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows: grouped.setdefault(tuple(str(row[k]) for k in key_fields), []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        if "abs_error_bpm" in values[0]:
            errors = [float(row["abs_error_bpm"]) for row in values]
            hop_count = len(values); mae = sum(errors) / hop_count
        else:
            hop_count = sum(int(row["hop_count"]) for row in values)
            mae = sum(float(row["mae_bpm"]) * int(row["hop_count"]) for row in values) / hop_count
        item = {**dict(zip(key_fields, key)), "hop_count": hop_count, "mae_bpm": mae}
        if "subject_id" in key_fields and "clip_id" not in key_fields: item["clip_count"] = len(values)
        result.append(item)
    return result

def _identity_check(rows: Sequence[Mapping[str, Any]], plan: Any, provenance: Mapping[str, Any] | None = None) -> None:
    if {r["method_id"] for r in rows} != set(plan.payload["arms"]): _fail("primary method identities are incomplete")
    authenticated = {item["method_id"]: item for item in (provenance or {}).get("checkpoint_identities", [])}
    for row in rows:
        method = row["method_id"]
        if method == "full_face_pos":
            if row["family"] != "fixed_full_face" or row["seed"] != "": _fail("full-face identity is malformed")
        else:
            if not row["family"] or row["seed"] not in {"0", "1", "2"}: _fail("checkpoint seed identity is malformed")
            if len(row["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["checkpoint_sha256"]): _fail("checkpoint hash identity is malformed")
        if authenticated:
            expected = authenticated.get(method)
            if expected is None or row["family"] != expected["family"] or row["seed"] != expected["seed"] or row["checkpoint_sha256"] != expected["sha256"]: _fail(f"row identity is not bound to authenticated checkpoint {method}")

def _parse_rows(data: bytes | str) -> list[dict[str, Any]]:
    raw_rows = _read(data); typed: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        try:
            for field in ("hop_idx", "gt_observation_count", "proposed_action", "executed_action", "pre_hold_count", "post_hold_count"): row[field] = int(row[field])
            for field in ("hop_time_s", "gt_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm"): row[field] = float(row[field])
            for field in ("selected_valid", "legal", "causal_reset"):
                if row[field] not in ("true", "false"): raise ValueError(f"{field} is not strict boolean")
                row[field] = row[field] == "true"
        except (KeyError, ValueError) as exc: raise ContractValidationError("Gate 9 publication: typed per-hop decoding failed") from exc
        if not all(math.isfinite(row[f]) for f in ("hop_time_s", "gt_hr_bpm", "post_belief_hr_bpm", "abs_error_bpm")): _fail("non-finite primary value")
        if row["abs_error_bpm"] != abs(row["post_belief_hr_bpm"] - row["gt_hr_bpm"]): _fail("absolute error is not derived")
        if not 0 <= row["proposed_action"] < 12 or not 0 <= row["executed_action"] < 12: _fail("action is outside the canonical range")
        typed.append(row)
    return typed

def rebuild_gate9_artifacts(per_hop: bytes | str, plan: Any, *, oracle_b: bytes | str | None = None, oracle_c: bytes | str | None = None, provenance: Mapping[str, Any] | None = None) -> dict[str, bytes]:
    typed = _parse_rows(per_hop)
    validate_hop_rows(typed, plan); _identity_check(typed, plan, provenance)
    if {row["subject_id"] for row in typed} != set(plan.cohort_subject_ids): _fail("primary rows do not cover the sealed subject cohort")
    if oracle_b is None or oracle_c is None: _fail("production evaluation requires explicit Oracle B and Oracle C artifacts")
    b_rows, c_rows = _parse_rows(oracle_b), _parse_rows(oracle_c)
    for oracle_rows, method in ((b_rows, "oracle_b_greedy"), (c_rows, "oracle_c_beam")):
        if any(row["method_id"] != method for row in oracle_rows): _fail(f"{method} artifact contains another method")
        validate_hop_rows(oracle_rows, plan, primary_only=False)
    clip = _group(typed, ("method_id", "clip_id", "subject_id", "family", "seed", "checkpoint_sha256"))
    subject = _group(clip, ("method_id", "subject_id", "family", "seed", "checkpoint_sha256"))
    oracle_clip = _group(b_rows + c_rows, ("method_id", "clip_id", "subject_id", "family", "seed", "checkpoint_sha256"))
    oracle_subject = _group(oracle_clip, ("method_id", "subject_id", "family", "seed", "checkpoint_sha256"))
    equal_clip = {arm: sum(x["mae_bpm"] for x in clip if x["method_id"] == arm) / sum(x["method_id"] == arm for x in clip) for arm in plan.payload["arms"]}
    equal_subject = {arm: sum(x["mae_bpm"] for x in subject if x["method_id"] == arm) / sum(x["method_id"] == arm for x in subject) for arm in plan.payload["arms"]}
    by_arm_subject = {(row["method_id"], row["subject_id"]): float(row["mae_bpm"]) for row in subject}
    uncertainty = {}
    subjects = list(plan.cohort_subject_ids)
    for arm in plan.payload["arms"]:
        point = equal_subject[arm] - equal_subject["full_face_pos"]
        rng = random.Random(8101 + plan.payload["arms"].index(arm)); draws = []
        for _ in range(10000):
            sample = [subjects[rng.randrange(len(subjects))] for _ in subjects]
            draws.append(sum(by_arm_subject[(arm, s)] - by_arm_subject[("full_face_pos", s)] for s in sample) / len(sample))
        draws.sort(); uncertainty[arm] = {"difference_a_minus_full_face_bpm": point, "ci_95_percentile_bpm": [draws[249], draws[9749]], "replicates": 10000, "seed": 8101 + plan.payload["arms"].index(arm), "unit": "subject", "paired": True}
    report = {"schema": "gate9-mmpd-report-v1", "status": "complete", "clip_count": plan.clip_count, "subject_count": len(plan.cohort_subject_ids), "arm_count": len(plan.payload["arms"]), "hop_count": len(typed), "equal_clip": equal_clip, "equal_subject": equal_subject, "oracle_equal_clip": {method: sum(x["mae_bpm"] for x in oracle_clip if x["method_id"] == method) / sum(x["method_id"] == method for x in oracle_clip) for method in ORACLE_METHODS}, "oracle_equal_subject": {method: sum(x["mae_bpm"] for x in oracle_subject if x["method_id"] == method) / sum(x["method_id"] == method for x in oracle_subject) for method in ORACLE_METHODS}, "subject_block": {"unit": "subject", "paired": True, "bootstrap": {"replicates": 10000, "seed": 8101, "percentile": [0.025, 0.975], "comparison": "all primary arms against full_face_pos", "deterministic_metadata": True, "results": uncertainty}}}
    manifest = {"schema": "gate9-mmpd-run-manifest-v1", "status": "complete", "plan_id": plan.plan_id, "plan_sha256": plan.plan_sha256, "plan_payload": plan.payload, "arms": list(plan.payload["arms"]), "mmpd_evaluation_only": True, "provenance": dict(provenance or {})}
    return {"per_hop.csv": _csv_bytes(HOP_FIELDS, typed), "oracle_b.csv": _csv_bytes(HOP_FIELDS, b_rows), "oracle_c.csv": _csv_bytes(HOP_FIELDS, c_rows), "per_clip.csv": _csv_bytes(tuple(clip[0]) if clip else (), clip), "per_subject.csv": _csv_bytes(tuple(subject[0]) if subject else (), subject), "report.json": canonical_json_bytes(report), "run_manifest.json": canonical_json_bytes(manifest)}

def _exclusive(path: Path, data: bytes) -> None:
    created = False
    complete = False
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        complete = True
    except FileExistsError as exc: raise ContractValidationError(f"Gate 9 publication: existing member {path.name}") from exc
    finally:
        if descriptor is not None:
            try: os.close(descriptor)
            except OSError: pass
        if created and not complete:
            # A close/fsync failure can happen after bytes become visible.  A
            # publication member is valid only after the whole write closes.
            try: path.unlink()
            except OSError: pass

def start_gate9_evaluation(destination: str | Path, manifest: Mapping[str, Any]) -> Path:
    root = Path(destination)
    if root.exists() or root.is_symlink(): _fail("output root must be new")
    if root.parent.is_symlink() or not root.parent.is_dir(): _fail("output parent must be an existing real directory")
    root.mkdir()
    _exclusive(root / "STARTED.json", canonical_json_bytes({"state": "STARTED", "schema": "gate9-evaluation-publication-v1", "manifest": dict(manifest), "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()})); return root

def publish_gate9_evaluation(root: str | Path, artifacts: Mapping[str, bytes], manifest: Mapping[str, Any]) -> None:
    base = Path(root)
    if (base / "COMPLETE.json").exists() or (base / "FAILED.json").exists(): _fail("publication is already terminal")
    if set(artifacts) != set(ARTIFACTS) - {"artifacts.sha256"}: _fail("substantive artifact set is incomplete")
    started = json.loads((base / "STARTED.json").read_text(encoding="utf-8"))
    if started.get("manifest_sha256") != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(): _fail("STARTED manifest binding differs")
    published_manifest = json.loads(artifacts["run_manifest.json"].decode("utf-8"))
    if published_manifest.get("provenance") != dict(manifest): _fail("run manifest provenance differs from STARTED")
    try:
        from .plan import Gate9Plan
        rebuilt = rebuild_gate9_artifacts(artifacts["per_hop.csv"], Gate9Plan(manifest["plan_payload"], manifest["plan_id"], manifest["plan_sha256"]), oracle_b=artifacts["oracle_b.csv"], oracle_c=artifacts["oracle_c.csv"], provenance=manifest)
        for name in ("per_hop.csv", "oracle_b.csv", "oracle_c.csv", "per_clip.csv", "per_subject.csv", "report.json"):
            if rebuilt[name] != artifacts[name]: _fail(f"artifact {name} is not independently rebuildable")
    except (KeyError, json.JSONDecodeError, TypeError, ContractValidationError) as exc:
        raise ContractValidationError("Gate 9 publication: artifacts failed independent rebuild") from exc
    for name, data in artifacts.items(): _exclusive(base / name, data)
    digest_lines = [f"{hashlib.sha256(artifacts[name]).hexdigest()}  {name}" for name in sorted(artifacts)]
    _exclusive(base / "artifacts.sha256", ("\n".join(digest_lines) + "\n").encode())
    inventory = {name: {"sha256": hashlib.sha256((base / name).read_bytes()).hexdigest(), "bytes": (base / name).stat().st_size} for name in ARTIFACTS}
    started_hash = hashlib.sha256((base / "STARTED.json").read_bytes()).hexdigest()
    _exclusive(base / "COMPLETE.json", canonical_json_bytes({"state": "COMPLETE", "schema": "gate9-evaluation-publication-v1", "started_sha256": started_hash, "outputs": inventory}))

def fail_gate9_evaluation(root: str | Path, error: BaseException) -> None:
    base = Path(root)
    if (base / "COMPLETE.json").exists() or (base / "FAILED.json").exists(): return
    # A failed run is terminal and intentionally contains no ambiguous partial
    # artifact set.  Only files created by this publication protocol are
    # removed; STARTED.json remains as the immutable run binding.
    for name in (*ARTIFACTS, "COMPLETE.json"):
        target = base / name
        if target.exists() or target.is_symlink():
            target.unlink()
    _exclusive(base / "FAILED.json", canonical_json_bytes({"state": "FAILED", "schema": "gate9-evaluation-publication-v1", "error": type(error).__name__ + ": " + str(error)}))

def validate_gate9_evaluation_tree(root: str | Path, *, require_complete: bool = True) -> None:
    """Validate marker exclusivity and every published artifact hash."""
    base = Path(root)
    if base.is_symlink() or not base.is_dir(): _fail("evaluation root is not a regular directory")
    names = {item.name for item in base.iterdir()}
    if "STARTED.json" not in names: _fail("STARTED.json is required")
    if "COMPLETE.json" in names and "FAILED.json" in names: _fail("tree cannot contain both terminal markers")
    if require_complete and "COMPLETE.json" not in names: _fail("COMPLETE.json is required")
    if "COMPLETE.json" in names:
        expected = {"STARTED.json", "COMPLETE.json", *ARTIFACTS}
        if names != expected: _fail("complete tree has unexpected or missing members")
        marker = json.loads((base / "COMPLETE.json").read_text(encoding="utf-8"))
        if marker.get("state") != "COMPLETE" or set(marker.get("outputs", {})) != set(ARTIFACTS): _fail("COMPLETE inventory is incomplete")
        if marker.get("started_sha256") != hashlib.sha256((base / "STARTED.json").read_bytes()).hexdigest(): _fail("COMPLETE marker is not bound to STARTED.json")
        started = json.loads((base / "STARTED.json").read_text(encoding="utf-8"))
        if started.get("state") != "STARTED" or started.get("manifest_sha256") != hashlib.sha256(canonical_json_bytes(started.get("manifest", {}))).hexdigest(): _fail("STARTED manifest binding is invalid")
        try:
            run_manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
            if run_manifest.get("provenance") != started.get("manifest"): _fail("run manifest is not bound to STARTED provenance")
        except json.JSONDecodeError as exc:
            raise ContractValidationError("Gate 9 publication: run manifest is not valid JSON") from exc
        checksum_lines = (base / "artifacts.sha256").read_text(encoding="utf-8").splitlines()
        expected_checksums = [f"{hashlib.sha256((base / name).read_bytes()).hexdigest()}  {name}" for name in sorted(ARTIFACTS) if name != "artifacts.sha256"]
        if checksum_lines != expected_checksums: _fail("artifacts.sha256 does not match published artifacts")
        for name in ARTIFACTS:
            item = marker["outputs"][name]
            actual = base / name
            if actual.is_symlink() or not actual.is_file() or item != {"sha256": hashlib.sha256(actual.read_bytes()).hexdigest(), "bytes": actual.stat().st_size}: _fail("COMPLETE inventory does not match artifact")
        try:
            manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8"))
            from .plan import Gate9Plan
            rebuilt = rebuild_gate9_artifacts((base / "per_hop.csv").read_bytes(), Gate9Plan(manifest["plan_payload"], manifest["plan_id"], manifest["plan_sha256"]), oracle_b=(base / "oracle_b.csv").read_bytes(), oracle_c=(base / "oracle_c.csv").read_bytes(), provenance=manifest.get("provenance", {}))
            for name in ("per_hop.csv", "oracle_b.csv", "oracle_c.csv", "per_clip.csv", "per_subject.csv", "report.json"):
                if rebuilt[name] != (base / name).read_bytes(): _fail(f"published {name} is not independently rebuildable")
        except (KeyError, json.JSONDecodeError, TypeError, ContractValidationError) as exc:
            raise ContractValidationError("Gate 9 publication: completed artifacts cannot be independently rebuilt") from exc
    elif "FAILED.json" in names:
        if names != {"STARTED.json", "FAILED.json"}: _fail("failed tree has partial terminal artifacts")

__all__ = ["HOP_FIELDS", "ARTIFACTS", "rebuild_gate9_artifacts", "start_gate9_evaluation", "publish_gate9_evaluation", "fail_gate9_evaluation", "validate_gate9_evaluation_tree"]
