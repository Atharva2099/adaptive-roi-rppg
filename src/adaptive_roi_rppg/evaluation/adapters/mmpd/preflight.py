"""Fail-closed Gate 9 provenance preflight."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib, importlib, inspect, json
from .plan import load_gate9_plan
from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.adapters.mmpd.rules import MMPD_GT_RULE_ID

PRIMARY = ("full_face_pos", "dagger_seed0", "dagger_seed1", "dagger_seed2", "standard_ppo_seed0", "standard_ppo_seed1", "standard_ppo_seed2", "advantage_ppo_seed0", "advantage_ppo_seed1", "advantage_ppo_seed2")
def _fail(message: str) -> None: raise ContractValidationError("Gate 9 preflight: " + message)
def _safe_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file(): _fail(f"{label} must be a regular non-symlink file")
    current = path
    while current != current.parent:
        if current not in {Path("/tmp"), Path("/var"), Path("/private")} and current.is_symlink(): _fail(f"{label} has a symlinked path component")
        current = current.parent
def _canonical(path: Path, label: str) -> dict[str, Any]:
    _safe_file(path, label); raw = path.read_bytes()
    try: payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ContractValidationError(f"Gate 9 preflight: {label} is not JSON") from exc
    if canonical_json_bytes(payload) != raw: _fail(f"{label} is not canonical JSON")
    if not isinstance(payload, dict): _fail(f"{label} must be an object")
    return payload
def _inventory(path: Path, plan: Any, root: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = _canonical(path, "raw source inventory"); records = payload.get("clips")
    if not isinstance(records, list) or len(records) != plan.clip_count: _fail("raw source inventory must contain all 299 clips")
    expected = {x["clip_id"]: x["subject_id"] for x in plan.payload["clips"]}; seen = set()
    for record in records:
        if set(record) != {"clip_id", "subject_id", "locator", "sha256", "bytes"} or record["clip_id"] in seen: _fail("raw source inventory fields or uniqueness are invalid")
        if expected.get(record["clip_id"]) != record["subject_id"]: _fail("raw source inventory cohort binding differs")
        locator = Path(record["locator"])
        if locator.is_absolute() or ".." in locator.parts or locator.is_symlink() or not locator.name: _fail("raw source locator is not a safe relative path")
        if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in record["sha256"]): _fail("raw source hash is malformed")
        if not isinstance(record["bytes"], int) or record["bytes"] <= 0: _fail("raw source size is malformed")
        source = root / locator; _safe_file(source, f"raw source {record['clip_id']}")
        if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]: _fail(f"raw source authentication failed for {record['clip_id']}")
        seen.add(record["clip_id"])
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), records
def _rules(path: Path, root: Path) -> str:
    payload = _canonical(path, "rule identity manifest")
    required = {"corrected_gt_rule_id", "corrected_gt_rule_sha256", "corrected_gt_rule_locator", "corrected_gt_rule_bytes", "metadata_coding_id", "metadata_coding_sha256", "metadata_coding_locator", "metadata_coding_bytes"}
    if set(payload) != required: _fail("rule identity manifest fields are not exact")
    for prefix in ("corrected_gt_rule", "metadata_coding"):
        locator = Path(payload[f"{prefix}_locator"])
        if locator.is_absolute() or ".." in locator.parts: _fail("rule locator is unsafe")
        source = root / locator; _safe_file(source, f"{prefix} rule")
        digest = payload[f"{prefix}_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest): _fail("rule hash is malformed")
        if not isinstance(payload[f"{prefix}_bytes"], int) or payload[f"{prefix}_bytes"] <= 0: _fail("rule byte size is malformed")
        if source.stat().st_size != payload[f"{prefix}_bytes"] or sha256_file(source) != digest: _fail(f"{prefix} rule authentication failed")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
def _checkpoints(path: Path, root: Path) -> str:
    payload = _canonical(path, "checkpoint identity manifest"); records = payload.get("checkpoints")
    if not isinstance(records, list) or len(records) != len(PRIMARY) or {x.get("method_id") for x in records} != set(PRIMARY): _fail("checkpoint identity manifest must contain ten arms")
    expected = {
        "full_face_pos": ("fixed_full_face", "", "full_face"),
        "dagger_seed0": ("dagger", "0", "checkpoint"), "dagger_seed1": ("dagger", "1", "checkpoint"), "dagger_seed2": ("dagger", "2", "checkpoint"),
        "standard_ppo_seed0": ("standard_ppo", "0", "checkpoint"), "standard_ppo_seed1": ("standard_ppo", "1", "checkpoint"), "standard_ppo_seed2": ("standard_ppo", "2", "checkpoint"),
        "advantage_ppo_seed0": ("advantage_ppo", "0", "checkpoint"), "advantage_ppo_seed1": ("advantage_ppo", "1", "checkpoint"), "advantage_ppo_seed2": ("advantage_ppo", "2", "checkpoint"),
    }
    for record in records:
        if set(record) != {"method_id", "family", "seed", "sha256", "bytes", "kind", "locator"}: _fail("checkpoint identity fields are not exact")
        method = record["method_id"]
        if method not in expected or (record["family"], record["seed"], record["kind"]) != expected[method]: _fail(f"checkpoint identity mapping is invalid for {method}")
        if record["kind"] == "checkpoint" and (not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or not isinstance(record["bytes"], int) or record["bytes"] <= 0): _fail("checkpoint hash or size is missing")
        if record["kind"] not in {"checkpoint", "full_face"} or record["kind"] == "full_face" and (record["sha256"] or record["bytes"] != 0 or record["locator"] != ""): _fail("checkpoint kind is invalid")
        if record["kind"] == "checkpoint":
            locator = Path(record["locator"])
            if locator.is_absolute() or ".." in locator.parts: _fail("checkpoint locator is unsafe")
            source = root / locator; _safe_file(source, f"checkpoint {record['method_id']}")
            if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]: _fail(f"checkpoint authentication failed for {record['method_id']}")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
def _source_closure(code: Path, extractor_id: str) -> list[dict[str, Any]]:
    if not extractor_id.endswith("production:extract_gate9"):
        return [{"locator": str(code), "sha256": sha256_file(code), "bytes": code.stat().st_size}]
    parents = list(code.parents)
    src_parent = next((parent for parent in parents if parent.name == "src"), None)
    if src_parent is None:
        _fail("production extractor must be inside a repository src tree")
    repo = src_parent.parent
    relative = ("src/adaptive_roi_rppg/evaluation/adapters/mmpd/production.py", "src/adaptive_roi_rppg/evaluation/adapters/mmpd/extraction.py", "src/adaptive_roi_rppg/evaluation/adapters/mmpd/replay.py", "src/adaptive_roi_rppg/signal/pos.py", "src/adaptive_roi_rppg/control/core.py", "src/adaptive_roi_rppg/evaluation/adapters/sb3_recurrent.py")
    result = []
    for locator in relative:
        target = repo / locator
        _safe_file(target, f"source closure {locator}")
        result.append({"locator": locator, "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return result

def preflight_gate9_inputs(plan_path: str | Path, *, raw_root: str | Path | None = None, checkpoint_root: str | Path | None = None, rule_root: str | Path | None = None, raw_inventory: str | Path | None = None, rule_manifest: str | Path | None = None, checkpoint_manifest: str | Path | None = None, code_snapshot: str | Path | None = None, extractor_id: str | None = None, extraction_code_sha256: str | None = None, face_landmarker_model: str | Path | None = None, face_landmarker_model_sha256: str | None = None, face_landmarker_model_bytes: int | None = None) -> dict[str, Any]:
    plan = load_gate9_plan(plan_path)
    if any(value is None for value in (raw_root, checkpoint_root, rule_root, raw_inventory, rule_manifest, checkpoint_manifest, code_snapshot, extractor_id, extraction_code_sha256)): _fail("raw, rule, checkpoint, extraction, and code provenance are all required")
    for label, root_value in (("raw", raw_root), ("checkpoint", checkpoint_root), ("rule", rule_root)):
        root = Path(root_value)
        if root.is_symlink() or not root.is_dir(): _fail(f"{label} root must be a regular directory")
    inventory_hash, inventory_records = _inventory(Path(raw_inventory), plan, Path(raw_root)); rules_hash = _rules(Path(rule_manifest), Path(rule_root)); rule_payload = _canonical(Path(rule_manifest), "rule identity manifest"); checkpoints_payload = _canonical(Path(checkpoint_manifest), "checkpoint identity manifest"); checkpoints_hash = _checkpoints(Path(checkpoint_manifest), Path(checkpoint_root))
    code = Path(code_snapshot); _safe_file(code, "code snapshot"); code_hash = sha256_file(code)
    if ":" not in extractor_id: _fail("extractor identity must be module:function")
    module_name, function_name = extractor_id.split(":", 1)
    try: source = Path(inspect.getsourcefile(getattr(importlib.import_module(module_name), function_name)) or "")
    except (ImportError, AttributeError, TypeError) as exc: raise ContractValidationError("Gate 9 preflight: extractor identity cannot be resolved") from exc
    _safe_file(source, "extractor source")
    if source.resolve() != code.resolve() or sha256_file(source) != code_hash: _fail("extractor source is not the authenticated code snapshot")
    extraction_source = Path(__file__).with_name("extraction.py") if extractor_id.endswith("production:extract_gate9") else code
    _safe_file(extraction_source, "extraction source")
    if extraction_code_sha256 != sha256_file(extraction_source): _fail("extraction code hash does not match extraction source")
    closure = _source_closure(code, extractor_id)
    model_identity = {}
    if extractor_id.endswith("production:extract_gate9"):
        if face_landmarker_model is None or face_landmarker_model_sha256 is None or face_landmarker_model_bytes is None:
            _fail("production extractor requires authenticated face-landmarker model identity")
        model = Path(face_landmarker_model); _safe_file(model, "face-landmarker model")
        if model.stat().st_size != face_landmarker_model_bytes or sha256_file(model) != face_landmarker_model_sha256: _fail("face-landmarker model authentication failed")
        model_identity = {"face_landmarker_model_locator": str(model.resolve()), "face_landmarker_model_sha256": face_landmarker_model_sha256, "face_landmarker_model_bytes": face_landmarker_model_bytes}
        rule_source = Path(__file__).with_name("rules.py")
        _safe_file(rule_source, "repository MMPD rule source")
        if rule_payload["corrected_gt_rule_id"] != MMPD_GT_RULE_ID or rule_payload["corrected_gt_rule_sha256"] != sha256_file(rule_source) or rule_payload["corrected_gt_rule_bytes"] != rule_source.stat().st_size:
            _fail("corrected-GT manifest is not bound to repository MMPD rule source")
        model_identity["corrected_gt_rule_source_locator"] = str(rule_source)
        model_identity["corrected_gt_rule_source_sha256"] = sha256_file(rule_source)
        model_identity["corrected_gt_rule_source_bytes"] = rule_source.stat().st_size
    return {"plan_id": plan.plan_id, "plan_sha256": plan.plan_sha256, "plan_payload": plan.payload, "subjects": len(plan.cohort_subject_ids), "clips": plan.clip_count, "primary_rows": plan.clip_count * 10 * 53, "raw_inventory_sha256": inventory_hash, "raw_inventory_records": inventory_records, "rule_manifest_sha256": rules_hash, "corrected_gt_rule_id": rule_payload["corrected_gt_rule_id"], "corrected_gt_rule_sha256": rule_payload["corrected_gt_rule_sha256"], "corrected_gt_rule_bytes": rule_payload["corrected_gt_rule_bytes"], "metadata_coding_id": rule_payload["metadata_coding_id"], "metadata_coding_sha256": rule_payload["metadata_coding_sha256"], "metadata_coding_bytes": rule_payload["metadata_coding_bytes"], "corrected_gt_rule_locator": rule_payload["corrected_gt_rule_locator"], "metadata_coding_locator": rule_payload["metadata_coding_locator"], "checkpoint_manifest_sha256": checkpoints_hash, "checkpoint_identities": checkpoints_payload["checkpoints"], "source_closure": closure, "roi_mapping_id": "engineering_face_relative_rectangles_v1", "roi_mapping_compatibility": "not_compatible_with_frozen_checkpoints", "checkpoint_results_status": "engineering_only_unverified_roi_mapping", "code_snapshot_sha256": code_hash, "extractor_id": extractor_id, "extraction_code_sha256": extraction_code_sha256, "raw_root": str(Path(raw_root).resolve()), "checkpoint_root": str(Path(checkpoint_root).resolve()), "rule_root": str(Path(rule_root).resolve()), **model_identity}

__all__ = ["preflight_gate9_inputs"]
