"""Repository-owned Gate 9 engineering cohort and immutable plan loader."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError

PLAN_SCHEMA = "gate9-mmpd-plan-v1"
ENGINEERING_SCHEMA = "gate9-mmpd-transfer-engineering-p29_3-excluded-v1"
SUBJECTS = ("1", "2", "5", "7", "10", "11", "12", "15", "16", "17", "20", "21", "25", "28", "29")
EXCLUDED_CLIP = "p29_3"

def _fail(message: str) -> None:
    raise ContractValidationError("Gate 9 plan: " + message)

@dataclass(frozen=True, slots=True)
class Gate9Plan:
    payload: dict[str, Any]
    plan_id: str
    plan_sha256: str
    @property
    def cohort_subject_ids(self) -> tuple[str, ...]: return tuple(self.payload["subjects"])
    @property
    def expected_clip_ids(self) -> tuple[str, ...]: return tuple(x["clip_id"] for x in self.payload["clips"])
    @property
    def expected_clip_counts(self) -> dict[str, int]:
        return {s: sum(x["subject_id"] == s for x in self.payload["clips"]) for s in self.cohort_subject_ids}
    @property
    def clip_count(self) -> int: return len(self.payload["clips"])
    @property
    def excluded_clip_ids(self) -> tuple[str, ...]: return tuple(self.payload["excluded_clip_ids"])

def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "plan_id", "profile_id", "dataset_id", "subjects", "clips", "excluded_clip_ids", "arms", "expected_hops", "engineering_only"}
    if set(payload) != required: _fail("fields are missing or unknown")
    if payload["schema"] not in (PLAN_SCHEMA, ENGINEERING_SCHEMA) or payload["dataset_id"] != "mmpd": _fail("schema or dataset identity mismatch")
    if tuple(payload["subjects"]) != SUBJECTS or not payload["engineering_only"]: _fail("only the engineering-only plan is loadable here")
    excluded = tuple(payload["excluded_clip_ids"])
    if excluded != (EXCLUDED_CLIP,): _fail("the only engineering exclusion is p29_3")
    clips = payload["clips"]
    # The checked-in config is a compact sealed template. Expansion is
    # deterministic and remains part of the authenticated plan payload.
    expected_clips = build_engineering_plan()["clips"]
    if clips == []:
        clips = expected_clips
        payload = dict(payload); payload["clips"] = clips
    elif clips != expected_clips:
        _fail("clip inventory does not match the sealed engineering cohort")
    if len(clips) != 299: _fail("engineering cohort must contain 299 clips")
    keys = {(x.get("subject_id"), x.get("clip_id")) for x in clips}
    if len(keys) != len(clips) or any(set(x) != {"subject_id", "clip_id", "view", "condition", "expected_hops"} for x in clips): _fail("clip inventory is not exact")
    if any(x["expected_hops"] != 53 for x in clips): _fail("each clip must have 53 hops")
    if any(x["subject_id"] not in SUBJECTS for x in clips) or any(x["clip_id"] == EXCLUDED_CLIP for x in clips): _fail("clip is outside the sealed cohort")
    if len(payload["arms"]) != 10 or len(set(payload["arms"])) != 10: _fail("engineering plan must have ten primary arms")
    if payload["expected_hops"] != 53 or payload["plan_id"] != "gate9-engineering-p29_3-excluded-v1": _fail("plan identity or hop count mismatch")
    return dict(payload)

def load_gate9_plan(path: str | Path) -> Gate9Plan:
    target = Path(path)
    try: raw = target.read_bytes(); payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise ContractValidationError("Gate 9 plan: cannot read canonical JSON") from exc
    if canonical_json_bytes(payload) != raw: _fail("bytes are not canonical JSON")
    payload = _validate(payload)
    # The hash identifies the effective expanded cohort, not the compact
    # template bytes.  This makes a changed inventory or expansion visible in
    # every downstream manifest.
    effective_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return Gate9Plan(payload, payload["plan_id"], effective_hash)

def build_engineering_plan() -> dict[str, Any]:
    clips = []
    for subject in SUBJECTS:
        for index in range(20):
            clip = f"p{subject}_{index}"
            if clip == EXCLUDED_CLIP: continue
            clips.append({"subject_id": subject, "clip_id": clip, "view": "unknown", "condition": "unknown", "expected_hops": 53})
    return {"schema": ENGINEERING_SCHEMA, "plan_id": "gate9-engineering-p29_3-excluded-v1", "profile_id": "mmpd-clean-causal-no-fill-v1", "dataset_id": "mmpd", "subjects": list(SUBJECTS), "clips": clips, "excluded_clip_ids": [EXCLUDED_CLIP], "arms": ["full_face_pos", "dagger_seed0", "dagger_seed1", "dagger_seed2", "standard_ppo_seed0", "standard_ppo_seed1", "standard_ppo_seed2", "advantage_ppo_seed0", "advantage_ppo_seed1", "advantage_ppo_seed2"], "expected_hops": 53, "engineering_only": True}
