"""Strict Gate 8 frozen-checkpoint plan loading and cohort binding."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID
from adaptive_roi_rppg.signal import POS_CONFIG_ID

EXCLUSIONS = frozenset({"4952_FullHDwebcam_after","4952_FullHDwebcam_before","4952_IriunWebcam_after","4952_IriunWebcam_before","4952_USBVideo_after","4952_USBVideo_before","6066_USBVideo_before"})
FAMILIES = frozenset({"dagger","standard_ppo","advantage_ppo"})

def _fail(message: str) -> None: raise ContractValidationError(message)

@dataclass(frozen=True, slots=True)
class FrozenModelPlan:
    payload: dict[str,Any]; plan_id: str
    @property
    def checkpoints(self): return tuple(self.payload["checkpoints"])

def load_frozen_model_plan(path: str | Path) -> FrozenModelPlan:
    try: payload=json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: raise ContractValidationError("frozen model plan cannot be read") from exc
    if not isinstance(payload,dict) or payload.get("schema") != "gate8-mcd-frozen-model-plan-v1": _fail("frozen model plan schema is invalid")
    if (payload.get("dataset_id"),payload.get("split"),payload.get("expected_clip_count"),payload.get("expected_subject_count")) != ("mcd","eval",533,89): _fail("frozen model cohort contract is invalid")
    if set(payload.get("exclusions",())) != EXCLUSIONS or len(payload.get("exclusions",())) != 7: _fail("frozen model exclusions differ from Gate 6")
    if (payload.get("signal_config_id"),payload.get("observation_schema_id"),payload.get("control_config_id"),payload.get("deterministic")) != (POS_CONFIG_ID,OBSERVATION_SCHEMA_ID,CONTROL_CONFIG_ID,True): _fail("frozen model control contract is invalid")
    if payload.get("descriptive_checkpoint_provenance_allowed") is not True: _fail("plan must state the descriptive-provenance policy")
    checkpoints=payload.get("checkpoints")
    if not isinstance(checkpoints,list) or len(checkpoints)!=9: _fail("frozen model plan requires exactly nine checkpoints")
    keys=set()
    for spec in checkpoints:
        if not isinstance(spec,dict) or set(spec) != {"method_id","family","seed","locator","sha256","byte_size","provenance_status","training_config_provenance"}: _fail("checkpoint spec fields are invalid")
        key=(spec["family"],spec["seed"])
        if key in keys or spec["family"] not in FAMILIES or not isinstance(spec["seed"],int) or spec["seed"] not in (0,1,2) or not all(isinstance(spec[x],str) and spec[x] for x in ("method_id","locator","training_config_provenance")) or spec["provenance_status"] != "descriptive_unverified" or not isinstance(spec["sha256"],str) or len(spec["sha256"])!=64 or not isinstance(spec["byte_size"],int) or spec["byte_size"]<=0: _fail("checkpoint spec identity is invalid")
        keys.add(key)
    if keys != {(f,s) for f in FAMILIES for s in range(3)}: _fail("frozen model plan lacks a family/seed")
    identity={k:v for k,v in payload.items() if k not in {"plan_id"}}
    return FrozenModelPlan(payload, "gate8-frozen-model-"+hashlib.sha256(canonical_json_bytes(identity)).hexdigest())

__all__=["EXCLUSIONS","FrozenModelPlan","load_frozen_model_plan"]
