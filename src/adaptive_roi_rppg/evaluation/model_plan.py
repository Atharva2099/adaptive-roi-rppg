"""Strict Gate 8 frozen-checkpoint plan loading and cohort binding."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID
from adaptive_roi_rppg.signal import POS_CONFIG_ID

EXCLUSIONS = frozenset({"4952_FullHDwebcam_after","4952_FullHDwebcam_before","4952_IriunWebcam_after","4952_IriunWebcam_before","4952_USBVideo_after","4952_USBVideo_before","6066_USBVideo_before"})
FAMILIES = frozenset({"dagger","standard_ppo","advantage_ppo"})
SHA256 = frozenset("0123456789abcdef")
PLAN_SCHEMA = "gate8-mcd-frozen-model-plan-v1"
TOP_LEVEL_FIELDS = frozenset({"schema","dataset_id","split","expected_clip_count","expected_subject_count","exclusions","signal_config_id","observation_schema_id","control_config_id","deterministic","descriptive_checkpoint_provenance_allowed","checkpoints","backend","policy_class","observation_dim","action_count","lstm_hidden_size","n_lstm_layers","bootstrap"})
CHECKPOINT_FIELDS = frozenset({"method_id","family","seed","locator","sha256","byte_size","provenance_status","training_config_provenance"})
BOOTSTRAP_FIELDS = frozenset({"replicates","seed","unit","percentile_rule","comparison_seeds"})
ARCHITECTURE = {"backend":"sb3_contrib.RecurrentPPO", "policy_class":"RecurrentActorCriticPolicy", "observation_dim":101, "action_count":12, "lstm_hidden_size":128, "n_lstm_layers":1}
BOOTSTRAP = {"replicates":10000, "seed":8101, "unit":"subject", "percentile_rule":"2.5/97.5"}
COMPARISON_SEEDS = (8101,8111,8121,8131,8102,8112,8122,8132,8103,8113,8123,8133)
PROVENANCE_STATUSES = frozenset({"descriptive_unverified"})


def _fail(message: str) -> None: raise ContractValidationError(message)


def _int(value: object) -> bool: return isinstance(value, int) and not isinstance(value, bool)
def _sha256(value: object) -> bool: return isinstance(value, str) and len(value) == 64 and value == value.lower() and set(value) <= SHA256


@dataclass(frozen=True, slots=True)
class FrozenModelPlan:
    payload: dict[str, Any]; plan_id: str
    @property
    def checkpoints(self): return tuple(self.payload["checkpoints"])


def _validate_checkpoint(spec: object, keys: set[tuple[str, int]], method_ids: set[str]) -> None:
    if not isinstance(spec, Mapping) or set(spec) != CHECKPOINT_FIELDS: _fail("checkpoint spec fields are invalid")
    family, seed, method_id = spec["family"], spec["seed"], spec["method_id"]
    if not isinstance(family, str) or family not in FAMILIES or not _int(seed) or seed not in (0, 1, 2) or not isinstance(method_id, str) or method_id != f"{family}_seed{seed}": _fail("checkpoint family/seed/method identity is invalid")
    if (family, seed) in keys or method_id in method_ids: _fail("checkpoint identity is duplicated")
    locator = spec["locator"]
    if not isinstance(locator, str) or not locator: _fail("checkpoint locator is invalid")
    path = Path(locator)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}: _fail("checkpoint locator is not package-relative")
    if not _sha256(spec["sha256"]) or not _int(spec["byte_size"]) or spec["byte_size"] <= 0: _fail("checkpoint file identity is invalid")
    if spec["provenance_status"] not in PROVENANCE_STATUSES or not isinstance(spec["training_config_provenance"], str) or not spec["training_config_provenance"]: _fail("checkpoint provenance is invalid")
    keys.add((family, seed)); method_ids.add(method_id)


def load_frozen_model_plan(path: str | Path) -> FrozenModelPlan:
    try: payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ContractValidationError("frozen model plan cannot be read") from exc
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS or payload.get("schema") != PLAN_SCHEMA: _fail("frozen model plan schema is invalid")
    if (payload["dataset_id"], payload["split"], payload["expected_clip_count"], payload["expected_subject_count"]) != ("mcd", "eval", 533, 89) or not _int(payload["expected_clip_count"]) or not _int(payload["expected_subject_count"]): _fail("frozen model cohort contract is invalid")
    exclusions = payload["exclusions"]
    if not isinstance(exclusions, list) or any(not isinstance(value, str) or not value for value in exclusions) or set(exclusions) != EXCLUSIONS or len(exclusions) != len(EXCLUSIONS): _fail("frozen model exclusions differ from Gate 6")
    if (payload["signal_config_id"], payload["observation_schema_id"], payload["control_config_id"]) != (POS_CONFIG_ID, OBSERVATION_SCHEMA_ID, CONTROL_CONFIG_ID) or type(payload["deterministic"]) is not bool or payload["deterministic"] is not True: _fail("frozen model control contract is invalid")
    if payload["descriptive_checkpoint_provenance_allowed"] is not True: _fail("plan must state the descriptive-provenance policy")
    if any(payload[key] != value or type(payload[key]) is not type(value) for key, value in ARCHITECTURE.items()): _fail("frozen model architecture contract is invalid")
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != BOOTSTRAP_FIELDS or any(bootstrap[key] != value or type(bootstrap[key]) is not type(value) for key, value in BOOTSTRAP.items()) or not isinstance(bootstrap["comparison_seeds"], list) or tuple(bootstrap["comparison_seeds"]) != COMPARISON_SEEDS or any(not _int(seed) or seed < 0 for seed in bootstrap["comparison_seeds"]): _fail("frozen model bootstrap contract is invalid")
    checkpoints = payload["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 9: _fail("frozen model plan requires exactly nine checkpoints")
    keys: set[tuple[str, int]] = set(); method_ids: set[str] = set()
    for spec in checkpoints: _validate_checkpoint(spec, keys, method_ids)
    if keys != {(family, seed) for family in FAMILIES for seed in range(3)}: _fail("frozen model plan lacks a family/seed")
    return FrozenModelPlan(payload, "gate8-frozen-model-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest())


__all__ = ["EXCLUSIONS", "FrozenModelPlan", "load_frozen_model_plan"]
