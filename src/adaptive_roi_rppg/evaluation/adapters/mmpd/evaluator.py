"""Small injectable replay boundary for Gate 9; no MediaPipe dependency."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity
from .replay import rollout_gate9_policy

@dataclass(frozen=True, slots=True)
class Gate9Arm:
    method_id: str; family: str; seed: int | None = None; checkpoint_sha256: str | None = None; kind: str = "checkpoint"

FULL_FACE_ARM = Gate9Arm("full_face_pos", "fixed_full_face", None, None, "full_face")
ORACLE_B_ARM = Gate9Arm("oracle_b_greedy", "oracle_b", None, None, "oracle")
ORACLE_C_ARM = Gate9Arm("oracle_c_beam", "oracle_c", None, None, "oracle")
def validate_gate9_arm(arm: Gate9Arm) -> None:
    if not arm.method_id or arm.kind not in {"checkpoint", "full_face", "oracle"}: raise ContractValidationError("Gate 9 arm identity is incomplete")
    if arm.kind == "oracle" and arm.checkpoint_sha256 is not None: raise ContractValidationError("Oracle arm cannot have a checkpoint")
def validate_evaluator_arms(arms: Sequence[Gate9Arm]) -> None:
    for arm in arms: validate_gate9_arm(arm)
    if len(arms) != 10 or len({a.method_id for a in arms}) != 10: raise ContractValidationError("Gate 9 requires exactly ten primary arms")
def evaluate_primary_clip(frames: Sequence[Any], policies: Mapping[str, Any], *, clip_id: str) -> dict[str, tuple[Any, ...]]:
    if not frames: raise ContractValidationError("Gate 9 rejects an empty clip")
    return {method: rollout_gate9_policy(frames, policy, clip_id=clip_id) for method, policy in policies.items()}
def join_labels_after_replay(rollout: Sequence[Any], labels: Sequence[Any], metadata: Mapping[str, str]) -> tuple[dict[str, Any], ...]:
    if not rollout or len(rollout) != len(labels):
        raise ContractValidationError("Gate 9 labels do not cover replay")
    if set(metadata) != {"subject_id", "view", "condition"}:
        raise ContractValidationError("Gate 9 metadata is incomplete")
    result = []
    for row, label in zip(rollout, labels):
        if label.dataset_id != "mmpd" or label.clip_id != row["clip_id"] or label.hop_idx != row["hop_idx"] or not label.valid or label.gt_hr_bpm is None:
            raise ContractValidationError("Gate 9 label identity or validity is invalid")
        result.append({**row, **metadata, "dataset_id": "mmpd", "gt_hr_bpm": float(label.gt_hr_bpm), "abs_error_bpm": abs(float(row["post_belief_hr_bpm"]) - float(label.gt_hr_bpm)), "view": metadata["view"], "condition": metadata["condition"]})
    return tuple(result)
