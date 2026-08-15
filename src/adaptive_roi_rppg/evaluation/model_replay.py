"""GT-free recurrent-policy replay for frozen MCD controllers.

This module deliberately imports no learned-model framework.  Labels are joined
only by :func:`score_frozen_rollout`, after all policy decisions are complete.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID, control_step, initial_control_state


def _fail(message: str) -> None: raise ContractValidationError(message)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    method_id: str; family: str; seed: int | None; checkpoint_sha256: str | None
    backend: str = "sb3_contrib.RecurrentPPO"
    observation_schema_id: str = OBSERVATION_SCHEMA_ID
    observation_dim: int = 101; action_count: int = 12


class FrozenPolicy(Protocol):
    @property
    def identity(self) -> CheckpointIdentity: ...
    def initial_state(self) -> object | None: ...
    def predict(self, observation: np.ndarray, recurrent_state: object | None, *, episode_start: bool) -> tuple[int, object | None]: ...


@dataclass(frozen=True, slots=True)
class UnscoredModelHop:
    clip_id: str; hop_idx: int; hop_time_s: float; identity: CheckpointIdentity
    observation_sha256: str; frame_provenance_id: str; signal_config_id: str
    proposed_action: int; executed_action: int; previous_action: int | None
    legal: bool; override_reason: str | None; pre_hold_count: int; post_hold_count: int
    selected_valid: bool; selected_invalid_reason: str | None; selected_hr_bpm: float | None
    selected_confidence: float; selected_ppr: float; selected_coverage: float | None
    pre_belief_hr_bpm: float; post_belief_hr_bpm: float; post_belief_velocity: float; post_belief_std_bpm: float


def rollout_frozen_policy(frames: Sequence[MeasurementFrame], policy: FrozenPolicy, *, clip_id: str) -> tuple[UnscoredModelHop, ...]:
    """Replay one clip.  The function intentionally has no label parameter."""
    if not frames: _fail("model rollout rejects an empty clip")
    identity = policy.identity
    if identity.observation_schema_id != OBSERVATION_SCHEMA_ID or identity.observation_dim != 101 or identity.action_count != 12:
        _fail("policy identity does not match the current controller contract")
    state, recurrent, episode_start, output = initial_control_state("mcd", clip_id), policy.initial_state(), True, []
    for expected_hop, frame in enumerate(frames):
        if frame.dataset_id != "mcd" or frame.clip_id != clip_id or frame.hop_idx != expected_hop: _fail("model frame identity is not contiguous")
        # control_step builds the canonical pre-action observation; build it once here for the policy.
        from adaptive_roi_rppg.control import build_observation
        values = build_observation(frame, state).array()
        if values.shape != (101,) or values.dtype != np.float32 or not np.isfinite(values).all(): _fail("model observation must be finite float32[101]")
        batched = values.reshape(1, 101)
        action, recurrent = policy.predict(batched, recurrent, episode_start=episode_start)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < 12: _fail("policy proposed action is invalid")
        state, transition = control_step(frame, state, int(action)); d, m, b = transition.action_decision, transition.selected_measurement, transition.post_belief
        output.append(UnscoredModelHop(clip_id, frame.hop_idx, frame.hop_time_s, identity,
            hashlib.sha256(values.tobytes()).hexdigest(), transition.frame_provenance_id, transition.signal_config_id,
            d.proposed_action, d.executed_action, d.previous_action, d.legal, d.override_reason, d.pre_hold_count, d.post_hold_count,
            m.valid, m.invalid_reason, m.hr_bpm, m.confidence, m.peak_power_ratio, m.coverage,
            transition.pre_belief.mean_hr, b.mean_hr, b.velocity, math.sqrt(max(b.covariance[0][0], 0.0))))
        episode_start = False
    return tuple(output)


def score_frozen_rollout(rollout: Sequence[UnscoredModelHop], labels: Sequence[LabelFrame], metadata: dict[str, str]) -> tuple[dict[str, Any], ...]:
    """Join labels after inference; labels can only add GT and absolute error."""
    if not rollout or len(rollout) != len(labels): _fail("model rollout/label coverage differs")
    required = {"subject_id", "view", "condition"}
    if set(metadata) != required or any(not isinstance(metadata[k], str) or not metadata[k] for k in required): _fail("model metadata is invalid")
    rows = []
    for hop, label in zip(rollout, labels):
        if (label.dataset_id, label.clip_id, label.hop_idx, label.hop_time_s) != ("mcd", hop.clip_id, hop.hop_idx, hop.hop_time_s) or not label.valid or label.gt_hr_bpm is None or not math.isfinite(label.gt_hr_bpm): _fail("model labels are missing, invalid, or misaligned")
        rows.append({**metadata, "dataset_id":"mcd", "clip_id":hop.clip_id, "hop_idx":hop.hop_idx, "hop_time_s":hop.hop_time_s,
            "method_id":hop.identity.method_id, "family":hop.identity.family, "seed":hop.identity.seed, "checkpoint_sha256":hop.identity.checkpoint_sha256,
            "observation_schema_id":hop.identity.observation_schema_id, "observation_sha256":hop.observation_sha256, "frame_provenance_id":hop.frame_provenance_id,
            "signal_config_id":hop.signal_config_id, "control_config_id":CONTROL_CONFIG_ID, "gt_rule_id":label.gt_rule_id, "gt_hr_bpm":label.gt_hr_bpm,
            "abs_error_bpm":abs(hop.post_belief_hr_bpm-label.gt_hr_bpm), **{name:getattr(hop, name) for name in UnscoredModelHop.__dataclass_fields__ if name not in {"clip_id","hop_idx","hop_time_s","identity","observation_sha256","frame_provenance_id","signal_config_id"}}})
    return tuple(rows)


def summarize_model_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Exact row-level clip and subject aggregates; seed rows are never pooled."""
    if not rows: _fail("model rows are empty")
    clips: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows: clips.setdefault((row["method_id"], row["clip_id"]), []).append(row)
    clip_rows = []
    for (method_id, clip_id), values in sorted(clips.items()):
        values.sort(key=lambda r:r["hop_idx"])
        if [r["hop_idx"] for r in values] != list(range(len(values))): _fail("model rows have non-contiguous hops")
        actions=[r["executed_action"] for r in values]; first=values[0]
        def _jump(field):
            values_=[r.get(field) for r in values]
            pairs=[abs(float(a)-float(b)) for a,b in zip(values_,values_[1:]) if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
            return sum(pairs)/len(pairs) if pairs else 0.0
        jumps=[abs(a-b) for a,b in zip(actions,actions[1:])]
        proposed=[r.get("proposed_action", r["executed_action"]) for r in values]
        switches=sum(a!=b for a,b in zip(actions, actions[1:]))
        clip_rows.append({"method_id":method_id,"family":first["family"],"seed":first["seed"],"checkpoint_sha256":first["checkpoint_sha256"],"clip_id":clip_id,"subject_id":first["subject_id"],"view":first["view"],"condition":first["condition"],"hop_count":len(values),"mae_bpm":sum(r["abs_error_bpm"] for r in values)/len(values),"proposed_action_count":len(proposed),"executed_action_count":len(actions),"override_count":sum(not r["legal"] for r in values),"invalid_selected_count":sum(not r["selected_valid"] for r in values),"hold_mean":sum(r["post_hold_count"] for r in values)/len(values),"hold_max":max(r["post_hold_count"] for r in values),"switch_count":switches,"switches_per_hop":switches/len(values),"roi_index_jump_mean_abs":sum(jumps)/len(jumps) if jumps else 0.0,"selected_hr_jump_mean_abs_bpm":_jump("selected_hr_bpm"),"belief_jump_mean_abs_bpm":_jump("post_belief_hr_bpm"),**{f"action_count_{i:02d}":actions.count(i) for i in range(12)}})
    subjects: dict[tuple[str,str], list[dict[str,Any]]] = {}
    for row in clip_rows: subjects.setdefault((row["method_id"],row["subject_id"]),[]).append(row)
    behavior=("switches_per_hop","hold_mean","roi_index_jump_mean_abs","selected_hr_jump_mean_abs_bpm","belief_jump_mean_abs_bpm","invalid_selected_count")
    subject_rows=[]
    for (method, subject), values in sorted(subjects.items()):
        first=values[0]
        subject_rows.append({"method_id":method,"family":first["family"],"seed":first["seed"],"checkpoint_sha256":first["checkpoint_sha256"],"subject_id":subject,"clip_count":len(values),"equal_clip_mae_bpm":sum(x["mae_bpm"] for x in values)/len(values),"proposed_action_count":sum(x["proposed_action_count"] for x in values),"executed_action_count":sum(x["executed_action_count"] for x in values),"override_count":sum(x["override_count"] for x in values),"invalid_selected_count":sum(x["invalid_selected_count"] for x in values),"hold_mean":sum(x["hold_mean"] for x in values)/len(values),"hold_max":max(x["hold_max"] for x in values),"switches_per_hop":sum(x["switches_per_hop"] for x in values)/len(values),"roi_index_jump_mean_abs":sum(x["roi_index_jump_mean_abs"] for x in values)/len(values),"selected_hr_jump_mean_abs_bpm":sum(x["selected_hr_jump_mean_abs_bpm"] for x in values)/len(values),"belief_jump_mean_abs_bpm":sum(x["belief_jump_mean_abs_bpm"] for x in values)/len(values)})
    by_method: dict[str,list[dict[str,Any]]] = {}
    for row in clip_rows: by_method.setdefault(row["method_id"],[]).append(row)
    checkpoint_results={}
    for method, values in sorted(by_method.items()):
        subs=[row for row in subject_rows if row["method_id"]==method]
        hops=sum(x["hop_count"] for x in values); actions={f"action_{i:02d}":sum(x[f"action_count_{i:02d}"] for x in values) for i in range(12)}
        first=values[0]
        checkpoint_results[method]={"method_id":method,"family":first["family"],"seed":first["seed"],"checkpoint_sha256":first["checkpoint_sha256"],"clip_count":len(values),"subject_count":len(subs),"hop_count":hops,"equal_clip_mae_bpm":sum(x["mae_bpm"] for x in values)/len(values),"equal_subject_mae_bpm":sum(x["equal_clip_mae_bpm"] for x in subs)/len(subs),"proposed_action_count":sum(x["proposed_action_count"] for x in values),"executed_action_count":sum(x["executed_action_count"] for x in values),"override_count":sum(x["override_count"] for x in values),"switches_per_hop":sum(x["switch_count"] for x in values)/hops,"hold_mean":sum(x["hold_mean"]*x["hop_count"] for x in values)/hops,"hold_max":max(x["hold_max"] for x in values),"roi_index_jump_mean_abs":sum(x["roi_index_jump_mean_abs"]*max(x["hop_count"]-1,0) for x in values)/max(sum(max(x["hop_count"]-1,0) for x in values),1),"selected_hr_jump_mean_abs_bpm":sum(x["selected_hr_jump_mean_abs_bpm"] for x in values)/len(values),"belief_jump_mean_abs_bpm":sum(x["belief_jump_mean_abs_bpm"] for x in values)/len(values),"invalid_selected_count":sum(x["invalid_selected_count"] for x in values),"action_distribution":actions}
    cells=[]; view_rows=[]; condition_rows=[]
    for method, values in sorted(by_method.items()):
        for view, condition in sorted({(x["view"],x["condition"]) for x in values}):
            selected=[x for x in values if (x["view"],x["condition"])==(view,condition)]
            cells.append({"method_id":method,"view":view,"condition":condition,"clip_count":len(selected),"equal_clip_mae_bpm":sum(x["mae_bpm"] for x in selected)/len(selected)})
        for view in sorted({x["view"] for x in values}):
            selected=[x for x in values if x["view"]==view]; view_rows.append({"method_id":method,"view":view,"clip_count":len(selected),"equal_clip_mae_bpm":sum(x["mae_bpm"] for x in selected)/len(selected)})
        for condition in sorted({x["condition"] for x in values}):
            selected=[x for x in values if x["condition"]==condition]; condition_rows.append({"method_id":method,"condition":condition,"clip_count":len(selected),"equal_clip_mae_bpm":sum(x["mae_bpm"] for x in selected)/len(selected)})
    family_results=[]
    for family in sorted({x["family"] for x in clip_rows if x["family"]!="fixed_full_face"}):
        members=sorted({x["method_id"] for x in clip_rows if x["family"]==family})
        if len(members)==3:
            seed_values=[checkpoint_results[x]["equal_clip_mae_bpm"] for x in members]
            family_results.append({"family":family,"member_identities":[{"method_id":checkpoint_results[x]["method_id"],"seed":checkpoint_results[x]["seed"],"checkpoint_sha256":checkpoint_results[x]["checkpoint_sha256"]} for x in members],"seed_results":[checkpoint_results[x] for x in members],"seed_mean_equal_clip_mae_bpm":sum(seed_values)/3,"seed_range_equal_clip_mae_bpm":[min(seed_values),max(seed_values)],"not_an_ensemble":True})
    return {"checkpoint_results":checkpoint_results,"family_results":family_results,"view_rows":view_rows,"condition_rows":condition_rows,"view_condition_rows":cells,"clip_rows":clip_rows,"subject_rows":subject_rows}


def paired_subject_bootstrap(clip_rows: Sequence[dict[str, Any]], method_a: str, method_b: str, *, estimand: str = "equal_clip", replicates: int = 10_000, seed: int = 8101) -> dict[str, Any]:
    """Subject-block paired interval for two complete checkpoint arms.

    A sampled subject carries every one of its clips in both arms.  This keeps
    the seed/controller comparison paired and avoids treating hops as samples.
    """
    if replicates < 1: _fail("bootstrap replicates must be positive")
    by_key: dict[tuple[str,str], list[float]] = {}
    for row in clip_rows:
        if row["method_id"] in (method_a,method_b): by_key.setdefault((row["method_id"],row["subject_id"]),[]).append(float(row["mae_bpm"]))
    subjects=sorted({s for m,s in by_key if m==method_a})
    if not subjects or subjects != sorted({s for m,s in by_key if m==method_b}): _fail("paired bootstrap arms do not share subjects")
    if estimand not in {"equal_clip","equal_subject"}: _fail("bootstrap estimand is invalid")
    values={m:{s:by_key[(m,s)] for s in subjects} for m in (method_a,method_b)}
    def estimate(selected):
        if estimand=="equal_subject": return float(np.mean([np.mean(values[method_a][subjects[i]])-np.mean(values[method_b][subjects[i]]) for i in selected]))
        a=[value for i in selected for value in values[method_a][subjects[i]]]; b=[value for i in selected for value in values[method_b][subjects[i]]]
        return float(np.mean(a)-np.mean(b))
    point=estimate(range(len(subjects)))
    rng=np.random.default_rng(seed); draws=[]
    for selected in rng.integers(0,len(subjects),size=(replicates,len(subjects))):
        draws.append(estimate(selected))
    draws.sort(); lo=draws[int(.025*(replicates-1))]; hi=draws[int(.975*(replicates-1))]
    return {"method_a":method_a,"method_b":method_b,"estimand":f"{estimand}_mae_difference_a_minus_b_bpm","difference_a_minus_b_bpm":point,"ci_95_percentile_bpm":[lo,hi],"replicates":replicates,"seed":seed,"resampling_unit":"subject","subject_draw_keeps_all_clip_rows":True}


__all__ = ["CheckpointIdentity", "FrozenPolicy", "UnscoredModelHop", "rollout_frozen_policy", "score_frozen_rollout", "summarize_model_rows", "paired_subject_bootstrap"]
