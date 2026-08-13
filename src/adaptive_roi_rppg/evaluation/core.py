"""Strict Gate 5 joins, canonical clip bindings, and immutable publication."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from adaptive_roi_rppg.contracts import (ClipManifest, LabelFrame, ManifestStatus,
    ROI_NAMES, canonical_json_bytes, read_json_object, require_structurally_complete,
    sha256_file)
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID, control_step, initial_control_state
from adaptive_roi_rppg.data.mcd import (MCD_DATASET_ID, MCD_SCHEMA_ID,
    MCDManifestBundle, load_mcd_manifest_tree, read_mcd_canonical_frames)
from adaptive_roi_rppg.labels import GT_RULE_ID, read_mcd_labels
from adaptive_roi_rppg.signal import POS_CONFIG_ID, build_pos_measurements

METRIC_ID = "strict_post_belief_mae_v1"
AGGREGATION_ID = "equal_clip_mean_then_subject_clip_mean_v1"
UNCERTAINTY_ID = "bootstrap_deferred_gate6"
METHOD_ID = "full_face_pos_deterministic_v1"
ACTION_COUNT_FIELDS = tuple(f"action_count_{i:02d}" for i in range(12))
HOP_FIELDS = ("run_id", "plan_id", "method_id", "dataset_id", "clip_id", "subject_id", "view", "condition", "hop_idx", "hop_time_s", "seed", "checkpoint_sha256", "prediction_post_belief_hr_bpm", "gt_hr_bpm", "abs_error_bpm", "selected_roi_index", "selected_roi_name", "selected_hr_bpm", "selected_valid", "selected_invalid_reason", "pre_belief_mean_hr", "post_belief_mean_hr", "proposed_action", "executed_action", "previous_action", "legal", "override_reason", "signal_config_id", "control_config_id", "observation_schema_id", "gt_rule_id", "frame_provenance_id", "gt_source_sha256")
CLIP_FIELDS = ("run_id", "plan_id", "method_id", "dataset_id", "clip_id", "subject_id", "view", "condition", "seed", "checkpoint_sha256", "expected_hops", "scored_hops", "invalid_selected_count", "mae_clip", "proposed_switches", "executed_switches", "override_count", *ACTION_COUNT_FIELDS, "signal_config_id", "control_config_id", "observation_schema_id", "gt_rule_id", "gt_source_sha256")
SUBJECT_FIELDS = ("subject_id", "clip_count", "mean_clip_mae")
RUN_FIELDS = ("run_id", "plan_id", "method_id", "dataset_id", "split", "clip_count", "subject_count", "hop_count", "equal_clip_mean_mae", "invalid_selected_count", "override_count", "metric_id", "aggregation_id", "uncertainty_id")
_PLAN_FIELDS = ("dataset_id", "split", "dataset_manifest_id", "dataset_manifest_sha256", "split_manifest_id", "split_manifest_sha256", "exact_clip_keys", "expected_hops", "clip_bindings", "method_id", "seed", "checkpoint_sha256", "signal_config_id", "control_config_id", "observation_schema_id", "gt_rule_id", "metric_id", "aggregation_id", "uncertainty_id")
SUBSTANTIVE_FILES = ("per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256")
_MARKER_FILES = ("STARTED.json", "COMPLETE.json", "FAILED.json")
_PUBLICATION_SCHEMA = "gate5-publication-state-v1"


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value: _fail(f"{field}: must be a nonempty string")
    return value


def _sha(value: Any, field: str, optional: bool = False) -> str | None:
    if optional and value is None: return None
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): _fail(f"{field}: must be lowercase SHA-256")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): _fail(f"{field}: must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ClipBinding:
    dataset_id: str
    clip_id: str
    clip_manifest_id: str
    canonical_clip_sha256: str
    dataset_ref_position: int
    state_sha256: str
    gt_sha256: str
    gt_row_count: int
    subject_id: str
    view: str
    condition: str
    camera_id: str
    camera_fps: float
    split_id: str
    schema_id: str

    def __post_init__(self) -> None:
        for name in ("dataset_id", "clip_id", "clip_manifest_id", "subject_id", "view", "condition", "camera_id", "split_id", "schema_id"):
            _string(getattr(self, name), name)
        for name in ("canonical_clip_sha256", "state_sha256", "gt_sha256"): _sha(getattr(self, name), name)
        if type(self.dataset_ref_position) is not int or self.dataset_ref_position < 0: _fail("dataset_ref_position: must be a nonnegative integer")
        if type(self.gt_row_count) is not int or self.gt_row_count <= 0: _fail("gt_row_count: must be a positive integer")
        fps = _finite(self.camera_fps, "camera_fps")
        if fps not in (24.0, 30.0): _fail("camera_fps: must be exactly 24 or 30")
        object.__setattr__(self, "camera_fps", fps)

    @classmethod
    def from_clip(cls, clip: ClipManifest, dataset_ref_position: int = 0) -> "ClipBinding":
        if clip.gt_sha256 is None or clip.gt_row_count is None or clip.gt_row_count != clip.state_row_count: _fail("clip binding requires authenticated GT/state row equality")
        return cls(clip.dataset_id, clip.clip_id, clip.clip_manifest_id,
            hashlib.sha256(canonical_json_bytes(clip.to_dict())).hexdigest(),
            dataset_ref_position, clip.state_sha256, clip.gt_sha256, clip.gt_row_count, clip.subject_id,
            clip.view, clip.condition, clip.camera_id, float(clip.camera_fps),
            clip.split_id, clip.schema_id)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ("dataset_id", "clip_id", "clip_manifest_id", "canonical_clip_sha256", "dataset_ref_position", "state_sha256", "gt_sha256", "gt_row_count", "subject_id", "view", "condition", "camera_id", "camera_fps", "split_id", "schema_id")}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClipBinding":
        fields = {"dataset_id", "clip_id", "clip_manifest_id", "canonical_clip_sha256", "dataset_ref_position", "state_sha256", "gt_sha256", "gt_row_count", "subject_id", "view", "condition", "camera_id", "camera_fps", "split_id", "schema_id"}
        if not isinstance(value, Mapping) or set(value) != fields: _fail("clip binding fields are not exact")
        return cls(**dict(value))


def _binding_matches(binding: ClipBinding, clip: ClipManifest) -> bool:
    return binding == ClipBinding.from_clip(clip, binding.dataset_ref_position)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    run_id: str
    code_snapshot_sha256: str
    command: str
    environment_versions: Mapping[str, str]
    slurm_job_id: str | None = None
    slurm_node: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _string(self.run_id, "run_id")); object.__setattr__(self, "code_snapshot_sha256", _sha(self.code_snapshot_sha256, "code_snapshot_sha256")); object.__setattr__(self, "command", _string(self.command, "command"))
        if not isinstance(self.environment_versions, Mapping) or not self.environment_versions or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in self.environment_versions.items()): _fail("environment_versions: must map nonempty strings to strings")
        if (self.slurm_job_id is None) != (self.slurm_node is None): _fail("slurm_job_id and slurm_node must both be null or both be nonempty")
        for name in ("slurm_job_id", "slurm_node"):
            value = getattr(self, name)
            if value is not None: _string(value, name)
        object.__setattr__(self, "environment_versions", dict(sorted(self.environment_versions.items())))


@dataclass(frozen=True, slots=True)
class FrozenEvaluationPlan:
    dataset_id: str
    split: str
    dataset_manifest_id: str
    dataset_manifest_sha256: str
    split_manifest_id: str
    split_manifest_sha256: str
    exact_clip_keys: tuple[tuple[str, str], ...]
    expected_hops: tuple[tuple[str, int], ...]
    clip_bindings: tuple[ClipBinding, ...]
    method_id: str = METHOD_ID
    seed: int | None = None
    checkpoint_sha256: str | None = None
    signal_config_id: str = POS_CONFIG_ID
    control_config_id: str = CONTROL_CONFIG_ID
    observation_schema_id: str = OBSERVATION_SCHEMA_ID
    gt_rule_id: str = GT_RULE_ID
    metric_id: str = METRIC_ID
    aggregation_id: str = AGGREGATION_ID
    uncertainty_id: str = UNCERTAINTY_ID
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("dataset_id", "split", "dataset_manifest_id", "split_manifest_id", "method_id", "signal_config_id", "control_config_id", "observation_schema_id", "gt_rule_id", "metric_id", "aggregation_id", "uncertainty_id"): _string(getattr(self, name), name)
        if self.dataset_id != MCD_DATASET_ID or self.split != "train" or (self.method_id, self.signal_config_id, self.control_config_id, self.observation_schema_id, self.gt_rule_id, self.metric_id, self.aggregation_id, self.uncertainty_id) != (METHOD_ID, POS_CONFIG_ID, CONTROL_CONFIG_ID, OBSERVATION_SCHEMA_ID, GT_RULE_ID, METRIC_ID, AGGREGATION_ID, UNCERTAINTY_ID): _fail("Gate 5 plan constants are not frozen")
        _sha(self.dataset_manifest_sha256, "dataset_manifest_sha256"); _sha(self.split_manifest_sha256, "split_manifest_sha256"); _sha(self.checkpoint_sha256, "checkpoint_sha256", True)
        if self.seed is not None and type(self.seed) is not int: _fail("seed: must be an integer or null")
        keys = tuple(tuple(item) for item in self.exact_clip_keys); hops = tuple(tuple(item) for item in self.expected_hops); bindings = tuple(item if isinstance(item, ClipBinding) else ClipBinding.from_dict(item) for item in self.clip_bindings)
        if not keys or len(keys) != len(set(keys)) or any(len(k) != 2 or not all(isinstance(v, str) and v for v in k) for k in keys): _fail("exact_clip_keys are not exact and unique")
        if len(hops) != len(keys) or tuple(k[1] for k in keys) != tuple(h[0] for h in hops) or any(len(h) != 2 or type(h[1]) is not int or h[1] <= 0 for h in hops): _fail("expected_hops are not exact")
        if len(bindings) != len(keys) or tuple((b.dataset_id, b.clip_id) for b in bindings) != keys or len({b.clip_id for b in bindings}) != len(bindings): _fail("clip bindings are not exact")
        if self.method_id == METHOD_ID and (self.seed is not None or self.checkpoint_sha256 is not None): _fail("deterministic full-face plan requires null seed and checkpoint")
        object.__setattr__(self, "exact_clip_keys", keys); object.__setattr__(self, "expected_hops", hops); object.__setattr__(self, "clip_bindings", bindings)
        identity = {name: getattr(self, name) for name in _PLAN_FIELDS}
        identity["clip_bindings"] = [binding.to_dict() for binding in bindings]
        object.__setattr__(self, "plan_id", "frozen-plan-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, **{name: ([b.to_dict() for b in self.clip_bindings] if name == "clip_bindings" else getattr(self, name)) for name in _PLAN_FIELDS}}


def build_train_full_face_plan(manifest_tree: str | os.PathLike[str]) -> FrozenEvaluationPlan:
    bundle = load_mcd_manifest_tree(manifest_tree)
    if not isinstance(bundle, MCDManifestBundle): _fail("bundle must be an authenticated MCDManifestBundle")
    require_structurally_complete(bundle.split_manifest); require_structurally_complete(bundle.dataset_manifest)
    if bundle.dataset_manifest.dataset_id != MCD_DATASET_ID or bundle.dataset_manifest.schema_id != MCD_SCHEMA_ID or bundle.split_manifest.dataset_id != MCD_DATASET_ID: _fail("bundle is not complete MCD")
    root = Path(manifest_tree)
    if root.is_symlink() or not root.is_dir(): _fail("manifest_tree must be a real directory")
    dataset_path, split_path = root / "dataset_manifest.json", root / "split_manifest.json"
    if any(p.is_symlink() or not p.is_file() for p in (dataset_path, split_path)): _fail("manifest tree files must be direct regular files")
    if read_json_object(dataset_path) != bundle.dataset_manifest.to_dict() or read_json_object(split_path) != bundle.split_manifest.to_dict(): _fail("manifest tree JSON does not bind to the supplied authenticated bundle")
    canonical = {clip.clip_id: clip for clip in bundle.clip_manifests}
    if len(canonical) != len(bundle.clip_manifests) or len(canonical) != len(bundle.dataset_manifest.clip_manifest_refs) or any(clip.status is not ManifestStatus.complete for clip in bundle.clip_manifests): _fail("bundle clip manifests are not complete and canonical")
    ref_hash = dict(zip(bundle.dataset_manifest.clip_manifest_refs, bundle.dataset_manifest.clip_manifest_hashes))
    subjects = sorted(bundle.split_manifest.train_subject_ids)[:2]
    if len(subjects) != 2: _fail("train split must contain at least two subjects")
    clips = tuple(sorted((c for c in bundle.clip_manifests if c.subject_id in subjects), key=lambda c: (c.subject_id, c.clip_id)))
    expected_inventory = {(subject, camera, condition) for subject in subjects for camera in ("FullHDwebcam", "USBVideo", "IriunWebcam") for condition in ("before", "after")}
    if len(clips) != 12 or {(c.subject_id, c.camera_id, c.condition) for c in clips} != expected_inventory:
        _fail("first two train subjects must have exactly six camera-condition clips each")
    for clip in clips:
        if canonical.get(clip.clip_id) != clip: _fail(f"{clip.clip_id}: supplied clip is not the canonical bundle value")
        if clip.dataset_id != MCD_DATASET_ID or clip.schema_id != MCD_SCHEMA_ID or clip.status is not ManifestStatus.complete or clip.split_id != bundle.split_manifest.split_id or clip.clip_id not in bundle.split_manifest.train_clip_ids or clip.subject_id not in bundle.split_manifest.train_subject_ids: _fail("ordered clips are not authenticated train clips")
        if clip.clip_manifest_id not in ref_hash or ref_hash[clip.clip_manifest_id] != hashlib.sha256(canonical_json_bytes(clip.to_dict())).hexdigest(): _fail(f"{clip.clip_id}: dataset manifest reference/hash mismatch")
    positions = {ref: index for index, ref in enumerate(bundle.dataset_manifest.clip_manifest_refs)}
    bindings = tuple(ClipBinding.from_clip(c, positions[c.clip_manifest_id]) for c in clips)
    expected = tuple((c.clip_id, max(0, (c.state_row_count - round(8 * c.camera_fps)) // round(c.camera_fps) + 1)) for c in clips)
    if any(n <= 0 for _, n in expected): _fail("selected clips must have at least one hop")
    return FrozenEvaluationPlan(MCD_DATASET_ID, "train", bundle.dataset_manifest.manifest_id, sha256_file(dataset_path), bundle.split_manifest.split_id, sha256_file(split_path), tuple((c.dataset_id, c.clip_id) for c in clips), expected, bindings)


def _same_belief(left: Any, right: Any) -> bool:
    return left == right


def _validate_sources(plan: FrozenEvaluationPlan, bundle: MCDManifestBundle, state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str]) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[LabelFrame, ...]]]:
    transitions = {}; labels = {}
    clips = {c.clip_id: c for c in bundle.clip_manifests}
    bindings = {b.clip_id: b for b in plan.clip_bindings}; expected = dict(plan.expected_hops)
    if not set(cid for _, cid in plan.exact_clip_keys).issubset(clips): _fail("fresh bundle does not contain the plan cohort")
    for _, clip_id in plan.exact_clip_keys:
        clip = clips[clip_id]
        binding = bindings[clip.clip_id]
        if not _binding_matches(binding, clip): _fail(f"{clip.clip_id}: supplied clip does not exactly match plan binding")
        canonical = read_mcd_canonical_frames(bundle, state_root, clip.clip_id, "train")
        fs = build_pos_measurements(canonical)
        ls = read_mcd_labels(bundle, gt_root, clip.clip_id, "train")
        state = initial_control_state(clip.dataset_id, clip.clip_id); ts = []
        for frame in fs:
            state, transition = control_step(frame, state, 0); ts.append(transition)
        ts = tuple(ts); transitions[clip.clip_id] = ts
        if len(canonical) != clip.state_row_count or len(fs) != expected[clip.clip_id] or len(ts) != expected[clip.clip_id] or len(ls) != expected[clip.clip_id]: _fail(f"{clip.clip_id}: wrong hop count")
        tkeys = [(x.dataset_id, x.clip_id, x.hop_idx) for x in ts]; lkeys = [(x.dataset_id, x.clip_id, x.hop_idx) for x in ls]; wanted = [(clip.dataset_id, clip.clip_id, i) for i in range(expected[clip.clip_id])]
        if tkeys != wanted or lkeys != wanted or tkeys != lkeys: _fail(f"{clip.clip_id}: duplicate/missing/extra/nonordered keys")
        previous_post = None; previous_action = None
        for t, label in zip(ts, ls):
            d, m = t.action_decision, t.selected_measurement
            if t.dataset_id != clip.dataset_id or t.clip_id != clip.clip_id or t.signal_config_id != plan.signal_config_id or t.control_config_id != plan.control_config_id or t.observation.schema_id != plan.observation_schema_id or label.dataset_id != clip.dataset_id or label.clip_id != clip.clip_id or label.gt_rule_id != plan.gt_rule_id or not label.valid or not math.isfinite(label.gt_hr_bpm): _fail(f"{clip.clip_id}/{t.hop_idx}: identity or label invalid")
            if d.proposed_action != 0 or d.executed_action != 0 or d.previous_action != previous_action or m.roi_index != 0 or m.roi_name is not ROI_NAMES[0] or m.signal_config_id != plan.signal_config_id: _fail(f"{clip.clip_id}/{t.hop_idx}: full-face/action semantics invalid")
            expected_time = (round(8 * clip.camera_fps) + t.hop_idx * round(clip.camera_fps)) / clip.camera_fps
            if previous_post is None and (t.pre_belief != initial_control_state(clip.dataset_id, clip.clip_id).belief or d.previous_action is not None): _fail(f"{clip.clip_id}/{t.hop_idx}: initial belief/action mismatch")
            if previous_post is not None and (not _same_belief(t.pre_belief, previous_post) or d.previous_action != previous_action): _fail(f"{clip.clip_id}/{t.hop_idx}: belief/action continuity failed")
            if abs(label.hop_time_s - expected_time) > 1e-12: _fail(f"{clip.clip_id}/{t.hop_idx}: hop time mismatch")
            if not t.frame_provenance_id: _fail(f"{clip.clip_id}/{t.hop_idx}: missing provenance")
            if m.valid != (m.hr_bpm is not None and math.isfinite(m.hr_bpm)) or (m.valid and m.invalid_reason is not None) or (not m.valid and (m.hr_bpm is not None or not m.invalid_reason)): _fail(f"{clip.clip_id}/{t.hop_idx}: selected validity semantics invalid")
            if not math.isfinite(t.post_belief.mean_hr): _fail(f"{clip.clip_id}/{t.hop_idx}: nonfinite prediction")
            previous_post, previous_action = t.post_belief, d.executed_action
        labels[clip.clip_id] = ls
    return transitions, labels

def evaluate(manifest_tree: str | os.PathLike[str], state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str], provenance: RunProvenance, plan: FrozenEvaluationPlan | None = None) -> dict[str, Any]:
    if not isinstance(provenance, RunProvenance): _fail("provenance must be RunProvenance")
    bundle = load_mcd_manifest_tree(manifest_tree)
    fresh_plan = build_train_full_face_plan(manifest_tree)
    if plan is not None and (not isinstance(plan, FrozenEvaluationPlan) or plan != fresh_plan): _fail("supplied plan is not the freshly built canonical plan")
    plan = fresh_plan
    transitions, labels = _validate_sources(plan, bundle, state_root, gt_root)
    clips = {c.clip_id: c for c in bundle.clip_manifests}
    clip_map = clips; hop_rows, clip_rows = [], []
    for _, cid in plan.exact_clip_keys:
        clip, ts, ls = clip_map[cid], tuple(transitions[cid]), tuple(labels[cid]); errors=[]; counts=[0]*12; proposed_switches=executed_switches=overrides=invalid=0
        for t, label in zip(ts, ls):
            d, m = t.action_decision, t.selected_measurement; error=abs(t.post_belief.mean_hr-label.gt_hr_bpm); errors.append(error); counts[d.executed_action]+=1; proposed_switches += int(d.previous_action is not None and d.proposed_action != d.previous_action); executed_switches += int(d.previous_action is not None and d.executed_action != d.previous_action); overrides += int(not d.legal); invalid += int(not m.valid)
            hop_rows.append({"run_id":provenance.run_id,"plan_id":plan.plan_id,"method_id":plan.method_id,"dataset_id":clip.dataset_id,"clip_id":cid,"subject_id":clip.subject_id,"view":clip.view,"condition":clip.condition,"hop_idx":t.hop_idx,"hop_time_s":label.hop_time_s,"seed":plan.seed,"checkpoint_sha256":plan.checkpoint_sha256,"prediction_post_belief_hr_bpm":t.post_belief.mean_hr,"gt_hr_bpm":label.gt_hr_bpm,"abs_error_bpm":error,"selected_roi_index":m.roi_index,"selected_roi_name":m.roi_name.value,"selected_hr_bpm":m.hr_bpm,"selected_valid":m.valid,"selected_invalid_reason":m.invalid_reason,"pre_belief_mean_hr":t.pre_belief.mean_hr,"post_belief_mean_hr":t.post_belief.mean_hr,"proposed_action":d.proposed_action,"executed_action":d.executed_action,"previous_action":d.previous_action,"legal":d.legal,"override_reason":d.override_reason,"signal_config_id":t.signal_config_id,"control_config_id":t.control_config_id,"observation_schema_id":t.observation.schema_id,"gt_rule_id":label.gt_rule_id,"frame_provenance_id":t.frame_provenance_id,"gt_source_sha256":clip.gt_sha256})
        clip_rows.append({"run_id":provenance.run_id,"plan_id":plan.plan_id,"method_id":plan.method_id,"dataset_id":clip.dataset_id,"clip_id":cid,"subject_id":clip.subject_id,"view":clip.view,"condition":clip.condition,"seed":plan.seed,"checkpoint_sha256":plan.checkpoint_sha256,"expected_hops":len(ts),"scored_hops":len(errors),"invalid_selected_count":invalid,"mae_clip":sum(errors)/len(errors),"proposed_switches":proposed_switches,"executed_switches":executed_switches,"override_count":overrides,**{ACTION_COUNT_FIELDS[i]:counts[i] for i in range(12)},"signal_config_id":plan.signal_config_id,"control_config_id":plan.control_config_id,"observation_schema_id":plan.observation_schema_id,"gt_rule_id":plan.gt_rule_id,"gt_source_sha256":clip.gt_sha256})
    subjects=sorted({r["subject_id"] for r in clip_rows}); subject_rows=[{"subject_id":s,"clip_count":sum(r["subject_id"]==s for r in clip_rows),"mean_clip_mae":sum(r["mae_clip"] for r in clip_rows if r["subject_id"]==s)/sum(r["subject_id"]==s for r in clip_rows)} for s in subjects]
    run={"run_id":provenance.run_id,"plan_id":plan.plan_id,"method_id":plan.method_id,"dataset_id":plan.dataset_id,"split":plan.split,"clip_count":len(clip_rows),"subject_count":len(subject_rows),"hop_count":len(hop_rows),"equal_clip_mean_mae":sum(r["mae_clip"] for r in clip_rows)/len(clip_rows),"invalid_selected_count":sum(r["invalid_selected_count"] for r in clip_rows),"override_count":sum(r["override_count"] for r in clip_rows),"metric_id":plan.metric_id,"aggregation_id":plan.aggregation_id,"uncertainty_id":plan.uncertainty_id}
    return {"plan":plan,"hop_rows":hop_rows,"clip_rows":clip_rows,"subject_rows":subject_rows,"run_summary":run,"provenance":provenance}


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, extrasaction="raise"); writer.writeheader(); writer.writerows({k:row.get(k) for k in fields} for row in rows)


def _check_result(result: Mapping[str, Any]) -> None:
    clips=result["clip_rows"]; hops=result["hop_rows"]
    if not clips or sum(int(r["scored_hops"]) for r in clips)!=len(hops): _fail("summary hop count disagrees with rows")


def _parse_csv(path: Path, fields: tuple[str, ...], kind: str) -> list[dict[str, Any]]:
    nullable={"seed","checkpoint_sha256","previous_action","selected_hr_bpm","selected_invalid_reason","override_reason"}; numeric={"hop_time_s","prediction_post_belief_hr_bpm","gt_hr_bpm","abs_error_bpm","selected_hr_bpm","pre_belief_mean_hr","post_belief_mean_hr","mae_clip","mean_clip_mae"}; integer={"hop_idx","selected_roi_index","proposed_action","executed_action","previous_action","expected_hops","scored_hops","invalid_selected_count","proposed_switches","executed_switches","override_count","clip_count",*ACTION_COUNT_FIELDS}; boolean={"selected_valid","legal"}
    with path.open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        if tuple(reader.fieldnames or ())!=fields: _fail(f"{kind} CSV header is invalid")
        result=[]
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()): _fail(f"{kind} CSV row width is invalid")
            row=dict(raw)
            for field in fields:
                value=row[field]
                if field in nullable and value=="": row[field]=None; continue
                if field in boolean:
                    if value not in ("True","False"): _fail(f"{kind}.{field} is not a strict boolean")
                    row[field]=value=="True"; continue
                if field in integer:
                    if not value.isdigit() or (len(value)>1 and value[0]=="0"): _fail(f"{kind}.{field} is not a strict integer")
                    row[field]=int(value); continue
                if field in numeric:
                    try: row[field]=float(value)
                    except ValueError as exc: raise ContractValidationError(f"{kind}.{field} is not numeric") from exc
                    if not math.isfinite(row[field]): _fail(f"{kind}.{field} is not finite")
                elif value=="": _fail(f"{kind}.{field} must not be empty")
            result.append(row)
        return result


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))


def _marker_payload(path: Path, expected: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file(): _fail("publication marker must be a direct regular file")
    try:
        payload = read_json_object(path)
    except Exception as exc:
        raise ContractValidationError("publication marker is malformed") from exc
    if payload.get("schema") != _PUBLICATION_SCHEMA or payload.get("state") != expected:
        _fail("publication marker state is invalid")
    return payload


def _verify_markers(root: Path, run: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    names = {p.name for p in root.iterdir()}
    if names != set(SUBSTANTIVE_FILES) | {"STARTED.json", "COMPLETE.json"}: _fail("publication marker set is not complete")
    started = _marker_payload(root / "STARTED.json", "started")
    complete = _marker_payload(root / "COMPLETE.json", "complete")
    if set(started) != {"schema", "state", "run_id", "plan_id", "expected_files"} or started["expected_files"] != list(SUBSTANTIVE_FILES): _fail("STARTED marker fields are invalid")
    if not isinstance(started["run_id"], str) or (started["plan_id"] is not None and not isinstance(started["plan_id"], str)): _fail("STARTED marker identity is invalid")
    if set(complete) != {"schema", "state", "run_id", "plan_id", "outputs"} or complete["run_id"] != started["run_id"] or (started["plan_id"] is not None and complete["plan_id"] != started["plan_id"]): _fail("COMPLETE marker identity is invalid")
    if not isinstance(complete["plan_id"], str): _fail("COMPLETE marker plan identity is invalid")
    outputs = complete["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != set(SUBSTANTIVE_FILES): _fail("COMPLETE marker outputs are invalid")
    if run.get("status", "complete") != "complete" or run.get("run_id") != complete["run_id"] or run.get("plan_id") != complete["plan_id"] or manifest.get("status") != "complete": _fail("publication manifest binding is invalid")
    for name in SUBSTANTIVE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file(): _fail("publication artifacts must be direct regular files")
        entry = outputs[name]
        if not isinstance(entry, Mapping) or set(entry) != {"sha256", "bytes"} or _sha(entry.get("sha256"), "output hash") is None or type(entry.get("bytes")) is not int or entry["bytes"] < 0 or entry["sha256"] != sha256_file(path) or entry["bytes"] != path.stat().st_size: _fail("COMPLETE marker output mismatch")


def _verify_internal_rows(root: Path, expected_plan: FrozenEvaluationPlan | None = None, expected_provenance: RunProvenance | None = None) -> None:
    for name in SUBSTANTIVE_FILES:
        if (root / name).is_symlink() or not (root / name).is_file(): _fail("publication artifacts must be direct regular files")
    sidecar={}
    for line in (root/"artifacts.sha256").read_text(encoding="utf-8").splitlines():
        parts=line.split("  ")
        if len(parts)!=2 or parts[1] in sidecar: _fail("invalid publication sidecar")
        _sha(parts[0],"sidecar hash"); sidecar[parts[1]]=parts[0]
    if set(sidecar)!=set(SUBSTANTIVE_FILES[:-1]) or any(sidecar[n]!=sha256_file(root/n) for n in SUBSTANTIVE_FILES[:-1]): _fail("publication sidecar hash mismatch")
    hops=_parse_csv(root/"per_hop.csv",HOP_FIELDS,"per_hop"); clips=_parse_csv(root/"per_clip.csv",CLIP_FIELDS,"per_clip"); subjects=_parse_csv(root/"subject_summary.csv",SUBJECT_FIELDS,"subject_summary"); run=read_json_object(root/"run_summary.json"); manifest=read_json_object(root/"run_manifest.json")
    if set(run)!=set(RUN_FIELDS) or set(manifest)!={"status","plan_id","frozen_plan","run_provenance","outputs"} or manifest["status"]!="complete": _fail("run JSON fields are not exact")
    payload=dict(manifest["frozen_plan"])
    if set(payload)!=set(_PLAN_FIELDS)|{"plan_id"}: _fail("frozen plan fields are not exact")
    serialized_id=payload["plan_id"]; plan_payload={key: value for key, value in payload.items() if key != "plan_id"}; plan=FrozenEvaluationPlan(**plan_payload)
    if serialized_id!=plan.plan_id or manifest["plan_id"]!=plan.plan_id: _fail("plan ID mismatch")
    provenance=manifest["run_provenance"]
    if set(provenance)!={"run_id","code_snapshot_sha256","command","environment_versions","slurm_job_id","slurm_node"}: _fail("run provenance fields are not exact")
    p=RunProvenance(**provenance)
    if set(manifest["outputs"])!=set(SUBSTANTIVE_FILES[:4]): _fail("manifest output list is not exact")
    for name in SUBSTANTIVE_FILES[:4]:
        out=manifest["outputs"][name]
        if set(out)!={"sha256","bytes"} or type(out["bytes"]) is not int or out["bytes"]<0 or _sha(out["sha256"], "output hash") is None or out["sha256"]!=sha256_file(root/name) or out["bytes"]!=(root/name).stat().st_size: _fail("manifest output mismatch")
    if run["run_id"]!=p.run_id or run["plan_id"]!=plan.plan_id or len(clips)!=len(plan.clip_bindings) or tuple(r["clip_id"] for r in clips)!=tuple(b.clip_id for b in plan.clip_bindings): _fail("run/cohort identity mismatch")
    binding={b.clip_id:b for b in plan.clip_bindings}; by_clip={cid:[] for cid in binding}
    for row in hops:
        b=binding.get(row["clip_id"])
        expected_time = (round(8 * b.camera_fps) + row["hop_idx"] * round(b.camera_fps)) / b.camera_fps if b else None
        if b is None or row["run_id"]!=p.run_id or row["plan_id"]!=plan.plan_id or row["method_id"]!=plan.method_id or row["dataset_id"]!=b.dataset_id or row["subject_id"]!=b.subject_id or row["view"]!=b.view or row["condition"]!=b.condition or row["gt_source_sha256"]!=b.gt_sha256 or row["seed"] != plan.seed or row["checkpoint_sha256"] != plan.checkpoint_sha256 or row["signal_config_id"]!=plan.signal_config_id or row["control_config_id"]!=plan.control_config_id or row["observation_schema_id"]!=plan.observation_schema_id or row["gt_rule_id"]!=plan.gt_rule_id or row["proposed_action"]!=0 or row["executed_action"]!=0 or row["selected_roi_index"]!=0 or row["selected_roi_name"]!=ROI_NAMES[0].value or row["prediction_post_belief_hr_bpm"]!=row["post_belief_mean_hr"] or abs(row["hop_time_s"]-expected_time)>1e-12 or not row["frame_provenance_id"] or not (0.0 <= row["pre_belief_mean_hr"] <= 240.0) or not (0.0 <= row["post_belief_mean_hr"] <= 240.0) or abs(abs(row["prediction_post_belief_hr_bpm"]-row["gt_hr_bpm"])-row["abs_error_bpm"])>1e-12 or (row["legal"] and row["override_reason"] is not None) or (not row["legal"] and row["override_reason"] != "minimum_hold"): _fail("per_hop semantic reconstruction failed")
        if row["selected_valid"] and (row["selected_hr_bpm"] is None or row["selected_invalid_reason"] is not None): _fail("selected valid semantics failed")
        if not row["selected_valid"] and (row["selected_hr_bpm"] is not None or not row["selected_invalid_reason"]): _fail("selected invalid semantics failed")
        by_clip[row["clip_id"]].append(row)
    for b in plan.clip_bindings:
        rows=by_clip[b.clip_id]; n=dict(plan.expected_hops)[b.clip_id]
        if len(rows)!=n or tuple(r["hop_idx"] for r in rows)!=tuple(range(n)) or rows[0]["previous_action"] is not None: _fail("ordered hop coverage failed")
        if any(row["previous_action"] != rows[i-1]["executed_action"] for i, row in enumerate(rows) if i) or abs(rows[0]["pre_belief_mean_hr"] - 70.0) > 1e-12 or any(abs(row["pre_belief_mean_hr"] - rows[i-1]["post_belief_mean_hr"]) > 1e-12 for i, row in enumerate(rows) if i): _fail("state continuity failed")
        errors=[r["abs_error_bpm"] for r in rows]; counts=[sum(r["executed_action"]==i for r in rows) for i in range(12)]; first=rows[0]
        expected={"run_id":p.run_id,"plan_id":plan.plan_id,"method_id":plan.method_id,"dataset_id":b.dataset_id,"clip_id":b.clip_id,"subject_id":b.subject_id,"view":b.view,"condition":b.condition,"seed":first["seed"],"checkpoint_sha256":first["checkpoint_sha256"],"expected_hops":n,"scored_hops":n,"invalid_selected_count":sum(not r["selected_valid"] for r in rows),"mae_clip":sum(errors)/n,"proposed_switches":sum(r["previous_action"] is not None and r["proposed_action"]!=r["previous_action"] for r in rows),"executed_switches":sum(r["previous_action"] is not None and r["executed_action"]!=r["previous_action"] for r in rows),"override_count":sum(not r["legal"] for r in rows),**{ACTION_COUNT_FIELDS[i]:counts[i] for i in range(12)},"signal_config_id":plan.signal_config_id,"control_config_id":plan.control_config_id,"observation_schema_id":plan.observation_schema_id,"gt_rule_id":plan.gt_rule_id,"gt_source_sha256":b.gt_sha256}
        actual=next(r for r in clips if r["clip_id"]==b.clip_id)
        if any(expected[f]!=actual[f] and not (isinstance(expected[f],float) and abs(expected[f]-actual[f])<=1e-12) for f in CLIP_FIELDS): _fail("per_clip reconstruction failed")
    expected_subjects=[{"subject_id": subject, "clip_count": sum(row["subject_id"] == subject for row in clips), "mean_clip_mae": sum(row["mae_clip"] for row in clips if row["subject_id"] == subject) / sum(row["subject_id"] == subject for row in clips)} for subject in sorted({row["subject_id"] for row in clips})]
    if len(subjects) != len(expected_subjects) or any(any(subject[field] != expected[field] for field in SUBJECT_FIELDS) for subject, expected in zip(subjects, expected_subjects)): _fail("subject summary reconstruction failed")
    expected_run={"run_id":p.run_id,"plan_id":plan.plan_id,"method_id":plan.method_id,"dataset_id":plan.dataset_id,"split":plan.split,"clip_count":len(clips),"subject_count":len(subjects),"hop_count":len(hops),"equal_clip_mean_mae":sum(r["mae_clip"] for r in clips)/len(clips),"invalid_selected_count":sum(r["invalid_selected_count"] for r in clips),"override_count":sum(r["override_count"] for r in clips),"metric_id":plan.metric_id,"aggregation_id":plan.aggregation_id,"uncertainty_id":plan.uncertainty_id}
    if any(expected_run[f]!=run[f] and not (isinstance(expected_run[f],float) and abs(expected_run[f]-run[f])<=1e-12) for f in RUN_FIELDS): _fail("run summary reconstruction failed")


def _verify_substantive_tree(root: Path) -> None:
    extras = {p.name for p in root.iterdir()} - set(SUBSTANTIVE_FILES)
    if root.is_symlink() or not root.is_dir() or not extras.issubset({"STARTED.json", "COMPLETE.json"}) or "STARTED.json" not in extras: _fail("publication must contain substantive outputs and STARTED")
    for name in SUBSTANTIVE_FILES:
        if (root / name).is_symlink() or not (root / name).is_file(): _fail("publication artifacts must be direct regular files")
    sidecar={}
    for line in (root/"artifacts.sha256").read_text(encoding="utf-8").splitlines():
        parts=line.split("  ")
        if len(parts)!=2 or parts[1] in sidecar: _fail("invalid publication sidecar")
        _sha(parts[0],"sidecar hash"); sidecar[parts[1]]=parts[0]
    if set(sidecar)!=set(SUBSTANTIVE_FILES[:-1]) or any(sidecar[n]!=sha256_file(root/n) for n in SUBSTANTIVE_FILES[:-1]): _fail("publication sidecar hash mismatch")


def verify_publication_structure(destination: str | os.PathLike[str]) -> None:
    root=Path(destination)
    if root.is_symlink() or not root.is_dir(): _fail("publication must be a real directory")
    _verify_substantive_tree_without_markers = _verify_substantive_tree
    try:
        _verify_substantive_tree_without_markers(root)
        run=read_json_object(root/"run_summary.json")
        manifest=read_json_object(root/"run_manifest.json")
        _verify_markers(root, run, manifest)
        _verify_internal_rows(root)
    except FileNotFoundError as exc:
        raise ContractValidationError("publication is incomplete") from exc


def _write_substantive_files(result: Mapping[str, Any], root: Path) -> None:
    _check_result(result)
    _write_csv(root/"per_hop.csv", HOP_FIELDS, result["hop_rows"])
    _write_csv(root/"per_clip.csv", CLIP_FIELDS, result["clip_rows"])
    _write_csv(root/"subject_summary.csv", SUBJECT_FIELDS, result["subject_rows"])
    _write_json_exclusive(root/"run_summary.json", result["run_summary"])
    prov=result["provenance"]
    manifest={"status":"complete","plan_id":result["plan"].plan_id,"frozen_plan":result["plan"].to_dict(),"run_provenance":{"run_id":prov.run_id,"code_snapshot_sha256":prov.code_snapshot_sha256,"command":prov.command,"environment_versions":prov.environment_versions,"slurm_job_id":prov.slurm_job_id,"slurm_node":prov.slurm_node},"outputs":{n:{"sha256":sha256_file(root/n),"bytes":(root/n).stat().st_size} for n in SUBSTANTIVE_FILES[:4]}}
    _write_json_exclusive(root/"run_manifest.json", manifest)
    _write_csv_sidecar(root)


def _write_csv_sidecar(root: Path) -> None:
    _write_text_exclusive(root/"artifacts.sha256", "".join(f"{sha256_file(root/n)}  {n}\n" for n in SUBSTANTIVE_FILES[:-1]))


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle: handle.write(text)


def _prepare_destination(destination: str | os.PathLike[str]) -> Path:
    root=Path(destination)
    if root.exists() or root.is_symlink(): _fail("destination must not already exist")
    if root.parent.is_symlink() or not root.parent.is_dir(): _fail("destination parent must be an existing real directory")
    try: root.mkdir()
    except FileExistsError: _fail("destination appeared during publication")
    return root


def _publish_started(root: Path, provenance: RunProvenance, plan: FrozenEvaluationPlan | None) -> None:
    _write_json_exclusive(root/"STARTED.json", {"schema":_PUBLICATION_SCHEMA,"state":"started","run_id":provenance.run_id,"plan_id":None if plan is None else plan.plan_id,"expected_files":list(SUBSTANTIVE_FILES)})


def _publish_failed(root: Path, provenance: RunProvenance, plan: FrozenEvaluationPlan | None, error_code: str, stage: str) -> None:
    _write_json_exclusive(root/"FAILED.json", {"schema":_PUBLICATION_SCHEMA,"state":"failed","run_id":provenance.run_id,"plan_id":None if plan is None else plan.plan_id,"error_code":error_code,"stage":stage,"message":"Gate 5 evaluation or publication failed"})


def _publish_complete(root: Path, provenance: RunProvenance, plan: FrozenEvaluationPlan) -> None:
    outputs={n:{"sha256":sha256_file(root/n),"bytes":(root/n).stat().st_size} for n in SUBSTANTIVE_FILES}
    _write_json_exclusive(root/"COMPLETE.json", {"schema":_PUBLICATION_SCHEMA,"state":"complete","run_id":provenance.run_id,"plan_id":plan.plan_id,"outputs":outputs})


def _publish_result(result: Mapping[str, Any], destination: str | os.PathLike[str], provenance: RunProvenance | None = None) -> Path:
    if provenance is not None and result.get("provenance") != provenance: _fail("provenance mismatch")
    root=_prepare_destination(destination); actual=result["provenance"]; plan=result["plan"]
    _publish_started(root, actual, plan)
    try:
        _write_substantive_files(result, root); _verify_substantive_tree(root)
        _verify_internal_rows(root, plan, actual)
        _publish_complete(root, actual, plan)
        verify_publication_structure(root)
        return root
    except Exception as original:
        try: _publish_failed(root, actual, plan, "EVALUATION_FAILED" if not (root/"per_hop.csv").exists() else "PUBLICATION_FAILED", "evaluate" if not (root/"per_hop.csv").exists() else "publish")
        except Exception: pass
        raise original


def publish_evaluation(result: Mapping[str, Any], destination: str | os.PathLike[str], provenance: RunProvenance | None = None) -> Path:
    return _publish_result(result, destination, provenance)


def evaluate_and_publish(manifest_tree: str | os.PathLike[str], state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str], provenance: RunProvenance, destination: str | os.PathLike[str], plan: FrozenEvaluationPlan | None = None) -> Path:
    root=_prepare_destination(destination); _publish_started(root, provenance, plan)
    try:
        result=evaluate(manifest_tree, state_root, gt_root, provenance, plan)
        _write_substantive_files(result, root); _verify_substantive_tree(root); _verify_internal_rows(root, result["plan"], provenance); _publish_complete(root, provenance, result["plan"]); verify_publication_structure(root); return root
    except Exception as original:
        try: _publish_failed(root, provenance, plan, "EVALUATION_FAILED" if not (root/"per_hop.csv").exists() else "PUBLICATION_FAILED", "evaluate" if not (root/"per_hop.csv").exists() else "publish")
        except Exception: pass
        raise original


def _provenance_from_publication(root: Path) -> RunProvenance:
    manifest = read_json_object(root / "run_manifest.json")
    payload = manifest.get("run_provenance")
    if not isinstance(payload, Mapping): _fail("publication provenance is invalid")
    return RunProvenance(**dict(payload))


def verify_publication_against_sources(destination: str | os.PathLike[str], manifest_tree: str | os.PathLike[str], state_root: str | os.PathLike[str], gt_root: str | os.PathLike[str]) -> None:
    """Verify structure, then recompute and byte-compare every published artifact."""
    root = Path(destination)
    verify_publication_structure(root)
    manifest = read_json_object(root / "run_manifest.json")
    plan_payload = manifest.get("frozen_plan")
    if not isinstance(plan_payload, Mapping): _fail("publication frozen plan is invalid")
    serialized_id = plan_payload.get("plan_id")
    plan = FrozenEvaluationPlan(**{key: value for key, value in plan_payload.items() if key != "plan_id"})
    if serialized_id != plan.plan_id or manifest.get("plan_id") != plan.plan_id: _fail("publication plan ID mismatch")
    result = evaluate(manifest_tree, state_root, gt_root, _provenance_from_publication(root), plan)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.authority-", dir=root.parent) as temporary:
        recomputed = Path(temporary) / "recomputed"
        publish_evaluation(result, recomputed)
        names = ("per_hop.csv", "per_clip.csv", "subject_summary.csv", "run_summary.json", "run_manifest.json", "artifacts.sha256")
        for name in names:
            if (root / name).read_bytes() != (recomputed / name).read_bytes(): _fail(f"authoritative publication mismatch: {name}")


verify_publication = verify_publication_structure

__all__=["CLIP_FIELDS","HOP_FIELDS","SUBJECT_FIELDS","ClipBinding","FrozenEvaluationPlan","RunProvenance","build_train_full_face_plan","evaluate","evaluate_and_publish","publish_evaluation","verify_publication_structure","verify_publication_against_sources","verify_publication"]
