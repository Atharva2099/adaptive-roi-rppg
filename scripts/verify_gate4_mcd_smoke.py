"""Run the Gate 4 state-only MCD smoke; intended for a Slurm compute node."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import shlex
import sys
from pathlib import Path

import numpy
import scipy

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import CONTROL_CONFIG_ID, CONTROL_CONFIG_PAYLOAD, FIELD_NAMES, OBSERVATION_SCHEMA_ID, belief_step, control_step, initial_control_state
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.signal import POS_CONFIG_ID, build_pos_measurements

CSV_FIELDS = ("clip_id", "arm", "passes", "hops", "overrides", "invalid_selected", "transition_sha256")
MAX_EXTRA = 30

def accepted_selected_invalid(rows):
    """Return whether accepted control runs actually executed an invalid ROI."""
    return sum(int(row["invalid_selected"]) for row in rows) > 0

def _hash_transitions(transitions):
    digest = hashlib.sha256()
    for item in transitions:
        decision = item.action_decision
        payload = {"identity": [item.dataset_id, item.clip_id, item.hop_idx], "frame_provenance_id": item.frame_provenance_id, "observation_schema_id": item.observation.schema_id, "observation_config_id": item.observation.config_id, "field_names": list(item.observation.field_names), "signal_config_id": item.signal_config_id, "control_config_id": item.control_config_id, "observation": list(item.observation.values), "decision": [decision.proposed_action, decision.executed_action, decision.previous_action, decision.pre_hold_count, decision.post_hold_count, decision.legal, decision.override_reason], "selected": item.selected_measurement.to_dict(), "pre": [item.pre_belief.mean_hr, item.pre_belief.velocity, item.pre_belief.covariance, item.pre_belief.hops_since_confident], "post": [item.post_belief.mean_hr, item.post_belief.velocity, item.post_belief.covariance, item.post_belief.hops_since_confident]}
        digest.update(canonical_json_bytes(payload) + b"\n")
    return digest.hexdigest()

def _run(frames, clip, arm):
    measurements = build_pos_measurements(frames)
    expected_hops = max(0, (len(frames) - round(8 * clip.camera_fps)) // round(clip.camera_fps) + 1)
    if len(measurements) != expected_hops: raise ContractValidationError(f"{clip.clip_id}: hop count mismatch")
    state = initial_control_state(clip.dataset_id, clip.clip_id); transitions = []; invalid_selected = 0; overrides = 0
    for hop, measurement_frame in enumerate(measurements):
        proposed = 0 if arm == "full_face" else hop % 12
        state, transition = control_step(measurement_frame, state, proposed)
        if transition.hop_idx != hop or transition.observation.array().shape != (101,) or transition.observation.array().dtype != numpy.float32 or not numpy.isfinite(transition.observation.array()).all(): raise ContractValidationError(f"{clip.clip_id}: observation/progression assertion failed")
        if arm == "full_face" and (transition.action_decision.executed_action != 0 or not transition.action_decision.legal): raise ContractValidationError(f"{clip.clip_id}: full-face action was overridden")
        if not transition.action_decision.legal: overrides += 1
        if not transition.selected_measurement.valid:
            invalid_selected += 1
            if transition.post_belief != belief_step(transition.pre_belief, None, None): raise ContractValidationError(f"{clip.clip_id}: invalid selection updated belief")
        transitions.append(transition)
    if state.next_hop_idx != expected_hops: raise ContractValidationError(f"{clip.clip_id}: state hop progression mismatch")
    if arm == "deterministic" and len(transitions) >= 3 and (overrides == 0 or not any(t.action_decision.legal and t.action_decision.proposed_action != t.action_decision.previous_action for t in transitions[1:])): raise ContractValidationError(f"{clip.clip_id}: deterministic action legality trace incomplete")
    return measurements, transitions, {"clip_id": clip.clip_id, "arm": arm, "passes": 2, "hops": len(transitions), "overrides": overrides, "invalid_selected": invalid_selected, "transition_sha256": _hash_transitions(transitions)}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--code-snapshot-sha256", required=True)
    args = parser.parse_args()
    if len(args.code_snapshot_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.code_snapshot_sha256): raise SystemExit("--code-snapshot-sha256 must be 64 lowercase hexadecimal characters")
    job, node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
    if not job or not node: raise SystemExit("SLURM_JOB_ID and SLURMD_NODENAME are required")
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit("output-dir must not exist or must be empty")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_mcd_manifest_tree(args.manifest_tree)
    train = sorted((c for c in bundle.clip_manifests if c.subject_id in bundle.split_manifest.train_subject_ids), key=lambda c: (c.subject_id, c.clip_id))
    if not train: raise SystemExit("no train clips")
    subject = train[0].subject_id; selected = sorted((c for c in train if c.subject_id == subject), key=lambda c: c.clip_id)
    if len(selected) != 6 or {(c.camera_id, c.condition) for c in selected} != {(a, b) for a in ("FullHDwebcam", "USBVideo", "IriunWebcam") for b in ("before", "after")}: raise SystemExit("first train subject does not have exact six-clip inventory")
    def probe(clip):
        frames = read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "train")
        return [_run(frames, clip, arm)[2] for arm in ("full_face", "deterministic")]
    exploratory = [row for clip in selected for row in probe(clip)]
    if not accepted_selected_invalid(exploratory):
        extra = [c for c in train if c not in selected][:MAX_EXTRA]
        invalid = next((c for c in extra if accepted_selected_invalid(probe(c))), None)
        if invalid is None: raise SystemExit(f"no invalid example found within {MAX_EXTRA} additional sorted train clips")
        selected.append(invalid)
    rows = []
    for clip in selected:
        frames = read_mcd_canonical_frames(bundle, args.state_root, clip.clip_id, "train")
        for arm in ("full_face", "deterministic"):
            first = _run(frames, clip, arm); second = _run(frames, clip, arm)
            if first[2]["transition_sha256"] != second[2]["transition_sha256"]: raise ContractValidationError(f"{clip.clip_id}/{arm}: repeated hashes differ")
            rows.append(first[2])
    if not accepted_selected_invalid(rows): raise ContractValidationError("accepted cohort has no executed invalid ROI measurement")
    csv_path = output / "gate4_mcd_smoke.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(rows)
    manifest_hashes = {name: sha256_file(Path(args.manifest_tree) / name) for name in ("dataset_manifest.json", "split_manifest.json", "source_inventory.json")}
    summary = {"dataset_id": bundle.dataset_manifest.dataset_id, "split_id": bundle.split_manifest.split_id, "primary_subject_id": subject, "accepted_clips": [{"clip_id": c.clip_id, "subject_id": c.subject_id, "state_sha256": c.state_sha256} for c in selected], "clip_count": len(selected), "arms": ["full_face", "deterministic"], "control_config_id": CONTROL_CONFIG_ID, "control_config_payload_sha256": hashlib.sha256(canonical_json_bytes(CONTROL_CONFIG_PAYLOAD)).hexdigest(), "observation_schema_id": OBSERVATION_SCHEMA_ID, "field_names_sha256": hashlib.sha256(canonical_json_bytes(FIELD_NAMES)).hexdigest(), "signal_config_id": POS_CONFIG_ID, "manifest_hashes": manifest_hashes, "state_root": str(Path(args.state_root)), "code_snapshot_sha256": args.code_snapshot_sha256, "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "slurm_job_id": job, "slurm_nodename": node, "command": shlex.join(sys.argv), "rows": rows, "assertions": "passed"}
    json_path = output / "gate4_mcd_smoke.json"; json_path.write_bytes(canonical_json_bytes(summary))
    sidecar = output / "artifacts.sha256"
    sidecar.write_text(f"{hashlib.sha256(csv_path.read_bytes()).hexdigest()}  {csv_path.name}\n{hashlib.sha256(json_path.read_bytes()).hexdigest()}  {json_path.name}\n", encoding="utf-8")
    return 0

if __name__ == "__main__": raise SystemExit(main())
