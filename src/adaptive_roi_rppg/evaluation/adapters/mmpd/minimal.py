"""Small, local MMPD clip evaluator.

This module intentionally has no provenance, lifecycle, or publication framework.
It is a one-clip runner for parallel Slurm tasks and keeps labels out of replay.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import control_step, initial_control_state
from adaptive_roi_rppg.control import build_observation
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity
from adaptive_roi_rppg.signal import build_pos_measurements
from .extraction import extract_mmpd_canonical_frames, trim_mmpd_terminal_zero_suffix
from .rules import MMPD_FPS, build_mmpd_labels

EXPECTED_LEARNED_METHOD_IDS = (
    "dagger_seed0", "dagger_seed1", "dagger_seed2",
    "standard_ppo_seed0", "standard_ppo_seed1", "standard_ppo_seed2",
    "advantage_ppo_seed0", "advantage_ppo_seed1", "advantage_ppo_seed2",
)
EXPECTED_METHOD_IDS = ("full_face_pos",) + EXPECTED_LEARNED_METHOD_IDS


@dataclass(frozen=True, slots=True)
class MinimalSource:
    video: np.ndarray
    gt_ppg: np.ndarray
    trimmed_tail_frames: int
    original_frame_count: int
    processed_frame_count: int


def load_minimal_source(path: str | Path) -> MinimalSource:
    """Load and minimally repair one MMPD MAT source.

    Only a contiguous terminal run of exactly-zero RGB frames is removed.
    In particular, an interior zero frame is an input error, not a missing frame.
    """
    try:
        values = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    except Exception as exc:  # scipy exposes several MAT-specific exceptions
        raise ContractValidationError("minimal MMPD source: MAT cannot be loaded") from exc
    if "video" not in values or "GT_ppg" not in values:
        raise ContractValidationError("minimal MMPD source: video and GT_ppg are required")
    video = np.asarray(values["video"])
    gt = np.asarray(values["GT_ppg"], dtype=np.float64).reshape(-1)
    trimmed, trimmed_gt, original, tail = trim_mmpd_terminal_zero_suffix(video, gt, context="minimal MMPD source")
    return MinimalSource(trimmed, trimmed_gt, tail, original, int(trimmed.shape[0]))


class FullFacePolicy:
    """Fixed action-zero policy used as the full-face arm."""

    identity = CheckpointIdentity("full_face_pos", "fixed_full_face", None, None)

    def initial_state(self) -> None:
        return None

    def predict(self, observation: np.ndarray, recurrent_state: None, *, episode_start: bool):
        if observation.shape != (1, 101) or observation.dtype != np.float32:
            raise ContractValidationError("minimal MMPD replay: invalid observation")
        return 0, None


def validate_checkpoint_manifest(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Require exactly one record for each of the nine learned arms."""
    methods = [record.get("method_id") for record in records]
    if len(records) != len(EXPECTED_LEARNED_METHOD_IDS) or set(methods) != set(EXPECTED_LEARNED_METHOD_IDS):
        raise ContractValidationError("minimal MMPD evaluator: checkpoint manifest must contain exactly nine learned arms")
    if len(methods) != len(set(methods)):
        raise ContractValidationError("minimal MMPD evaluator: checkpoint manifest contains duplicate method IDs")
    return tuple(records)


def _validate_arm_set(methods: Sequence[str]) -> None:
    if len(methods) != 10 or len(set(methods)) != 10 or set(methods) != set(EXPECTED_METHOD_IDS):
        raise ContractValidationError("minimal MMPD evaluator: replay must contain exactly ten expected arms")


def load_minimal_recurrent_policy(path: str | Path, *, method_id: str, family: str = "ppo", seed: Any = "") -> Any:
    """Load an SB3 recurrent policy without file hashes or identity checks."""
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(path), device="cpu")
    except ImportError as exc:
        raise ContractValidationError("minimal MMPD replay: sb3-contrib is required for recurrent arms") from exc
    except Exception as exc:
        raise ContractValidationError("minimal MMPD replay: checkpoint cannot be loaded") from exc

    class Policy:
        identity = CheckpointIdentity(method_id, family, seed, None)

        def initial_state(self):
            return None

        def predict(self, observation, recurrent_state, *, episode_start):
            action, state = model.predict(observation, state=recurrent_state,
                                          episode_start=np.asarray([episode_start]), deterministic=True)
            return int(np.asarray(action).reshape(-1)[0]), state

    return Policy()


def replay_shared_measurements(measurements: Sequence[Any], policies: Mapping[str, Any], *, clip_id: str,
                               trimmed_tail_frames: int = 0) -> tuple[dict[str, Any], ...]:
    """Replay every arm over the same already-extracted measurement tuple."""
    _validate_arm_set(tuple(policies))
    rows: list[dict[str, Any]] = []
    for method, policy in policies.items():
        state = initial_control_state("mmpd", clip_id)
        recurrent = policy.initial_state()
        episode_start = True
        for frame in measurements:
            observation = build_observation(frame, state).array()
            proposed, recurrent = policy.predict(observation.reshape(1, 101), recurrent, episode_start=episode_start)
            state, transition = control_step(frame, state, int(proposed))
            selected = transition.selected_measurement
            decision = transition.action_decision
            rows.append({
                "clip_id": clip_id, "method_id": method, "hop_idx": frame.hop_idx,
                "selected_valid": selected.valid,
                "selected_invalid_reason": selected.invalid_reason or "",
                "selected_hr_bpm": selected.hr_bpm, "selected_confidence": selected.confidence,
                "selected_ppr": selected.peak_power_ratio, "pre_belief_hr_bpm": transition.pre_belief.mean_hr,
                "post_belief_hr_bpm": transition.post_belief.mean_hr,
                "proposed_action": decision.proposed_action, "executed_action": decision.executed_action,
                "trimmed_tail_frames": trimmed_tail_frames, "gt_hr_bpm": None, "abs_error_bpm": None,
            })
            episode_start = False
    return tuple(rows)


def evaluate_minimal_clip(source_path: str | Path, *, clip_id: str, model_asset_path: str,
                          policies: Mapping[str, Any] | None = None,
                          extractor: Callable[..., Sequence[Any]] | None = None) -> tuple[dict[str, Any], ...]:
    source = load_minimal_source(source_path)
    if extractor is None:
        extractor = extract_mmpd_canonical_frames
    frames = extractor(source.video, clip_id=clip_id, source_provenance_id=str(source_path),
                       model_asset_path=model_asset_path, fps=MMPD_FPS)
    measurements = build_pos_measurements(frames)
    if not measurements:
        raise ContractValidationError("minimal MMPD evaluator: no complete extraction hops")
    labels = build_mmpd_labels(source.gt_ppg, clip_id=clip_id)[:len(measurements)]
    if len(labels) != len(measurements) or any(not label.valid for label in labels):
        raise ContractValidationError("minimal MMPD evaluator: emitted hops lack valid GT labels")
    arms = dict(policies or {})
    _validate_arm_set(tuple(arms))
    rows = list(replay_shared_measurements(measurements, arms, clip_id=clip_id,
                                           trimmed_tail_frames=source.trimmed_tail_frames))
    by_hop = {label.hop_idx: label for label in labels}
    for row in rows:
        label = by_hop[row["hop_idx"]]
        row["gt_hr_bpm"] = label.gt_hr_bpm
        row["abs_error_bpm"] = abs(float(row["post_belief_hr_bpm"]) - float(label.gt_hr_bpm))
    _assert_clip_coverage(rows, len(measurements))
    return tuple(rows)


CSV_FIELDS = ("clip_id", "method_id", "hop_idx", "selected_valid", "selected_invalid_reason",
              "selected_hr_bpm", "selected_confidence", "selected_ppr", "pre_belief_hr_bpm",
              "post_belief_hr_bpm", "proposed_action", "executed_action", "gt_hr_bpm",
              "abs_error_bpm", "trimmed_tail_frames")


def _assert_clip_coverage(rows: Sequence[Mapping[str, Any]], available_hops: int) -> None:
    pairs = {(row.get("method_id"), row.get("hop_idx")) for row in rows}
    if len(rows) != available_hops * 10 or len(pairs) != available_hops * 10:
        raise ContractValidationError("minimal MMPD evaluator: clip CSV does not cover exactly available_hops * 10 pairs")
    if {row.get("method_id") for row in rows} != set(EXPECTED_METHOD_IDS):
        raise ContractValidationError("minimal MMPD evaluator: clip CSV has an unexpected arm set")
    if {row.get("hop_idx") for row in rows} != set(range(available_hops)):
        raise ContractValidationError("minimal MMPD evaluator: clip CSV has incomplete hop coverage")


def write_clip_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    hops = {row.get("hop_idx") for row in rows}
    if not hops or any(not isinstance(hop, int) or hop < 0 for hop in hops):
        raise ContractValidationError("minimal MMPD evaluator: clip CSV has no valid hops")
    _assert_clip_coverage(rows, len(hops))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["MinimalSource", "EXPECTED_LEARNED_METHOD_IDS", "EXPECTED_METHOD_IDS", "load_minimal_source",
           "FullFacePolicy", "validate_checkpoint_manifest", "load_minimal_recurrent_policy",
           "replay_shared_measurements", "evaluate_minimal_clip", "write_clip_csv", "CSV_FIELDS"]
