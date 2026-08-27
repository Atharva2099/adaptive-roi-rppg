import csv
import hashlib
import importlib.util
import json
import math
import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.long_term_regret import (
    CLIP_SELECTION_FIELDS, SELECTION_SCHEMA, build_anchor_regret, csv_bytes, read_frozen_selection, replay_one_time_interventions, score_branches,
    select_worst_clips_from_replay, subject_block_bootstrap,
)
from adaptive_roi_rppg.evaluation.repeated_correction_beam import (
    eligible_anchors, factual_snapshots, rank1_action, repeated_correction_beam, run_widths,
)
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity, rollout_frozen_policy
from adaptive_roi_rppg.evaluation.model_publication import marker_payload, validate_directory, validate_precomplete_directory, write_marker
from adaptive_roi_rppg.labels.mcd import GT_RULE_ID
from adaptive_roi_rppg.signal import POS_CONFIG_ID


def frame(hop, clip="clip", invalid=False):
    measurements = tuple(ROIMeasurement(i, ROI_NAMES[i], None if invalid else 70.0 + i, 0.0 if invalid else 0.8, 0.5, 0.6, not invalid, "synthetic_invalid" if invalid else None, (), None, 0, 10, POS_CONFIG_ID) for i in range(12))
    return MeasurementFrame("mcd", clip, hop, 8.0 + hop, measurements, POS_CONFIG_ID, not invalid, "synthetic_invalid" if invalid else None, f"frame-{hop}")


class Policy:
    identity = CheckpointIdentity("advantage_ppo_seed0", "advantage_ppo", 0, "a" * 64)
    def initial_state(self): return (np.asarray([[0]], dtype=np.int64),)
    def predict(self, observation, recurrent_state, *, episode_start):
        count = int(recurrent_state[0][0, 0])
        # New arrays make accidental recurrence sharing visible to this test.
        return count % 12, (np.asarray([[count + 1]], dtype=np.int64),)


class LongTermRegretTests(unittest.TestCase):
    def test_repeated_beam_runner_resolves_task_before_checkpoint_boundary(self):
        """The production runner parses task arguments without stale pair access."""
        path = Path(__file__).resolve().parents[1] / "scripts/verify_mcd_repeated_correction_beam.py"
        spec = importlib.util.spec_from_file_location("verify_mcd_repeated_correction_beam_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        with patch.dict(sys.modules, {"verify_mcd_long_term_regret": importlib.import_module("scripts.verify_mcd_long_term_regret")}):
            spec.loader.exec_module(module)
        task = {"clip_id": "clip-000", "subject_id": "subject-000", "seed": 0, "_source_sha256": "a" * 64, "_plan_sha256": "b" * 64}
        selected = ({"clip_id": "clip-000", "subject_id": "subject-000"},)
        class ReachedCheckpointBoundary(Exception): pass
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "subsets").mkdir()
            (root / "subsets" / "index.json").write_text(json.dumps({"checkpoint_sha256": {"0": "a" * 64, "1": "b" * 64, "2": "c" * 64}}))
            argv = ["beam.py", "--output-dir", str(root / "out"), "--selection-dir", str(root / "selection"), "--source-subsets-dir", str(root / "subsets"), "--manifest-tree", str(root / "manifests"), "--state-root", str(root / "state"), "--gt-root", str(root / "gt"), "--plan", str(root / "plan"), "--checkpoint-root", str(root / "checkpoints")]
            with patch.object(module, "_resolve_task", return_value=(task, [], ("clip-000", 0), selected)), patch.object(module, "_runtime_checkpoints", side_effect=ReachedCheckpointBoundary), patch.object(module, "load_frozen_model_plan", return_value=object()), patch.object(module, "_load_config", return_value={}), patch.object(module, "_config", return_value={}), patch.object(module, "_fresh"):
                with patch.object(sys, "argv", argv), self.assertRaises(ReachedCheckpointBoundary): module.main()

    def test_boundary_and_locked_eligibility(self):
        anchors, branches = replay_one_time_interventions([frame(i) for i in range(15)], Policy(), clip_id="clip")
        self.assertTrue(anchors[0]["eligible"])
        self.assertEqual(sorted({row.offset for row in branches if row.anchor_hop_idx == 0}), list(range(15)))
        self.assertFalse(anchors[1]["eligible"])
        self.assertEqual(anchors[1]["ineligible_reason"], "insufficient_future_hops")
        short, _ = replay_one_time_interventions([frame(i) for i in range(14)], Policy(), clip_id="clip")
        self.assertFalse(short[0]["eligible"])
        # With 16 hops, anchor one has enough remaining hops but min-hold locks switches.
        locked, _ = replay_one_time_interventions([frame(i) for i in range(16)], Policy(), clip_id="clip")
        self.assertFalse(locked[1]["eligible"])
        self.assertEqual(locked[1]["ineligible_reason"], "locked_or_nonfree_action_set")

    def test_factual_parity_same_action_zero_and_recurrence_timing(self):
        frames = [frame(i) for i in range(16)]
        anchors, branches = replay_one_time_interventions(frames, Policy(), clip_id="clip")
        factual_action = anchors[0]["factual_proposed_action"]
        trace = [row for row in branches if row.anchor_hop_idx == 0 and row.branch_action == factual_action]
        uninterrupted = rollout_frozen_policy(frames, Policy(), clip_id="clip")
        self.assertEqual([row.post_belief_hr_bpm for row in trace], [row.post_belief_hr_bpm for row in uninterrupted[:15]])
        self.assertEqual(trace[0].post_predict_recurrent_sha256, anchors[0]["post_predict_recurrent_sha256"])
        self.assertNotEqual(trace[1].pre_predict_recurrent_sha256, anchors[0]["pre_predict_recurrent_sha256"])
        labels = [LabelFrame("mcd", "clip", i, 8.0 + i, 70.0, GT_RULE_ID, True, None) for i in range(16)]
        scores = [dict(row) for row in score_branches(branches, labels)]
        for row in scores: row["is_factual"] = row["branch_action"] == anchors[row["anchor_hop_idx"]]["factual_proposed_action"]
        regrets = build_anchor_regret(scores)
        for regret in regrets:
            if regret["anchor_hop_idx"] == 0:
                # Same action is a branch in the action set and cannot differ from itself.
                self.assertGreaterEqual(regret["factual_regret_bpm"], 0.0)

    def test_branch_isolation_gt_independence_and_predict_only_invalid(self):
        frames = [frame(i, invalid=True) for i in range(15)]
        first = replay_one_time_interventions(frames, Policy(), clip_id="clip")
        second = replay_one_time_interventions(frames, Policy(), clip_id="clip")
        self.assertEqual(first, second)
        self.assertTrue(all(math.isfinite(row.post_belief_hr_bpm) for row in first[1]))
        labels_a = [LabelFrame("mcd", "clip", i, 8.0 + i, 70.0, GT_RULE_ID, True, None) for i in range(15)]
        labels_b = [LabelFrame("mcd", "clip", i, 8.0 + i, 170.0, GT_RULE_ID, True, None) for i in range(15)]
        scores_a = score_branches(first[1], labels_a); scores_b = score_branches(first[1], labels_b)
        self.assertNotEqual(scores_a, scores_b)
        self.assertEqual(first[0], second[0])

    def test_selection_exact_ties_and_mixed_identity_reject(self):
        fields = ["dataset_id", "method_id", "family", "seed", "checkpoint_sha256", "clip_id", "subject_id", "hop_idx", "abs_error_bpm"]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_hop.csv"
            rows = []
            for seed in range(3):
                for index in range(533):
                    rows.append({"dataset_id": "mcd", "method_id": f"advantage_ppo_seed{seed}", "family": "advantage_ppo", "seed": seed, "checkpoint_sha256": chr(97 + seed) * 64, "clip_id": f"clip-{index:03d}", "subject_id": f"subject-{index % 89:03d}", "hop_idx": 0, "abs_error_bpm": 10.0 if index < 55 else 1.0})
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            selection = select_worst_clips_from_replay(path)
            self.assertEqual(len(selection), 54)
            self.assertEqual([row["clip_id"] for row in selection], [f"clip-{index:03d}" for index in range(54)])
            with self.assertRaises(ContractValidationError):
                select_worst_clips_from_replay(path, expected_checkpoint_sha256={0: "0" * 64, 1: "b" * 64, 2: "c" * 64})
            rows[1]["subject_id"] = "wrong"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ContractValidationError): select_worst_clips_from_replay(path)

    def test_aggregation_is_clip_then_subject_and_reversal_pattern(self):
        rows = [{"subject_id": "s1", "clip_id": "a", "factual_regret_bpm": 0.0}, {"subject_id": "s1", "clip_id": "a", "factual_regret_bpm": 2.0}, {"subject_id": "s1", "clip_id": "b", "factual_regret_bpm": 8.0}, {"subject_id": "s2", "clip_id": "c", "factual_regret_bpm": 10.0}]
        result = subject_block_bootstrap(rows, replicates=25, seed=2)
        # s1 = mean((0+2)/2, 8) = 4.5; s2 = 10, then equal subject mean = 7.25.
        self.assertAlmostEqual(result["point_estimate_bpm"], 7.25)
        scores = []
        for horizon, factual, alternate in ((1, 5.0, 1.0), (5, 1.0, 4.0), (15, 1.0, 4.0)):
            scores.extend(({"anchor_hop_idx": 0, "horizon": horizon, "branch_action": action, "mean_abs_error_bpm": factual if action == 0 else alternate, "is_factual": action == 0} for action in range(12)))
        regrets = build_anchor_regret(scores)
        self.assertGreater(regrets[0]["factual_regret_bpm"], 0.0)
        self.assertEqual(regrets[1]["factual_regret_bpm"], 0.0)
        self.assertEqual(regrets[2]["factual_regret_bpm"], 0.0)

    def test_frozen_selection_rejects_tamper_and_duplicate_rows(self):
        rows = tuple({"clip_id": f"clip-{index:03d}", "subject_id": f"subject-{index % 89:03d}", "seed0_mae_bpm": float(index), "seed1_mae_bpm": float(index), "seed2_mae_bpm": float(index), "three_seed_mean_mae_bpm": float(index), "selection_rank": index + 1} for index in range(54))
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "selection"; root.mkdir(); raw = csv_bytes(rows, CLIP_SELECTION_FIELDS)
            (root / "clip_selection.csv").write_bytes(raw)
            provenance = {"schema": SELECTION_SCHEMA, "config_sha256": "c" * 64, "source_sha256": "p" * 64, "source_manifest_sha256": "m" * 64, "selection_sha256": hashlib.sha256(raw).hexdigest(), "clip_count": 54, "selection_rule": "mean over Advantage PPO seeds 0,1,2; descending MAE then clip_id tie-break"}
            (root / "selection_provenance.json").write_text(json.dumps(provenance))
            self.assertEqual(len(read_frozen_selection(root, config_sha256="c" * 64, source_sha256="p" * 64, source_manifest_sha256="m" * 64)), 54)
            (root / "clip_selection.csv").write_bytes(raw.replace(b"clip-000", b"clip-999", 1))
            with self.assertRaises(ContractValidationError):
                read_frozen_selection(root, config_sha256="c" * 64, source_sha256="p" * 64, source_manifest_sha256="m" * 64)

    def test_repeated_beam_factual_is_pinned_outside_capacity_and_rank_tie(self):
        frames = [frame(index) for index in range(5)]
        labels = [LabelFrame("mcd", "clip", index, 8.0 + index, 80.0, GT_RULE_ID, True, None) for index in range(5)]
        snapshots = factual_snapshots(frames, labels, Policy(), clip_id="clip")
        self.assertEqual(eligible_anchors(snapshots, window=4), (0,))
        result = repeated_correction_beam(frames, labels, Policy(), clip_id="clip", anchor_hop_idx=0, width=1, window=4)
        self.assertEqual(len(result.factual), 4)
        self.assertEqual([hop.immediate_abs_error_bpm for hop in result.factual], [item.abs_error_bpm for item in snapshots[:4]])
        self.assertLessEqual(len(result.retained), 1)
        tied_frames = [frame(index, invalid=True) for index in range(4)]
        tied_labels = [LabelFrame("mcd", "clip", index, 8.0 + index, 70.0, GT_RULE_ID, True, None) for index in range(4)]
        action, _error = rank1_action(tied_frames[0], tied_labels[0], snapshots[0].state)
        self.assertEqual(action, 0)

    def test_repeated_beam_legal_children_deterministic_width_and_gt_boundary(self):
        frames = [frame(index) for index in range(5)]
        labels_a = [LabelFrame("mcd", "clip", index, 8.0 + index, 80.0, GT_RULE_ID, True, None) for index in range(5)]
        labels_b = [LabelFrame("mcd", "clip", index, 8.0 + index, 160.0, GT_RULE_ID, True, None) for index in range(5)]
        results = run_widths(frames, labels_a, Policy(), clip_id="clip", anchor_hop_idx=0, widths=(1, 2, 4, 8), window=4)
        self.assertEqual(results, run_widths(frames, labels_a, Policy(), clip_id="clip", anchor_hop_idx=0, widths=(1, 2, 4, 8), window=4))
        best = [min(path[-1].cumulative_abs_error_bpm for path in item.retained) for item in results]
        self.assertEqual(best, sorted(best, reverse=True))
        for result in results:
            for path in result.retained:
                for hop in path:
                    self.assertIn(hop.proposed_action, range(12))
                    self.assertEqual(hop.proposed_action, hop.executed_action)
                    self.assertIn(hop.source, {"rank1_gt", "ppo_legal"})
        factual_a = factual_snapshots(frames, labels_a, Policy(), clip_id="clip")
        factual_b = factual_snapshots(frames, labels_b, Policy(), clip_id="clip")
        self.assertEqual([(row.proposed_action, row.executed_action, row.post_belief_hr_bpm) for row in factual_a], [(row.proposed_action, row.executed_action, row.post_belief_hr_bpm) for row in factual_b])

    def test_repeated_beam_unpruned_complete_binary_h4_matches_all_leaves(self):
        """A mocked all-free controller makes the H=4 binary expansion exact."""
        from adaptive_roi_rppg.evaluation import repeated_correction_beam as beam
        class BinaryPolicy:
            identity = CheckpointIdentity("advantage_ppo_seed0", "advantage_ppo", 0, "a" * 64)
            def initial_state(self): return 0
            def predict(self, observation, recurrent_state, *, episode_start): return 1, recurrent_state + 1
        frames = [frame(index) for index in range(4)]
        labels = [LabelFrame("mcd", "clip", index, 8.0 + index, 0.0, GT_RULE_ID, True, None) for index in range(4)]
        def step(_frame, state, action):
            next_state = SimpleNamespace(path=state.path + (action,))
            decision = SimpleNamespace(executed_action=action, legal=True, override_reason=None)
            return next_state, SimpleNamespace(action_decision=decision, post_belief=SimpleNamespace(mean_hr=float(action)))
        observation = SimpleNamespace(array=lambda: np.zeros(101, dtype=np.float32))
        with patch.object(beam, "initial_control_state", return_value=SimpleNamespace(path=())), patch.object(beam, "build_observation", return_value=observation), patch.object(beam, "control_step", side_effect=step), patch.object(beam, "legal_actions", side_effect=lambda _state: (0, 1)), patch.object(beam, "control_state_digest", side_effect=lambda state: repr(state.path)), patch.object(beam, "ACTION_COUNT", 2):
            result = beam.repeated_correction_beam(frames, labels, BinaryPolicy(), clip_id="clip", anchor_hop_idx=0, width=16, window=4)
        leaves = {path[-1].origin_sequence for path in result.retained}
        self.assertEqual(leaves, set(product((0, 1), repeat=4)))
        self.assertEqual(min(path[-1].cumulative_abs_error_bpm for path in result.retained), 0.0)

    def test_lifecycle_complete_is_last_and_hash_tamper_rejects(self):
        provenance = {"plan_id":"beam-plan", "code_snapshot_sha256":"a" * 64, "source_inventory_sha256":"b" * 64, "environment_sha256":"c" * 64, "job_id":"smoke", "node":"local", "command":"beam-smoke"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "published"; root.mkdir()
            write_marker(root, marker_payload("STARTED", "shard", provenance, shard_index=0, shard_count=54))
            (root / "beam_hops.csv").write_text("row\n")
            outputs = validate_precomplete_directory(root, ("beam_hops.csv",), kind="shard", provenance=provenance)
            write_marker(root, marker_payload("COMPLETE", "shard", provenance, shard_index=0, shard_count=54, outputs=outputs))
            validate_directory(root, {"STARTED.json", "COMPLETE.json", "beam_hops.csv"}, kind="shard", provenance=provenance, complete_outputs=("beam_hops.csv",))
            (root / "beam_hops.csv").write_text("tampered\n")
            with self.assertRaises(ContractValidationError):
                validate_directory(root, {"STARTED.json", "COMPLETE.json", "beam_hops.csv"}, kind="shard", provenance=provenance, complete_outputs=("beam_hops.csv",))


if __name__ == "__main__": unittest.main()
