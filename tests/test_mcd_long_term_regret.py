import math
import csv
import io
import unittest
from unittest.mock import patch
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.long_term_regret import (
    build_anchor_regret, csv_bytes, replay_one_time_interventions, score_branches, subject_block_bootstrap,
)
from adaptive_roi_rppg.evaluation.repeated_correction_beam import (
    eligible_anchors, factual_snapshots, rank1_action, repeated_correction_beam, run_widths,
)
from scripts.verify_mcd_repeated_correction_beam import HOP_FIELDS, _rows
from scripts.analyze_mcd_repeated_beam_behavior import _beam
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
    def test_beam_producer_rows_round_trip_into_analyzer(self):
        frames = [frame(index) for index in range(15)]
        labels = [LabelFrame("mcd", "clip", index, 8.0 + index, 80.0, GT_RULE_ID, True, None) for index in range(15)]
        snapshots = factual_snapshots(frames, labels, Policy(), clip_id="clip")
        results = [repeated_correction_beam(frames, labels, Policy(), clip_id="clip", anchor_hop_idx=0, width=width, snapshots=snapshots) for width in (1, 2, 4, 8, 16, 32)]
        produced, stats = [], []
        for result in results:
            rows, result_stats = _rows(result, method_id="advantage_ppo_seed0", seed=0, clip_id="clip", subject_id="subject")
            produced.extend(rows); stats.extend(result_stats)
        serialized = csv_bytes(produced, HOP_FIELDS).decode("utf-8")
        parsed = list(csv.DictReader(io.StringIO(serialized)))
        factual = {item.hop_idx: {"gt": labels[item.hop_idx].gt_hr_bpm, "belief": snapshots[item.hop_idx].post_belief_hr_bpm,
                                  "error": item.immediate_abs_error_bpm, "proposed": item.proposed_action,
                                  "executed": item.executed_action} for item in result.factual}
        analyzed = _beam(parsed, stats, factual)
        self.assertIn(0, analyzed)

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
