import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation import mcd_failure_runner_v2 as runner


def row(*, clip="clip-a", hop=0, action=3, selected=3):
    alternatives = [{"requested_action": value, "proposed_action": value, "executed_action": value, "measurement_valid": True, "measurement_invalid_reason": None, "measurement_hr_bpm": 70.0 + value, "post_belief_hr_bpm": 70.0 + value, "abs_error_bpm": abs(70.0 + value - 72.0)} for value in range(12)]
    return {"subject_id": "s1", "clip_id": clip, "camera_id": "FullHDwebcam", "view": "Frontal", "condition": "before", "pose_status": "unavailable_in_source", "pose_yaw": None, "pose_pitch": None, "pose_roll": None, "hop_idx": hop, "hop_time_s": 8.0 + hop, "method_id": "advantage_ppo_seed0", "family": "advantage_ppo", "seed": 0, "checkpoint_sha256": "a" * 64, "proposed_action": selected, "executed_action": selected, "previous_action": None, "legal": True, "override_reason": None, "pre_hold_count": 0, "post_hold_count": 1, "selected_measurement_valid": True, "selected_measurement_invalid_reason": None, "selected_measurement_hr_bpm": 73.0, "selected_post_belief_hr_bpm": 73.0, "gt_rule_id": "gt", "gt_hr_bpm": 72.0, "selected_abs_error_bpm": 1.0, "failure_category": "ok", "best_valid_actions_json": "[2]", "best_valid_error_bpm": 0.0, "policy_minus_best_immediate_regret_bpm": 1.0, "any_valid_immediate_action": True, "better_actionable_alternative": True, "random_control_requested_action": action, "alternatives_json": json.dumps(alternatives, sort_keys=True, separators=(",", ":"))}


class RunnerV2Tests(unittest.TestCase):
    def test_stable_load_balanced_assignment_is_not_modulo(self):
        assigned = runner.balanced_subject_assignment({"s3": 9, "s1": 9, "s2": 8, "s4": 1}, 2)
        self.assertEqual(assigned, (("s1", "s2"), ("s3", "s4")))

    def test_compact_json_requires_all_12_in_canonical_order(self):
        item = row(); runner.validate_hop_rows([item])
        bad = dict(item); values = json.loads(bad["alternatives_json"]); values[-1]["requested_action"] = 10; bad["alternatives_json"] = json.dumps(values, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ContractValidationError): runner.validate_hop_rows([bad])

    def test_pose_uses_degree_named_canonical_fields_and_preserves_all_missing(self):
        finite = SimpleNamespace(camera_fps=30.0, head_yaw_deg=12.5, head_pitch_deg=-3.0, head_roll_deg=1.25)
        missing = SimpleNamespace(camera_fps=30.0, head_yaw_deg=None, head_pitch_deg=None, head_roll_deg=None)
        measurement = SimpleNamespace(hop_time_s=1 / 30)
        self.assertEqual(runner._pose((finite,), measurement), ("mcd_state_end_frame", 12.5, -3.0, 1.25))
        self.assertEqual(runner._pose((missing,), measurement), ("unavailable_in_source", None, None, None))

    def test_exact_coverage_rejects_missing_checkpoint_hop(self):
        identity = [{"method_id": "advantage_ppo_seed0", "family": "advantage_ppo", "seed": 0, "checkpoint_sha256": "a" * 64}]
        runner.validate_hop_rows([row(hop=0), row(hop=1)], {"clip-a": 2}, identity)
        with self.assertRaises(ContractValidationError): runner.validate_hop_rows([row(hop=0)], {"clip-a": 2}, identity)

    def test_completed_retry_is_reused_and_mixed_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "shard-000" / "attempts" / "first"; root.mkdir(parents=True)
            (root / "COMPLETE.json").write_text(json.dumps({"state": "COMPLETE", "audit_plan_sha256": "a" * 64, "logical_shard_index": 0}))
            self.assertEqual(runner._matching_completed(root.parent.parent, "a" * 64, 0), root)
            self.assertIsNone(runner._matching_completed(root.parent.parent, "b" * 64, 0))

    def test_stale_benchmark_and_underprovision_are_rejected(self):
        plan = {"code_snapshot_sha256": "a" * 64, "source_inventory_sha256": "b" * 64, "environment_sha256": "c" * 64, "plan_id": "p", "checkpoint_identities": [], "canonical_roi_names": [], "resource_shape": {"cpus_per_worker": 1, "gpu": False, "logical_shards": 1}, "logical_shards": [{"expected_hops_per_checkpoint": 100}], "cohort": {"expected_hops": {"x": 100}}}
        record = {"schema": runner.BENCHMARK_SCHEMA, "status": "complete", "compatibility": {key: plan[key] for key in ("code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "plan_id", "checkpoint_identities", "canonical_roi_names", "resource_shape")}, "measured": {"worker_seconds_per_hop_checkpoint": 1.0, "bytes_per_hop_checkpoint": 2.0, "merge_seconds_per_byte": 1.0}}
        projection = runner._projection(record, plan); self.assertGreater(projection["shard_seconds"], 100)
        record["compatibility"]["plan_id"] = "old"
        with self.assertRaises(ContractValidationError): runner._projection(record, plan)

    def test_parallel_benchmark_uses_slowest_individual_shard_rate(self):
        plan = {"audit_plan_sha256": "a" * 64, "code_snapshot_sha256": "a" * 64, "source_inventory_sha256": "b" * 64, "environment_sha256": "c" * 64, "plan_id": "p", "checkpoint_identities": [], "canonical_roi_names": [], "resource_shape": {"cpus_per_worker": 1, "gpu": False, "logical_shards": 2}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); shards = []
            for name, elapsed, work, bytes_ in (("left", 10.0, 10, 100), ("right", 20.0, 100, 1000)):
                shard = root / name; shard.mkdir(); (shard / "shard_manifest.json").write_text(json.dumps({"elapsed_seconds": elapsed, "work_units": work, "per_hop_bytes": bytes_})); shards.append(shard)
            report = root / "report.json"; report.write_text(json.dumps({"merge_elapsed_seconds": 5.0}))
            record = root / "benchmark.json"; runner._benchmark(record, plan, shards, report)
            self.assertEqual(json.loads(record.read_text())["measured"]["worker_seconds_per_hop_checkpoint"], 1.0)

    def test_frozen_worker_assignment_budgets_each_explicit_local_list(self):
        plan = {"logical_shard_count": 3, "logical_shards": [{"expected_hops_per_checkpoint": 5}, {"expected_hops_per_checkpoint": 4}, {"expected_hops_per_checkpoint": 3}]}
        assignment = runner.frozen_worker_assignment(3, (0, 1, 2), 2)
        self.assertEqual(assignment["worker_logical_shard_indices"], [[0, 2], [1]])
        self.assertEqual(runner._local_shard_work(plan, assignment["worker_logical_shard_indices"][0]), 72)
        self.assertEqual(runner._local_shard_work(plan, assignment["worker_logical_shard_indices"][1]), 36)
        self.assertEqual(runner.frozen_worker_assignment(1, (0,), 4)["worker_logical_shard_indices"], [[0], [], [], []])  # S < W
        self.assertEqual(runner.frozen_worker_assignment(2, (0, 1), 2)["worker_logical_shard_indices"], [[0], [1]])  # S = W

    def test_underprovision_is_rejected_before_replay_budget(self):
        plan = {"code_snapshot_sha256": "a" * 64, "source_inventory_sha256": "b" * 64, "environment_sha256": "c" * 64, "plan_id": "p", "checkpoint_identities": [], "canonical_roi_names": [], "resource_shape": {"cpus_per_worker": 1, "gpu": False, "logical_shards": 3}, "logical_shard_count": 3, "logical_shards": [{"expected_hops_per_checkpoint": 5}, {"expected_hops_per_checkpoint": 4}, {"expected_hops_per_checkpoint": 3}], "cohort": {"expected_hops": {"x": 12}}}
        record = {"schema": runner.BENCHMARK_SCHEMA, "status": "complete", "compatibility": {key: plan[key] for key in ("code_snapshot_sha256", "source_inventory_sha256", "environment_sha256", "plan_id", "checkpoint_identities", "canonical_roi_names", "resource_shape")}, "measured": {"worker_seconds_per_hop_checkpoint": 1.0, "bytes_per_hop_checkpoint": 1.0, "merge_seconds_per_byte": 1.0}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "benchmark.json"; path.write_text(json.dumps(record)); args = SimpleNamespace(split="eval", benchmark_record=str(path), audit_walltime_seconds=1, cpus_per_worker=1, worker_count=2, allocated_task_count=2)
            with self.assertRaises(ContractValidationError): runner._require_budget(args, plan, "shard", logical_indices=(0, 2))

    def test_retry_requires_frozen_rank_local_list(self):
        plan = {"logical_shard_count": 3, "worker_assignment": runner.frozen_worker_assignment(3, (0, 1, 2), 2)}
        args = SimpleNamespace(worker_count=2, allocated_task_count=2, worker_rank=0)
        runner._validate_worker_assignment(args, plan, (0, 2))
        with self.assertRaises(ContractValidationError): runner._validate_worker_assignment(args, plan, (0, 1))

    def test_completed_reuse_rejects_corrupt_marker_or_artifact(self):
        plan = {"audit_plan_sha256": "a" * 64, "run_id": "run", "split": "train", "logical_shard_count": 1, "logical_shards": [{"clip_ids": ["clip-a"]}], "cohort": {"expected_hops": {"clip-a": 1}}, "checkpoint_identities": []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "COMPLETE.json").write_text(json.dumps({"schema": runner.SCHEMA, "state": "COMPLETE", "audit_plan_sha256": "a" * 64, "run_id": "run", "split": "train", "logical_shard_index": 0, "logical_shard_count": 1, "outputs": {}})); (root / "shard_manifest.json").write_text(json.dumps({}))
            with self.assertRaises(ContractValidationError): runner._validate_completed_attempt(root, plan, 0)

    def test_safe_tokens_reject_path_and_dot_forms(self):
        self.assertEqual(runner._safe_token("run-01.a", "run"), "run-01.a")
        for value in ("", ".", "..", "a/b", "a b", "a\\b"):
            with self.assertRaises(ContractValidationError): runner._safe_token(value, "run")

    def test_pos_is_built_once_per_clip_before_all_policy_replays_and_labels(self):
        events = []; old_build, old_replay = runner.build_pos_measurements, runner.replay_measurements_without_labels
        try:
            runner.build_pos_measurements = lambda frames: events.append("pos") or ("m",)
            runner.replay_measurements_without_labels = lambda measurements, frames, policy, clip: events.append(f"replay-{policy}") or ()
            self.assertEqual(runner._replay_all_policies_without_labels((object(),), ("p1", "p2"), SimpleNamespace()), ((), ()))
            self.assertEqual(events, ["pos", "replay-p1", "replay-p2"])
        finally:
            runner.build_pos_measurements, runner.replay_measurements_without_labels = old_build, old_replay

    def test_evaluate_reads_labels_only_after_all_policy_replays(self):
        events = []; saved = (runner.load_frozen_recurrent_policy, runner.read_mcd_canonical_frames, runner._replay_all_policies_without_labels, runner.read_mcd_labels, runner.score_replay_after_labels)
        try:
            runner.load_frozen_recurrent_policy = lambda item: item["method_id"]
            runner.read_mcd_canonical_frames = lambda *args: events.append("frames") or (object(),)
            runner._replay_all_policies_without_labels = lambda frames, policies, clip: events.append("all-replays") or ((), ())
            runner.read_mcd_labels = lambda *args: events.append("labels") or ()
            runner.score_replay_after_labels = lambda replay, labels: events.append("score") or ()
            args = SimpleNamespace(state_root="state", gt_root="gt", split="train")
            clip = SimpleNamespace(clip_id="clip-a")
            self.assertEqual(list(runner._evaluate(args, None, (clip,), ({"method_id": "p1"}, {"method_id": "p2"}))), [])
            self.assertEqual(events, ["frames", "all-replays", "labels", "score", "score"])
        finally:
            runner.load_frozen_recurrent_policy, runner.read_mcd_canonical_frames, runner._replay_all_policies_without_labels, runner.read_mcd_labels, runner.score_replay_after_labels = saved

    def test_dispatch_smoke_accepts_no_data_paths_and_rejects_them_if_supplied(self):
        root = Path(__file__).resolve().parents[1]; script = root / "scripts" / "run_mcd_frozen_failure_audit_v2.py"; env = {**os.environ, "PYTHONPATH": str(root / "src")}
        common = [sys.executable, str(script), "--mode", "dispatch-smoke", "--run-id", "smoke-1", "--code-snapshot-sha256", "a" * 64, "--audit-walltime-seconds", "60", "--worker-count", "2", "--allocated-task-count", "2", "--worker-rank", "1"]
        result = subprocess.run(common, check=True, capture_output=True, text=True, env=env)
        self.assertEqual(json.loads(result.stdout)["proof"], "NO_DATA_ACCESS")
        rejected = subprocess.run([*common, "--manifest-tree", "must-not-be-opened"], capture_output=True, text=True, env=env)
        self.assertNotEqual(rejected.returncode, 0)
        rejected_benchmark = subprocess.run([*common, "--benchmark-record", "must-not-be-opened"], capture_output=True, text=True, env=env)
        self.assertNotEqual(rejected_benchmark.returncode, 0)

    def test_streamed_output_round_trips_without_metadata_duplication(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.csv"; runner._write_csv(path, [row()], runner.HOP_FIELDS)
            loaded = list(runner._stream(path, runner.HOP_FIELDS)); self.assertEqual(len(loaded), 1); self.assertEqual(loaded[0]["alternatives_json"], row()["alternatives_json"])


if __name__ == "__main__": unittest.main()
