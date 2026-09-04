import tempfile
import unittest
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from scripts.verify_gate6_1_mcd_crossover import _args

from adaptive_roi_rppg.evaluation.crossover import (
    ARM_IDS, LEARNED_ARM_STATUS, HistoricalCacheBinding, _clip_row,
    aggregate_crossover, merge_crossover_shards, publish_crossover_shard,
    score_current_arm, score_historical_arm,
)
from adaptive_roi_rppg.contracts.errors import ContractValidationError


def _npz(path: Path, hr: float = 70.0, confidence: float = 1.0) -> None:
    shape = (4, 12)
    np.savez(path, stem=np.array(path.stem), view=np.array("Frontal"), condition=np.array("before"), fs=np.array(30.0),
             hr_meas=np.full(shape, hr), conf=np.full(shape, confidence), ppr=np.ones(shape), cov=np.ones(shape),
             gt_hr=np.full(4, 70.0), cache_schema_version=np.array(3), gt_subharmonic_labels=np.array(False))


def _rows(clip="c", subject="s"):
    return [{"clip_id": clip, "subject_id": subject, "view": "Frontal", "condition": "before", "arm_id": arm, "action": index, "historical_mae_bpm": 1.0, "current_mae_bpm": 2.0, "current_minus_historical_bpm": 1.0, "historical_invalid_count": 0, "current_invalid_count": 0, "historical_action_stats": {"hops": 1, "override_count": 0, "switch_count": 0, "roi_index_jump_count": 0, "proposed_action": index, "hold_mean": 1.0, "hold_max": 1, "executed_action_distribution": {str(index): 1}, "roi_index_jump_mean_abs": 0.0, "hr_jump_mean_abs_bpm": 0.0, "belief_jump_mean_abs_bpm": 0.0}, "current_action_stats": {"hops": 1, "override_count": 0, "switch_count": 0, "roi_index_jump_count": 0, "proposed_action": index, "hold_mean": 1.0, "hold_max": 1, "executed_action_distribution": {str(index): 1}, "roi_index_jump_mean_abs": 0.0, "hr_jump_mean_abs_bpm": 0.0, "belief_jump_mean_abs_bpm": 0.0}} for index, arm in enumerate(ARM_IDS)]


class Gate61Tests(unittest.TestCase):
    def test_launcher_accepts_repeated_shard_directories(self):
        args = _args(["--merge", "--output-dir", "merged", "--shard-dir", "shard-0", "--shard-dir", "shard-1"])
        self.assertEqual(args.shard_dir, ["shard-0", "shard-1"])

    def test_known_answer_crossover_and_fixed_action_legality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"; _npz(path)
            rows, stats = score_historical_arm(path, 7, expected_sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(), clip_id="x")
            self.assertEqual([row["executed_action"] for row in rows], [7] * 4)
            self.assertEqual(stats["override_count"], 0)
            self.assertAlmostEqual(rows[-1]["abs_error_bpm"], 0.0)
            self.assertEqual([row["hop_time_s"] for row in rows[:3]], [8.0, 9.0, 10.0])
            binding = HistoricalCacheBinding("x", "s", "Frontal", "before", "x.npz", "unused", str(path), 4, 0.0)
            current = [dict(row) for row in rows[:3]]
            result = _clip_row(binding, ARM_IDS[7], 7, rows[:3], current, stats, stats)
            self.assertEqual(result["current_minus_historical_bpm"], 0.0)

    def test_equal_clip_aggregation_and_unavailable_learned_arm(self):
        rows = []
        stats = {"hops": 1, "override_count": 0, "switch_count": 0, "roi_index_jump_count": 0, "proposed_action": 0, "hold_mean": 1.0, "hold_max": 1, "executed_action_distribution": {"0": 1}, "roi_index_jump_mean_abs": 0.0, "hr_jump_mean_abs_bpm": 0.0, "belief_jump_mean_abs_bpm": 0.0}
        for arm in ARM_IDS:
            action = ARM_IDS.index(arm)
            for subject, values in (("s1", (1.0, 3.0)), ("s2", (5.0, 7.0))):
                for index, value in enumerate(values):
                    rows.append({"arm_id": arm, "action": action, "subject_id": subject, "clip_id": f"{subject}-{index}", "historical_mae_bpm": 10.0, "current_mae_bpm": 10.0 + value, "current_minus_historical_bpm": value, "historical_invalid_count": 0, "current_invalid_count": 0, "historical_action_stats": stats, "current_action_stats": stats})
        _, subjects, summary = aggregate_crossover(rows, bootstrap_replicates=100, seed=9)
        self.assertEqual(len(subjects), 24)
        self.assertEqual(summary["learned_arms"]["status"], LEARNED_ARM_STATUS)
        self.assertEqual(summary["arms"][0]["current_minus_historical_bpm"], 4.0)

    def test_shard_tamper_and_fresh_output_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shard = root / "shard"; destination = root / "merged"
            dummy = _rows()
            self.assertIsNone(publish_crossover_shard(dummy, shard, shard_index=0, shard_count=1, code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, command="test", node="node", job_id="job", subject_ids=["s"]))
            (shard / "shard.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ContractValidationError): merge_crossover_shards([shard], destination, bootstrap_replicates=5)
            self.assertEqual({p.name for p in destination.iterdir()}, {"STARTED.json", "FAILED.json"})
            publish_crossover_shard(dummy, root / "other", shard_index=0, shard_count=1, code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, command="test", node="node", job_id="job", subject_ids=["s"])
            with self.assertRaises(ContractValidationError): publish_crossover_shard([], root / "other", shard_index=0, shard_count=1)

    def test_historical_hash_and_exact_key_gt_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"; _npz(path)
            with self.assertRaises(ContractValidationError): score_historical_arm(path, 0, expected_sha256="0" * 64, clip_id="x")
            binding = HistoricalCacheBinding("x", "s", "Frontal", "before", "x.npz", "a" * 64, str(path), 1, 0.0)
            historical = [{"dataset_id": "mcd", "clip_id": "x", "hop_idx": 0, "hop_time_s": 8.0, "gt_hr_bpm": 70.0, "abs_error_bpm": 0.0}]
            current = [{**historical[0], "hop_time_s": 9.0}]
            with self.assertRaises(ContractValidationError): _clip_row(binding, "fixed_full_face", 0, historical, current, {}, {})
            current = [{**historical[0], "gt_hr_bpm": 71.0}]
            with self.assertRaises(ContractValidationError): _clip_row(binding, "fixed_full_face", 0, historical, current, {}, {})

    def test_current_measurements_must_match_label_identity_and_time(self):
        frame = SimpleNamespace(dataset_id="mcd", clip_id="x", hop_idx=0, hop_time_s=99.0)
        label = SimpleNamespace(dataset_id="mcd", clip_id="x", hop_idx=0, hop_time_s=8.0, valid=True, gt_hr_bpm=70.0)
        with self.assertRaises(ContractValidationError): score_current_arm([frame], [label], 0, "x")

    def test_strict_merge_requires_twelve_arms_and_matching_provenance(self):
        def rows(arm_count=12):
            return [{"clip_id": "c", "subject_id": "s", "view": "Frontal", "condition": "before", "arm_id": ARM_IDS[i], "action": i, "historical_mae_bpm": 1.0, "current_mae_bpm": 2.0, "current_minus_historical_bpm": 1.0, "historical_invalid_count": 0, "current_invalid_count": 0, "historical_action_stats": {"hops": 1, "override_count": 0, "switch_count": 0, "roi_index_jump_count": 0, "proposed_action": i, "hold_mean": 1.0, "hold_max": 1, "executed_action_distribution": {str(i): 1}, "roi_index_jump_mean_abs": 0.0, "hr_jump_mean_abs_bpm": 0.0, "belief_jump_mean_abs_bpm": 0.0}, "current_action_stats": {"hops": 1, "override_count": 0, "switch_count": 0, "roi_index_jump_count": 0, "proposed_action": i, "hold_mean": 1.0, "hold_max": 1, "executed_action_distribution": {str(i): 1}, "roi_index_jump_mean_abs": 0.0, "hr_jump_mean_abs_bpm": 0.0, "belief_jump_mean_abs_bpm": 0.0}} for i in range(arm_count)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shard = root / "shard"; out = root / "out"
            with self.assertRaises(ContractValidationError):
                publish_crossover_shard(rows(11), shard, shard_index=0, shard_count=1, plan_id="p", phase="fixture", code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, subject_ids=["s"], command="test", node="n", job_id="job")
            self.assertEqual({p.name for p in shard.iterdir()}, {"STARTED.json", "FAILED.json"})

    def test_publication_rejects_null_roi_jump_statistic(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = _rows()
            rows[0]["historical_action_stats"]["roi_index_jump_mean_abs"] = None
            destination = Path(directory) / "shard"
            with self.assertRaises(ContractValidationError):
                publish_crossover_shard(rows, destination, shard_index=0, shard_count=1, code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, command="test", node="n", job_id="job", subject_ids=["s"])
            self.assertEqual({p.name for p in destination.iterdir()}, {"STARTED.json", "FAILED.json"})

    def test_publication_rejects_impossible_fixed_action_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = _rows()
            rows[0]["historical_action_stats"]["executed_action_distribution"] = {"1": 1}
            rows[0]["historical_action_stats"]["override_count"] = 1
            destination = Path(directory) / "shard"
            with self.assertRaises(ContractValidationError):
                publish_crossover_shard(rows, destination, shard_index=0, shard_count=1, code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, command="test", node="n", job_id="job", subject_ids=["s"])
            self.assertEqual({p.name for p in destination.iterdir()}, {"STARTED.json", "FAILED.json"})

    def test_valid_merge_publishes_provenance_only_after_validation(self):
        rows = _rows("c1") + _rows("c2")
        provenance = {"plan_id": "p", "phase": "fixture", "code_snapshot_sha256": "a" * 64, "source_inventory_sha256": "b" * 64, "command": "merge", "node": "n", "job_id": "job"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shard = root / "shard"; out = root / "out"
            publish_crossover_shard(rows, shard, shard_index=0, shard_count=1, plan_id="p", phase="fixture", code_snapshot_sha256="a" * 64, source_inventory_sha256="b" * 64, subject_ids=["s"], command="run", node="n", job_id="job")
            result = merge_crossover_shards([shard], out, expected_phase="fixture", expected_plan_id="p", expected_code_snapshot_sha256="a" * 64, expected_source_inventory_sha256="b" * 64, merged_provenance=provenance, bootstrap_replicates=5)
            self.assertEqual(result["provenance"], provenance)
            self.assertEqual(json.loads((out / "COMPLETE.json").read_text())["state"], "complete")


if __name__ == "__main__": unittest.main()
