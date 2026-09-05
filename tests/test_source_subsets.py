import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.evaluation.source_subsets import FULL_TASK_COUNT, _validate_full_subject_inventory, audit_plan_body_sha256, read_task_subset


class SourceSubsetTests(unittest.TestCase):
    def test_full_subject_set_must_equal_source_plan(self):
        clips = tuple(f"c{index}" for index in range(533))
        plan_subjects = tuple(f"s{index}" for index in range(89))
        source_subjects = {clip: plan_subjects[index % 89] for index, clip in enumerate(clips)}
        _validate_full_subject_inventory(clips, plan_subjects, source_subjects)
        # Same cardinality, but one plan subject is replaced by an intruder.
        source_subjects[clips[0]] = "intruder"
        with self.assertRaises(ContractValidationError):
            _validate_full_subject_inventory(clips, plan_subjects, source_subjects)

    def test_full_index_mapping_boundaries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); full = root / "full"; full.mkdir()
            tasks = []
            for task in range(FULL_TASK_COUNT):
                clip_position, seed = divmod(task, 3)
                path = full / f"task-{task:04d}.csv"
                payload = ("subject_id,clip_id,hop_idx,method_id,family,seed,checkpoint_sha256,proposed_action,executed_action,selected_post_belief_hr_bpm,gt_hr_bpm,selected_abs_error_bpm\n"
                           f"s{clip_position},c{clip_position},0,advantage_ppo_seed{seed},advantage_ppo,{seed},{chr(97 + seed) * 64},0,0,70,70,0\n").encode()
                path.write_bytes(payload)
                tasks.append({"task_index": task, "clip_position": clip_position, "clip_id": f"c{clip_position}", "subject_id": f"s{clip_position}", "seed": seed, "row_count": 1, "sha256": sha256_file(path), "bytes": len(payload)})
            index = {"schema": "mcd-repeated-correction-beam-full-subsets-v1", "task_count": FULL_TASK_COUNT,
                     "source_sha256": "a" * 64, "plan_sha256": "b" * 64, "complete_sha256": "c" * 64, "tasks": tasks}
            (full / "index.json").write_bytes(canonical_json_bytes(index))
            for task, expected in ((0, (0, 0)), (1, (0, 1)), (2, (0, 2)), (1596, (532, 0)), (1597, (532, 1)), (1598, (532, 2))):
                entry, rows = read_task_subset(full, task, task_count=FULL_TASK_COUNT)
                self.assertEqual((entry["clip_position"], entry["seed"]), expected)
                self.assertEqual(rows[0]["clip_id"], f"c{expected[0]}")
            with self.assertRaises(ContractValidationError):
                read_task_subset(full, 1599, task_count=FULL_TASK_COUNT)

    def test_audit_plan_hash_is_canonical_body_hash(self):
        body = {"schema": "mcd-frozen-failure-audit-plan-v1", "split": "eval", "value": {"b": 2, "a": 1}}
        digest = audit_plan_body_sha256({**body, "audit_plan_sha256": "ignored"})
        self.assertEqual(digest, hashlib.sha256(__import__("adaptive_roi_rppg.contracts", fromlist=["canonical_json_bytes"]).canonical_json_bytes(body)).hexdigest())
        self.assertNotEqual(digest, audit_plan_body_sha256({**body, "value": {"b": 3, "a": 1}}))


if __name__ == "__main__": unittest.main()
