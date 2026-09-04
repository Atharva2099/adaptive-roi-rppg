import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.evaluation.source_subsets import FULL_TASK_COUNT, SOURCE_FIELDS, TASK_COUNT, _validate_full_subject_inventory, audit_plan_body_sha256, prepare_task_subsets, read_task_subset


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

    def test_full_index_mapping_boundaries_and_worst54_coexistence(self):
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

    def test_streamed_mapping_has_all_162_unique_tasks_and_seed_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); run_root = root / "run"; merged = run_root / "merged"; merged.mkdir(parents=True); source = merged / "per_hop_counterfactual.csv"; plan = run_root / "audit_plan.json"; complete = merged / "COMPLETE.json"; selection = root / "selection.csv"; out = root / "subsets"
            selected = [{"clip_id": f"c{i:03d}", "subject_id": f"s{i:03d}"} for i in range(54)]
            with selection.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("clip_id", "subject_id")); writer.writeheader(); writer.writerows(selected)
            rows = []
            for item in selected:
                for seed in range(3):
                    rows.append({"subject_id": item["subject_id"], "clip_id": item["clip_id"], "hop_idx": "0", "method_id": f"advantage_ppo_seed{seed}", "family": "advantage_ppo", "seed": str(seed), "checkpoint_sha256": "a" * 64, "proposed_action": "0", "executed_action": "0", "selected_post_belief_hr_bpm": "70", "gt_hr_bpm": "70", "selected_abs_error_bpm": "0"})
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(*SOURCE_FIELDS, "extra_audit_field")); writer.writeheader(); writer.writerows([{**row, "extra_audit_field": "kept-in-source"} for row in rows])
            clips = [f"c{i:03d}" for i in range(533)]
            expected_hops = {clip: (1 if index < 54 else 190 + (1 if index - 54 < 163 else 0)) for index, clip in enumerate(clips)}
            plan_body = {"schema":"mcd-frozen-failure-audit-plan-v1", "split":"eval", "run_id":"audit-48121", "cohort":{"clip_ids":clips, "subject_ids":[str(i) for i in range(89)], "expected_hops":expected_hops}, "checkpoint_identities":[{"seed":s,"family":"advantage_ppo","method_id":f"advantage_ppo_seed{s}","checkpoint_sha256":"a"*64} for s in range(3)] + [{"seed":s,"family":"dagger","method_id":f"dagger_seed{s}","checkpoint_sha256":f"{s + 1:x}" * 64} for s in range(3)] + [{"seed":s,"family":"standard_ppo","method_id":f"standard_ppo_seed{s}","checkpoint_sha256":f"{s + 4:x}" * 64} for s in range(3)]}
            plan = plan.with_suffix(".json")
            plan.write_text(json.dumps({**plan_body, "audit_plan_sha256": audit_plan_body_sha256(plan_body)}))
            complete.write_text(json.dumps({"schema":"mcd-frozen-failure-audit-v2", "state":"COMPLETE", "run_id":"audit-48121", "audit_plan_sha256":audit_plan_body_sha256(plan_body), "outputs":{"per_hop_counterfactual.csv":{"bytes":source.stat().st_size,"sha256":hashlib.sha256(source.read_bytes()).hexdigest()}}}))
            index = prepare_task_subsets(source, plan, complete, selection, out)
            self.assertEqual(index["task_count"], TASK_COUNT)
            self.assertEqual(len(index["tasks"]), TASK_COUNT)
            self.assertEqual({item["task_index"] for item in index["tasks"]}, set(range(TASK_COUNT)))
            entry, task_rows = read_task_subset(out, 1)
            self.assertEqual((entry["clip_id"], entry["seed"]), ("c000", 1))
            self.assertEqual(task_rows[0]["seed"], "1")
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaises(ContractValidationError):
                prepare_task_subsets(source, plan, complete, selection, root / "tampered-subsets")


if __name__ == "__main__": unittest.main()
