import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.source_subsets import SOURCE_FIELDS, TASK_COUNT, audit_plan_body_sha256, prepare_task_subsets, read_task_subset


class SourceSubsetTests(unittest.TestCase):
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
