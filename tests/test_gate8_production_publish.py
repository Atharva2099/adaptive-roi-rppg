"""Regression coverage for the actual Gate 8 shard and merge publishers.

The expensive data readers are replaced only at their boundary.  Publication,
marker validation, ownership checks, CSV serialization, and merge are the real
production functions under test.
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.model_publication import HOP_FIELDS, MARKER_NAMES, artifact_map, marker_payload, write_marker


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_gate8_mcd_frozen_models.py"
SPEC = importlib.util.spec_from_file_location("gate8_runner_under_test", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)


PROVENANCE = {"plan_id":"plan", "code_snapshot_sha256":"a" * 64, "source_inventory_sha256":"b" * 64, "environment_sha256":"c" * 64, "job_id":"1", "node":"node", "command":"gate8-frozen-models:plan"}


def plan():
    checkpoints=[]
    for family in ("dagger", "standard_ppo", "advantage_ppo"):
        for seed in range(3):
            checkpoints.append({"method_id":f"{family}_seed{seed}", "family":family, "seed":seed, "sha256":f"{seed + 1:x}" * 64, "byte_size":1, "provenance_status":"descriptive_unverified", "training_config_provenance":"unresolved"})
    return SimpleNamespace(plan_id="plan", checkpoints=tuple(checkpoints), payload={"bootstrap":{"comparison_seeds":[8101,8111,8121,8131,8102,8112,8122,8132,8103,8113,8123,8133]}})


def clip(subject, clip_id): return SimpleNamespace(subject_id=subject, clip_id=clip_id, view="front", condition="before", state_row_count=9, camera_fps=1.0)


def rows(clips):
    out=[]
    identities=[("full_face","fixed_full_face",None,None)]+[(x["method_id"],x["family"],x["seed"],x["sha256"]) for x in plan().checkpoints]
    for current in clips:
        for method_id, family, seed, sha in identities:
            row={key:"x" for key in HOP_FIELDS}
            row.update(subject_id=current.subject_id, view=current.view, condition=current.condition, dataset_id="mcd", clip_id=current.clip_id, hop_idx=0, hop_time_s=0.0, method_id=method_id, family=family, seed=seed, checkpoint_sha256=sha, observation_schema_id="obs", observation_sha256="d"*64, frame_provenance_id="frame", signal_config_id="sig", control_config_id="control", gt_rule_id="gt", gt_hr_bpm=70.0, abs_error_bpm=1.0, proposed_action=0, executed_action=0, previous_action=None, legal=True, override_reason=None, pre_hold_count=0, post_hold_count=1, selected_valid=True, selected_invalid_reason=None, selected_hr_bpm=70.0, selected_confidence=1.0, selected_ppr=1.0, selected_coverage=1.0, pre_belief_hr_bpm=70.0, post_belief_hr_bpm=70.0, post_belief_velocity=0.0, post_belief_std_bpm=1.0)
            out.append(row)
    return sorted(out, key=lambda row:(row["method_id"],row["clip_id"],row["hop_idx"]))


class ProductionPublishTests(unittest.TestCase):
    def args(self, root, **extra):
        return SimpleNamespace(output_dir=str(root), shard_index=extra.get("shard_index", 0), shard_count=extra.get("shard_count", 2), shard_dir=extra.get("shard_dir"))

    def test_paired_comparisons_consume_the_declared_seed_schedule(self):
        declared=plan(); calls=[]
        def paired(rows, method_a, method_b, *, estimand, seed):
            calls.append((method_a, method_b, estimand, seed)); return {"method_a":method_a,"method_b":method_b,"estimand":estimand,"seed":seed}
        with patch.object(runner, "paired_subject_bootstrap", side_effect=paired):
            report_rows=runner._paired(declared, {"clip_rows":[]})
        self.assertEqual([row["seed"] for row in report_rows], [seed for seed in declared.payload["bootstrap"]["comparison_seeds"] for _ in range(2)])
        mutated=plan(); mutated.payload["bootstrap"]["comparison_seeds"] = list(reversed(mutated.payload["bootstrap"]["comparison_seeds"]))
        calls.clear()
        with patch.object(runner, "paired_subject_bootstrap", side_effect=paired): runner._paired(mutated, {"clip_rows":[]})
        self.assertEqual([call[3] for call in calls], [seed for seed in mutated.payload["bootstrap"]["comparison_seeds"] for _ in range(2)])

    def test_actual_publish_merge_and_complete_last(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "_evaluate", side_effect=lambda args, selected, checkpoints: rows(selected)), patch.object(runner, "_expected_hops", side_effect=lambda selected:{x.clip_id:1 for x in selected}), patch.object(runner, "_validate_learned", return_value=None), patch.object(runner, "_paired", return_value=[]):
            base=Path(tmp); shards=[]
            for index in range(2):
                path=base/f"shard-{index}"; runner._publish_shard(self.args(path, shard_index=index), plan(), clips, ()); self.assertTrue((path/MARKER_NAMES["COMPLETE"]).is_file()); shards.append(str(path))
            merged=base/"merged"; runner._merge(self.args(merged, shard_dir=shards), plan(), clips, ())
            self.assertTrue((merged/MARKER_NAMES["COMPLETE"]).is_file())
            self.assertFalse((merged/MARKER_NAMES["FAILED"]).exists())

    def test_precomplete_failure_writes_failed_without_complete_for_shard_and_merge(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "_evaluate", side_effect=lambda args, selected, checkpoints: rows(selected)), patch.object(runner, "_expected_hops", side_effect=lambda selected:{x.clip_id:1 for x in selected}), patch.object(runner, "_validate_learned", return_value=None), patch.object(runner, "_paired", return_value=[]):
            base=Path(tmp); original=runner.write_marker
            for target, publish in ((base/"failed-shard", "shard"),):
                def fail_before_complete(root, payload):
                    if Path(root) == target and payload["state"] == "COMPLETE": raise OSError("injected before COMPLETE")
                    original(root, payload)
                with patch.object(runner, "write_marker", side_effect=fail_before_complete), self.assertRaises(OSError):
                    runner._publish_shard(self.args(target, shard_index=0), plan(), clips, ())
                self.assertTrue((target/MARKER_NAMES["STARTED"]).is_file()); self.assertTrue((target/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((target/MARKER_NAMES["COMPLETE"]).exists())
            shards=[]
            for index in range(2):
                path=base/f"shard-{index}"; runner._publish_shard(self.args(path, shard_index=index), plan(), clips, ()); shards.append(str(path))
            target=base/"failed-merge"
            def fail_before_complete(root, payload):
                if Path(root) == target and payload["state"] == "COMPLETE": raise OSError("injected before COMPLETE")
                original(root, payload)
            with patch.object(runner, "write_marker", side_effect=fail_before_complete), self.assertRaises(OSError):
                runner._merge(self.args(target, shard_dir=shards), plan(), clips, ())
            self.assertTrue((target/MARKER_NAMES["STARTED"]).is_file()); self.assertTrue((target/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((target/MARKER_NAMES["COMPLETE"]).exists())

    def test_complete_write_is_the_last_production_operation(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "_evaluate", side_effect=lambda args, selected, checkpoints: rows(selected)), patch.object(runner, "_expected_hops", side_effect=lambda selected:{x.clip_id:1 for x in selected}), patch.object(runner, "_validate_learned", return_value=None), patch.object(runner, "_paired", return_value=[]):
            events=[]; original_marker=runner.write_marker; original_precomplete=runner.validate_precomplete_directory
            def record_marker(root, payload): events.append(payload["state"]); original_marker(root, payload)
            def record_precomplete(*args, **kwargs): events.append("precomplete"); return original_precomplete(*args, **kwargs)
            with patch.object(runner, "write_marker", side_effect=record_marker), patch.object(runner, "validate_precomplete_directory", side_effect=record_precomplete):
                runner._publish_shard(self.args(Path(tmp)/"ordered", shard_index=0), plan(), clips, ())
            self.assertEqual(events[-2:], ["precomplete", "COMPLETE"])

    def test_ownership_failure_after_destination_creation_is_classified(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "expected_shard_clip_ids", side_effect=ContractValidationError("injected ownership failure")):
            target=Path(tmp)/"ownership-failure"
            with self.assertRaises(ContractValidationError): runner._publish_shard(self.args(target, shard_index=0), plan(), clips, ())
            self.assertTrue((target/MARKER_NAMES["STARTED"]).is_file()); self.assertTrue((target/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((target/MARKER_NAMES["COMPLETE"]).exists())

    def test_missing_shard_after_started_is_classified_and_bad_count_creates_nothing(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})):
            base=Path(tmp); missing=base/"missing-shard"; target=base/"missing-merge"
            with self.assertRaises(ContractValidationError): runner._merge(self.args(target, shard_dir=[str(missing), str(missing)]), plan(), clips, ())
            self.assertTrue((target/MARKER_NAMES["STARTED"]).is_file()); self.assertTrue((target/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((target/MARKER_NAMES["COMPLETE"]).exists())
            invalid=base/"invalid-count"
            with self.assertRaises(ContractValidationError): runner._merge(self.args(invalid, shard_count=3, shard_dir=[str(missing)]), plan(), clips, ())
            self.assertFalse(invalid.exists())

    def test_actual_merge_rejects_marker_and_shard_tampering(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "_evaluate", side_effect=lambda args, selected, checkpoints: rows(selected)), patch.object(runner, "_expected_hops", side_effect=lambda selected:{x.clip_id:1 for x in selected}), patch.object(runner, "_validate_learned", return_value=None), patch.object(runner, "_paired", return_value=[]):
            base=Path(tmp); shards=[]
            for index in range(2):
                path=base/f"shard-{index}"; runner._publish_shard(self.args(path, shard_index=index), plan(), clips, ()); shards.append(str(path))
            # Re-sign a deliberately malformed STARTED marker: directory validation
            # must reject its schema before trusting any shard content.
            started=Path(shards[0])/"STARTED.json"; started.write_bytes(canonical_json_bytes({"state":"STARTED"}))
            merged=base/"bad-merge"
            with self.assertRaises(ContractValidationError): runner._merge(self.args(merged, shard_dir=shards), plan(), clips, ())
            self.assertTrue((merged/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((merged/MARKER_NAMES["COMPLETE"]).exists())

    def test_actual_merge_rejects_swapped_shards_and_report_tampering(self):
        clips=(clip("s1","c1"), clip("s2","c2"))
        with TemporaryDirectory() as tmp, patch.object(runner, "_marker_provenance", return_value=(PROVENANCE,{"source":"inventory"})), patch.object(runner, "_evaluate", side_effect=lambda args, selected, checkpoints: rows(selected)), patch.object(runner, "_expected_hops", side_effect=lambda selected:{x.clip_id:1 for x in selected}), patch.object(runner, "_validate_learned", return_value=None), patch.object(runner, "_paired", return_value=[]):
            base=Path(tmp); shards=[]
            for index in range(2):
                path=base/f"shard-{index}"; runner._publish_shard(self.args(path, shard_index=index), plan(), clips, ()); shards.append(str(path))
            swapped=base/"swapped"
            with self.assertRaises(ContractValidationError): runner._merge(self.args(swapped, shard_dir=list(reversed(shards))), plan(), clips, ())
            self.assertTrue((swapped/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((swapped/MARKER_NAMES["COMPLETE"]).exists())
            original=runner._write_json
            def corrupt_report(path, value):
                original(path, value)
                if path.name == "report.json": path.write_bytes(canonical_json_bytes({"schema":"tampered"}))
            tampered=base/"tampered"
            with patch.object(runner, "_write_json", side_effect=corrupt_report), self.assertRaises(ContractValidationError):
                runner._merge(self.args(tampered, shard_dir=shards), plan(), clips, ())
            self.assertTrue((tampered/MARKER_NAMES["FAILED"]).is_file()); self.assertFalse((tampered/MARKER_NAMES["COMPLETE"]).exists())


if __name__ == "__main__": unittest.main()
