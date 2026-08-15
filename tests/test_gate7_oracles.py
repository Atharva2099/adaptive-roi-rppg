import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.oracles import ORACLE_SCHEMA, aggregate_report, legal_actions, replay_action_sequence, run_oracle_clip, validate_aggregation
from adaptive_roi_rppg.labels.mcd import GT_RULE_ID
from adaptive_roi_rppg.signal import POS_CONFIG_ID
from scripts.verify_gate7_mcd_oracles import (TEACHER_EXCLUSION_CONTRACT, _expected_hop, _merge, _plan,
                                               _run, _same_semantics, _validate_complete_marker,
                                               _validate_payload, _validate_rows, _write_json_exclusive)


def _frame(hop, values):
    measurements = tuple(ROIMeasurement(i, ROI_NAMES[i], float(value), 1.0, 0.5, 1.0, True, None, (), None, 0, 10, POS_CONFIG_ID) for i, value in enumerate(values))
    return MeasurementFrame("mcd", "clip", hop, 8.0 + hop, measurements, POS_CONFIG_ID, True, None, f"frame-{hop}")


def _label(hop, gt):
    return LabelFrame("mcd", "clip", hop, 8.0 + hop, float(gt), GT_RULE_ID, True, None)


def _declared_invalid_labels(clip_id, count=171, *, reason="degenerate_variation", valid=False):
    return tuple(LabelFrame("mcd", clip_id, hop, 8.0 + hop, 60.0 if valid else None,
                            GT_RULE_ID, valid, None if valid else reason)
                 for hop in range(count))


def _plan_fixture(*, missing_exclusion=False, eval_clips_per_subject=6):
    train_clips = []
    subjects = ["7412"] + [f"subject-{index:03d}" for index in range(1, 510)]
    exclusion_ids = [record.clip_id for record in TEACHER_EXCLUSION_CONTRACT]
    for subject_index, subject_id in enumerate(subjects):
        for clip_index in range(6):
            clip_id = f"{subject_id}_clip_{clip_index}"
            if subject_index == 0 and clip_index < len(exclusion_ids) and not (missing_exclusion and clip_index == 0):
                clip_id = exclusion_ids[clip_index]
            train_clips.append(SimpleNamespace(clip_id=clip_id, subject_id=subject_id, view="Frontal",
                                                condition="after", camera_fps=30.0,
                                                clip_manifest_id=f"manifest-{clip_id}", state_sha256="a" * 64,
                                                gt_sha256="b" * 64, state_row_count=300))
    eval_clips = [SimpleNamespace(clip_id=f"eval-{subject_index}-{clip_index}", subject_id=f"eval-subject-{subject_index}")
                  for subject_index in range(90) for clip_index in range(eval_clips_per_subject)]
    return SimpleNamespace(
        split_manifest=SimpleNamespace(dataset_id="mcd", train_clip_ids=tuple(clip.clip_id for clip in train_clips),
                                       eval_clip_ids=tuple(clip.clip_id for clip in eval_clips), split_id="split"),
        dataset_manifest=SimpleNamespace(manifest_id="dataset"), clip_manifests=tuple(train_clips + eval_clips))


class Gate7OracleTests(unittest.TestCase):
    def test_declared_exclusion_contract_is_ordered_and_immutable(self):
        self.assertEqual(
            [(record.clip_id, record.expected_invalid_label_count, record.invalid_reason)
             for record in TEACHER_EXCLUSION_CONTRACT],
            [("7412_FullHDwebcam_after", 171, "degenerate_variation"),
             ("7412_IriunWebcam_after", 171, "degenerate_variation"),
             ("7412_USBVideo_after", 171, "degenerate_variation")],
        )
        with self.assertRaises(AttributeError):
            TEACHER_EXCLUSION_CONTRACT[0].clip_id = "other"

    def test_plan_freezes_manifest_and_eligible_teacher_provenance(self):
        bundle = _plan_fixture()
        args = SimpleNamespace(manifest_tree="unused", gt_root="unused", beam_width=8)
        with patch("scripts.verify_gate7_mcd_oracles.load_mcd_manifest_tree", return_value=bundle), \
             patch("scripts.verify_gate7_mcd_oracles.sha256_file", return_value="a" * 64), \
             patch("scripts.verify_gate7_mcd_oracles.read_mcd_labels", side_effect=lambda _bundle, _root, clip_id, *_: _declared_invalid_labels(clip_id)):
            plan = _plan(args)
        self.assertEqual(len(plan.bindings), 3057)
        self.assertEqual(len(plan.subjects), 510)
        self.assertEqual(plan.payload["schema"], "gate7-mcd-oracle-plan-v3")
        self.assertEqual(plan.payload["manifest_counts"], {"train_subject_count": 510, "train_clip_count": 3060,
                                                             "eval_clip_count": 540, "eval_subject_count": 90})
        self.assertEqual(plan.payload["eligible_teacher_counts"], {"train_subject_count": 510, "train_clip_count": 3057})
        self.assertEqual(plan.payload["teacher_exclusions"], [dict(clip_id=record.clip_id,
                         expected_invalid_label_count=171, invalid_reason="degenerate_variation")
                         for record in TEACHER_EXCLUSION_CONTRACT])

        old_payload = dict(plan.payload, schema="gate7-mcd-oracle-plan-v2")
        common = {"phase": "full", "code_snapshot_sha256": "c" * 64,
                  "source_inventory_sha256": "a" * 64, "signal_config_id": POS_CONFIG_ID,
                  "control_config_id": "control", "gt_rule_id": GT_RULE_ID, "beam_width": 8,
                  "shard_count": 1}
        with self.assertRaisesRegex(ContractValidationError, "plan provenance mismatch"):
            _validate_payload({"schema": ORACLE_SCHEMA, "plan_id": plan.plan_id,
                               "plan_payload": old_payload}, common, plan)

    def test_plan_rejects_declared_exclusion_label_contract_drift(self):
        cases = {
            "wrong count": lambda clip_id: _declared_invalid_labels(clip_id, count=170),
            "wrong reason": lambda clip_id: _declared_invalid_labels(clip_id, reason="degenerate_spectrum"),
            "valid label": lambda clip_id: _declared_invalid_labels(clip_id, valid=True),
        }
        args = SimpleNamespace(manifest_tree="unused", gt_root="unused", beam_width=8)
        for label, labels_for_exclusion in cases.items():
            with self.subTest(label=label), \
                 patch("scripts.verify_gate7_mcd_oracles.load_mcd_manifest_tree", return_value=_plan_fixture()), \
                 patch("scripts.verify_gate7_mcd_oracles.sha256_file", return_value="a" * 64), \
                 patch("scripts.verify_gate7_mcd_oracles.read_mcd_labels", side_effect=lambda _bundle, _root, clip_id, *_: labels_for_exclusion(clip_id)):
                with self.assertRaises(ContractValidationError):
                    _plan(args)

    def test_plan_rejects_missing_declared_exclusion_id(self):
        args = SimpleNamespace(manifest_tree="unused", gt_root="unused", beam_width=8)
        with patch("scripts.verify_gate7_mcd_oracles.load_mcd_manifest_tree", return_value=_plan_fixture(missing_exclusion=True)), \
             patch("scripts.verify_gate7_mcd_oracles.sha256_file", return_value="a" * 64):
            with self.assertRaisesRegex(ContractValidationError, "not a train clip"):
                _plan(args)

    def test_plan_rejects_ninety_eval_clips_across_ninety_subjects(self):
        args = SimpleNamespace(manifest_tree="unused", gt_root="unused", beam_width=8)
        with patch("scripts.verify_gate7_mcd_oracles.load_mcd_manifest_tree", return_value=_plan_fixture(eval_clips_per_subject=1)), \
             patch("scripts.verify_gate7_mcd_oracles.sha256_file", return_value="a" * 64):
            with self.assertRaisesRegex(ContractValidationError, "540 held-out eval clips"):
                _plan(args)

    def test_known_answer_one_step_greedy_and_beam(self):
        frames = (_frame(0, [60.0] * 12),)
        labels = (_label(0, 60.0),)
        greedy, beam = run_oracle_clip(frames, labels, clip_id="clip", subject_id="s", view="Frontal", condition="before", beam_width=8)
        self.assertEqual(greedy[0]["executed_action"], 0)
        self.assertEqual(beam[0]["executed_action"], 0)
        self.assertGreater(greedy[0]["abs_error_bpm"], 0.0)
        self.assertLess(greedy[0]["abs_error_bpm"], 10.0)

    def test_minimum_hold_restricts_teacher_actions(self):
        frames = (_frame(0, [60.0] * 12), _frame(1, [70.0] * 12))
        labels = (_label(0, 60.0), _label(1, 70.0))
        greedy, _ = run_oracle_clip(frames, labels, clip_id="clip", subject_id="s", view="Frontal", condition="before", beam_width=8)
        self.assertEqual([row["executed_action"] for row in greedy], [0, 0])

    def test_b_is_prefix_causal_and_uses_current_gt(self):
        frames = (_frame(0, [60.0] * 12), _frame(1, [90.0] + [60.0] * 11))
        prefix, _ = run_oracle_clip(frames[:1], (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        full, _ = run_oracle_clip(frames, (_label(0, 60.0), _label(1, 90.0)), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        self.assertEqual(prefix[0], full[0])
        gt_sensitive = (_frame(0, [45, 55, 65, 75, 90, 110, 50, 60, 70, 80, 100, 120]),)
        low, _ = run_oracle_clip(gt_sensitive, (_label(0, 45.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        high, _ = run_oracle_clip(gt_sensitive, (_label(0, 100.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        self.assertNotEqual(low[0]["proposed_action"], high[0]["proposed_action"])

    def test_c_first_choice_can_depend_on_future_gt(self):
        values = (
            [55, 90, 110, 110, 90, 55, 45, 90, 55, 75, 75, 55],
            [55, 75, 75, 75, 75, 45, 65, 75, 55, 55, 90, 65],
            [75, 65, 90, 110, 55, 55, 90, 75, 90, 65, 90, 55],
            [75, 45, 110, 45, 55, 110, 75, 90, 55, 90, 45, 65],
        )
        frames = tuple(_frame(hop, row) for hop, row in enumerate(values))
        _, baseline = run_oracle_clip(frames, tuple(_label(hop, 60.0) for hop in range(4)), clip_id="clip", subject_id="s", view="Frontal", condition="before", beam_width=32)
        _, changed = run_oracle_clip(frames, tuple(_label(hop, value) for hop, value in enumerate((60.0, 90.0, 45.0, 120.0))), clip_id="clip", subject_id="s", view="Frontal", condition="before", beam_width=32)
        self.assertNotEqual(baseline[0]["proposed_action"], changed[0]["proposed_action"])

    def test_ties_use_lowest_canonical_action(self):
        frames = (_frame(0, [60.0] * 12),)
        greedy, beam = run_oracle_clip(frames, (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        self.assertEqual(greedy[0]["proposed_action"], 0)
        self.assertEqual(beam[0]["proposed_action"], 0)

    def test_replay_rejects_changed_post_belief(self):
        frames = (_frame(0, [60.0] * 12),)
        labels = (_label(0, 60.0),)
        _, beam = run_oracle_clip(frames, labels, clip_id="clip", subject_id="s", view="Frontal", condition="before")
        tampered = [dict(beam[0], post_belief_hr_bpm=beam[0]["post_belief_hr_bpm"] + 1.0)]
        with self.assertRaises(ContractValidationError):
            replay_action_sequence(frames, labels, tampered, clip_id="clip")

    def test_invalid_label_identity_and_rule_are_rejected(self):
        frames = (_frame(0, [60.0] * 12),)
        with self.assertRaises(ContractValidationError):
            run_oracle_clip(frames, (LabelFrame("mcd", "other", 0, 8.0, 60.0, GT_RULE_ID, True, None),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        with self.assertRaises(ContractValidationError):
            run_oracle_clip(frames, (LabelFrame("mcd", "clip", 0, 8.0, 60.0, "wrong-rule", True, None),), clip_id="clip", subject_id="s", view="Frontal", condition="before")

    def test_row_validation_helper_rejects_duplicate_coverage(self):
        frames = (_frame(0, [60.0] * 12),)
        rows, _ = run_oracle_clip(frames, (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        plan = {"clips": [{"clip_id": "clip", "subject_id": "s", "view": "Frontal", "condition": "before", "camera_fps": 30.0, "clip_manifest_id": "manifest", "expected_hops": 1}]}
        rows[0]["frame_provenance_id"] = _expected_hop(plan["clips"][0], 0)[1]
        _validate_rows(rows, plan, "greedy_b")
        with self.assertRaises(ContractValidationError):
            _validate_rows(rows + rows, plan, "greedy_b")

    def test_row_validation_rejects_wrong_hop_action_nan_belief_and_false_error(self):
        frames = (_frame(0, [60.0] * 12),)
        rows, _ = run_oracle_clip(frames, (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        plan = {"clips": [{"clip_id": "clip", "subject_id": "s", "view": "Frontal", "condition": "before", "camera_fps": 30.0, "clip_manifest_id": "manifest", "expected_hops": 1}]}
        rows[0]["frame_provenance_id"] = _expected_hop(plan["clips"][0], 0)[1]
        for field, value in (("hop_idx", 1), ("proposed_action", 12), ("pre_belief_hr_bpm", float("nan")), ("abs_error_bpm", rows[0]["abs_error_bpm"] + 1.0)):
            tampered = [dict(rows[0], **{field: value})]
            with self.assertRaises(ContractValidationError):
                _validate_rows(tampered, plan, "greedy_b")

    def test_complete_marker_rejects_partial_or_mixed_provenance(self):
        common = {"plan_id": "plan", "phase": "full", "code_snapshot_sha256": "a" * 64, "source_inventory_sha256": "b" * 64,
                  "signal_config_id": POS_CONFIG_ID, "control_config_id": "control", "gt_rule_id": GT_RULE_ID,
                  "beam_width": 8, "job_id": "job", "node": "node", "command": "command"}
        valid = {"state": "complete", **common, "sha256": "c" * 64}
        _validate_complete_marker(valid, common, "c" * 64)
        for tampered in (dict(valid, code_snapshot_sha256="d" * 64), {"state": "complete", "sha256": "c" * 64}):
            with self.assertRaises(ContractValidationError):
                _validate_complete_marker(tampered, common, "c" * 64)

    def test_aggregation_and_source_recomputation_mismatch_are_rejected(self):
        frames = (_frame(0, [60.0] * 12),)
        b, c = run_oracle_clip(frames, (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        rows = {"greedy_b": b, "beam_c": c}
        aggregation = aggregate_report(rows)
        validate_aggregation(rows, aggregation)
        with self.assertRaises(ContractValidationError):
            validate_aggregation(rows, {**aggregation, "comparison": {}})
        with self.assertRaises(ContractValidationError):
            _same_semantics(rows, {"greedy_b": [dict(b[0], gt_hr_bpm=61.0)], "beam_c": c}, "source vs shard")

    def test_complete_file_is_exclusive_and_launcher_isolated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "COMPLETE.json"
            _write_json_exclusive(path, {"state": "complete"})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(path, {"state": "complete"})
        launcher = Path("slurm/gate7_oracles_parallel_full.slurm").read_text(encoding="utf-8")
        self.assertNotIn("git ", launcher.lower())
        self.assertNotIn("/usr/bin/git", launcher)
        self.assertIn('unset PYTHONPATH', launcher)
        self.assertIn('excluded_dirs = {".git", ".venv", "__pycache__"}', launcher)
        self.assertIn('entry.suffix != ".pyc"', launcher)
        self.assertIn('entry.is_symlink()', launcher)
        self.assertIn('unexpected symlink in BASE', launcher)
        self.assertIn('unexpected non-regular entry in BASE', launcher)
        self.assertIn('key=lambda path: path.as_posix().encode("utf-8")', launcher)
        self.assertIn('h.update(path.as_posix().encode("utf-8")); h.update(b"\\0"); h.update(data); h.update(b"\\0")', launcher)
        self.assertIn('export PYTHONPATH="$BASE/src"', launcher)
        self.assertIn("export PYTHONNOUSERSITE=1", launcher)

    def test_run_writes_complete_last_and_failure_never_writes_complete(self):
        frames = (_frame(0, [60.0] * 12),)
        b, c = run_oracle_clip(frames, (_label(0, 60.0),), clip_id="clip", subject_id="s", view="Frontal", condition="before")
        clip = {"clip_id": "clip", "subject_id": "s", "view": "Frontal", "condition": "before", "camera_fps": 30.0, "clip_manifest_id": "manifest", "expected_hops": 1}
        for row in b + c:
            row["frame_provenance_id"] = _expected_hop(clip, 0)[1]
        result = {"schema": "gate7-mcd-oracle-shard-v2", "beam_width": 8,
                  "rows": {"greedy_b": b, "beam_c": c},
                  "summary": aggregate_report({"greedy_b": b, "beam_c": c})["methods"]}
        plan_payload = {"clips": [clip], "source_inventory_sha256": "b" * 64}
        plan = SimpleNamespace(plan_id="plan", bindings=(SimpleNamespace(clip_id="clip", subject_id="s"),), subjects=["s"], payload=plan_payload)
        args_base = dict(shard_count=1, shard_index=0, code_snapshot_sha256="a" * 64,
                         beam_width=8, manifest_tree="unused", state_root="unused", gt_root="unused")
        old_job, old_node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
        os.environ["SLURM_JOB_ID"], os.environ["SLURMD_NODENAME"] = "job", "node"
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory) / "ok"; calls = []
                import scripts.verify_gate7_mcd_oracles as gate7
                original = gate7._write_json_exclusive
                def recorded(path, payload):
                    calls.append(Path(path).name); return original(path, payload)
                with patch("scripts.verify_gate7_mcd_oracles._plan", return_value=plan), patch("scripts.verify_gate7_mcd_oracles._load_and_evaluate_bindings", return_value=result), patch("scripts.verify_gate7_mcd_oracles._write_json_exclusive", side_effect=recorded):
                    _run(SimpleNamespace(output_dir=str(root), **args_base))
                self.assertEqual(calls[-1], "COMPLETE.json")
                self.assertTrue((root / "COMPLETE.json").exists())
            with TemporaryDirectory() as directory:
                root = Path(directory) / "failed"
                with patch("scripts.verify_gate7_mcd_oracles._plan", return_value=plan), patch("scripts.verify_gate7_mcd_oracles._load_and_evaluate_bindings", side_effect=RuntimeError("synthetic failure")):
                    with self.assertRaises(RuntimeError):
                        _run(SimpleNamespace(output_dir=str(root), **args_base))
                self.assertTrue((root / "FAILED.json").exists())
                self.assertFalse((root / "COMPLETE.json").exists())
        finally:
            if old_job is None: os.environ.pop("SLURM_JOB_ID", None)
            else: os.environ["SLURM_JOB_ID"] = old_job
            if old_node is None: os.environ.pop("SLURMD_NODENAME", None)
            else: os.environ["SLURMD_NODENAME"] = old_node

    def test_run_marker_write_failure_preserves_original_error(self):
        plan = SimpleNamespace(plan_id="plan", bindings=(), subjects=[], payload={"source_inventory_sha256": "b" * 64})
        args = SimpleNamespace(output_dir=None, shard_count=1, shard_index=0, code_snapshot_sha256="a" * 64,
                               beam_width=8, manifest_tree="unused", state_root="unused", gt_root="unused")
        old_job, old_node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
        os.environ["SLURM_JOB_ID"], os.environ["SLURMD_NODENAME"] = "job", "node"
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory) / "failed"; args.output_dir = str(root)
                import scripts.verify_gate7_mcd_oracles as gate7
                original_marker = gate7._write_marker

                def fail_failed_marker(root, state, common, **extra):
                    if state == "failed":
                        raise OSError("marker failure")
                    return original_marker(root, state, common, **extra)

                class BrokenStderr:
                    def write(self, _):
                        raise OSError("stderr failure")

                original_error = RuntimeError("original run failure")
                with patch("scripts.verify_gate7_mcd_oracles._plan", return_value=plan), patch("scripts.verify_gate7_mcd_oracles._load_and_evaluate_bindings", side_effect=original_error), patch("scripts.verify_gate7_mcd_oracles._write_marker", side_effect=fail_failed_marker), patch("scripts.verify_gate7_mcd_oracles.sys.stderr", BrokenStderr()):
                    try:
                        _run(args)
                    except RuntimeError as caught:
                        self.assertIs(caught, original_error)
                        self.assertIsNotNone(caught.__traceback__)
                    else:
                        self.fail("original run failure did not escape")
                self.assertFalse((root / "COMPLETE.json").exists())
        finally:
            if old_job is None: os.environ.pop("SLURM_JOB_ID", None)
            else: os.environ["SLURM_JOB_ID"] = old_job
            if old_node is None: os.environ.pop("SLURMD_NODENAME", None)
            else: os.environ["SLURMD_NODENAME"] = old_node

    def test_merge_marker_write_failure_preserves_original_error_and_never_completes(self):
        plan = SimpleNamespace(plan_id="plan", bindings=(), subjects=[], payload={"source_inventory_sha256": "b" * 64})
        old_job, old_node = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURMD_NODENAME")
        os.environ["SLURM_JOB_ID"], os.environ["SLURMD_NODENAME"] = "job", "node"
        try:
            with TemporaryDirectory() as directory:
                root, shard = Path(directory) / "failed", Path(directory) / "empty-shard"
                shard.mkdir()
                args = SimpleNamespace(output_dir=str(root), shard_dir=[str(shard)], shard_count=1, merge_workers=1,
                                       code_snapshot_sha256="a" * 64, beam_width=8, manifest_tree="unused", state_root="unused", gt_root="unused")
                import scripts.verify_gate7_mcd_oracles as gate7
                original_marker = gate7._write_marker

                def fail_failed_marker(root, state, common, **extra):
                    if state == "failed":
                        raise OSError("marker failure")
                    return original_marker(root, state, common, **extra)

                class BrokenStderr:
                    def write(self, _):
                        raise OSError("stderr failure")

                original_error = ContractValidationError("oracle shard marker/file state is invalid")
                with patch("scripts.verify_gate7_mcd_oracles._plan", return_value=plan), patch("scripts.verify_gate7_mcd_oracles._fail", side_effect=original_error), patch("scripts.verify_gate7_mcd_oracles._write_marker", side_effect=fail_failed_marker), patch("scripts.verify_gate7_mcd_oracles.sys.stderr", BrokenStderr()):
                    try:
                        _merge(args)
                    except ContractValidationError as caught:
                        self.assertIs(caught, original_error)
                        self.assertIsNotNone(caught.__traceback__)
                    else:
                        self.fail("original merge failure did not escape")
                self.assertFalse((root / "COMPLETE.json").exists())
        finally:
            if old_job is None: os.environ.pop("SLURM_JOB_ID", None)
            else: os.environ["SLURM_JOB_ID"] = old_job
            if old_node is None: os.environ.pop("SLURMD_NODENAME", None)
            else: os.environ["SLURMD_NODENAME"] = old_node


if __name__ == "__main__":
    unittest.main()
