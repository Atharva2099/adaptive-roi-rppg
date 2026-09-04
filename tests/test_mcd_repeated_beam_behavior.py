import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_mcd_repeated_beam_behavior import (
    COMPARISON_TOLERANCE_BPM, SENSITIVITY_WIDTHS, _anchor, _beam, _estimate, _expected_task_indices, _full_subjects_from_task_entries, _result_inventory, _source_identity, recovery_classification,
)
def factual(values=(1.0,) * 15):
    return {i: {"gt": 60, "belief": 60 + value, "error": float(value), "proposed": 3, "executed": 3}
            for i, value in enumerate(values)}
def branches(value=2.0, anchors=(0,)):
    return {(anchor, action): [float(value)] * 15 for anchor in anchors for action in range(12)}
def fixture(corrections=(0, 2), locked=()):
    hops, stats = [], []
    for width in SENSITIVITY_WIDTHS:
        factual_total = repeated_total = 0.0
        sequence = []
        for offset in range(15):
            legal = [3] if offset in locked else list(range(12))
            factual_total += 1
            base = {"anchor_hop_idx": 0, "width": width, "path_rank": 0, "offset": offset, "hop_idx": offset,
                    "source": "factual_ppo", "origin_sequence": "[]", "executed_sequence": "[]",
                    "ppo_action": 3, "legal_actions": json.dumps(legal), "ppo_legal": True, "rank1_action": -1,
                    "proposed_action": 3, "executed_action": 3, "legal": True, "override_reason": "",
                    "immediate_abs_error_bpm": 1, "cumulative_abs_error_bpm": factual_total, "endpoint_abs_error_bpm": 1}
            proposed = 1 if offset in corrections else 3
            sequence.append(proposed)
            repeated_total += .5
            retained = {**base, "source": "rank1_gt" if offset in corrections else "ppo_legal",
                        "origin_sequence": json.dumps(sequence), "executed_sequence": json.dumps(sequence), "rank1_action": 1,
                        "proposed_action": proposed, "executed_action": proposed, "immediate_abs_error_bpm": .5,
                        "cumulative_abs_error_bpm": repeated_total, "endpoint_abs_error_bpm": .5}
            hops.extend((base, retained))
            stats.append({"anchor_hop_idx": 0, "width": width, "offset": offset, "generated": 1,
                          "deduped": 0, "retained": 1, "pruned": 0,
                          "factual_cumulative_abs_error_bpm": factual_total, "factual_endpoint_abs_error_bpm": 1,
                          "best_cumulative_abs_error_bpm": repeated_total if offset == 14 else 999,
                          "best_endpoint_abs_error_bpm": .5 if offset == 14 else 999})
    return hops, stats
class BehaviorTests(unittest.TestCase):
    def test_seed_subset_task_mapping_and_inventory(self):
        self.assertEqual(_expected_task_indices(533, (1,)), tuple(range(1, 1599, 3)))
        self.assertEqual(len(_expected_task_indices(533, (1,))), 533)
        self.assertEqual(len(_expected_task_indices(533, (0, 1, 2))), 1599)
        with self.assertRaisesRegex(ValueError, "requested seed task mapping"):
            _expected_task_indices(533, ())
        with self.assertRaisesRegex(ValueError, "requested seed task mapping"):
            _expected_task_indices(533, (3,))

    def test_full_task_triplets_bind_clip_and_subject(self):
        entries = {3 * position + seed: {"clip_id": f"c{position}", "subject_id": f"s{position}"}
                   for position in range(2) for seed in (0, 1, 2)}
        self.assertEqual(_full_subjects_from_task_entries(entries, clip_count=2), {"c0": "s0", "c1": "s1"})
        entries[4]["subject_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "triplet identity"):
            _full_subjects_from_task_entries(entries, clip_count=2)

    def test_seed_one_inventory_rejects_missing_extra_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = [1, 4, 7]
            for task in expected:
                (root / f"mcd-repeated-beam-full-48580-{task}").mkdir()
            inventory = _result_inventory(root, "mcd-repeated-beam-full", ("48580",), expected)
            self.assertEqual(sorted(path.name for path in inventory.values()), [f"mcd-repeated-beam-full-48580-{task}" for task in expected])
            (root / "mcd-repeated-beam-full-48580-10").mkdir()
            with self.assertRaisesRegex(ValueError, "undeclared"):
                _result_inventory(root, "mcd-repeated-beam-full", ("48580",), expected)
            (root / "mcd-repeated-beam-full-48580-10").rmdir()
            (root / "mcd-repeated-beam-full-48580-4").rmdir()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _result_inventory(root, "mcd-repeated-beam-full", ("48580",), expected)
            (root / "mcd-repeated-beam-full-48580-4").mkdir()
            (root / "mcd-repeated-beam-full-48581-4").mkdir()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _result_inventory(root, "mcd-repeated-beam-full", ("48580", "48581"), expected)

    def test_analysis_launcher_forwards_default_and_seed_one(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_beam_behavior.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "python"
            args_file = root / "args"
            fake.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$@\" > '{args_file}'\n")
            fake.chmod(0o755)
            common = {"BASE": str(root), "RESULTS_ROOT": str(root), "OUT": str(root / "out"),
                      "SOURCE_SUBSETS_DIR": str(root), "JOB_ID": "48579", "PYTHON": str(fake),
                      "PATH": os.environ["PATH"]}
            (root / "out").mkdir()
            for value, expected in ((None, ["--seeds", "0", "1", "2"]), ("1", ["--seeds", "1"])):
                env = {**common, **({} if value is None else {"ANALYSIS_SEEDS": value})}
                self.assertEqual(subprocess.run(["bash", str(launcher)], env=env, capture_output=True).returncode, 0)
                args = args_file.read_text().splitlines()
                start = args.index("--seeds")
                self.assertEqual(args[start:], expected)
            for value in ("", "1,1", "3"):
                env = {**common, "ANALYSIS_SEEDS": value}
                self.assertNotEqual(subprocess.run(["bash", str(launcher)], env=env, capture_output=True).returncode, 0)
    def test_path_identity_and_repeated_threshold(self):
        for corrections, count in (((), 0), ((0,), 0), ((0, 2), 1)):
            hops, stats = fixture(corrections)
            self.assertEqual(_beam(hops, stats, factual())[0]["repeated_path_count"], count)
    def test_duplicate_offset_and_final_stat_rejected(self):
        hops, stats = fixture()
        hops[-1]["offset"] = 13
        with self.assertRaisesRegex(ValueError, "coverage"):
            _beam(hops, stats, factual())
        hops, stats = fixture()
        stats[-1]["best_cumulative_abs_error_bpm"] = 123
        with self.assertRaisesRegex(ValueError, "final beam best"):
            _beam(hops, stats, factual())
        hops, stats = fixture()
        stats[-1]["factual_endpoint_abs_error_bpm"] = 123
        with self.assertRaisesRegex(ValueError, "final factual endpoint"):
            _beam(hops, stats, factual())
    def test_illegal_ppo_forced_rank1_is_not_correction(self):
        hops, stats = fixture()
        for row in hops:
            if row["source"] == "rank1_gt" and row["offset"] == 2:
                row.update(legal_actions="[1]", ppo_legal=False)
        self.assertEqual(_beam(hops, stats, factual())[0]["repeated_path_count"], 0)
    def test_fixed_horizon_and_missing_coverage(self):
        hops, stats = fixture()
        beam = _beam(hops, stats, factual())[0]
        values = branches(20, range(15))
        values[(14, 0)] = [0] * 15
        self.assertEqual(_anchor(0, factual(), values, beam)["best_single_offset"], 14)
        hops, stats = fixture(locked=(14,))
        beam = _beam(hops, stats, factual())[0]
        values = branches(2, range(14))
        self.assertEqual(_anchor(0, factual(), values, beam)["analysis_eligible"], 1)
        beam["factual"][-1]["legal_actions"] = json.dumps(list(range(12)))
        self.assertEqual(_anchor(0, factual(), values, beam)["exclusion_reason"], "incomplete_single_coverage")
    def test_same_step_reversal_and_first_post_correction(self):
        hops, stats = fixture(locked=tuple(i for i in range(1, 15) if i != 2))
        for width in SENSITIVITY_WIDTHS:
            path = [row for row in hops if row["width"] == width and row["source"] != "factual_ppo"]
            for row in path[1:]:
                sequence = json.loads(row["origin_sequence"])
                sequence[1] = 4
                row["origin_sequence"] = row["executed_sequence"] = json.dumps(sequence)
            path[1].update(ppo_action=4, proposed_action=4, executed_action=4, legal_actions="[3,4]")
        beam = _beam(hops, stats, factual())[0]
        values = branches(20, (0, 2))
        values[(0, 0)] = [2] + [0] * 14
        row = _anchor(0, factual(), values, beam)
        self.assertEqual((row["same_step_myopic_reversal"], row["downstream_ppo_proposal_change_count"]), (1, 1))
        self.assertNotIn("ppo_immediate_better_than_best_single_action", row)
    def test_recovery_classes_and_tied_end(self):
        cases = {"better_throughout": [1, 2], "briefly_better_then_ends_worse": [1, 0, -1],
                 "only_ends_better_late": [0, -1, 1], "mixed_intermittent": [1, -1, 1],
                 "never_better": [0, -1]}
        for label, gains in cases.items():
            self.assertEqual(recovery_classification(gains), label)
        self.assertEqual(recovery_classification([1, 0, 0]), "mixed_intermittent")
        self.assertEqual(recovery_classification([1, -1, 0]), "mixed_intermittent")

    def test_comparison_tolerance_near_equal_recovery(self):
        self.assertEqual(COMPARISON_TOLERANCE_BPM, 1e-6)
        self.assertEqual(recovery_classification([1e-13, -1e-9]), "never_better")
        self.assertEqual(recovery_classification([1e-6, -1e-6]), "never_better")
        self.assertEqual(recovery_classification([1.0001e-6, 0, -1.0001e-6]), "briefly_better_then_ends_worse")
        self.assertEqual(recovery_classification([-1.0001e-6, 1.0001e-6]), "only_ends_better_late")

    def test_comparison_tolerance_same_step_single_and_repeated(self):
        hops, stats = fixture()
        beam = _beam(hops, stats, factual())[0]
        values = branches(2, range(15))
        values[(0, 0)] = [1 + 1e-9] + [0] * 14
        row = _anchor(0, factual(), values, beam)
        self.assertEqual((row["same_step_myopic_reversal"], row["positive_immediate_regret"]), (0, 0))
        values[(0, 0)] = [1 + 2e-6] + [0] * 14
        row = _anchor(0, factual(), values, beam)
        self.assertEqual((row["same_step_myopic_reversal"], row["positive_immediate_regret"]), (1, 0))
        values[(0, 0)] = [1 - 2e-6] + [2] * 14
        self.assertEqual(_anchor(0, factual(), values, beam)["positive_immediate_regret"], 1)

        values = branches(1, range(15))
        row = _anchor(0, factual(), values, beam)
        self.assertEqual(row["best_single_beats_factual"], 0)
        # A total difference above 1e-6 but below 15e-6 remains an H15-MAE tie.
        values = branches(1, range(15))
        values[(0, 0)] = [1 - 2e-6] + [1] * 14
        self.assertEqual(_anchor(0, factual(), values, beam)["best_single_beats_factual"], 0)
        values = branches(1 - 2e-6, range(15))
        row = _anchor(0, factual(), values, beam)
        self.assertEqual(row["best_single_beats_factual"], 1)

        for row in beam["repeated"]:
            row["immediate_abs_error_bpm"] = 1
        row = _anchor(0, factual(), branches(1, range(15)), beam)
        self.assertEqual(row["repeated_beats_best_single"], 0)
        for row in beam["repeated"]:
            row["immediate_abs_error_bpm"] = 1 - 2e-6
        row = _anchor(0, factual(), branches(1, range(15)), beam)
        self.assertEqual(row["repeated_beats_best_single"], 1)

    def test_h15_tolerance_is_on_mae_scale_for_reversal_and_repeated(self):
        hops, stats = fixture()
        beam = _beam(hops, stats, factual())[0]
        values = branches(1, range(15))
        values[(0, 0)] = [1 + 2e-6] + [1] * 14
        self.assertEqual(_anchor(0, factual(), values, beam)["same_step_myopic_reversal"], 0)
        for row in beam["repeated"]:
            row["immediate_abs_error_bpm"] = 1 - 2e-6 / 15
        self.assertEqual(_anchor(0, factual(), branches(1, range(15)), beam)["repeated_beats_best_single"], 0)
    def test_known_answer_anchor_clip_subject_aggregation(self):
        rows = [{"subject_id": "a", "clip_id": "c1", "v": value} for value in (0, 2)]
        rows += [{"subject_id": "a", "clip_id": "c2", "v": 3}, {"subject_id": "b", "clip_id": "c3", "v": 6}]
        self.assertEqual(_estimate(rows, "v")["point"], 4)
    def test_source_model_identity_rejects_family_and_checkpoint_swaps(self):
        row = {"method_id": "advantage_ppo_seed0", "family": "advantage_ppo", "checkpoint_sha256": "a" * 64}
        self.assertTrue(_source_identity([row], 0, "a" * 64))
        self.assertFalse(_source_identity([{**row, "family": "other"}], 0, "a" * 64))
        self.assertFalse(_source_identity([row], 0, "b" * 64))


if __name__ == "__main__":
    unittest.main()
