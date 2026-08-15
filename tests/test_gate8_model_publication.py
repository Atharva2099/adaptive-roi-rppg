import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.model_publication import (
    HOP_FIELDS, MARKER_NAMES, artifact_map, canonical_subject_shard, csv_bytes, expected_shard_clip_ids,
    marker_payload, read_strict_csv, validate_directory, validate_hop_rows,
    validate_marker, write_marker,
)
from adaptive_roi_rppg.evaluation.model_replay import summarize_model_rows


def provenance():
    return {"plan_id":"plan", "code_snapshot_sha256":"a" * 64,
            "source_inventory_sha256":"b" * 64, "environment_sha256":"c" * 64,
            "job_id":"1", "node":"node", "command":"command"}


def hop(clip="c", idx=0):
    row = {key: "x" for key in HOP_FIELDS}
    row.update(subject_id="s", view="front", condition="before", dataset_id="mcd", clip_id=clip,
               hop_idx=idx, hop_time_s=float(idx), method_id="dagger_seed0", family="dagger", seed=0,
               checkpoint_sha256="d" * 64, observation_schema_id="obs", observation_sha256="e" * 64,
               frame_provenance_id="frame", signal_config_id="sig", control_config_id="control", gt_rule_id="gt",
               gt_hr_bpm=70.0, abs_error_bpm=1.0, proposed_action=1, executed_action=1, previous_action=None,
               legal=True, override_reason=None, pre_hold_count=0, post_hold_count=1, selected_valid=True,
               selected_invalid_reason=None, selected_hr_bpm=70.0, selected_confidence=1.0, selected_ppr=1.0,
               selected_coverage=1.0, pre_belief_hr_bpm=70.0, post_belief_hr_bpm=70.0,
               post_belief_velocity=0.0, post_belief_std_bpm=1.0)
    return row


class Gate8PublicationTests(unittest.TestCase):
    def test_canonical_stride_is_deterministic(self):
        self.assertEqual(canonical_subject_shard(["s3", "s1", "s2"], 1, 2), ("s2",))
        with self.assertRaises(ContractValidationError): canonical_subject_shard(["s1", "s1"], 0, 2)
        clips=[{"subject_id":"s1","clip_id":"c2"},{"subject_id":"s1","clip_id":"c1"},{"subject_id":"s2","clip_id":"c3"}]
        self.assertEqual(expected_shard_clip_ids(clips, 0, 2), ("c1","c2"))

    def test_strict_hop_csv_rejects_reordered_and_tampered_hops(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_hop.csv"; path.write_bytes(csv_bytes([hop("a", 0), hop("a", 1)], HOP_FIELDS))
            rows = read_strict_csv(path, HOP_FIELDS); validate_hop_rows(rows, ["a"], {"a": 2})
            bad = [hop("a", 0), hop("a", 2)]
            with self.assertRaises(ContractValidationError): validate_hop_rows(bad)
            path.write_text(",".join(reversed(HOP_FIELDS)) + "\n")
            with self.assertRaises(ContractValidationError): read_strict_csv(path, HOP_FIELDS)

    def test_invalid_selected_measurement_round_trips_with_blank_confidence_and_ppr(self):
        row = hop(); row.update(selected_valid=False, selected_invalid_reason="missing_required_rgb", selected_hr_bpm=None, selected_confidence=None, selected_ppr=None, selected_coverage=0.3478)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_hop.csv"; path.write_bytes(csv_bytes([row], HOP_FIELDS))
            self.assertEqual(read_strict_csv(path, HOP_FIELDS), [row])

    def test_selected_confidence_and_ppr_remain_strict_numeric_fields(self):
        with TemporaryDirectory() as tmp:
            for field in ("selected_confidence", "selected_ppr"):
                row = hop(); row[field] = "not-a-number"
                path = Path(tmp) / f"{field}.csv"; path.write_bytes(csv_bytes([row], HOP_FIELDS))
                with self.assertRaises(ContractValidationError): read_strict_csv(path, HOP_FIELDS)
            row = hop(); row["selected_confidence"] = None
            path = Path(tmp) / "valid_blank.csv"; path.write_bytes(csv_bytes([row], HOP_FIELDS))
            with self.assertRaises(ContractValidationError): read_strict_csv(path, HOP_FIELDS)

    def test_markers_require_exact_lifecycle_provenance_and_outputs(self):
        p = provenance()
        started = marker_payload("STARTED", "run", p)
        validate_marker(started, expected_kind="run", expected_provenance=p)
        bad = dict(started); bad["extra"] = True
        with self.assertRaises(ContractValidationError): validate_marker(bad, expected_kind="run")
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "result"; root.mkdir(); write_marker(root, started)
            (root / "per_hop.csv").write_bytes(b"rows\n")
            complete = marker_payload("COMPLETE", "run", p, outputs=artifact_map(root, ["per_hop.csv"]))
            write_marker(root, complete)
            validate_directory(root, {"STARTED.json", "COMPLETE.json", "per_hop.csv"}, kind="run", provenance=p, complete_outputs=["per_hop.csv"])
            (root / "FAILED.json").write_text(json.dumps(marker_payload("FAILED", "run", p)))
            with self.assertRaises(ContractValidationError): validate_directory(root, {"STARTED.json", "FAILED.json", "COMPLETE.json", "per_hop.csv"}, kind="run", provenance=p, complete_outputs=["per_hop.csv"])

    def test_complete_hash_does_not_authorize_content_change(self):
        p = provenance()
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "result"; root.mkdir(); write_marker(root, marker_payload("STARTED", "run", p))
            target = root / "per_hop.csv"; target.write_bytes(b"original\n")
            write_marker(root, marker_payload("COMPLETE", "run", p, outputs=artifact_map(root, ["per_hop.csv"])))
            target.write_bytes(b"changed\n")
            with self.assertRaises(ContractValidationError): validate_directory(root, {"STARTED.json", "COMPLETE.json", "per_hop.csv"}, kind="run", provenance=p, complete_outputs=["per_hop.csv"])

    def test_full_face_requires_null_model_identity(self):
        row = hop(); row.update(method_id="full_face", family="fixed_full_face", seed=None, checkpoint_sha256=None)
        validate_hop_rows([row])
        row["seed"] = 0
        with self.assertRaises(ContractValidationError): validate_hop_rows([row])

    def test_aggregation_rejects_conflicting_checkpoint_identity(self):
        first, second = hop("c", 0), hop("c", 1)
        second["checkpoint_sha256"] = "f" * 64
        with self.assertRaises(ContractValidationError): summarize_model_rows([first, second])


if __name__ == "__main__": unittest.main()
