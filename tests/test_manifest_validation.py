import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_roi_rppg.contracts.constants import ManifestStatus, OverlapResult
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.contracts.io import (
    canonical_json_bytes,
    read_json_object,
    sha256_file,
    verify_file_sha256,
    write_json_atomic,
)
from adaptive_roi_rppg.contracts.manifests import (
    ClipManifest,
    DatasetManifest,
    RunInputManifest,
    SplitManifest,
    require_structurally_complete,
    validate_manifest_transition,
    validate_status_transition,
)

HASH = "a" * 64
UTC = "2026-08-11T00:00:00Z"


def clip(status=ManifestStatus.complete):
    return ClipManifest("d", "c", "cm", "subject", "view", "condition", "camera", 30.0, "state", HASH, 2, None, None, None, "split", "schema", status)


def split(status=ManifestStatus.complete):
    return SplitManifest("split", "d", ("s1",), ("s2",), ("c1",), ("c2",), 1, 1, 1, 1, OverlapResult.zero, (HASH,), status)


def dataset(status=ManifestStatus.complete, cohort=None):
    return DatasetManifest("manifest", "d", "schema", ("cm",), (HASH,), cohort or {"nested": {"value": "ok"}}, 1, 1, HASH, ("transform",), UTC, "producer", status)


def run_input(status=ManifestStatus.complete):
    return RunInputManifest("run", "manifest", HASH, "split", HASH, (("d", "c"),), "signal", "obs", None, ("transform",), 0, "commit", HASH, UTC, status)


class ManifestTests(unittest.TestCase):
    def test_round_trips_and_optional_gt_triplet(self):
        for value in (clip(), split(), dataset(), run_input()):
            self.assertEqual(type(value).from_dict(value.to_dict()), value)
        with self.assertRaises(ContractValidationError):
            ClipManifest("d", "c", "cm", "s", "v", "x", "camera", 30.0, "state", HASH, 1, "gt", None, 1, "split", "schema", ManifestStatus.planned)

    def test_hash_timestamp_camera_and_gate1_boundaries(self):
        with self.assertRaises(ContractValidationError):
            clip().__class__("d", "c", "cm", "s", "v", "x", "camera", 30.0, "state", "A" * 64, 1, None, None, None, "split", "schema", ManifestStatus.planned)
        with self.assertRaises(ContractValidationError):
            dataset().__class__("m", "d", "s", ("cm",), (HASH,), {}, 1, 1, HASH, (), "2026-08-11T00:00:00+01:00Z", "p", ManifestStatus.planned)
        self.assertEqual(clip().camera_id, "camera")
        self.assertNotIn("MCD", clip().__class__.__module__)
        for malformed in ("20260811T000000Z", "2026-W33-2T00:00:00Z", "2026-08-11 00:00:00Z"):
            with self.assertRaises(ContractValidationError):
                DatasetManifest("m", "d", "s", ("cm",), (HASH,), {}, 1, 1, HASH, (), malformed, "p", ManifestStatus.planned)

    def test_split_validation_and_usability(self):
        self.assertEqual(split().require_usable(), split())
        with self.assertRaises(ContractValidationError):
            SplitManifest("split", "d", ("s1",), ("s1",), ("c1",), ("c2",), 1, 1, 1, 1, OverlapResult.overlap, (HASH,), ManifestStatus.complete)
        with self.assertRaises(ContractValidationError):
            SplitManifest("split", "d", ("s1",), ("s2",), ("c1",), ("c2",), 2, 1, 1, 1, OverlapResult.zero, (HASH,), ManifestStatus.complete)
        with self.assertRaises(ContractValidationError):
            split(ManifestStatus.complete).__class__("split", "d", ("s1",), ("s2",), ("c1",), ("c2",), 1, 1, 1, 1, OverlapResult.zero, (HASH,), ManifestStatus.failed).require_usable()
        with self.assertRaises(ContractValidationError):
            SplitManifest("split", "d", ("s1",), ("s2",), ("c1",), ("c2",), 1, 1, 1, 1, OverlapResult.zero, None, ManifestStatus.planned)
        with self.assertRaises(ContractValidationError):
            SplitManifest("split", "d", ("s1",), ("s2",), ("c1",), ("c2",), 1, 1, 1, 1, OverlapResult.zero, HASH, ManifestStatus.planned)

    def test_dataset_nested_defensive_copy_and_counts(self):
        cohort = {"nested": {"values": ["a"]}}
        value = dataset(cohort=cohort)
        cohort["nested"]["values"].append("changed")
        self.assertEqual(value.to_dict()["cohort"]["nested"]["values"], ["a"])
        output = value.to_dict()
        output["cohort"]["nested"]["values"].append("other")
        self.assertEqual(value.to_dict()["cohort"]["nested"]["values"], ["a"])
        with self.assertRaises(ContractValidationError):
            DatasetManifest("m", "d", "s", ("cm",), (), {}, 1, 1, HASH, (), UTC, "p", ManifestStatus.planned)
        with self.assertRaises(ContractValidationError):
            DatasetManifest("m", "d", "s", ("cm",), None, {}, 1, 1, HASH, (), UTC, "p", ManifestStatus.planned)
        with self.assertRaises(ContractValidationError):
            DatasetManifest("m", "d", "s", ("cm",), HASH, {}, 1, 1, HASH, (), UTC, "p", ManifestStatus.planned)

    def test_run_input_keys_transforms_and_no_dataset_class(self):
        value = run_input()
        self.assertFalse(hasattr(value, "allowed_dataset_class"))
        with self.assertRaises(ContractValidationError):
            RunInputManifest("run", "manifest", HASH, "split", HASH, (("d", "c"), ("d", "c")), "signal", "obs", None, ("t",), None, "commit", HASH, UTC, ManifestStatus.planned)
        with self.assertRaises(ContractValidationError):
            RunInputManifest("run", "manifest", HASH, "split", HASH, (("d", "c"),), "", "obs", None, (), None, "commit", HASH, UTC, ManifestStatus.planned)

    def test_lifecycle(self):
        validate_status_transition("planned", "building")
        validate_status_transition("building", "complete")
        validate_status_transition("complete", "complete")
        with self.assertRaises(ContractValidationError):
            validate_status_transition("complete", "failed")
        validate_manifest_transition("id", "planned", "id", "failed")
        with self.assertRaises(ContractValidationError):
            validate_manifest_transition("id", "planned", "other", "failed")
        self.assertEqual(require_structurally_complete(dataset()), require_structurally_complete(dataset()))
        with self.assertRaises(ContractValidationError):
            require_structurally_complete(dataset(ManifestStatus.failed))

    def test_file_hash_and_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_text("hello", encoding="utf-8")
            expected = hashlib.sha256(b"hello").hexdigest()
            self.assertEqual(sha256_file(path), expected)
            verify_file_sha256(path, expected)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ContractValidationError):
                verify_file_sha256(path, expected)
        self.assertEqual(canonical_json_bytes({"z": 1, "a": None, "é": "x"}), '{"a":null,"z":1,"é":"x"}\n'.encode("utf-8"))
        with self.assertRaises(ContractValidationError):
            canonical_json_bytes({"x": float("nan")})

    def test_json_read_duplicate_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "value.json"
            write_json_atomic(path, {"b": 2, "a": None})
            self.assertEqual(read_json_object(path), {"a": None, "b": 2})
            with self.assertRaises(ContractValidationError):
                write_json_atomic(path, {"new": True})
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaises(ContractValidationError):
                read_json_object(duplicate)
            with self.assertRaises(ContractValidationError):
                write_json_atomic(root / "missing" / "x.json", {})

    def test_atomic_publication_race_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "race.json"

            def create_target_then_fail(temporary, target):
                Path(target).write_bytes(b'{"existing":true}\n')
                raise FileExistsError(target)

            with patch("adaptive_roi_rppg.contracts.io.os.link", side_effect=create_target_then_fail):
                with self.assertRaises(ContractValidationError):
                    write_json_atomic(path, {"new": True})
            self.assertEqual(read_json_object(path), {"existing": True})
            self.assertEqual(list(root.glob(".*.tmp")), [])
