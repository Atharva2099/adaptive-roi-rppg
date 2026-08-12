import csv
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from adaptive_roi_rppg.contracts import ManifestStatus, canonical_json_bytes, read_json_object
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import *
import adaptive_roi_rppg.data.mcd.adapter as adapter


class MCDAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state, self.gt = self.root / "state", self.root / "gt"
        self.state.mkdir(); self.gt.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, stem="S1_FullHDwebcam_before", rows=(0, 2), *, state_rows=None, gt_rows=None):
        state_rows = rows if state_rows is None else state_rows
        gt_rows = rows if gt_rows is None else gt_rows
        with (self.state / f"{stem}{MCD_STATE_SUFFIX}").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(MCD_STATE_COLUMNS)
            writer.writerows([[i, *([1.0] * (len(MCD_STATE_COLUMNS) - 1))] for i in state_rows])
        with (self.gt / f"{stem}{MCD_GT_SUFFIX}").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(MCD_GT_COLUMNS); writer.writerows([[i, 70.0] for i in gt_rows])

    def add_raw(self, state_row, gt_row=(0, "70.0"), stem="S1_FullHDwebcam_before"):
        self.add_raw_rows((state_row,), (gt_row,), stem=stem)

    def add_raw_rows(self, state_rows, gt_rows, stem="S1_FullHDwebcam_before"):
        with (self.state / f"{stem}{MCD_STATE_SUFFIX}").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(MCD_STATE_COLUMNS); writer.writerows(state_rows)
        with (self.gt / f"{stem}{MCD_GT_SUFFIX}").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(MCD_GT_COLUMNS); writer.writerows(gt_rows)

    def split(self, subjects=("S1", "S2"), *, path=None):
        path = self.root / "split.csv" if path is None else path
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(("subject_id", "split"))
            for index, subject in enumerate(subjects): writer.writerow((subject, "train" if index == 0 else "eval"))
        return path

    def bundle(self):
        self.add(); self.add("S2_USBVideo_after")
        return build_mcd_manifest_bundle(self.state, self.gt, self.split(), created_at_utc="2026-08-11T00:00:00Z", producer_command="test")

    def rebind_clip(self, bundle, **changes):
        clip = replace(bundle.clip_manifests[0], **changes)
        clip = replace(clip, clip_manifest_id=adapter._id("mcd-clip", adapter._without(clip.to_dict(), "clip_manifest_id", "status")))
        clips = (clip, *bundle.clip_manifests[1:])
        inventory = json.loads(json.dumps(bundle.source_inventory)); record = inventory["clips"][0]
        record.update({"metadata": clip.to_dict() | {"clip_id": clip.clip_id}, "state_locator": clip.state_locator, "gt_locator": clip.gt_locator,
                       "state_sha256": clip.state_sha256, "gt_sha256": clip.gt_sha256, "state_row_count": clip.state_row_count, "gt_row_count": clip.gt_row_count})
        record["metadata"] = {key: clip.to_dict()[key] for key in ("clip_id", "subject_id", "camera_id", "view", "camera_fps", "condition")}
        inventory["inventory_id"] = adapter._id("mcd-inventory", {"dataset_id": MCD_DATASET_ID, "schema_id": MCD_SCHEMA_ID, "clips": inventory["clips"]})
        inventory_hash = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
        split = replace(bundle.split_manifest, source_hashes=(bundle.source_inventory["split_source_sha256"], inventory_hash))
        refs = tuple(item.clip_manifest_id for item in clips); hashes = tuple(hashlib.sha256(canonical_json_bytes(item.to_dict())).hexdigest() for item in clips)
        dataset = replace(bundle.dataset_manifest, clip_manifest_refs=refs, clip_manifest_hashes=hashes, source_inventory_sha256=inventory_hash,
                          cohort={**dict(bundle.dataset_manifest.cohort), "source_inventory_id": inventory["inventory_id"]})
        dataset = replace(dataset, manifest_id=adapter._id("mcd-dataset", adapter._without(dataset.to_dict(), "manifest_id", "created_at_utc", "producer_command", "status")))
        return bundle.__class__(inventory, split, clips, dataset)

    def rebind_dataset(self, bundle, **changes):
        dataset = replace(bundle.dataset_manifest, **changes)
        dataset = replace(dataset, manifest_id=adapter._id("mcd-dataset", adapter._without(dataset.to_dict(), "manifest_id", "created_at_utc", "producer_command", "status")))
        return replace(bundle, dataset_manifest=dataset)

    def test_schema_naming_and_camera_metadata(self):
        self.assertEqual(len(MCD_STATE_COLUMNS), 64)
        self.assertEqual(MCD_SCHEMA_ID, "mcd-semantic-state-64-structured-missing-paired-gt-v2")
        for camera, view, fps in (("FullHDwebcam", "Frontal", 30.0), ("USBVideo", "SideA", 30.0), ("IriunWebcam", "SideB", 24.0)):
            metadata = parse_mcd_stem(f"S1_{camera}_before")
            self.assertEqual((metadata.view, metadata.camera_fps), (view, fps))
        for stem in ("S_1_FullHDwebcam_before", "S1_unknown_before", "S1_FullHDwebcam_before_x"):
            with self.assertRaises(ContractValidationError): parse_mcd_stem(stem)

    def test_strict_roots_children_and_pairing(self):
        with self.assertRaises(ContractValidationError): discover_mcd_sources(self.state, self.gt)
        self.add()
        (self.gt / f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()
        with self.assertRaises(ContractValidationError): discover_mcd_sources(self.state, self.gt)
        self.add(); (self.state / "notes.txt").write_text("x")
        with self.assertRaises(ContractValidationError): discover_mcd_sources(self.state, self.gt)
        (self.state / "notes.txt").unlink(); (self.gt / "link").symlink_to(self.gt / f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}")
        with self.assertRaises(ContractValidationError): discover_mcd_sources(self.state, self.gt)

    def test_csv_frame_finite_coverage_hash_rows_and_join(self):
        self.add(state_rows=(0, 1), gt_rows=(0, 2))
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])

        (self.gt / f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").write_text("frame_idx,gt_ppg\n0,nan\n", encoding="utf-8")
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
        self.add(); state = self.state / f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}"
        state.write_text(state.read_text().replace(",1.0\n", ",1.2\n", 1), encoding="utf-8")
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
        state.write_text(state.read_text().replace(",1.2\n", ",nan\n", 1), encoding="utf-8")
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
        self.add()
        pair = discover_mcd_sources(self.state, self.gt)[0]; before = adapter._snapshot(pair.state_path); gt = adapter._snapshot(pair.gt_path)
        with patch.object(adapter, "_snapshot", side_effect=[before, gt, (before[0], before[1], before[2] + 1, before[3]), gt]):
            with self.assertRaises(ContractValidationError): validate_mcd_source_pair(pair)

    def test_structured_missing_semantics(self):
        present = ["1.0"] * 4 + ["1.0"]
        absent = ["", "", "", "", "0.0"]
        blank_pose = [0, "", "", ""] + present * 12
        mixed = [0, "", "", ""] + present + absent + present * 10
        present_row = [0, "1.0", "2.0", "3.0"] + present * 12
        absent_row = [2, "1.0", "2.0", "3.0"] + absent * 12
        valid_cases = (("blank pose", blank_pose), ("all ROI absent", present_row), ("mixed ROI", mixed))
        for name, row in valid_cases:
            with self.subTest(name=name):
                self.add_raw_rows((row, absent_row), ((0, "70.0"), (2, "71.0")))
                validated = validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
                self.assertEqual(validated.state_row_count, 2)
                self.assertEqual(validated.gt_row_count, 2)
                self.state.joinpath(f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}").unlink()
                self.gt.joinpath(f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()
        cases = {
            "partial pose": [0, "", "1.0", "2.0"] + present * 12,
            "positive coverage missing value": [0, "1.0", "2.0", "3.0", "", "1.0", "1.0", "1.0", "1.0"] + present * 11,
            "zero coverage present value": [0, "1.0", "2.0", "3.0", "1.0", "1.0", "1.0", "1.0", "0.0"] + present * 11,
            "blank coverage": [0, "1.0", "2.0", "3.0", "1.0", "1.0", "1.0", "1.0", ""] + present * 11,
            "whitespace pose": [0, " ", "1.0", "2.0"] + present * 12,
            "nan ROI": [0, "1.0", "2.0", "3.0", "nan", "1.0", "1.0", "1.0", "1.0"] + present * 11,
        }
        for name, row in cases.items():
            with self.subTest(name=name):
                self.add_raw(row)
                with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
                self.state.joinpath(f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}").unlink()
                self.gt.joinpath(f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()
        for token, location in (("NaN", "pose"), ("inf", "roi"), ("-inf", "gt"), ("null", "roi")):
            with self.subTest(token=token, location=location):
                row = [0, "1.0", "2.0", "3.0"] + present * 12
                if location == "pose": row[1] = token
                elif location == "roi": row[4] = token
                else: self.add_raw(row, (0, token)); row = None
                if row is not None: self.add_raw(row)
                with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
                self.state.joinpath(f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}").unlink()
                self.gt.joinpath(f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()

    def test_width_and_strict_gt_and_old_schema_rejection(self):
        row = [0, "1.0", "2.0", "3.0"] + ["1.0"] * 60
        self.add_raw(row, (0, ""))
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
        self.state.joinpath(f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}").unlink(); self.gt.joinpath(f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()
        self.add_raw(row[:-1])
        with self.assertRaises(ContractValidationError): validate_mcd_source_pair(discover_mcd_sources(self.state, self.gt)[0])
        self.state.joinpath(f"S1_FullHDwebcam_before{MCD_STATE_SUFFIX}").unlink(); self.gt.joinpath(f"S1_FullHDwebcam_before{MCD_GT_SUFFIX}").unlink()
        bundle = self.bundle(); destination = self.root / "old"
        write_mcd_manifest_bundle(bundle, destination)
        payload = read_json_object(destination / "dataset_manifest.json"); payload["schema_id"] = "mcd-semantic-state-64-paired-gt-v1"
        (destination / "dataset_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ContractValidationError): load_mcd_manifest_tree(destination)

    def test_split_exhaustive_duplicate_overlap_and_change(self):
        self.add(); self.add("S2_USBVideo_after")
        bad = self.root / "bad.csv"; bad.write_text("subject_id,split\nS1,train\nS1,eval\nS2,eval\n", encoding="utf-8")
        with self.assertRaises(ContractValidationError): load_mcd_split_assignment(bad, ("S1", "S2"))
        bad.write_text("subject_id,split\nS1,train\nS3,eval\n", encoding="utf-8")
        with self.assertRaises(ContractValidationError): load_mcd_split_assignment(bad, ("S1", "S2"))
        split = self.split(); before = adapter._snapshot(split)
        with patch.object(adapter, "_snapshot", side_effect=[before, (before[0], before[1], before[2] + 1, before[3])]):
            with self.assertRaises(ContractValidationError): load_mcd_split_assignment(split, ("S1", "S2"))

    def test_root_membership_change_during_build(self):
        self.add(); self.add("S2_USBVideo_after")
        original = adapter._enumerate
        calls = [0]
        def changing(root, suffix, label):
            result = original(root, suffix, label); calls[0] += 1
            if calls[0] == 3: return result[0], result[1] + ("S3_FullHDwebcam_before" + suffix,)
            return result
        with patch.object(adapter, "_enumerate", side_effect=changing):
            with self.assertRaises(ContractValidationError): self.bundle()

    def test_determinism_and_representative_tamper(self):
        first = self.bundle(); other = self.root / "other"; (other / "state").mkdir(parents=True); (other / "gt").mkdir()
        for source, target in ((self.state, other / "state"), (self.gt, other / "gt")):
            for path in source.iterdir(): target.joinpath(path.name).write_bytes(path.read_bytes())
        split = other / "split.csv"; split.write_bytes(self.split().read_bytes())
        second = build_mcd_manifest_bundle(other / "state", other / "gt", split, created_at_utc="2026-08-11T00:00:00Z", producer_command="test")
        self.assertEqual(first.dataset_manifest.manifest_id, second.dataset_manifest.manifest_id)
        for mutation in (("camera_fps", 99.0), ("state_row_count", 99), ("state_sha256", "0" * 64)):
            clip = first.clip_manifests[0].to_dict(); clip[mutation[0]] = mutation[1]
            with self.assertRaises(ContractValidationError): adapter._validate_bundle(first.__class__(first.source_inventory, first.split_manifest, (type(first.clip_manifests[0]).from_dict(clip), *first.clip_manifests[1:]), first.dataset_manifest))

    def test_semantic_revalidation_table(self):
        bundle = self.bundle()
        cases = [("planned split", lambda b: replace(b, split_manifest=replace(b.split_manifest, status=ManifestStatus.planned))),
                 ("wrong FPS", lambda b: self.rebind_clip(b, camera_fps=99.0)),
                 ("zero rows", lambda b: self.rebind_clip(b, state_row_count=0, gt_row_count=0)),
                 ("unequal rows", lambda b: self.rebind_clip(b, state_row_count=2, gt_row_count=3)),
                 ("wrong locator", lambda b: self.rebind_clip(b, state_locator="state/wrong.csv")),
                 ("wrong stem metadata", lambda b: self.rebind_clip(b, subject_id="S9")),
                 ("modified cohort", lambda b: self.rebind_dataset(b, cohort={"changed": True})),
                 ("nonempty transforms", lambda b: self.rebind_dataset(b, transform_ids=("unexpected",)))]
        for name, mutate in cases:
            with self.subTest(name=name), self.assertRaises(ContractValidationError): adapter._validate_bundle(mutate(bundle))

    def test_publication_load_existing_incomplete_extra_duplicate_and_failure(self):
        bundle = self.bundle(); destination = self.root / "out"
        output = write_mcd_manifest_bundle(bundle, destination); self.assertTrue(output.is_file())
        self.assertEqual(load_mcd_manifest_tree(destination), bundle)
        with self.assertRaises(ContractValidationError): write_mcd_manifest_bundle(bundle, destination)
        (destination / "extra").write_text("x")
        with self.assertRaises(ContractValidationError): load_mcd_manifest_tree(destination)
        (destination / "extra").unlink(); (destination / "clips" / next(iter(p.name for p in (destination / "clips").iterdir()))).write_text("{}")
        with self.assertRaises(ContractValidationError): load_mcd_manifest_tree(destination)
        destination = self.root / "symlink-out"; write_mcd_manifest_bundle(bundle, destination)
        for name in ("source_inventory.json", "split_manifest.json", "dataset_manifest.json"):
            target = destination / name; backup = destination / (name + ".real"); target.rename(backup); target.symlink_to(backup)
            with self.assertRaises(ContractValidationError): load_mcd_manifest_tree(destination)
            target.unlink(); backup.rename(target)
        duplicate = destination / "source_inventory.json"; duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaises(ContractValidationError): load_mcd_manifest_tree(destination)
        destination2 = self.root / "failed"
        with patch.object(adapter, "write_json_atomic", side_effect=OSError("write")):
            with self.assertRaises(OSError): write_mcd_manifest_bundle(bundle, destination2)
        self.assertFalse(destination2.exists())

    def test_cli_stdout_hashes_and_output_boundary(self):
        self.add(); self.add("S2_USBVideo_after")
        output = self.root / "cli-out"; env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
        command = ["python3", "scripts/build_mcd_manifests.py", "--state-root", str(self.state), "--gt-root", str(self.gt), "--split-csv", str(self.split()), "--output-dir", str(output), "--created-at-utc", "2026-08-11T00:00:00Z"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr); self.assertEqual(len(result.stdout.splitlines()), 1)
        payload = json.loads(result.stdout); self.assertEqual(payload["dataset_manifest_hash"], hashlib.sha256((output / "dataset_manifest.json").read_bytes()).hexdigest())
        bad = list(command); bad[bad.index("--output-dir") + 1] = str(self.state / "inside")
        self.assertNotEqual(subprocess.run(bad, env=env, capture_output=True).returncode, 0)

    def test_canonical_verifier_accepts_and_rejects_expectation(self):
        self.add(); self.add("S1_USBVideo_after"); self.add("S2_FullHDwebcam_before"); self.add("S2_USBVideo_after")
        output = self.root / "verify-out"; env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
        build = ["python3", "scripts/build_mcd_manifests.py", "--state-root", str(self.state), "--gt-root", str(self.gt), "--split-csv", str(self.split()), "--output-dir", str(output), "--created-at-utc", "2026-08-11T00:00:00Z"]
        self.assertEqual(subprocess.run(build, env=env, capture_output=True).returncode, 0)
        split_hash = read_json_object(output / "source_inventory.json")["split_source_sha256"]
        expectation = self.root / "expectation.json"
        payload = {"split_sha256": split_hash, "train_subjects": 1, "eval_subjects": 1, "total_subjects": 2, "total_clips": 4, "camera_condition_combinations": [["FullHDwebcam", "before"], ["USBVideo", "after"]]}
        expectation.write_text(json.dumps(payload), encoding="utf-8")
        verify = ["python3", "scripts/verify_mcd_gate2_acceptance.py", str(output), "--expectation", str(expectation)]
        accepted = subprocess.run(verify, env=env, capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        accepted_payload = json.loads(accepted.stdout); self.assertEqual(accepted_payload["status"], "accepted")
        self.assertEqual(accepted_payload["expectation_path"], str(expectation.resolve()))
        self.assertEqual(accepted_payload["expectation_sha256"], hashlib.sha256(expectation.read_bytes()).hexdigest())
        for name in ("dataset_manifest", "split", "inventory"):
            path = output / ("dataset_manifest.json" if name == "dataset_manifest" else f"{name}_manifest.json" if name == "split" else "source_inventory.json")
            self.assertEqual(accepted_payload[f"{name}_hash"], hashlib.sha256(path.read_bytes()).hexdigest())
        payload["split_sha256"] = "0" * 64; expectation.write_text(json.dumps(payload), encoding="utf-8")
        self.assertNotEqual(subprocess.run(verify, env=env, capture_output=True).returncode, 0)
        payload["split_sha256"] = split_hash; payload["camera_condition_combinations"] = [["FullHDwebcam", "before"]]; expectation.write_text(json.dumps(payload), encoding="utf-8")
        self.assertNotEqual(subprocess.run(verify, env=env, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
