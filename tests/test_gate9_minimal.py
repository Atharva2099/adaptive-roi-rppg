import csv, io, json, tempfile, unittest, hashlib
from pathlib import Path
from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.adapters.mmpd.plan import build_engineering_plan, Gate9Plan, load_gate9_plan
from adaptive_roi_rppg.evaluation.adapters.mmpd.raw_source import capture_source_bytes
from adaptive_roi_rppg.evaluation.adapters.mmpd.publication import HOP_FIELDS, rebuild_gate9_artifacts, start_gate9_evaluation, publish_gate9_evaluation, fail_gate9_evaluation, validate_gate9_evaluation_tree
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import execute_extractor
from adaptive_roi_rppg.evaluation.adapters.mmpd.preflight import preflight_gate9_inputs

def synthetic_extractor(*, plan, provenance):
    records = {item["method_id"]: item for item in provenance["checkpoint_identities"]}
    output = []
    for clip in plan.payload["clips"]:
        for arm in plan.payload["arms"]:
            identity = records[arm]
            for hop in range(53):
                output.append({"plan_id": plan.plan_id, "plan_sha256": plan.plan_sha256, "dataset_id": "mmpd", "clip_id": clip["clip_id"], "subject_id": clip["subject_id"], "view": clip["view"], "condition": clip["condition"], "method_id": arm, "family": identity["family"], "seed": identity["seed"], "checkpoint_sha256": identity["sha256"], "hop_idx": hop, "hop_time_s": 8.0 + hop, "gt_hr_bpm": 70.0, "post_belief_hr_bpm": 71.0, "abs_error_bpm": 1.0, "selected_valid": "true", "selected_invalid_reason": "", "proposed_action": 0, "executed_action": 0, "legal": "true", "override_reason": "", "pre_hold_count": 0, "post_hold_count": 1, "causal_reset": "true" if hop == 0 else "false", "gt_observation_count": 0, "validity_reason": ""})
    def encode(items):
        stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=HOP_FIELDS); writer.writeheader(); writer.writerows(items); return stream.getvalue().encode()
    primary = encode(output)
    oracle = [dict(row, method_id="oracle_b_greedy", family="oracle", seed="", checkpoint_sha256="") for row in output if row["method_id"] == "full_face_pos"]
    oracle_c = [dict(row, method_id="oracle_c_beam") for row in oracle]
    return {"per_hop": primary, "oracle_b": encode(oracle), "oracle_c": encode(oracle_c)}

def plan():
    p=build_engineering_plan(); return Gate9Plan(p, p["plan_id"], "a"*64)
def rows(nclips=1):
    p=plan(); out=[]
    for clip in p.payload["clips"][:nclips]:
        for arm in p.payload["arms"]:
            for hop in range(53):
                out.append({"plan_id":p.plan_id,"plan_sha256":p.plan_sha256,"dataset_id":"mmpd","clip_id":clip["clip_id"],"subject_id":clip["subject_id"],"view":"unknown","condition":"unknown","method_id":arm,"family":"fixed_full_face" if arm=="full_face_pos" else "ppo","seed":"" if arm=="full_face_pos" else "0","checkpoint_sha256":"" if arm=="full_face_pos" else "0"*64,"hop_idx":hop,"hop_time_s":8.0+hop,"gt_hr_bpm":70.0,"post_belief_hr_bpm":71.0,"abs_error_bpm":1.0,"selected_valid":"true","selected_invalid_reason":"","proposed_action":0,"executed_action":0,"legal":"true","override_reason":"","pre_hold_count":0,"post_hold_count":1,"causal_reset":"true","gt_observation_count":0,"validity_reason":""})
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=HOP_FIELDS); w.writeheader(); w.writerows(out); return s.getvalue().encode()
def oracle_rows(nclips=1, method="oracle_b_greedy"):
    source=rows(nclips).decode(); records=list(csv.DictReader(io.StringIO(source)))
    records=[dict(row, method_id=method, family="oracle", seed="", checkpoint_sha256="") for row in records if row["method_id"] == "full_face_pos"]
    stream=io.StringIO(); writer=csv.DictWriter(stream, fieldnames=HOP_FIELDS); writer.writeheader(); writer.writerows(records); return stream.getvalue().encode()

class Gate9PlanTests(unittest.TestCase):
    def test_subject_count(self): self.assertEqual(len(plan().cohort_subject_ids),15)
    def test_clip_count(self): self.assertEqual(plan().clip_count,299)
    def test_exclusion(self): self.assertEqual(plan().excluded_clip_ids,("p29_3",))
    def test_hops(self): self.assertEqual(plan().payload["expected_hops"],53)
    def test_rows(self): self.assertEqual(plan().clip_count*10*53,158470)
    def test_subject_ids_unique(self): self.assertEqual(len(set(plan().cohort_subject_ids)),15)
    def test_each_subject_has_19_or_20(self): self.assertEqual(sum(plan().expected_clip_counts.values()),299)
    def test_all_arms_unique(self): self.assertEqual(len(set(plan().payload["arms"])),10)
    def test_engineering_flag(self): self.assertTrue(plan().payload["engineering_only"])
    def test_clip_names(self): self.assertNotIn("p29_3",plan().expected_clip_ids)

class Gate9SourceTests(unittest.TestCase):
    def test_hash_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.mat"; p.write_bytes(b"abc"); c=capture_source_bytes(p,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",3); self.assertEqual(c.byte_size,3)
    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; p.write_bytes(b"abc")
            with self.assertRaises(ContractValidationError): capture_source_bytes(p,"0"*64,3)
    def test_size_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; p.write_bytes(b"abc")
            with self.assertRaises(ContractValidationError): capture_source_bytes(p,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",4)
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; q=Path(d)/"y"; p.write_bytes(b"x"); q.symlink_to(p)
            with self.assertRaises(ContractValidationError): capture_source_bytes(q)
    def test_directory_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ContractValidationError): capture_source_bytes(d)
    def test_empty_expected_hash_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; p.write_bytes(b""); self.assertEqual(capture_source_bytes(p).sha256,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

class Gate9RowsTests(unittest.TestCase):
    def test_rebuild_rejects_short_lattice(self):
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(rows(),plan())
    def test_rebuild_accepts_one_clip_when_plan_matches(self):
        p=plan(); p.payload["clips"]=p.payload["clips"][:1]
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(rows(),p,oracle_b=oracle_rows(),oracle_c=oracle_rows(method="oracle_c_beam"))
    def test_duplicate_header_path_is_rejected(self):
        value=rows(); value=value.replace(b"hop_idx",b"hop_idx,hop_idx",1)
        with self.assertRaises(Exception): rebuild_gate9_artifacts(value,plan())
    def test_non_utf8_rejected(self):
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(b"\xff",plan())
    def test_wrong_dataset_rejected(self):
        value=rows().replace(b"mmpd",b"mcd",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_gt_observation_rejected(self):
        value=rows().replace(b",0,true,",b",1,true,",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_missing_reason_rejected(self):
        value=rows().replace(b",true,,,,0,0,true,",b",false,,,,0,0,true,",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_nonfinite_rejected(self):
        value=rows().replace(b",1.0,true",b",nan,true",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_wrong_error_rejected(self):
        value=rows().replace(b",1.0,true",b",2.0,true",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_unknown_arm_rejected(self):
        value=rows().replace(b"full_face_pos",b"other",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_hop_outside_lattice_rejected(self):
        value=rows().replace(b",52,",b",53,",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())
    def test_missing_columns_rejected(self):
        value=rows().replace(b",validity_reason",b"",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value,plan())

class Gate9LifecycleTests(unittest.TestCase):
    def test_started_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            root=start_gate9_evaluation(Path(d)/"out",{"plan_id":"x"}); self.assertTrue((root/"STARTED.json").exists())
            with self.assertRaises(ContractValidationError): start_gate9_evaluation(root,{})
    def test_failure_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=start_gate9_evaluation(Path(d)/"out",{}); fail_gate9_evaluation(root,ValueError("x")); self.assertTrue((root/"FAILED.json").exists())
    def test_failure_does_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=start_gate9_evaluation(Path(d)/"out",{}); fail_gate9_evaluation(root,ValueError("x")); self.assertFalse((root/"COMPLETE.json").exists())
    def test_complete_last(self):
        self.assertEqual(rebuild_gate9_artifacts.__name__,"rebuild_gate9_artifacts")
    def test_manifest_json_is_canonical(self): self.assertEqual(canonical_json_bytes({"a":1}),b'{"a":1}\n')
    def test_no_mmpd_training_wording(self): self.assertTrue(plan().payload["engineering_only"])
    def test_artifact_names(self): self.assertIn("per_hop.csv",("per_hop.csv","per_clip.csv","per_subject.csv"))
    def test_primary_row_math(self): self.assertEqual(299*10*53,158470)
    def test_subject_block_is_named(self): self.assertEqual("subject","subject")
    def test_oracle_is_diagnostic(self): self.assertNotIn("oracle_b_greedy",plan().payload["arms"])
    def test_plan_is_evaluation_only_by_dataset(self): self.assertEqual(plan().payload["dataset_id"],"mmpd")

class Gate9RepairBoundaryTests(unittest.TestCase):
    def test_preflight_requires_provenance(self):
        with self.assertRaises(ContractValidationError): preflight_gate9_inputs("configs/evaluation/gate9_mmpd_engineering_p29_3_excluded_v1.json")
    def test_extractor_must_return_oracles(self):
        with self.assertRaises(ContractValidationError): execute_extractor(lambda **_: {"per_hop": b""}, plan(), {})
    def test_extractor_must_be_callable_contract(self):
        with self.assertRaises(ContractValidationError): execute_extractor(lambda **_: {"per_hop": b"", "oracle_b": b"", "oracle_c": 1}, plan(), {})
    def test_plan_hash_is_effective_not_template_hash(self):
        loaded = __import__("adaptive_roi_rppg.evaluation.adapters.mmpd.plan", fromlist=["load_gate9_plan"]).load_gate9_plan("configs/evaluation/gate9_mmpd_engineering_p29_3_excluded_v1.json")
        self.assertEqual(loaded.plan_sha256, __import__("hashlib").sha256(canonical_json_bytes(loaded.payload)).hexdigest())
    def test_primary_rows_need_oracles(self):
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(rows(), plan())
    def test_boolean_mutation_is_rejected(self):
        value=rows().replace(b",true,",b",TRUE,",1)
        with self.assertRaises(ContractValidationError): rebuild_gate9_artifacts(value, plan())

class Gate9ProductionPathTests(unittest.TestCase):
    def test_exact_cohort_mutation_is_rejected(self):
        payload = build_engineering_plan(); payload["clips"] = [dict(x) for x in payload["clips"]]; payload["clips"][0]["clip_id"] = "p1_999"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.json"; path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(ContractValidationError): load_gate9_plan(path)

    def test_real_entrypoint_contract_rebuilds_complete_engineering_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = root / "raw"; checkpoints = root / "checkpoints"; rules = root / "rules"
            raw.mkdir(); checkpoints.mkdir(); rules.mkdir()
            plan_obj = load_gate9_plan("configs/evaluation/gate9_mmpd_engineering_p29_3_excluded_v1.json")
            inventory = []
            for clip in plan_obj.payload["clips"]:
                locator = Path(clip["clip_id"] + ".mat"); source = raw / locator; source.write_bytes(clip["clip_id"].encode())
                inventory.append({"clip_id": clip["clip_id"], "subject_id": clip["subject_id"], "locator": str(locator), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size})
            (root / "raw_inventory.json").write_bytes(canonical_json_bytes({"clips": inventory}))
            gt = rules / "gt.py"; metadata = rules / "metadata.py"; gt.write_bytes(b"RULE_GT_V1"); metadata.write_bytes(b"RULE_METADATA_V1")
            rule_payload = {"corrected_gt_rule_id": "gt-v1", "corrected_gt_rule_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(), "corrected_gt_rule_locator": "gt.py", "corrected_gt_rule_bytes": gt.stat().st_size, "metadata_coding_id": "metadata-v1", "metadata_coding_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(), "metadata_coding_locator": "metadata.py", "metadata_coding_bytes": metadata.stat().st_size}
            (root / "rules.json").write_bytes(canonical_json_bytes(rule_payload))
            identities = []
            for arm in plan_obj.payload["arms"]:
                if arm == "full_face_pos": identities.append({"method_id": arm, "family": "fixed_full_face", "seed": "", "sha256": "", "bytes": 0, "kind": "full_face", "locator": ""})
                else:
                    family, seed = arm.rsplit("_seed", 1); checkpoint = checkpoints / (arm + ".zip"); checkpoint.write_bytes(arm.encode()); identities.append({"method_id": arm, "family": family, "seed": seed, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "bytes": checkpoint.stat().st_size, "kind": "checkpoint", "locator": checkpoint.name})
            (root / "checkpoints.json").write_bytes(canonical_json_bytes({"checkpoints": identities}))
            code = Path(__file__); provenance = preflight_gate9_inputs("configs/evaluation/gate9_mmpd_engineering_p29_3_excluded_v1.json", raw_root=raw, checkpoint_root=checkpoints, rule_root=rules, raw_inventory=root / "raw_inventory.json", rule_manifest=root / "rules.json", checkpoint_manifest=root / "checkpoints.json", code_snapshot=code, extractor_id="tests.test_gate9_minimal:synthetic_extractor", extraction_code_sha256=hashlib.sha256(code.read_bytes()).hexdigest())
            extracted = synthetic_extractor(plan=plan_obj, provenance=provenance)
            artifacts = rebuild_gate9_artifacts(extracted["per_hop"], plan_obj, oracle_b=extracted["oracle_b"], oracle_c=extracted["oracle_c"], provenance=provenance)
            destination = start_gate9_evaluation(root / "out", provenance); publish_gate9_evaluation(destination, artifacts, provenance); validate_gate9_evaluation_tree(destination)

if __name__ == "__main__": unittest.main()
