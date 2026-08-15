import json
import hashlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from adaptive_roi_rppg.contracts import LabelFrame, MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.model_plan import EXCLUSIONS, load_frozen_model_plan
from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity, paired_subject_bootstrap, rollout_frozen_policy, score_frozen_rollout, summarize_model_rows
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.labels.mcd import GT_RULE_ID
from adaptive_roi_rppg.signal import POS_CONFIG_ID


def frame(hop, clip="clip"):
    measurements=tuple(ROIMeasurement(i,ROI_NAMES[i],70.0+i,0.8,0.5,0.6,True,None,(),None,0,10,POS_CONFIG_ID) for i in range(12))
    return MeasurementFrame("mcd",clip,hop,8.0+hop,measurements,POS_CONFIG_ID,True,None,f"frame-{hop}")


class Policy:
    def __init__(self, actions=(1,2,3)):
        self._identity=CheckpointIdentity("dagger_seed0","dagger",0,"a"*64); self.actions=iter(actions); self.calls=[]
    @property
    def identity(self): return self._identity
    def initial_state(self): return {"fresh":True}
    def predict(self, observation, recurrent_state, *, episode_start):
        self.calls.append((observation.copy(), recurrent_state, episode_start)); return next(self.actions), {"fresh":False}


class FrozenModelReplayTests(unittest.TestCase):
    def test_gt_free_rollout_shape_reset_and_hold(self):
        policy=Policy(); rollout=rollout_frozen_policy([frame(i) for i in range(3)],policy,clip_id="clip")
        self.assertEqual([x.proposed_action for x in rollout],[1,2,3]); self.assertEqual([x.executed_action for x in rollout],[1,1,3])
        self.assertEqual([x.override_reason for x in rollout],[None,"minimum_hold",None])
        self.assertEqual([c[2] for c in policy.calls],[True,False,False])
        self.assertEqual(policy.calls[1][1],{"fresh":False})
        self.assertTrue(all(c[0].shape==(1,101) and c[0].dtype==np.float32 for c in policy.calls))
        second=rollout_frozen_policy([frame(0,"next")],Policy((4,)),clip_id="next")
        self.assertEqual(second[0].previous_action,None)

    def test_labels_do_not_change_rollout(self):
        first=rollout_frozen_policy([frame(i) for i in range(3)],Policy(),clip_id="clip")
        second=rollout_frozen_policy([frame(i) for i in range(3)],Policy(),clip_id="clip")
        self.assertEqual([(x.observation_sha256,x.proposed_action,x.post_belief_hr_bpm) for x in first],[(x.observation_sha256,x.proposed_action,x.post_belief_hr_bpm) for x in second])
        labels=[LabelFrame("mcd","clip",i,8.0+i,50.0+i,GT_RULE_ID,True,None) for i in range(3)]
        changed=[LabelFrame("mcd","clip",i,8.0+i,150.0+i,GT_RULE_ID,True,None) for i in range(3)]
        a=score_frozen_rollout(first,labels,{"subject_id":"s","view":"Frontal","condition":"before"})
        b=score_frozen_rollout(first,changed,{"subject_id":"s","view":"Frontal","condition":"before"})
        self.assertEqual([(x["observation_sha256"],x["proposed_action"],x["post_belief_hr_bpm"]) for x in a],[(x["observation_sha256"],x["proposed_action"],x["post_belief_hr_bpm"]) for x in b])
        self.assertNotEqual([x["abs_error_bpm"] for x in a],[x["abs_error_bpm"] for x in b])

    def test_invalid_action_and_label_fail(self):
        with self.assertRaises(ContractValidationError): rollout_frozen_policy([frame(0)],Policy((12,)),clip_id="clip")
        rollout=rollout_frozen_policy([frame(0)],Policy((1,)),clip_id="clip")
        with self.assertRaises(ContractValidationError): score_frozen_rollout(rollout,[],{"subject_id":"s","view":"v","condition":"c"})

    def test_row_aggregation_is_clip_then_subject(self):
        rows=[]
        for clip,subject,error in (("a","s1",1.0),("b","s1",3.0),("c","s2",9.0)):
            row={"method_id":"dagger_seed0","family":"dagger","seed":0,"checkpoint_sha256":"a"*64,"clip_id":clip,"subject_id":subject,"view":"Frontal","condition":"before","hop_idx":0,"abs_error_bpm":error,"executed_action":0,"legal":True,"selected_valid":True,"post_hold_count":1}
            rows.append(row)
        summary=summarize_model_rows(rows)
        self.assertAlmostEqual(summary["checkpoint_results"]["dagger_seed0"]["equal_clip_mae_bpm"],13/3)
        self.assertEqual(len(summary["subject_rows"]),2)

    def test_subject_bootstrap_is_reproducible_and_paired(self):
        rows=[]
        for method,values in (("a",(1.,4.)),("b",(2.,8.))):
            for subject,value in zip(("s1","s2"),values): rows.append({"method_id":method,"subject_id":subject,"mae_bpm":value})
        result=paired_subject_bootstrap(rows,"a","b",replicates=25,seed=7)
        self.assertEqual(result,paired_subject_bootstrap(rows,"a","b",replicates=25,seed=7)); self.assertAlmostEqual(result["difference_a_minus_b_bpm"],-2.5)

    def test_bootstrap_labels_equal_clip_and_equal_subject_separately(self):
        rows=[{"method_id":"a","subject_id":"s1","mae_bpm":0.},{"method_id":"a","subject_id":"s1","mae_bpm":0.},{"method_id":"a","subject_id":"s2","mae_bpm":9.},{"method_id":"b","subject_id":"s1","mae_bpm":0.},{"method_id":"b","subject_id":"s1","mae_bpm":0.},{"method_id":"b","subject_id":"s2","mae_bpm":0.}]
        clip=paired_subject_bootstrap(rows,"a","b",estimand="equal_clip",replicates=5,seed=2)
        subject=paired_subject_bootstrap(rows,"a","b",estimand="equal_subject",replicates=5,seed=2)
        self.assertEqual(clip["estimand"],"equal_clip_mae_difference_a_minus_b_bpm"); self.assertEqual(subject["estimand"],"equal_subject_mae_difference_a_minus_b_bpm")
        self.assertAlmostEqual(clip["difference_a_minus_b_bpm"],3.0); self.assertAlmostEqual(subject["difference_a_minus_b_bpm"],4.5)

    def test_three_seed_family_and_behavior_summary(self):
        rows=[]
        for seed in range(3):
            rows.append({"method_id":f"dagger_seed{seed}","family":"dagger","seed":seed,"checkpoint_sha256":str(seed)*64,"clip_id":"clip","subject_id":"s","view":"Frontal","condition":"before","hop_idx":0,"abs_error_bpm":float(seed),"executed_action":seed,"legal":True,"selected_valid":True,"post_hold_count":1,"selected_hr_bpm":70.,"post_belief_hr_bpm":70.})
        summary=summarize_model_rows(rows)
        self.assertEqual(summary["family_results"][0]["family"],"dagger")
        self.assertEqual(len(summary["family_results"][0]["seed_results"]),3)
        self.assertIn("action_distribution",summary["checkpoint_results"]["dagger_seed0"])

    def test_plan_is_exact_and_rejects_missing_seed(self):
        plan=load_frozen_model_plan(Path("configs/evaluation/mcd_frozen_models_v1.json"))
        self.assertEqual(len(plan.checkpoints),9); self.assertEqual(set(plan.payload["exclusions"]),EXCLUSIONS)
        bad=dict(plan.payload); bad["checkpoints"]=[dict(x) for x in plan.checkpoints[:-1]]
        path=Path("/tmp/gate8_bad_plan.json"); path.write_text(json.dumps(bad))
        try:
            with self.assertRaises(ContractValidationError): load_frozen_model_plan(path)
        finally: path.unlink(missing_ok=True)

    def test_plan_requires_truthful_provenance_status(self):
        plan=load_frozen_model_plan(Path("configs/evaluation/mcd_frozen_models_v1.json")); bad=dict(plan.payload); bad["checkpoints"]=[dict(x) for x in plan.checkpoints]; bad["checkpoints"][0]["provenance_status"]="verified"
        path=Path("/tmp/gate8_bad_provenance.json"); path.write_text(json.dumps(bad))
        try:
            with self.assertRaises(ContractValidationError): load_frozen_model_plan(path)
        finally: path.unlink(missing_ok=True)

    def test_adapter_rejects_wrong_file_identity_before_deserialization(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.zip"; path.write_bytes(b"data")
            spec={"locator":str(path),"byte_size":5,"sha256":"0"*64,"method_id":"x","family":"dagger","seed":0}
            with self.assertRaises(ContractValidationError): load_frozen_recurrent_policy(spec)

    def test_adapter_rejects_incompatible_architecture(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.zip"; path.write_bytes(b"data")
            spec={"locator":str(path),"byte_size":4,"sha256":hashlib.sha256(b"data").hexdigest(),"method_id":"x","family":"dagger","seed":0}
            class Space: shape=(101,); dtype=np.dtype(np.float32); n=12
            class FeedForward: pass
            class FakePPO:
                @staticmethod
                def load(*args,**kwargs): return types.SimpleNamespace(observation_space=Space(),action_space=Space(),policy=FeedForward())
            with patch.dict(sys.modules,{"sb3_contrib":types.SimpleNamespace(RecurrentPPO=FakePPO)}):
                with self.assertRaises(ContractValidationError): load_frozen_recurrent_policy(spec)
            spec["byte_size"]=4; spec["sha256"]=hashlib.sha256(b"data").hexdigest()
            with self.assertRaises(ContractValidationError): load_frozen_recurrent_policy(spec)

    def test_launcher_exports_run_to_child(self):
        source=Path("slurm/gate8_mcd_frozen_models_full.slurm").read_text()
        self.assertIn('RUN="$OUT/full-$SLURM_JOB_ID"; export RUN',source)


if __name__ == "__main__": unittest.main()
