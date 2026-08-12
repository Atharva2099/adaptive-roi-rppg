import ast
import hashlib
import math
import unittest
from dataclasses import replace

import numpy as np

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement, canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control.core import *


def measurement(i, hr=70.0, confidence=.8, valid=True, coverage=.6):
    return ROIMeasurement(i, ROI_NAMES[i], hr if valid else None, confidence if valid else None, .5 if valid else None, coverage, valid, None if valid else "missing", (), None, 0, 1, "sig-v1")

def frame(hop=0, values=None, clip="clip"):
    values = values or tuple(measurement(i, 70 + i) for i in range(12))
    return MeasurementFrame("mcd", clip, hop, float(hop + 1), tuple(values), "sig-v1", any(x.valid for x in values), None if any(x.valid for x in values) else "none", f"p-{hop}")


class ControlTests(unittest.TestCase):
    def test_config_is_deeply_immutable_and_recomputable(self):
        with self.assertRaises(TypeError): CONTROL_CONFIG_PAYLOAD["belief"] = None
        with self.assertRaises(TypeError): CONTROL_CONFIG_PAYLOAD["belief"]["dt"] = 2
        self.assertEqual(CONTROL_CONFIG_ID, "control-v1-" + hashlib.sha256(canonical_json_bytes(CONTROL_CONFIG_PAYLOAD)).hexdigest())
        self.assertEqual(len(FIELD_NAMES), 101); self.assertEqual(FIELD_NAMES, tuple(FIELD_NAMES))

    def test_known_answer_and_pure_repeat(self):
        state = initial_control_state("mcd", "c").belief
        result = belief_step(state, 72, .8)
        self.assertAlmostEqual(result.mean_hr, 71.2592958372426, places=12)
        self.assertAlmostEqual(result.velocity, .07421855710829095, places=12)
        expected = ((157.41197965532564, 9.277319638536365), (9.277319638536367, 24.17041257221866))
        np.testing.assert_allclose(result.covariance, expected, rtol=0, atol=1e-12)
        self.assertEqual(state, initial_control_state("mcd", "c").belief)
        self.assertEqual(result, belief_step(state, 72, .8))

    def test_predict_only_and_threshold(self):
        state = initial_control_state("mcd", "c").belief
        for z, c in ((float("nan"), .8), (72, float("inf")), (72, 0), (72, -1)):
            before = state; state = belief_step(state, z, c); self.assertEqual(state.hops_since_confident, before.hops_since_confident + 1)
        state = belief_step(state, 72, .199999); self.assertEqual(state.hops_since_confident, 5)
        self.assertEqual(belief_step(state, 72, .2).hops_since_confident, 0)
        self.assertTrue(np.isfinite(state.covariance_array).all())
        with self.assertRaises(ContractValidationError): BeliefState(70, 0, ((1, 2), (3, 1)), 0)
        with self.assertRaises(ContractValidationError): BeliefState(70, 0, ((1, 2), (2, 1)), 0)

    def test_observation_layout_and_invalid_semantics(self):
        invalid = measurement(0, valid=False, coverage=.75)
        values = (invalid,) + tuple(measurement(i, 70 + i) for i in range(1, 12))
        obs = build_observation(frame(values=values), initial_control_state("mcd", "clip"))
        self.assertEqual(obs.array().shape, (101,)); self.assertEqual(obs.array().dtype, np.float32); self.assertTrue(np.isfinite(obs.array()).all())
        self.assertEqual(obs.values[:7], ((70 - 90) / 40, (70 - 70) / 40, 0, 0, .75, 0.0, 0.0))
        self.assertEqual(obs.values[84:88], ((70 - 90) / 40, 20 / 40, 0, 0))
        self.assertEqual(obs.values[88:100], (0,) * 12); self.assertEqual(obs.values[100], 0)
        all_invalid = tuple(measurement(i, valid=False, coverage=None if i == 0 else 0) for i in range(12))
        obs2 = build_observation(frame(values=all_invalid), initial_control_state("mcd", "clip")); self.assertEqual(obs2.values[4], 0)

    def test_agreement_excludes_self_invalid_and_is_inclusive(self):
        values = [measurement(0, 70), measurement(1, 75), measurement(2, 65), measurement(3, valid=False)] + [measurement(i, 100) for i in range(4, 12)]
        obs = build_observation(frame(values=tuple(values)), initial_control_state("mcd", "clip"))
        self.assertAlmostEqual(obs.values[5], 2 / 10); self.assertEqual(obs.values[3 * 7 + 5], 0)
        only = tuple(measurement(i, 70, valid=(i == 0)) for i in range(12)); self.assertEqual(build_observation(frame(values=only), initial_control_state("mcd", "clip")).values[5], 0)

    def test_actions_and_control_step_boundaries(self):
        s = initial_control_state("mcd", "clip")
        s, t = control_step(frame(0), s, 0); self.assertEqual((t.action_decision.executed_action, s.hold_count), (0, 1))
        s, t = control_step(frame(1), s, 1); self.assertEqual((t.action_decision.executed_action, t.action_decision.legal, s.hold_count), (0, False, 2))
        s, t = control_step(frame(2), s, 1); self.assertEqual((t.action_decision.executed_action, s.hold_count), (1, 1))
        self.assertEqual(t.observation.values[88:100], (1.0,) + (0.0,) * 11)
        with self.assertRaises(ContractValidationError): control_step(frame(4), s, 0)
        with self.assertRaises(ContractValidationError): control_step(frame(3, clip="other"), s, 0)
        fresh = initial_control_state("mcd", "other"); self.assertEqual(fresh.next_hop_idx, 0)

    def test_strict_action_patterns(self):
        valid = (
            ActionDecision(2, 2, None, 0, 1, True, None),
            ActionDecision(2, 2, 2, 1, 2, True, None),
            ActionDecision(3, 2, 2, 1, 2, False, "minimum_hold"),
            ActionDecision(3, 3, 2, 2, 1, True, None),
        )
        self.assertEqual(len(valid), 4)
        impossible = ((2, 3, None, 0, 1, True, None), (2, 2, 2, 1, 1, True, None), (3, 3, 2, 1, 1, True, None), (3, 2, 2, 2, 1, False, "minimum_hold"))
        for fields in impossible:
            with self.subTest(fields=fields), self.assertRaises(ContractValidationError): ActionDecision(*fields)

    def test_invalid_selected_predicts_only_and_replay(self):
        values = tuple(measurement(i, valid=(i != 2)) for i in range(12)); s, t = control_step(frame(values=values), initial_control_state("mcd", "clip"), 2)
        expected = belief_step(initial_control_state("mcd", "clip").belief, None, None)
        self.assertEqual(t.selected_measurement.valid, False); self.assertEqual(s.belief, expected)
        self.assertEqual(control_step(frame(values=values), initial_control_state("mcd", "clip"), 2), (s, t))
        self.assertEqual(t.frame_provenance_id, "p-0")
        with self.assertRaises(ContractValidationError): ControlTransition(t.dataset_id, t.clip_id, t.hop_idx, t.observation, t.action_decision, t.selected_measurement, t.pre_belief, t.post_belief, "", t.signal_config_id, t.control_config_id)
        with self.assertRaises(ContractValidationError):
            ControlTransition(t.dataset_id, t.clip_id, t.hop_idx, t.observation, t.action_decision, t.selected_measurement, t.pre_belief, t.post_belief, t.frame_provenance_id, "other", t.control_config_id)
        with self.assertRaises(ContractValidationError):
            ControlTransition(t.dataset_id, t.clip_id, t.hop_idx, t.observation, t.action_decision, t.selected_measurement, t.pre_belief, t.post_belief, t.frame_provenance_id, t.signal_config_id, "other")

    def test_smoke_acceptance_requires_executed_invalid(self):
        from scripts.verify_gate4_mcd_smoke import accepted_selected_invalid
        self.assertFalse(accepted_selected_invalid([{"invalid_selected": 0}, {"invalid_selected": 0}]))
        self.assertTrue(accepted_selected_invalid([{"invalid_selected": 0}, {"invalid_selected": 1}]))

    def test_import_closure_and_no_label_or_reward_surface(self):
        path = "src/adaptive_roi_rppg/control/core.py"
        with open(path, encoding="utf-8") as handle: text = handle.read().lower()
        tree = ast.parse(text)
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        for forbidden in ("data", "labels", "evaluation", "training", "legacy", "mmpd", "gymnasium", "stable_baselines3", "sb3_contrib", "torch", "reward"):
            if forbidden in ("data",): continue
            self.assertNotIn(forbidden, text)
        self.assertFalse(any(any(x in item.lower() for x in ("label", "evaluation", "training", "legacy", "mmpd", "gymnasium", "stable_baselines3", "sb3_contrib", "torch")) for item in imports))
        self.assertNotIn("gt", text)


if __name__ == "__main__": unittest.main()
