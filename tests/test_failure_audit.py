import unittest

from adaptive_roi_rppg.contracts import MeasurementFrame, ROI_NAMES, ROIMeasurement
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.control import control_step, initial_control_state
from adaptive_roi_rppg.evaluation.failure_audit import (
    build_failure_audit, random_legal_requested_action, score_failure_audit,
)


def measurement(index, hr=70.0, valid=True):
    return ROIMeasurement(
        index, ROI_NAMES[index], hr if valid else None, .8 if valid else None,
        .5 if valid else None, .6 if valid else 0.0, valid,
        None if valid else "missing", (), None, 0, 1, "sig-v1",
    )


def frame(hop=0, values=None):
    values = tuple(values or (measurement(i, 70 + i) for i in range(12)))
    return MeasurementFrame("mcd", "clip", hop, float(hop + 1), values, "sig-v1",
                            any(value.valid for value in values),
                            None if any(value.valid for value in values) else "none",
                            f"p-{hop}")


class FailureAuditTests(unittest.TestCase):
    def test_known_best_roi_and_gt_is_joined_only_when_scoring(self):
        values = tuple(measurement(i, 72.0 if i == 2 else 120.0) for i in range(12))
        audit = build_failure_audit(frame(values=values), initial_control_state("mcd", "clip"), 5)
        self.assertEqual(tuple(item.requested_action for item in audit.alternatives), tuple(range(12)))
        self.assertEqual(audit.selected.proposed_action, 5)
        self.assertFalse(any(name in audit.__dataclass_fields__ for name in ("gt_hr_bpm", "absolute_error_bpm")))
        scored = score_failure_audit(audit, 72.0)
        self.assertEqual(scored.category, "ok")
        self.assertEqual(scored.best_valid_actions, (2,))
        self.assertGreater(scored.policy_minus_best_immediate_regret_bpm, 0.0)

    def test_all_invalid_measurements_have_no_valid_immediate_action(self):
        values = tuple(measurement(i, valid=False) for i in range(12))
        scored = score_failure_audit(build_failure_audit(frame(values=values), initial_control_state("mcd", "clip"), 2), 72)
        self.assertEqual(scored.category, "no_valid_immediate_action")
        self.assertIsNone(scored.best_valid_error_bpm)

    def test_selected_invalid_still_reports_best_valid_alternative_and_regret(self):
        values = tuple(measurement(i, 72.0 if i == 2 else 120.0, valid=i != 0) for i in range(12))
        scored = score_failure_audit(build_failure_audit(frame(values=values), initial_control_state("mcd", "clip"), 0), 72.0)
        self.assertEqual(scored.category, "selected_measurement_invalid")
        self.assertEqual(scored.best_valid_actions, (2,))
        self.assertAlmostEqual(scored.best_valid_error_bpm, scored.alternatives[2].absolute_error_bpm)
        self.assertAlmostEqual(scored.policy_minus_best_immediate_regret_bpm,
                               scored.selected.absolute_error_bpm - scored.best_valid_error_bpm)

    def test_tied_best_requested_actions_are_all_exposed(self):
        values = tuple(measurement(i, 72.0 if i in (2, 3) else 120.0) for i in range(12))
        scored = score_failure_audit(build_failure_audit(frame(values=values), initial_control_state("mcd", "clip"), 5), 72.0)
        self.assertEqual(scored.best_valid_actions, (2, 3))

    def test_alternatives_share_pre_state_and_do_not_chain(self):
        state = initial_control_state("mcd", "clip")
        audit = build_failure_audit(frame(), state, 4)
        self.assertTrue(all(item.transition.pre_belief == state.belief for item in audit.alternatives))
        self.assertTrue(all(item.transition.hop_idx == state.next_hop_idx for item in audit.alternatives))
        self.assertEqual(audit.alternatives[0].transition, control_step(frame(), state, 0)[1])

    def test_minimum_hold_preserves_requested_and_executed_actions(self):
        state, _ = control_step(frame(), initial_control_state("mcd", "clip"), 0)
        audit = build_failure_audit(frame(hop=1), state, 3)
        self.assertEqual((audit.selected.proposed_action, audit.selected.executed_action), (3, 0))
        alternative = audit.alternatives[3]
        self.assertEqual((alternative.proposed_action, alternative.executed_action), (3, 0))
        self.assertFalse(alternative.transition.action_decision.legal)

    def test_random_selector_is_stable_and_legal(self):
        state, _ = control_step(frame(), initial_control_state("mcd", "clip"), 0)
        a = random_legal_requested_action(state, 91)
        self.assertEqual(a, random_legal_requested_action(state, 91))
        self.assertEqual(a, 0)  # only staying is legal during the minimum hold
        state, _ = control_step(frame(hop=1), state, 0)
        self.assertIn(random_legal_requested_action(state, 91), range(12))

    def test_nonfinite_gt_is_rejected(self):
        audit = build_failure_audit(frame(), initial_control_state("mcd", "clip"), 0)
        for value in (float("nan"), float("inf"), True, "72"):
            with self.subTest(value=value), self.assertRaises(ContractValidationError):
                score_failure_audit(audit, value)


if __name__ == "__main__":
    unittest.main()
