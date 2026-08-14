import inspect
import unittest

from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels, read_mcd_labels


class LabelCompatibilityTests(unittest.TestCase):
    def test_gate5_signature_remains_unchanged_and_eval_is_separate(self):
        self.assertEqual(tuple(inspect.signature(read_mcd_labels).parameters), ("bundle", "gt_root", "clip_id", "required_split", "clip_manifest"))
        self.assertEqual(tuple(inspect.signature(read_mcd_eval_labels).parameters), ("bundle", "gt_root", "clip_id", "clip_manifest"))


if __name__ == "__main__": unittest.main()
