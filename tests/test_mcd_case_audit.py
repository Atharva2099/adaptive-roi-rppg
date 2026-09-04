import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_mcd_cases import load_selection


class MCDCaseAuditTests(unittest.TestCase):
    def test_fixed_selection_contract(self):
        path = Path(__file__).resolve().parents[1] / "configs/evaluation/mcd_case_audit_v1.json"
        value = load_selection(path)
        self.assertEqual(value["seed"], 1)
        self.assertEqual(value["cases"][-1]["role"], "median_control")

    def test_selection_rejects_swapped_clip(self):
        source = Path(__file__).resolve().parents[1] / "configs/evaluation/mcd_case_audit_v1.json"
        value = json.loads(source.read_text())
        value["cases"][0]["clip_id"] = "not-a-fixed-case"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_selection(path)


if __name__ == "__main__":
    unittest.main()
