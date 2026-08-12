import ast
import pathlib
import tomllib
import unittest

from adaptive_roi_rppg.contracts.records import validate_observation_field_names
from adaptive_roi_rppg.contracts.errors import ContractValidationError


class BoundaryTests(unittest.TestCase):
    def test_observation_boundary(self):
        safe = tuple(f"field_{index}" for index in range(101))
        self.assertEqual(validate_observation_field_names(safe), safe)
        self.assertEqual(validate_observation_field_names(("dataset_id_provenance",)), ("dataset_id_provenance",))
        with self.assertRaises(ContractValidationError):
            validate_observation_field_names(())
        for forbidden in ("gt_ppg", "gt_hr", "gt_hr_bpm", "c_seq", "b_seq", "dataset_id", "subject", "subject_id", "view", "condition", "gt_custom"):
            with self.assertRaises(ContractValidationError):
                validate_observation_field_names((forbidden,))
        with self.assertRaises(ContractValidationError):
            validate_observation_field_names(("x", "x"))

    def test_contract_import_closure_and_no_legacy_paths(self):
        source_root = pathlib.Path(__file__).parents[1] / "src" / "adaptive_roi_rppg" / "contracts"
        forbidden_modules = {"numpy", "scipy", "pydantic", "gymnasium", "stable_baselines", "legacy", "signal", "labels", "control", "training", "evaluation", "mmpd", "data"}
        for path in source_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    self.assertFalse(any(part in forbidden_modules for part in name.lower().split(".")), f"forbidden import {name} in {path}")
            self.assertNotIn("/Users/atharva/Desktop/RL for RoI", path.read_text(encoding="utf-8"))
            self.assertNotIn("MMPD", path.read_text(encoding="utf-8"))

    def test_no_gate1_capability_or_dataset_class(self):
        import adaptive_roi_rppg.contracts.manifests as manifests

        self.assertFalse(hasattr(manifests, "MCDTrainingManifest"))
        self.assertFalse(hasattr(manifests, "EvaluationDatasetManifest"))
        self.assertFalse(any("Training" in name or "Evaluation" in name for name in dir(manifests)))

    def test_exact_pyproject(self):
        root = pathlib.Path(__file__).parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        project = config["project"]
        self.assertEqual(project["name"], "adaptive-roi-rppg")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["dependencies"], ["numpy>=2.2,<3", "scipy>=1.17,<2"])
        self.assertFalse(config["tool"]["uv"]["package"])
