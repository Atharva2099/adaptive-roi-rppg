import copy
import math
import unittest

from adaptive_roi_rppg.contracts import (
    CanonicalFrame,
    LabelFrame,
    MeasurementFrame,
    ROIName,
    ROI_NAMES,
    ROIFrameValue,
    ROIMeasurement,
)
from adaptive_roi_rppg.contracts.errors import ContractValidationError


def frame_value(index=0, *, valid=True, origin=None, age=None):
    return ROIFrameValue(
        index,
        ROI_NAMES[index],
        1.0 if valid else None,
        2.0 if valid else None,
        3.0 if valid else None,
        0.1 if valid else None,
        0.5 if valid else None,
        valid,
        None if valid else "missing",
        {"r_mean": age} if age is not None else None,
        {"r_mean": origin} if origin is not None else None,
    )


def measurement(index=0, *, valid=True, config="signal-v1"):
    return ROIMeasurement(
        index,
        ROI_NAMES[index],
        70.0 if valid else None,
        0.8 if valid else None,
        0.5 if valid else None,
        0.5 if valid else None,
        valid,
        None if valid else "invalid",
        (),
        None,
        0,
        10,
        config,
    )


class RecordTests(unittest.TestCase):
    def test_round_trip_and_exact_roi_order(self):
        values = tuple(frame_value(index) for index in range(12))
        record = CanonicalFrame("d", "c", 4, 4.0, 30.0, 0.0, 1.0, 2.0, values, "p")
        self.assertEqual(CanonicalFrame.from_dict(record.to_dict()), record)
        self.assertEqual(record.roi_values[0].roi_name, ROIName.full_face)

    def test_missing_reordered_duplicate_and_mismatch_rejected(self):
        values = [frame_value(index) for index in range(12)]
        with self.assertRaises(ContractValidationError):
            CanonicalFrame("d", "c", 0, 0.0, 30.0, 0.0, 0.0, 0.0, values[:-1], "p")
        values[0], values[1] = values[1], values[0]
        with self.assertRaises(ContractValidationError):
            CanonicalFrame("d", "c", 0, 0.0, 30.0, 0.0, 0.0, 0.0, values, "p")
        values = [frame_value(index) for index in range(12)]
        values[1] = frame_value(0)
        with self.assertRaises(ContractValidationError):
            CanonicalFrame("d", "c", 0, 0.0, 30.0, 0.0, 0.0, 0.0, values, "p")

    def test_validity_numeric_and_coverage_rules(self):
        with self.assertRaises(ContractValidationError):
            frame_value(0, valid=True).__class__(0, ROI_NAMES[0], math.nan, 1.0, 1.0, 1.0, 0.5, True, None, None, None)
        with self.assertRaises(ContractValidationError):
            frame_value(0, valid=False).__class__(0, ROI_NAMES[0], None, None, None, None, 1.1, False, "missing", None, None)
        with self.assertRaises(ContractValidationError):
            frame_value(0, valid=False).__class__(0, ROI_NAMES[0], None, None, None, None, None, False, None, None, None)

    def test_imputation_pair_and_causality(self):
        future = [frame_value(0, origin=1, age=0)] + [frame_value(i) for i in range(1, 12)]
        with self.assertRaises(ContractValidationError):
            CanonicalFrame("d", "c", 0, 0.0, 30.0, 0.0, 0.0, 0.0, future, "p")
        CanonicalFrame("d", "c", 4, 4.0, 30.0, 0.0, 0.0, 0.0, [frame_value(0, origin=2, age=2)] + [frame_value(i) for i in range(1, 12)], "p")
        with self.assertRaises(ContractValidationError):
            CanonicalFrame("d", "c", 4, 4.0, 30.0, 0.0, 0.0, 0.0, [frame_value(0, origin=2, age=1)] + [frame_value(i) for i in range(1, 12)], "p")

    def test_defensive_copies(self):
        age = {"r_mean": 2}
        origin = {"r_mean": 3}
        record = frame_value(0)
        record = ROIFrameValue(0, ROI_NAMES[0], 1.0, 2.0, 3.0, 0.1, 0.5, True, None, age, origin)
        age["r_mean"] = 99
        origin["r_mean"] = 99
        self.assertEqual(record.imputation_age_frames["r_mean"], 2)
        self.assertEqual(record.to_dict()["imputation_origin_frame_idx"]["r_mean"], 3)
        nested = copy.deepcopy(record.to_dict())
        nested["imputation_age_frames"]["r_mean"] = 100
        self.assertEqual(record.imputation_age_frames["r_mean"], 2)

    def test_measurement_and_label_contracts(self):
        valid_measurements = tuple(measurement(index) for index in range(12))
        frame = MeasurementFrame("d", "c", 0, 0.0, valid_measurements, "signal-v1", True, None, "p")
        self.assertEqual(MeasurementFrame.from_dict(frame.to_dict()), frame)
        with self.assertRaises(ContractValidationError):
            MeasurementFrame("d", "c", 0, 0.0, valid_measurements, "other", True, None, "p")
        label = LabelFrame("d", "c", 0, 0.0, 70.0, "gt-v1", True, None)
        self.assertEqual(LabelFrame.from_dict(label.to_dict()), label)
        invalid = LabelFrame("d", "c", 0, 0.0, None, "gt-v1", False, "missing")
        self.assertEqual(LabelFrame.from_dict(invalid.to_dict()), invalid)
        with self.assertRaises(ContractValidationError):
            LabelFrame("d", "c", 0, 0.0, 70.0, "gt-v1", False, "missing")
        with self.assertRaises(ContractValidationError):
            MeasurementFrame("d", "c", 0, -0.1, valid_measurements, "signal-v1", True, None, "p")
        with self.assertRaises(ContractValidationError):
            LabelFrame("d", "c", 0, -0.1, 70.0, "gt-v1", True, None)

    def test_measurement_source_and_imputation_rules(self):
        with self.assertRaises(ContractValidationError):
            ROIMeasurement(0, ROI_NAMES[0], 70.0, 0.8, 0.5, 0.5, True, None, ("r_mean",), None, 0, 10, "s")
        with self.assertRaises(ContractValidationError):
            ROIMeasurement(0, ROI_NAMES[0], 70.0, 0.8, 0.5, 0.5, True, None, (), 1, 0, 10, "s")
        with self.assertRaises(ContractValidationError):
            ROIMeasurement(0, ROI_NAMES[0], 70.0, 0.8, 0.5, 0.5, True, None, (), None, 3, 2, "s")

    def test_strict_record_deserialization(self):
        payload = frame_value(0).to_dict()
        payload.pop("valid")
        with self.assertRaises(ContractValidationError):
            ROIFrameValue.from_dict(payload)
        payload = frame_value(0).to_dict()
        payload["extra"] = True
        with self.assertRaises(ContractValidationError):
            ROIFrameValue.from_dict(payload)
