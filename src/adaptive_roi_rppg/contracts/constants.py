from enum import Enum


class ROIName(str, Enum):
    full_face = "full_face"
    forehead_left = "forehead_left"
    forehead_right = "forehead_right"
    temple_left = "temple_left"
    temple_right = "temple_right"
    nose = "nose"
    cheek_upper_left = "cheek_upper_left"
    cheek_upper_right = "cheek_upper_right"
    cheek_lower_left = "cheek_lower_left"
    cheek_lower_right = "cheek_lower_right"
    lips = "lips"
    chin = "chin"


ROI_NAMES: tuple[ROIName, ...] = tuple(ROIName)


class ManifestStatus(str, Enum):
    planned = "planned"
    building = "building"
    complete = "complete"
    failed = "failed"


class OverlapResult(str, Enum):
    zero = "zero"
    overlap = "overlap"


FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "gt_ppg",
        "gt_hr",
        "gt_hr_bpm",
        "c_seq",
        "b_seq",
        "dataset_id",
        "subject",
        "subject_id",
        "view",
        "condition",
    }
)
