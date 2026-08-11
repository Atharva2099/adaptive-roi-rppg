# Data contract

## Dataset custody

MCD is the only training and development dataset. MMPD is evaluation-only for frozen MCD-trained checkpoints. MMPD local custody operations require explicit per-turn approval naming the exact source, destination, and operation. No MMPD-derived threshold, bucket, error pattern, or result may change MCD design.

## Identifiers, ordering, and ROI names

`clip_id` (or stem) is opaque stable text. Uniqueness keys are `(dataset_id, clip_id, frame_idx)` for raw frames, `(dataset_id, clip_id, hop_idx)` for measurement/label hops, and `(run_id, method_id, seed, checkpoint_sha256, dataset_id, clip_id)` for per-clip results.

`dataset_id` is provenance on every canonical frame, measurement, label, transition, and result. It is never controller input. The same clip ID may occur in two datasets because identity is `(dataset_id, clip_id)`. Frame indices and timestamps are strictly increasing within a clip. Subject, view, and condition are manifest fields, not ad hoc string splits.

The exact ROI order is:

| Index | ROI |
|---:|---|
| 0 | `full_face` |
| 1 | `forehead_left` |
| 2 | `forehead_right` |
| 3 | `temple_left` |
| 4 | `temple_right` |
| 5 | `nose` |
| 6 | `cheek_upper_left` |
| 7 | `cheek_upper_right` |
| 8 | `cheek_lower_left` |
| 9 | `cheek_lower_right` |
| 10 | `lips` |
| 11 | `chin` |

## Raw MCD frames

Required state columns, in order, are `frame_idx`, `head_yaw`, `head_pitch`, `head_roll`, followed for every ROI in canonical order by `r_mean`, `g_mean`, `b_mean`, `std`, `coverage`. This is 64 columns: 4 global plus 12 x 5. Frame index is integer >=0; pose is in degrees; RGB/statistics are finite numeric values in the source scale recorded by the manifest; coverage is expected in [0,1]. Rescaling requires a transformation record.

GT CSV requires `frame_idx`, `gt_ppg`. Alignment is an exact one-to-one frame-index join; missing or duplicate keys fail unless a separately approved policy is named. FullHDwebcam and USBVideo are 30 Hz; IriunWebcam is 24 Hz. Unknown cameras fail closed.

## Canonical records

`CanonicalFrame` contains dataset ID, clip ID, frame index, timestamp, camera FPS, pose, 12 ordered `ROIFrameValue` records, and provenance. It has no GT.

`ROIMeasurement` contains `roi_index`, `roi_name`, `hr_bpm`, `confidence`, `peak_power_ratio`, `coverage`, `valid`, `invalid_reason`, `imputed_channels`, `max_imputation_age_frames`, `source_frame_start`, `source_frame_end`, and `signal_config_id`. `MeasurementFrame` contains dataset ID, clip ID, hop index/time, 12 measurements, signal-config ID, and causal validity/provenance. It has no GT.

`LabelFrame` contains dataset ID, clip ID, hop index/time, GT HR, GT-rule ID, and validity. GT and teacher labels live in separate files/contracts from inference measurements.

Required field contract:

| Record | Exact field or field group | Type / unit | Nullability | Validation |
|---|---|---|---|---|
| `CanonicalFrame` | `dataset_id` | `str` / identifier | non-null | stable dataset identity; never model input |
| `CanonicalFrame` | `clip_id` | `str` / opaque identifier | non-null | stable clip identity |
| `CanonicalFrame` | `frame_idx` | `int` / frame index | non-null | integer >=0; strictly increasing within clip |
| `CanonicalFrame` | `timestamp_s` | `float` / seconds | non-null | finite; strictly increasing within clip |
| `CanonicalFrame` | `camera_fps` | `float` / Hz | non-null | finite; FullHDwebcam/USBVideo 30, IriunWebcam 24; unknown fails |
| `CanonicalFrame` | `head_yaw_deg`, `head_pitch_deg`, `head_roll_deg` | `float` / degrees | non-null when frame valid | finite; source validity is manifest-declared |
| `CanonicalFrame` | `roi_values` | fixed ordered 12 `ROIFrameValue` records | non-null | exact canonical ROI order; no GT |
| `CanonicalFrame` | `provenance_id` | `str` / provenance identifier | non-null | resolves to source manifest; no GT |
| `ROIFrameValue` | `roi_index` | `int` / index | non-null | exactly 0-11 and matches order |
| `ROIFrameValue` | `roi_name` | enum `str` / canonical name | non-null | exact index/name mapping |
| `ROIFrameValue` | `r_mean`, `g_mean`, `b_mean`, `std` | `float` / manifest-declared source scale | nullable only when invalid | finite when valid; range is manifest-declared, not invented |
| `ROIFrameValue` | `coverage` | `float` / fraction | nullable only when invalid | finite; expected [0,1] when valid |
| `ROIFrameValue` | `valid` | `bool` / unitless | non-null | true only when required numeric fields are finite |
| `ROIFrameValue` | `invalid_reason` | `str` / reason | nullable | required when `valid=false`; null when valid |
| `ROIFrameValue` | `imputation_age_frames` | mapping channel `str` to nonnegative `int` | mapping or null | only filled channels appear; ages are nonnegative |
| `ROIFrameValue` | `imputation_origin_frame_idx` | mapping channel `str` to nonnegative `int` / frame index | mapping or null | only filled channels appear; each origin index <= current `frame_idx`; pairs one-to-one with `imputation_age_frames` |
| `ROIMeasurement` | `roi_index` | `int` / index | non-null | exactly 0-11 and matches canonical order |
| `ROIMeasurement` | `roi_name` | enum `str` / canonical name | non-null | exact index/name mapping |
| `ROIMeasurement` | `hr_bpm` | `float` / BPM | nullable only when invalid | finite when valid; source/config range is manifest-declared |
| `ROIMeasurement` | `confidence` | `float` / unitless | nullable only when invalid | finite when valid; confidence semantics belong to signal config |
| `ROIMeasurement` | `peak_power_ratio` | `float` / unitless | nullable only when invalid | finite when valid; ratio semantics belong to signal config |
| `ROIMeasurement` | `coverage` | `float` / fraction | nullable only when invalid | finite; expected [0,1] when valid |
| `ROIMeasurement` | `valid` | `bool` / unitless | non-null | false when required measurement is unavailable |
| `ROIMeasurement` | `invalid_reason` | `str` / reason | nullable | required when invalid; null when valid |
| `ROIMeasurement` | `imputed_channels` | `list[str]` / channel names | non-null, possibly empty | names are known RGB channels; agrees with source imputation flags |
| `ROIMeasurement` | `max_imputation_age_frames` | nonnegative `int` / frames | nullable | null when no channels were imputed; bound is declared, not guessed |
| `ROIMeasurement` | `source_frame_start`, `source_frame_end` | `int` / inclusive frame indices | non-null for a constructed measurement | start <= end; end is not after hop timestamp |
| `ROIMeasurement` | `signal_config_id` | `str` / configuration identifier | non-null | resolves to frozen signal settings |
| `MeasurementFrame` | `dataset_id` | `str` / identifier | non-null | provenance only; not observation input |
| `MeasurementFrame` | `clip_id` | `str` / opaque identifier | non-null | stable clip identity |
| `MeasurementFrame` | `hop_idx` | `int` / hop index | non-null | integer >=0; unique within clip |
| `MeasurementFrame` | `hop_time_s` | `float` / seconds | non-null | finite; monotonic and aligned to source frames |
| `MeasurementFrame` | `measurements` | fixed ordered 12 `ROIMeasurement` records | non-null | exact ROI order; no GT |
| `MeasurementFrame` | `signal_config_id` | `str` / configuration identifier | non-null | all measurements resolve to this config |
| `MeasurementFrame` | `valid` | `bool` / unitless | non-null | false when the frame has no usable measurement set |
| `MeasurementFrame` | `invalid_reason` | `str` / reason | nullable | required when invalid; null when valid |
| `MeasurementFrame` | `provenance_id` | `str` / provenance identifier | non-null | resolves to causal source range; no GT |
| `LabelFrame` | `dataset_id`, `clip_id` | `str` / identifiers | non-null | exact validated join provenance |
| `LabelFrame` | `hop_idx` | `int` / hop index | non-null | exact measurement join key |
| `LabelFrame` | `hop_time_s` | `float` / seconds | non-null | finite and aligned to hop |
| `LabelFrame` | `gt_hr_bpm` | `float` / BPM | nullable when invalid | finite when valid; generated under approved GT rule |
| `LabelFrame` | `gt_rule_id` | `str` / rule identifier | non-null | resolves to approved rule; separate from measurements |
| `LabelFrame` | `valid` | `bool` / unitless | non-null | false when GT is unavailable or invalid |
| `LabelFrame` | `invalid_reason` | `str` / reason | nullable | required when invalid; null when valid |
| `ClipManifest` | `dataset_id`, `clip_id` | `str` / identifiers | non-null | stable pair identity |
| `ClipManifest` | `clip_manifest_id` | `str` / identifier | non-null | stable clip-manifest identity |
| `ClipManifest` | `subject_id`, `view`, `condition`, `camera_id` | `str` / provenance-only metadata | non-null when manifest requires it | metadata only; never observation fields |
| `ClipManifest` | `camera_fps` | `float` / Hz | non-null | matches known camera mapping |
| `ClipManifest` | `state_locator`, `state_sha256`, `state_row_count` | `str`, 64-hex `str`, `int` / locator, hash, rows | required | hash and row count verify state source |
| `ClipManifest` | optional `gt_locator`, `gt_sha256`, `gt_row_count` | `str`, 64-hex `str`, `int` / label source | nullable only in measurement-only manifest | present only in label-capable manifest; exact frame join |
| `ClipManifest` | `split_id`, `schema_id` | `str` / identifiers | non-null | resolve to split and schema definitions |
| `ClipManifest` | `status` | enum `str` / `planned`, `building`, `complete`, `failed` | non-null | only `complete` manifests are accepted |
| `SplitManifest` | `split_id`, `dataset_id` | `str` / identifiers | non-null | stable split identity |
| `SplitManifest` | `train_subject_ids`, `eval_subject_ids`, `train_clip_ids`, `eval_clip_ids` or hashed referenced tables | lists of `str` or hash references | non-null | subject- and stem-disjoint; exact counts |
| `SplitManifest` | `train_subject_count`, `eval_subject_count`, `train_clip_count`, `eval_clip_count` | `int` / subjects and clips | non-null | agree with referenced tables |
| `SplitManifest` | `overlap_result` | enum `str` / `zero` or failure | non-null | must be `zero` for an accepted split |
| `SplitManifest` | `source_hashes` | list of 64-hex `str` | non-null | every referenced source hash verifies |
| `SplitManifest` | `status` | enum `str` / `planned`, `building`, `complete`, `failed` | non-null | only complete manifests can train/evaluate |
| `DatasetManifest` | `manifest_id`, `dataset_id`, `schema_id` | `str` / identifiers | non-null | stable manifest and dataset identity |
| `DatasetManifest` | `clip_manifest_refs`, `clip_manifest_hashes` | ordered lists of IDs and 64-hex hashes | non-null | order and hashes verify every clip manifest |
| `DatasetManifest` | `cohort`, `subject_count`, `clip_count` | structured metadata, `int`, `int` | non-null | agree with ordered references |
| `DatasetManifest` | `source_inventory_sha256` | 64-hex `str` | non-null | verifies source inventory |
| `DatasetManifest` | `transform_ids` | ordered list of `str` | non-null | every transform is named and reproducible |
| `DatasetManifest` | `created_at_utc` | timestamp `str` / UTC | non-null | valid UTC timestamp |
| `DatasetManifest` | `producer_command` | `str` / command text | non-null | records the producer invocation |
| `DatasetManifest` | `status` | enum `str` / `planned`, `building`, `complete`, `failed` | non-null | only complete manifests can train/evaluate |
| `RunInputManifest` | `run_input_id` | `str` / identifier | non-null | stable run-input identity |
| `RunInputManifest` | allowed dataset class | enum `MCDTrainingManifest` or `EvaluationDatasetManifest` | non-null | class boundary is enforced; no conversion method |
| `RunInputManifest` | `dataset_manifest_id`, `dataset_manifest_sha256`, `split_manifest_id`, `split_manifest_sha256` | IDs plus 64-hex hashes | non-null | referenced manifests are complete and hash-match |
| `RunInputManifest` | `exact_clip_keys` | list of `(dataset_id, clip_id)` keys | non-null | exact cohort; no unlisted clips |
| `RunInputManifest` | `signal_config_id`, `observation_schema_id` | `str` / configuration identifiers | non-null when applicable | resolve to frozen definitions |
| `RunInputManifest` | `gt_rule_id` | `str` / rule identifier | nullable only when not label-capable | required for label-capable input; never inference observation |
| `RunInputManifest` | `transform_ids` | ordered list of `str` | non-null | all transforms are named and hashed where applicable |
| `RunInputManifest` | `seed` | `int` / unitless | nullable for deterministic evaluation | bound to result identity when present |
| `RunInputManifest` | `code_commit`, `environment_lock_sha256` | `str`, 64-hex `str` | non-null | exact source and environment provenance |
| `RunInputManifest` | `created_at_utc` | timestamp `str` / UTC | non-null | valid UTC timestamp |
| `RunInputManifest` | `status` | enum `str` / `planned`, `building`, `complete`, `failed` | non-null | only complete manifests can train/evaluate |

`ClipManifest`, `SplitManifest`, `DatasetManifest`, and `RunInputManifest` carry their exact stable ID fields, cohort and split metadata, schema/config IDs, source paths as locators, hashes, row counts, and `status`. Each uses the enum `planned`, `building`, `complete`, or `failed`; only `complete` is accepted. An inference manifest may carry dataset ID, subject, view, and condition as provenance outside the model-input payload.

Serialized missing numeric values are null plus `valid=false` and an invalid reason, never magic zero. Numeric arrays must be finite when marked valid.

## Label separation and historical compatibility

Inference manifests reject `gt_ppg`, `gt_hr`, `c_seq`, `b_seq`, `dataset_id`, subject, view, and condition as observation fields. Training joins measurements and labels only by validated `(dataset_id, clip_id, hop_idx)` keys. The MCD GT rule is `causal_periodogram_peak_no_subharmonic_v3`, with a centered 8-second label window and MCD subharmonic correction false. Centering is permitted for labels, not online features. The MMPD corrected-GT rule is separate and exists only in evaluation manifests.

The legacy Phase-1 NPZ compatibility envelope records exactly `hr_meas`, `conf`, `ppr`, `cov`, `gt_hr`, `c_seq`, `b_seq`, `view`, `condition`, `fs`, `stem`, plus exact dataset/schema/GT metadata. It is not the new canonical storage design because it mixes measurements, labels, and metadata.

## Missing data and causality

Full-clip linear interpolation is prohibited in canonical production builds. Past-only fill requires declared `max_fill_age_frames`; until approved, the canonical builder fails on a missing required RGB value rather than guessing. Each filled channel stores `imputation_age_frames` and `imputation_origin_frame_idx`; origin indices are no later than the current `frame_idx` and pair one-to-one with ages. A measurement records the exact inclusive source-frame range and cannot use a frame after its hop timestamp.

Generic `nan_to_num` at load time is prohibited. Non-finite values are rejected or explicitly converted to invalid records with a reason before downstream processing. The fill-age bound is **Unknown** pending an MCD-only parity/impact audit and approval.

## Split, leakage, and provenance

MCD train and test90 are subject-disjoint and stem-disjoint. Currently verified scope is 510 training subjects, 89 represented test subjects, 3,057 training clips in the teacher substrate, and zero listed overlap. The exact rows consumed by every historical checkpoint remain unproven.

Every new training checkpoint binds to a completed input manifest with exact clip IDs, subject IDs, file hashes, row counts, split version, transforms, schema version, and GT rule. Require SHA-256 for raw files, derived shards, config, source inventory, checkpoint, and outputs; `code_commit`; `environment_lock_sha256`; `created_at_utc`; `producer_command`; schema ID; dataset ID; cohort; seed; and completion state.

Manifest states are `planned`, `building`, `complete`, and `failed`. Only `complete` can train or evaluate. A completed manifest cannot be edited; a change creates a new ID. Paths are locators, not identity; hashes and stable IDs provide identity.

## Machine-checkable validation

Validate required columns and order, types/ranges, uniqueness, cross-dataset duplicate clip IDs, monotonicity, frame alignment, FPS, exact ROI set/order, finite-valid invariant, no future source frames, split overlap, expected clip/subject counts, hashes, no forbidden provenance/label fields in the 101-vector, dataset-class match, seed/checkpoint result-key uniqueness, sealed MMPD plan mutation rejection, zero-valid-hop and partial-GT failure, method-specific missing-hop detection, exact hold-counter behavior over the first four hops, exact E-003 bootstrap reproduction, TS-CAN context exclusion from causal-controller ranking, and atomic completion.
