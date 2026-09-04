# MMPD extraction recovery chronology: invalid ROI measurements

**Status:** NEGATIVE RESULT / ENGINEERING FAILURE.  The completed 299-clip
MMPD run is not valid for any model, baseline, transfer, or condition claim.

## Phased chronology

The sequence below summarizes the engineering work from the initial broad
failure through the remaining 20 unresolved clips. The detailed evidence and
limitations remain below it.

| Phase | What we learned | Current status |
|---|---|---|
| 1. Initial 299-clip run | All 15,847 selected measurements were invalid, so every policy retained 70 BPM. | **NEGATIVE RESULT:** not a scientific evaluation. |
| 2. Input and source diagnosis | Non-contiguous MAT-derived buffers prevented MediaPipe detection; after that was fixed, exact-zero terminal frames and intermittent misses became visible. | **FACT:** current reader removes only a contiguous exact-zero suffix; other invalid data fail. |
| 3. `p1_1` recovery experiments | Same-frame IMAGE recovery worked for some misses; a one-frame geometry bridge completed `p1_1` using prior geometry on current pixels. | **SUPERSEDED:** the bridge was removed from the current extractor. |
| 4. Historical 86-clip audit | Terminal cleanup, IMAGE recovery, and the bridge reduced the recorded audit to 22 hard failures: 17 no-face and 5 empty ROI. | **FACT, historical:** recorded Polaris evidence, not freshly reverified here; its 64/86 count is not current coverage. |
| 5. Boundary and recovery attempts | Boundary-safe floor/ceil/clamp fixed empty ROI handling. Crop/detector-guided recovery could label a forearm as a face. | Boundary repair retained; crop/detector recovery **SUPERSEDED**. |
| 6. Current full-frame audit | Full-frame VIDEO, then same-frame IMAGE, either yields current-frame geometry or hard failure. Two of 22 completed and 20 remained unresolved. | **FACT:** 12,154 frames emitted, with zero bridges, blank rows, invalid ROI values, or unexpected sources. |
| 7. Remaining 20 and next work | The remaining cases are mostly profile/turn failures, with two occlusions and one motion-blur case; three are not yet classified at their exact new failure frames. | **NEXT:** keep this strict reference, compare an official ruler separately, then design and freeze any tracker on MCD only. |

## Historical evidence: initial 299-clip run

| Item | Recorded detail |
|---|---|
| Job | `47826`, completed with exit code 0 |
| Output | `/Users/924254653/mmpd_gate9_20260817/gate9_engineering_run_20260821_prod_retry1` |
| Cohort | 299 clips, 15 subjects; `p29_3` excluded by the engineering GT rule |
| Rows | 158,470 `per_hop`, 2,990 `per_clip`, and 150 `per_subject` rows |

The runner completed and wrote the expected rows, but that did not establish
valid physiological measurement.

## Observed failure in the initial run

The ten primary arms all reported exactly the same equal-clip MAE:
`19.07306982394143` BPM.

This was not a genuine tie.  Direct comparison of the authoritative CSV
between `full_face_pos` and `advantage_ppo_seed1` found:

| Check | Result |
|---|---:|
| Shared `(clip_id, hop_idx)` rows | 15,847 |
| Different executed ROI actions | 7,774 |
| Different post-belief HR values | 0 |
| Different absolute errors | 0 |
| Valid selected measurements | 0 / 15,847 |
| Invalid reason | `missing_required_rgb` on 15,847 / 15,847 |
| Post-belief HR | exactly `70.0` BPM on 15,847 / 15,847 |

Thus policies did issue and execute different ROI choices, but no selected ROI
ever supplied a valid POS measurement.  The controller therefore received no
measurement update and retained the initial 70 BPM belief.  Each arm's MAE is
only the error of that fixed 70 BPM value against the MMPD labels.

## Historical fault location: initial run

The failure is upstream of model replay and aggregation, in the MMPD
face-landmark / ROI-RGB / POS measurement path.

At that stage, `extract_mmpd_canonical_frames` marked ROI values invalid when a
face was unavailable, and `build_pos_measurements` emitted
`missing_required_rgb` when the causal POS window lacked RGB samples.
`control_step` skipped the belief update, while the runner recorded the action
and invalidity. The output did not include selected ROI HR, confidence, PPR,
or crop geometry, which made the complete collapse less visible before the
aggregate table was inspected. Later jobs identified the missing
C-contiguous MediaPipe input buffer as the cause of the all-frame collapse.

## Impact and disposition

- Do not compare the 19.0731 BPM values across models or against MCD.
- Do not use this run to evaluate conditions, subjects, Oracle arms, or
  cross-dataset transfer.
- Keep the immutable completed output and this report as the engineering
  failure record; do not overwrite it.
- This result must not guide MCD training, policy selection, or any tuning.

## Resolution of the initial follow-up requirements

The initial requests for a focused extraction diagnosis, distinct valid ROI
measurements, regression coverage, and a new engineering run were addressed
by jobs 47840–47844 and the later 86-clip and 22-clip audits documented below.
Those jobs established the input-layout cause, the terminal all-zero behavior,
the intermittent detection misses, and the repaired current-frame contract.
The original 299-clip result remains invalid evidence; no further launch is
pending from this historical section.

## Follow-up extraction diagnosis: 2026-08-22

This section records the narrow, one-clip investigation that identified why
the original MMPD run had no usable ROI measurements.  It is engineering
evidence only.  It does not evaluate a policy or report a transfer result.

### Question

Does the current MMPD reader deliver the visible `p1_0` face to MediaPipe in
the same usable RGB representation as the prior working implementation?

### What was observed

The raw `p1_0` video is shaped `(1800, 320, 240, 3)`, `float32`, and its first
frame visibly contains a frontal face.  Before the repair, the current
extractor reported `no_face` for all 1,800 frames.  The contact sheet showed
the face but no ROI rectangles, so the failure occurred before ROI cropping,
POS, policy replay, or belief update.

Read-only comparison with the historical implementation found one relevant
input-boundary difference:

| Step | Historical implementation | Current implementation before repair |
|---|---|---|
| MMPD pixel conversion | Clip `[0, 1]`, scale to RGB `uint8` | Scale to `uint8` |
| MediaPipe image buffer | Explicit `np.ascontiguousarray(rgb_u8)` | Passed the MAT-derived array directly |
| Detector mode and settings | VIDEO mode, one face, 0.5 thresholds, frame timestamps | Same |

The mode, thresholds, timestamps, and RGB convention already matched.  The
missing C-contiguous buffer conversion was therefore the only demonstrated
behavioral difference at the MediaPipe boundary.

### Repair

At `src/adaptive_roi_rppg/evaluation/adapters/mmpd/extraction.py`, the
extractor now builds `rgb_u8` with the existing scaling rule and passes
`np.ascontiguousarray(rgb_u8)` to `mp.Image`.  This changes only the memory
layout supplied to MediaPipe.  It does not alter pixel values, RGB order,
detector settings, timestamps, crop geometry, POS, controller logic, or the
hard failure for invalid source measurements.

`tests/test_current_extraction_diagnostic.py` now checks that the MediaPipe
test double receives the expected RGB values, `uint8` dtype, and a
C-contiguous array.  Nine focused synthetic tests passed after the repair.

### Polaris timeline

Times below are the scheduler timestamps reported by Polaris (`sacct` does
not attach a timezone to these fields).

| Scheduler interval | Job | Observation | Resulting action |
|---|---:|---|---|
| 2026-08-22 17:37:30–17:37:39 | 47840 | Rounded ROI bounds produced coverage slightly above 1.0. | Clamp coverage to 1.0 after a nonempty rounded crop. |
| 2026-08-22 17:39:28–17:42:02 | 47841 | Extraction completed, but image writing rejected float MMPD display data. | Convert only the contact-sheet display copy to `uint8`; extraction input unchanged. |
| 2026-08-22 17:44:27–17:46:55 | 47842 | All MMPD frames reported `no_face`; the saved contact sheet nevertheless showed a face. | Compare the actual MediaPipe input construction with the historical implementation. |
| 2026-08-22 18:07:34–18:10:10 | 47843 | With a C-contiguous input buffer, face detection succeeded on frames 0–1798. | Keep the repair; retain strict failure for the remaining invalid frame. |

### Result of the repaired `p1_0` run

Job `47843` wrote the per-frame CSV, per-hop CSV, and ROI-overlay contact
sheet to:

`/Users/924254653/mmpd_gate9_20260817/diagnostics/current_extraction_20260822_s1020_p1_contiguous`

The per-frame CSV records 1,799 frames with `face_valid=True` and one frame
with `face_valid=False`.  Frames 0 through 1798 have a detected face and
valid full-face RGB measurements.  Frame 1799 is not a detector-layout
failure: the raw source frame itself is all zeros (`min = max = mean = 0.0`).
The diagnostic therefore exits nonzero with `no_face` at frame 1799, after
writing the inspection artifacts.  This strict failure is intentional: a
black source frame must not be silently ignored, substituted, or
forward-filled.

### One-clip frozen-policy smoke

To check that extraction was the only blocker in the usable part of the
pipeline, Polaris job `47844` ran from 2026-08-22 18:37:50 to 18:38:42
(scheduler timestamps, timezone not attached by `sacct`).  It used the real
`advantage_ppo_seed1` frozen checkpoint and the existing MMPD extraction,
causal POS, MMPD replay, and causal-label code on `p1_0`.  This is a
one-clip engineering smoke, not an evaluation result.

The smoke wrote its 53 rows to:

`/Users/924254653/mmpd_gate9_20260817/diagnostics/p1_0_advantage_ppo_seed1_smoke_20260822/per_hop.csv`

| Check | Observed result |
|---|---:|
| Decision hops written | 53 / 53 |
| Valid selected measurements | 52 / 53 |
| Invalid selected measurement | hop 52, `missing_required_rgb` |
| Distinct executed ROI actions | 7 |
| Distinct post-belief values | 53 |
| Ground-truth values provided to the policy | 0 |

The job exited with status `2` after writing the CSV because hop 52 is
invalid.  That is the harness's intended hard failure: its final causal POS
window includes raw frame 1799, the all-zero frame.  The result demonstrates
that the repaired extractor now supplies real ROI measurements to a real
frozen recurrent policy, which changes actions and belief across the valid
hops.  It does not authorize use of the final invalid hop or a broader MMPD
evaluation.

### Parallel four-clip smoke

A requested 16-task array was rejected before it started by Polaris's
submission policy. A four-clip smoke was then accepted, although Polaris ran
two tasks at a time. It used one extraction per clip and replayed all ten arms
from the shared measurements. The task intervals were 2026-08-22
19:10:20–19:11:14 for `p1_0` and `p1_1`, then 19:11:14–19:12:02/19:12:05 for
`p1_2` and `p1_3` (scheduler timestamps without timezone).

| Clip | Rows | Complete hops | Trimmed terminal frames | Invalid selected rows |
|---|---:|---:|---:|---:|
| `p1_0` | 520 | 52 | 1 | 0 |
| `p1_1` | 520 | 52 | 1 | 150 |
| `p1_2` | 520 | 52 | 1 | 0 |
| `p1_3` | 520 | 52 | 1 | 0 |

Every clip had the same one-frame all-zero terminal suffix, which the minimal
reader removed. `p1_1` has normal finite raw pixels throughout its failed
interval, but all ten methods received `missing_required_rgb` at hops 36–49
and 51. This is therefore an intermittent face/ROI extraction failure on a
valid source, not a tail-padding problem or a policy-specific result.

A frame-level follow-up on `p1_1` found six `no_face` frames: 1310–1311,
1490, 1747, 1749, and 1799. Frame 1799 is the terminal all-zero padding
frame. The other five frames have finite, nonzero raw RGB values. Thus the
current missing RGB is caused by intermittent MediaPipe VIDEO detection loss,
not a corrupt or black source frame. The first two misses are consecutive; the
other three are isolated. Each missed RGB sample makes every causal POS window
that contains it invalid, which explains the longer run of invalid hops.

At that stage, the worker wrote its ordinary CSV and exited nonzero whenever
any selected measurement was invalid. The prior smoke predated that hard-fail
update, so it completed despite writing the diagnostic `p1_1` rows. The later
full-frame audit replaced this interim behavior with the current extractor
contract described below.

### Resolution of the initial extraction diagnosis

The all-frame face-detection collapse in the original engineering run was
caused by the non-contiguous MATLAB-derived buffer passed to MediaPipe. The
contiguous-buffer repair resolved that collapse for the valid frames in
`p1_0`. The later reader retained only a contiguous exact-zero terminal-suffix
removal rule; other invalid or nonfinite data remained hard failures. The
subsequent full-frame audit replaced this open diagnostic question with the
current contract documented below.

### Video-level repair for the isolated `p1_1` profile miss

**SUPERSEDED approach.** This section records the one-frame geometry bridge
experiment because it explains the historical recovery, but that bridge is not
part of the current extractor and must not be counted as current coverage.

MediaPipe's VIDEO mode already performs face tracking, but it can still return
no landmarks when its detection, face-presence, or tracking confidence is too
low. It does not accept a left/right-profile hint. A same-frame IMAGE retry
recovered frames 1310, 1490, 1747, and 1749, but both modes missed frame 1311.

The experimental extractor bridged exactly one such miss with the immediately
previous directly detected face box. It reused only four box coordinates. All
12 ROI RGB values were cropped from the missed frame's own pixels. A bridged
box could not bridge another frame. MCD diagnostic behavior, MediaPipe
confidence thresholds, ROI definitions, controller behavior, and signal
processing were unchanged. An exact all-zero MMPD frame still failed before
detection or bridging.

The diagnostic CSV recorded `geometry_source`,
`geometry_source_frame_idx`, and `geometry_bridged`. Focused synthetic tests
covered VIDEO detection, IMAGE recovery, one-frame bridging with distinct
current-frame RGB, non-chainable bridging, first-frame failure, exact-zero
failure after valid geometry, MCD/MMPD policy separation, and explicit overlay
selection. All 35 focused tests passed at that stage, and an independent code
review found no remaining material defect before the real run. The bridge was
then removed from the current extractor after the later full-frame contract
was adopted.

| Scheduler interval | Job | Result |
|---|---:|---|
| 2026-08-23 18:45:17–18:46:01 | 47851 | The bridge passed frame 1311 and processing stopped at the known exact-zero terminal frame 1799. |
| 2026-08-23 18:46:41–18:47:30 | 47852 | After removing the one-frame exact-zero terminal suffix, the 1,799-frame diagnostic completed and rendered frames 1310–1312. |
| 2026-08-23 18:48:24–18:49:19 | 47853 | The ten-policy `p1_1` evaluation completed and wrote 520 valid rows. |

Job 47852 wrote its row-level diagnostic and overlay to:

`/Users/924254653/mmpd_gate9_20260817/diagnostics/p1_1_video_bridge_trimmed_20260823`

The observed geometry sources were 1,794 VIDEO frames, four same-frame IMAGE
recoveries, and one previous-frame bridge. Frame 1311 used the face box from
frame 1310 while cropping frame 1311's pixels. Frames 1310, 1311, and 1312 had
no invalid ROI values, and all 52 selected diagnostic measurements were valid.

Job 47853 wrote:

`/Users/924254653/mmpd_gate9_20260817/p1_1_bridge_eval_20260823/p1_1.csv`

The CSV contains 10 methods x 52 hops = 520 rows, with zero invalid selected
measurements. The nine learned policies used between three and nine distinct
executed actions; every method produced 52 distinct post-belief values. The
reader removed one exact-zero terminal padding frame. This establishes that
the `p1_1` extraction and replay path now functions without copying prior RGB
values. The three-frame overlay remains the direct visual check of box
alignment during the bridged profile frame.

## Historical 86-clip extraction audit: job 47865, 2026-08-23

**FACT, historical evidence from the recorded Polaris audit.** The artifacts
were recorded directly in the prior run but were not freshly re-read during
this documentation pass. The audit covered 86 clips using 16 workers and took
2:57. It exited with code 123 because 22 unresolved clips were deliberately
hard-failed.

The summary and row-level files were recorded at:

`/Users/924254653/mmpd_gate9_20260817/diagnostics/extraction_audit_86_20260823/summary.json`

`/Users/924254653/mmpd_gate9_20260817/diagnostics/extraction_audit_86_20260823/per_clip_summary.csv`

Per-frame files were under the same directory's `clips/` subdirectory.

| Audit observation | Recorded result |
|---|---:|
| Total clips | 86 |
| Fully repaired by the new handling | 43 |
| Extracted cleanly on rerun | 21 |
| Failed hard | 22 |
| Recorded completed count | 64 / 86 |
| Clips with an exact-zero terminal suffix | 49 |
| Terminal frames removed | 64 |
| Frames recovered by same-frame IMAGE mode | 7 across 3 clips |
| Frames recovered by one-frame geometry bridge | 18 across 17 clips |

The 22 hard failures were 17 `no_face` clips and 5 `empty_roi` clips. The
empty-ROI cases were `p11_3` frame 0 (lips), `p17_3` frame 67 (chin), `p17_7`
frame 869 (chin), `p2_19` frame 923 (chin), and `p7_3` frame 80 (chin). The
one-frame bridge allowed only `p1_1` to complete in the later focused
experiment; it was subsequently removed. Therefore the historical 64/86
number describes that earlier mixed behavior and is not current coverage.

The 21 clips that extracted cleanly on rerun also showed that the old failure
behavior was not fully deterministic. This audit narrowed the problem, but it
did not establish reliable extraction for profiles, occlusions, or downstream
heart-rate measurement.

## Superseded bounded-recovery protocol: 2026-08-24

This was an intermediate engineering-only protocol for unresolved MMPD
extraction misses. MMPD remained evaluation-only and the experiment did not
support a transfer or angle-robustness claim. It tested boundary-safe ROI
conversion (`floor` start edges, `ceil` end edges, then clamp) together with a
bounded current-frame cropped IMAGE retry based on geometry from the preceding
frame. It prohibited blank RGB, zero fills, prior RGB, and chained geometry.

The protocol was superseded after crop- and detector-guided recovery produced
false-valid ROI geometry on a forearm in job 47986. The boundary repair was
retained; the cropped retry and any prior-frame geometry reuse were removed
from the current extractor.

## Superseded detector-guided recovery protocol: 2026-08-24

This was the follow-on protocol for current-frame detector-guided landmark
reacquisition on the original 22 clips. It used `p10_15` and `p1_9` as
descriptive occlusion controls and limited interpretation to execution and
visual classification. Job 47986 showed that restricted-crop recovery could
place ROIs on a forearm, so this protocol was stopped and did not select
thresholds, direct detector boxes, or a tracker.

## Three-clip hand-control smoke: job 47986, 2026-08-24

**Status:** NEGATIVE RESULT / ENGINEERING REFINEMENT. Crop- and
detector-guided landmark recovery has been removed. It could produce
false-valid ROI geometry on non-face pixels, so it is not a safe extraction
method.

- `p2_19` completed.
- `p10_3` hard-failed.
- `p10_15` advanced to frame 1382, but the frame 1378 overlay placed ROIs on
  the forearm.

The latter observation invalidated the crop/detector-guided recovery claim: a
returned landmark box after a restricted crop did not establish visible face
geometry. At that stage's final safe contract, the extractor accepted only
full-current-frame VIDEO landmarks, with full-current-frame IMAGE landmarks
only after a valid empty VIDEO result. Malformed landmark output and
unresolved full-frame results hard-failed without emitting an invalid frame.

Any future occlusion-aware tracker would require a separately justified
visibility/occlusion method, not another box or crop heuristic. This is
descriptive MMPD engineering evidence only; it does not support an HR,
transfer, or generalization claim.

## Full-current-frame repair and known-failure audit: 2026-08-24

The unsafe crop/detector recovery was removed. The current extractor accepts
validated landmark and ROI geometry only from the full current frame: VIDEO
FaceLandmarker first, then retries with same-frame full-frame IMAGE
FaceLandmarker only when VIDEO returns a valid empty landmark list. Malformed landmarks fail
immediately. If both full-frame modes return no face, the clip stops at that
frame without emitting an invalid frame. No earlier geometry or RGB values are
reused.

Focused verification passed 48 tests and the complete local suite passed 246
tests. Independent review found no material defect in this repaired contract.

| Job and interval | Clips | Result |
|---|---:|---|
| `47988`, 2026-08-24 20:29:56–20:30:35 | 3 | `p2_19` completed 1,794 usable frames; `p10_3` stopped at frame 621; `p10_15` stopped at frame 1,377. |
| `47989`, 2026-08-24 20:32:09–20:32:15 | 1 visual check | Frame 1,376 was the final accepted full-frame VIDEO result. Frames 1,377 and 1,378 had no accepted geometry or ROI values. |
| `47990`, 2026-08-24 20:35:24–20:36:02 | 22 | Two clips completed and 20 stopped at their first unresolved full-frame landmark miss. |

Jobs `47988` and `47990` returned code 123 because the parallel wrapper returns
nonzero when any clip intentionally hard-fails. Both jobs produced every
requested per-clip summary; the parallel runner itself did not crash.

| Full 22-clip result | Count |
|---|---:|
| Completed | 2 |
| Stopped at first unresolved frame | 20 |
| Total | 22 |

The completed clips were `p2_19` with 1,794 emitted frames and `p7_3` with
1,799 emitted frames. The 20 unresolved clips were `p10_15`, `p10_3`,
`p11_1`, `p11_3`, `p17_11`, `p17_17`, `p17_3`, `p17_7`, `p17_9`, `p1_17`,
`p1_9`, `p20_1`, `p20_17`, `p20_5`, `p20_9`, `p25_3`, `p7_1`, `p7_17`,
`p7_5`, and `p7_9`.

Across all 12,154 emitted frames, 12,152 used VIDEO landmarks and two used
same-frame IMAGE landmarks. Every accepted geometry record came from its own
current frame. There were zero bridged records, zero blank value rows, zero
invalid ROI values, and zero unexpected geometry sources. The five former
empty-lips/chin cases no longer fail at an out-of-bounds ROI crop; two now
complete and three proceed farther before stopping at a later no-face frame.

Outputs:

- Three-clip smoke:
  `/Users/924254653/mmpd_gate9_20260817/diagnostics/fullframe_repair_20260824/smoke/clips`
- Smoke visual:
  `/Users/924254653/mmpd_gate9_20260817/diagnostics/fullframe_repair_20260824/smoke/visuals/p10_15_overview.png`
- Full 22-clip audit:
  `/Users/924254653/mmpd_gate9_20260817/diagnostics/fullframe_repair_20260824/all22/clips`

What this establishes is narrow: the repaired extractor either emits complete,
finite, current-frame ROI measurements or stops at the first unresolved frame.
It does not establish reliable extraction through hands or severe profile
occlusion, heart-rate accuracy on these clips, or view-angle generalization.

## Remaining 20: cautious visual categories

The categories below are a visual interpretation from earlier nearby-frame
sheets. Exact current failure frames were not all freshly rendered, so these
labels are descriptive and provisional rather than verified failure causes.

| Earlier visual interpretation | Clips | Count |
|---|---|---:|
| Strong side-profile or large turn | `p10_3`, `p11_1`, `p17_17`, `p17_9`, `p1_17`, `p20_1`, `p20_17`, `p20_5`, `p20_9`, `p7_1`, `p7_17`, `p7_5`, `p7_9` | 13 |
| Hand or arm occlusion | `p10_15`, `p1_9` | 2 |
| Partial face or face out of frame | `p17_11` | 1 |
| Fast turn or motion blur | `p25_3` | 1 |
| Later no-face after the boundary repair; exact new failure not yet classified | `p11_3`, `p17_3`, `p17_7` | 3 |
| **Total unresolved** |  | **20** |

Of the five old empty-ROI cases, two completed in the current audit and three
advanced farther before later failing as `no_face`. That is evidence that the
boundary repair addressed the old crop failure, not evidence that those clips
are now fully solved.

## Official MMPD preprocessing as a separate comparison

The official MMPD preprocessing is a comparison ruler, not our current
solution and not a source of tuning decisions. The reviewed references are
the paper at `/Users/atharva/Desktop/RL for RoI/Master Thesis/Useful papers/MMPD Dataset details.pdf`,
the [official repository](https://github.com/THU-CS-PI/MMPD_rPPG_dataset), its
[BaseLoader implementation](https://github.com/THU-CS-PI/MMPD_rPPG_dataset/blob/main/rppg-Toolbox_MMPD/dataset/data_loader/BaseLoader.py),
and its [unsupervised inference configuration](https://github.com/THU-CS-PI/MMPD_rPPG_dataset/blob/main/rppg-Toolbox_MMPD/configs/infer_configs/MMPD_UNSUPERVISED.yaml).

That reference uses OpenCV Haar detection on the first frame, expands the box
by 1.5 times, resizes to 72x72, and keeps a fixed crop. If no face is found,
processing continues with a broad fallback crop. Continuing processing does
not establish that the fallback contains facial pixels or that its POS signal
is valid. No per-clip failed/exclusion list was found in the reviewed
paper/repository; condition-level subset exclusions do exist. This comparison
is useful for protocol documentation, but it does not validate the current
strict extractor or justify silently excluding failures. See
`docs/evaluation_protocol.md` for the evaluation-only reporting boundary.

## Next plan

This is a proposed plan, not implemented work:

1. Retain the strict current extractor as the reference: full-frame VIDEO,
   same-frame full-frame IMAGE fallback, validated current-frame landmark and
   ROI geometry, and hard failure when that geometry is unavailable.
2. Reproduce the official OpenCV method as a separate preprocessing ruler,
   reporting its fallback behavior and coverage separately from the strict
   method.
3. Any future MCD tracker or feature work must be justified and evaluated from
   independent MCD evidence; this MMPD failure does not select its design.
4. Rerun all MMPD clips once, then report coverage, failure categories, and HR
   together, with no silent exclusion. MMPD remains evaluation-only; its
   results must not be fed back into MCD training, tuning, calibration,
   checkpoint selection, reward design, or feature design.
