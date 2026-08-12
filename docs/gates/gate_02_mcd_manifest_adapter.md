# Gate 2: MCD manifest adapter

Gate 2 is a small, research-focused adapter. It discovers paired MCD semantic-state and frame-level GT CSVs, validates them, loads a subject split, and publishes immutable manifests. It does not emit canonical frames, calculate HR, evaluate MMPD, or train.

Inputs are two real nonempty source directories and one UTF-8 `subject_id,split` CSV. Direct children must be regular non-symlink files with exact names `<subject>_<camera>_<before|after>` plus the fixed state or GT suffix. State and GT stems, frame indices, schemas, finite values, coverage, hashes, sizes, and row counts must agree. GT is the frame-level `gt_ppg` signal, not an HR label. The structured-missing schema is `mcd-semantic-state-64-structured-missing-paired-gt-v2`: only raw `""` means absence; pose is all three absent or all three finite; coverage is always present and finite; zero coverage requires all four ROI measurements absent, while positive coverage requires all four finite. No row is dropped or filled.

The output tree is:

```text
output/
  source_inventory.json
  split_manifest.json
  clips/mcd-clip-<64 lowercase hex>.json
  dataset_manifest.json
```

The loader revalidates semantic stem metadata, locators, hashes, row counts, split membership, cohort, IDs, and rejects missing, extra, duplicate-key, malformed, tampered, incomplete, or incorrectly ordered artifacts. IDs are content-derived and independent of absolute source roots. Source snapshots are checked before and after each parse, and both roots are re-enumerated after the build. This protects against ordinary source maintenance during a long scan. Malicious same-user swaps during millisecond publication are explicitly out of scope.

Publication writes to a unique sibling temporary directory, validates it, writes `dataset_manifest.json` last, then uses one ordinary directory rename. Existing destinations and missing parents are rejected; a pre-rename error removes only the adapter-created temporary directory.

Canonical one-time acceptance is separate from the reusable adapter:

```bash
PYTHONPATH=src python3 scripts/build_mcd_manifests.py \
  --state-root /Users/924254653/mcd_stage/precomputed_rgb/semantic_state_vectors_12roi_full \
  --gt-root /Users/924254653/mcd_stage/precomputed_rgb/ground_truths_full \
  --split-csv /Users/924254653/adaptive_roi_gate2_20260811.VhX0D1/mcd_subject_split_510_90_v1.csv \
  --output-dir /path/to/published-gate2 \
  --created-at-utc 2026-08-11T00:00:00Z
PYTHONPATH=src python3 scripts/verify_mcd_gate2_acceptance.py \
  /path/to/published-gate2 \
  --expectation configs/data/mcd_gate2_acceptance_v1.json
```

Run the fast test suite from the repository root before using real data:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest tests.test_mcd_adapter -v
```

For the full Polaris inventory, run the two commands above through Slurm from a hashed copy of the reviewed repository files. Use a new output directory because publication refuses to replace an existing destination. The build command prints a summary containing the manifest IDs, hashes, counts, and overlap result. Run the acceptance verifier only after the build succeeds. It must print `"status":"accepted"`; otherwise Gate 2 has not passed. The generated manifests describe and authenticate the source files. They do not copy, preprocess, or modify the MCD dataset.

## Polaris MCD layout observed on 2026-08-11

The MCD dataset is on Polaris under `/Users/924254653/mcd_stage`. The relevant structure and direct-file counts observed by read-only inspection were:

```text
mcd_stage/
  hf_mcd_rppg/
    video/                                      3,600 files
    ppg_sync/                                   2,094 files
    db.csv
    db_subset_upto_4646.csv
  ppg_sync/                                     1,086 files
  logs/
  precomputed_rgb/
    ground_truths_full/                         3,600 files  [Gate 2 GT]
    semantic_state_vectors_12roi_full/          3,600 files  [Gate 2 state]
    face_alignment_fullface_20260428/           3,632 files
    semantic_state_vectors_12roi_full_foreheadfix_20260421/
                                                  3,601 files
    state_vectors/                              1,506 files
    tile_angles/                                1,506 files
    _precompute_ckpt/
```

Only `ground_truths_full` and `semantic_state_vectors_12roi_full` are canonical Gate 2 data roots. The other folders are observed historical or intermediate material and are not silently substituted. The accepted output is separate at `/Users/924254653/adaptive_roi_gate2_reviewed_clean_20260811.fdGR97/mcd_manifest_structured_v2`, containing 3,600 clip manifests plus `source_inventory.json`, `split_manifest.json`, and `dataset_manifest.json`.

The verifier records the expectation-file path and SHA-256 alongside the three published artifact hashes. The canonical expectation is split SHA `efed747047950bf885797769ed717882a577302a76355896d95f035380d17d6b`, 510 train subjects, 90 eval subjects, 600 total subjects, 3,600 clips, and all six camera-condition combinations per subject. Historical evidence represented 89 raw eval subjects and 533 clips; that is evidence context, not a Gate 2 acceptance rule.

Actual Polaris diagnostics identified structured missingness rather than malformed finite values. Job 46821 failed in 6 seconds because the old strict contract rejected blanks on the first real state row. Job 46822 reported 18,021,360 matching state+GT rows, with 824 state files, 1,697,674 blank-state rows, and 29,043,224 blank cells; all nonblank state and GT checks passed. Job 46824 found 473,236 rows with all 12 ROI groups absent, with no partial groups or coverage contradictions.

After one SOL review and Luna repair cycle, SOL passed the exact execution snapshot. Polaris job `46838` then completed in 8:56 on `lmn01` with exit code 0. The canonical verifier accepted 3,600 clips, 600 subjects, 3,060 train clips from 510 subjects, 540 evaluation clips from 90 subjects, and zero subject overlap. Independent manifest aggregation confirmed 18,021,360 state rows, the same number of GT rows, no per-clip row mismatch, correct camera/FPS mapping, all six camera-condition combinations for every subject, and zero bad clip-manifest hashes. E-020 records the exact remote paths and hashes. No training or Gate 3 work began.
