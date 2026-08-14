# Gate 6: historical MCD parity replay

Status as of 2026-08-13: accepted.

## Question and boundary

Gate 6 asks whether the rebuilt causal full-face path can be compared fairly with the frozen historical Level-A path on the exact shared MCD cohort. This is an observational ruler-difference report. The frozen historical inputs contain measurements and outputs, but not the raw source windows, interpolation masks, or spectra needed for a causal decomposition. The report therefore does not claim that any category explains a difference mechanistically.

MCD is the only dataset used. MMPD was not accessed. Gate 5 remains frozen and training remains prohibited until this gate is accepted.

## Inputs and exact comparison

The historical reference is the authenticated Phase-0 schema-v3 cache. Its full-face trajectory uses `hr_meas[:, 0]` and `conf[:, 0]`, replayed from a fresh action-0 belief state at every clip. The frozen `per_clip.csv` contains 3,198 rows: six policy arms (`A`, `Aprime`, `B`, `C`, `D`, `Dv2`) with 533 clips per arm. Only `level == A` is the historical reference.

The historical CSV SHA-256 is `7481c4e5bcecda77dd954e78d49d7e6cb435a5353bf89f6a4a30187ce2169f2e`. The cache manifest SHA-256 is `d9c63ab03582ae254e067ce103e6ddb563019c65f1087ee723bcd9614d55aadf`; the cache inventory SHA-256 is `ad2a2cfa8d11ac6978d72958a4cadd8027e7eb9502a9482b53df8cf3f77c2f4c`. Historical parity uses 533 clips and 89 represented subjects. The seven Gate 2 evaluation clips without historical cache members are excluded explicitly, not silently filled.

The current arm reads the authenticated Gate 2 evaluation manifest, MCD semantic-state root, and MCD ground-truth root; builds current POS measurements; and calls the active `control_step(..., proposed_action=0)` path. Historical and current rows are joined on dataset, clip, hop index, and hop time. Missing hops remain visible as historical-only or current-only rows.

## Publication and acceptance rules

Each run uses a fresh destination on Polaris. `STARTED.json` is written first. The substantive CSV/JSON files and canonical source inventory are validated and hashed before `COMPLETE.json` is written last. A failed run gets `FAILED.json`; terminal markers are mutually exclusive. Readers accept only a complete, hash-verified tree.

The fixture is exactly subjects `1107` and `1314`, six clips each. It passed on Polaris job `47311` on `lmn01` with 12 clips, 2,063 joined hops, zero unclassified rows, and `COMPLETE.json`. Its historical equal-clip MAE was `18.626108613255187` BPM and current equal-clip MAE was `17.459874442266` BPM. These are fixture diagnostics, not full-cohort claims.

The predeclared full run was Polaris job `47313`, using snapshot hash `c32e0882af57757e52420b9ed283460cf52b244e1d57318ec1edabaa36f163d3`. It completed with exit `0` in `2:04:28` on `lmn01`. The output is `/Users/924254653/adaptive_roi_gate6_runs_20260813_full/output`. It contains 533 clips, 89 subjects, 91,227 joined hops, zero unclassified rows, and valid STARTED/COMPLETE markers with no FAILED marker.

The full historical equal-clip MAE is `12.386529570502146` BPM, exactly reproducing the authenticated Level-A reference. The current equal-clip MAE is `12.631680651991282` BPM. The signed current-minus-historical difference is `+0.2451510814891351` BPM. Category counts are 84,560 confidence differences, 3,657 validity differences, 2,481 measurement differences, 506 belief differences, 23 exact observed matches, and zero unclassified rows.

## Interpretation limits

The report separates exact observed matches, historical-only hops, current-only hops, GT differences, validity differences, measurement differences, confidence differences, belief differences, and numerical differences. Because the frozen cache lacks causal source-window evidence, the report records `causal_decomposition_status = not_supported_by_frozen_inputs`. A parity difference is not evidence that the current signal implementation is wrong, and a smaller MAE is not evidence of controller quality or transfer.

## Reproduction

The reviewed launcher is `scripts/verify_gate6_mcd_parity.py`. It requires a Slurm allocation, the MCD manifest/state/GT roots, the authenticated historical cache artifacts, a code snapshot SHA-256, and a fresh output directory. The local checks before submission were the full unittest suite, Python compile check, launcher `--help` import smoke, and `git diff --check`. Gate 6 is observational parity evidence only; it does not establish controller quality, training value, transfer, or generalization.
