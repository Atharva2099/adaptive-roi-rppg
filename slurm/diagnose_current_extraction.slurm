#!/bin/bash
#SBATCH --job-name=current-extraction-diagnostic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=diagnose-current-extraction-%j.out
#SBATCH --error=diagnose-current-extraction-%j.err
set -euo pipefail
: "${REPO:?set REPO to the current repository on Polaris}"
: "${MCD_VIDEO:?set MCD_VIDEO}"
: "${MCD_STATE_CSV:?set MCD_STATE_CSV}"
: "${MCD_CLIP_ID:?set MCD_CLIP_ID}"
: "${MMPD_MAT:?set MMPD_MAT}"
: "${MMPD_CLIP_ID:?set MMPD_CLIP_ID}"
: "${FACE_LANDMARKER_MODEL:?set FACE_LANDMARKER_MODEL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd "$REPO"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "diagnose_current_extraction requires Slurm" >&2; exit 2; fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-$REPO/.venv/bin/python}" -u scripts/diagnose_current_extraction.py \
  --mcd-video "$MCD_VIDEO" --mcd-state-csv "$MCD_STATE_CSV" --mcd-clip-id "$MCD_CLIP_ID" \
  --mmpd-mat "$MMPD_MAT" --mmpd-clip-id "$MMPD_CLIP_ID" \
  --face-landmarker-model "$FACE_LANDMARKER_MODEL" --output-dir "$OUTPUT_DIR"
