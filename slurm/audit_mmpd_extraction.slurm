#!/bin/bash
# One extraction-only audit task per MMPD engineering-plan clip.
# Submit with --array=0-298%16.
#SBATCH --job-name=mmpd-extraction-audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
: "${RAW_INVENTORY:?set RAW_INVENTORY}"
: "${BASE:?set BASE}"
: "${RAW_ROOT:?set RAW_ROOT}"
: "${FACE_LANDMARKER_MODEL:?set FACE_LANDMARKER_MODEL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"
"${PYTHON:-$BASE/.venv/bin/python}" -u scripts/audit_mmpd_extraction.py \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --raw-inventory "$RAW_INVENTORY" --raw-root "$RAW_ROOT" \
  --face-landmarker-model "$FACE_LANDMARKER_MODEL" --output-dir "$OUTPUT_DIR"
