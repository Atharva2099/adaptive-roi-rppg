#!/bin/bash
# One ordinary CSV per clip. Submit smoke with --array=0-15%16.
#SBATCH --job-name=mmpd-clip
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
: "${RAW_INVENTORY:?set RAW_INVENTORY}"
: "${BASE:?set BASE}"
: "${RAW_ROOT:?set RAW_ROOT}"
: "${CHECKPOINT_MANIFEST:?set CHECKPOINT_MANIFEST}"
: "${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
: "${FACE_LANDMARKER_MODEL:?set FACE_LANDMARKER_MODEL}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"

"${PYTHON:-$BASE/.venv/bin/python}" "$BASE/scripts/evaluate_mmpd_clip.py" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --raw-inventory "${RAW_INVENTORY}" --raw-root "${RAW_ROOT}" \
  --checkpoint-manifest "${CHECKPOINT_MANIFEST}" --checkpoint-root "${CHECKPOINT_ROOT}" \
  --face-landmarker-model "${FACE_LANDMARKER_MODEL}" --output-dir "${OUTPUT_DIR}"
