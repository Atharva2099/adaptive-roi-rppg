#!/bin/bash
# One independent clip/seed task per array element.  Belief rollouts within a
# task remain sequential; clips and seeds are the parallel boundary.
#SBATCH --job-name=mcd-repeat-beam
#SBATCH --array=0-161
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${FROZEN_SELECTION_DIR:?}" "${SOURCE_SUBSETS_DIR:?}" "${MCD_MANIFEST_TREE:?}" "${MCD_STATE_ROOT:?}" "${MCD_GT_ROOT:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$BASE/src"
mkdir -p "$OUT"; RUN="$OUT/mcd-repeated-beam-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"; [[ ! -e "$RUN" && ! -L "$RUN" ]]
exec "$PYTHON" "$BASE/scripts/verify_mcd_repeated_correction_beam.py" --output-dir "$RUN" --selection-dir "$FROZEN_SELECTION_DIR" --source-subsets-dir "$SOURCE_SUBSETS_DIR" --manifest-tree "$MCD_MANIFEST_TREE" --state-root "$MCD_STATE_ROOT" --gt-root "$MCD_GT_ROOT" --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" --checkpoint-root "$CHECKPOINT_ROOT" --task-index "$SLURM_ARRAY_TASK_ID" --task-count 162
