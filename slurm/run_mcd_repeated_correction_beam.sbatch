#!/bin/bash
# MCD-only offline beam diagnostic.  Do not use this launcher for training.
# Freeze the canonical selection before submitting; workers must only consume it.
# One-clip smoke only: set SMOKE=1. Production uses the same task interface
# as an array, with one clip/seed task owning its sequential belief rollouts.
#SBATCH --job-name=mcd-repeat-beam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${FROZEN_SELECTION_DIR:?}" "${SOURCE_SUBSETS_DIR:?}" "${MCD_MANIFEST_TREE:?}" "${MCD_STATE_ROOT:?}" "${MCD_GT_ROOT:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src"
[[ ${SMOKE:-0} == 1 ]] || { echo "Refusing implicit full launch; set SMOKE=1 for one deterministic frozen-selection clip."; exit 2; }
mkdir -p "$OUT"; RUN="$OUT/mcd-repeated-beam-smoke-$SLURM_JOB_ID"; [[ ! -e "$RUN" && ! -L "$RUN" ]]
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}; [[ "$TASK_INDEX" == 0 ]] || { echo "Smoke requires task 0" >&2; exit 2; }
exec "$PYTHON" "$BASE/scripts/verify_mcd_repeated_correction_beam.py" --smoke --output-dir "$RUN" --selection-dir "$FROZEN_SELECTION_DIR" --source-subsets-dir "$SOURCE_SUBSETS_DIR" --manifest-tree "$MCD_MANIFEST_TREE" --state-root "$MCD_STATE_ROOT" --gt-root "$MCD_GT_ROOT" --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" --checkpoint-root "$CHECKPOINT_ROOT" --task-index "$TASK_INDEX" --task-count 162
