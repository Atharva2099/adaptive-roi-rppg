#!/bin/bash
# MCD-only, offline GT-informed full-cohort diagnostic. One clip/seed per task.
# Submit two dependency-linked waves with --array=0-799%8 and 800-1598%8.
#SBATCH --job-name=mcd-repeat-beam-full
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${SOURCE_SUBSETS_DIR:?}" "${MCD_MANIFEST_TREE:?}" "${MCD_STATE_ROOT:?}" "${MCD_GT_ROOT:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
BEAM_CONFIG=${BEAM_CONFIG:-$BASE/configs/evaluation/mcd_repeated_correction_beam_full_v1.json}
[[ -r "$BEAM_CONFIG" ]]
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$BASE/src"
[[ ${SLURM_ARRAY_TASK_ID:?} -ge 0 && ${SLURM_ARRAY_TASK_ID:?} -lt 1599 ]]
mkdir -p "$OUT"
RUN="$OUT/mcd-repeated-beam-full-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
[[ ! -e "$RUN" && ! -L "$RUN" ]]
exec "$PYTHON" "$BASE/scripts/verify_mcd_repeated_correction_beam.py" \
  --output-dir "$RUN" --source-subsets-dir "$SOURCE_SUBSETS_DIR" \
  --beam-config "$BEAM_CONFIG" --manifest-tree "$MCD_MANIFEST_TREE" \
  --state-root "$MCD_STATE_ROOT" --gt-root "$MCD_GT_ROOT" \
  --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" \
  --checkpoint-root "$CHECKPOINT_ROOT" --task-index "$SLURM_ARRAY_TASK_ID" --task-count 1599
