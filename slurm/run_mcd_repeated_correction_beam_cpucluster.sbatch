#!/bin/bash
# Two sequential task workers share one CPU-cluster node.  Belief rollouts
# within each task remain sequential; the two task ranges run in parallel.
#SBATCH --job-name=mcd-repeat-beam
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${FROZEN_SELECTION_DIR:?}" "${SOURCE_SUBSETS_DIR:?}" "${MCD_MANIFEST_TREE:?}" "${MCD_STATE_ROOT:?}" "${MCD_GT_ROOT:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]; export PYTHON
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$BASE/src"
mkdir -p "$OUT"

run_range() {
    local first=$1 last=$2 task_index run
    for ((task_index=first; task_index<=last; task_index++)); do
        run="$OUT/mcd-repeated-beam-$SLURM_JOB_ID-$task_index"
        [[ ! -e "$run" && ! -L "$run" ]]
        "$PYTHON" "$BASE/scripts/verify_mcd_repeated_correction_beam.py" --output-dir "$run" --selection-dir "$FROZEN_SELECTION_DIR" --source-subsets-dir "$SOURCE_SUBSETS_DIR" --manifest-tree "$MCD_MANIFEST_TREE" --state-root "$MCD_STATE_ROOT" --gt-root "$MCD_GT_ROOT" --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" --checkpoint-root "$CHECKPOINT_ROOT" --task-index "$task_index" --task-count 162
    done
}

export -f run_range
srun --exclusive --ntasks=1 --cpus-per-task=1 bash -euo pipefail -c 'run_range "$1" "$2"' _ 0 80 & worker_a=$!
srun --exclusive --ntasks=1 --cpus-per-task=1 bash -euo pipefail -c 'run_range "$1" "$2"' _ 81 161 & worker_b=$!
set +e
wait "$worker_a"; status_a=$?
wait "$worker_b"; status_b=$?
set -e
if (( status_a != 0 || status_b != 0 )); then
    exit 1
fi
