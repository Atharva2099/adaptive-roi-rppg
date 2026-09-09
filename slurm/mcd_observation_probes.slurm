#!/bin/bash
# Offline MCD observation probes over the job 49143 shard outputs.
# Runs the best-ROI persistence probe and the relative-feature screen.
# Set BASE, SHARDS, OUT. PYTHON defaults to the Polaris miniforge interpreter.
#SBATCH --job-name=mcd-probes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
: "${BASE:?}" "${SHARDS:?}" "${OUT:?}"
PYTHON=${PYTHON:-/Users/924254653/miniforge3/bin/python3}
[[ -x "$PYTHON" ]]
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONPATH="$BASE/src"
RUN="$OUT/probes-$SLURM_JOB_ID"; mkdir -p "$RUN"
"$PYTHON" "$BASE/scripts/probe_best_roi_persistence.py" \
  --shard-root "$SHARDS" --output "$RUN/best_roi_persistence.json"
"$PYTHON" "$BASE/scripts/probe_relative_feature_screen.py" \
  --shard-root "$SHARDS" --output "$RUN/relative_feature_screen.json"
