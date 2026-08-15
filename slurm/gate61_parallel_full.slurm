#!/bin/bash
# One Slurm allocation runs all 16 Gate 6.1 shards, then merges them.
# The allocation is intentionally one node so shard processes share the
# requested cores without creating 16 scheduler jobs.
#SBATCH --job-name=gate61-parallel
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=gate61-parallel-%j.out
#SBATCH --error=gate61-parallel-%j.err

set -euo pipefail

: "${BASE:?set BASE to the verified Gate 6.1 snapshot}"
: "${OUT:?set OUT to the run root}"
: "${HASH:?set HASH to the shard code hash}"

PYTHON=${PYTHON:-/Users/924254653/RL-for-RoI/.venv/bin/python}
MANIFEST=/Users/924254653/adaptive_roi_gate2_reviewed_clean_20260811.fdGR97/mcd_manifest_structured_v2
STATE=/Users/924254653/mcd_stage/precomputed_rgb/semantic_state_vectors_12roi_full
GT=/Users/924254653/mcd_stage/precomputed_rgb/ground_truths_full
CACHE=/Users/924254653/RL-for-RoI/results/streaming/phase0_v3nc/cache
CACHE_MANIFEST=/Users/924254653/RL-for-RoI/results/streaming/phase0_v3nc/cache_manifest_v3.json
CACHE_INVENTORY=/Users/924254653/adaptive_roi_gate6_snapshot_20260813/streaming_v3nc_cache_inventory.json
HISTORICAL_CSV=/Users/924254653/RL-for-RoI/results/streaming/phase0_v3nc/per_clip.csv
RUN_ROOT="$OUT/full-parallel-${SLURM_JOB_ID}"
SHARD_ROOT="$RUN_ROOT/shards"
MERGED_ROOT="$RUN_ROOT/merged"
mkdir -p "$SHARD_ROOT"
export RUN_ROOT PYTHON

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

COMMON=(
  --phase full --shard-count 16
  --manifest-tree "$MANIFEST"
  --state-root "$STATE"
  --gt-root "$GT"
  --historical-cache-root "$CACHE"
  --historical-cache-manifest "$CACHE_MANIFEST"
  --historical-cache-inventory "$CACHE_INVENTORY"
  --historical-test90-csv "$HISTORICAL_CSV"
  --code-snapshot-sha256 "$HASH"
)

cd "$BASE"
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"
srun --ntasks=16 --ntasks-per-node=16 --cpus-per-task=1 --exclusive \
  bash -c 'exec "$1" scripts/verify_gate6_1_mcd_crossover.py "${@:2}" --shard-index "$SLURM_PROCID" --output-dir "$RUN_ROOT/shards/shard-$SLURM_PROCID"' \
  _ "$PYTHON" "${COMMON[@]}"

SHARDS=()
for i in $(seq 0 15); do SHARDS+=(--shard-dir "$SHARD_ROOT/shard-$i"); done

"$PYTHON" scripts/verify_gate6_1_mcd_crossover.py --merge --phase full \
  --output-dir "$MERGED_ROOT" \
  --manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" \
  --historical-cache-root "$CACHE" \
  --historical-cache-manifest "$CACHE_MANIFEST" \
  --historical-cache-inventory "$CACHE_INVENTORY" \
  --historical-test90-csv "$HISTORICAL_CSV" \
  --code-snapshot-sha256 "$HASH" "${SHARDS[@]}"

echo "Gate 6.1 parallel run complete: $MERGED_ROOT"
