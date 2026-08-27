#!/bin/bash
# Diagnostic-only MCD job. The caller must provide accepted replay-source paths.
#SBATCH --job-name=mcd-long-regret
#SBATCH --nodes=1
#SBATCH --ntasks=27
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${SOURCE_PER_HOP:?}" "${SOURCE_MANIFEST:?}" "${FROZEN_SELECTION_DIR:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src"
MANIFEST=${MCD_MANIFEST_TREE:?}; STATE=${MCD_STATE_ROOT:?}; GT=${MCD_GT_ROOT:?}
RUN="$OUT/long-term-regret-$SLURM_JOB_ID"; mkdir -p "$RUN"
# Smoke before a full launch: set SHARD_COUNT=54 and run one seed/index manually.
# Each worker loads one checkpoint once.  Clip partition and final CSV merge are
# deterministic; the runner validates all 27 seed/shard identities before merge.
SHARD_COUNT=${SHARD_COUNT:-9}; SHARDS=()
for SEED in 0 1 2; do for SHARD in $(seq 0 $((SHARD_COUNT - 1))); do
  DEST="$RUN/shard-seed$SEED-$SHARD"; SHARDS+=(--shard-dir "$DEST")
  srun --exclusive --ntasks=1 --cpus-per-task=1 "$PYTHON" "$BASE/scripts/verify_mcd_long_term_regret.py" \
    --output-dir "$DEST" --source-per-hop "$SOURCE_PER_HOP" --source-manifest "$SOURCE_MANIFEST" --selection-dir "$FROZEN_SELECTION_DIR" \
    --manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" \
    --checkpoint-root "$CHECKPOINT_ROOT" --seed "$SEED" --shard-index "$SHARD" --shard-count "$SHARD_COUNT" &
done; done
wait
"$PYTHON" "$BASE/scripts/verify_mcd_long_term_regret.py" --merge --output-dir "$RUN/merged" \
  --source-per-hop "$SOURCE_PER_HOP" --source-manifest "$SOURCE_MANIFEST" --selection-dir "$FROZEN_SELECTION_DIR" --manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" \
  --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" --checkpoint-root "$CHECKPOINT_ROOT" --shard-count "$SHARD_COUNT" "${SHARDS[@]}"
