#!/bin/bash
# Full frozen MCD simple-selector diagnostic. Heavy work runs under Slurm as
# subject-disjoint shards followed by an independent merge. Set BASE, OUT,
# MANIFEST, STATE, GT, CHECKPOINT_ROOT. Optionally SMOKE_CLIPS, CLIP_OFFSET, SHARDS.
# The runner refuses an existing output directory.
#SBATCH --job-name=mcd-simple
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${MANIFEST:?}" "${STATE:?}" "${GT:?}" "${CHECKPOINT_ROOT:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}
[[ -x "$PYTHON" ]]
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src"
SHARDS=${SHARDS:-16}
MAX_TASKS=${SLURM_NTASKS:-16}
if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || [ "$SHARDS" -gt "$MAX_TASKS" ]; then
  echo "SHARDS must be an integer in 1..$MAX_TASKS (got '$SHARDS')" >&2; exit 1
fi
RUN="$OUT/full-$SLURM_JOB_ID"; export RUN; mkdir -p "$RUN/shards"
ARGS=(--manifest-tree "$MANIFEST" --state-root "$STATE" \
  --gt-root "$GT" --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" \
  --checkpoint-root "$CHECKPOINT_ROOT" --shard-count "$SHARDS")
[[ -n "${SMOKE_CLIPS:-}" ]] && ARGS+=(--limit-clips "$SMOKE_CLIPS")
[[ -n "${CLIP_OFFSET:-}" ]] && ARGS+=(--clip-offset "$CLIP_OFFSET")
srun --ntasks="$SHARDS" --ntasks-per-node="$SHARDS" --cpus-per-task=2 --exclusive bash -c 'exec "$1" "$2" "${@:3}" --shard-index "$SLURM_PROCID" --output-dir "$RUN/shards/shard-$SLURM_PROCID"' _ "$PYTHON" "$BASE/scripts/evaluate_mcd_simple_selectors.py" "${ARGS[@]}"
SHARD_DIRS=(); for i in $(seq 0 $((SHARDS - 1))); do SHARD_DIRS+=(--shard-dir "$RUN/shards/shard-$i"); done
"$PYTHON" "$BASE/scripts/evaluate_mcd_simple_selectors.py" --merge --output-dir "$RUN/merged" "${ARGS[@]}" "${SHARD_DIRS[@]}"
