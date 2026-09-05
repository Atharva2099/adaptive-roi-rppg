#!/bin/bash
# MCD-only offline diagnostic. One allocation runs eight exclusive sequential
# workers; submit exactly two allocations for ranges 0-799 and 800-1598.
#SBATCH --job-name=mcd-repeat-beam-full
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
set -euo pipefail

partition_range() {
  local first=${1-} last=${2-} workers=${3:-8} count base remainder worker start size end
  [[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ && "$workers" =~ ^[1-9][0-9]*$ ]] || return 2
  (( first >= 0 && last >= first && last <= 1598 )) || return 2
  count=$((last - first + 1)); (( count >= workers )) || return 2
  base=$((count / workers)); remainder=$((count % workers))
  for ((worker = 0; worker < workers; worker++)); do
    start=$((first + worker * base + (worker < remainder ? worker : remainder)))
    size=$((base + (worker < remainder ? 1 : 0)))
    end=$((start + size - 1))
    printf '%s:%s\n' "$start" "$end"
  done
}

partition_clip_range() {
  local first=${1-} last=${2-} workers=${3:-8} count base remainder worker start size end
  [[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ && "$workers" =~ ^[1-9][0-9]*$ ]] || return 2
  (( first >= 0 && last >= first && last <= 532 )) || return 2
  count=$((last - first + 1)); (( count >= workers )) || return 2
  base=$((count / workers)); remainder=$((count % workers))
  for ((worker = 0; worker < workers; worker++)); do
    start=$((first + worker * base + (worker < remainder ? worker : remainder)))
    size=$((base + (worker < remainder ? 1 : 0)))
    end=$((start + size - 1))
    printf '%s:%s\n' "$start" "$end"
  done
}

seed_task_indices() {
  local seed=${1-} first=${2-} last=${3-} clip
  [[ "$seed" =~ ^[012]$ && "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || return 2
  (( first >= 0 && last >= first && last <= 532 )) || return 2
  for ((clip = first; clip <= last; clip++)); do
    printf '%s\n' "$((3 * clip + seed))"
  done
}

: "${BASE:?}" "${OUT:?}" "${SOURCE_SUBSETS_DIR:?}" "${MCD_MANIFEST_TREE:?}" "${MCD_STATE_ROOT:?}" "${MCD_GT_ROOT:?}" "${CHECKPOINT_ROOT:?}"
# RUN_SEED is the deliberately narrower full-cohort mode.  It covers every
# clip once while retaining the global 3*clip_position+seed task identity.
seed_mode=0
if [[ -n ${RUN_SEED:-} ]]; then
  seed_mode=1
  [[ "$RUN_SEED" =~ ^[012]$ && "${FIRST_CLIP_POSITION:-}" == 0 && "${LAST_CLIP_POSITION:-}" == 532 ]] || {
    echo "seed mode requires RUN_SEED in {0,1,2}, FIRST_CLIP_POSITION=0, LAST_CLIP_POSITION=532" >&2
    exit 2
  }
else
  : "${FIRST_TASK:?}" "${LAST_TASK:?}"
  # The all-seed cohort is intentionally split into these two fixed
  # allocations. Rejecting any other range prevents a gap or overlap.
  if ! { [[ "$FIRST_TASK" == 0 && "$LAST_TASK" == 799 ]] ||
         [[ "$FIRST_TASK" == 800 && "$LAST_TASK" == 1598 ]]; }; then
    echo "FIRST_TASK/LAST_TASK must be exactly 0/799 or 800/1598" >&2
    exit 2
  fi
fi
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
BEAM_CONFIG=${BEAM_CONFIG:-$BASE/configs/evaluation/mcd_repeated_correction_beam_full_v1.json}
[[ -r "$BEAM_CONFIG" ]]
unset PYTHONPATH
# The parent batch is submitted with --export=NONE. Export every value read by
# the nested bash -c so srun cannot start a child with an incomplete env.
export BASE OUT SOURCE_SUBSETS_DIR MCD_MANIFEST_TREE MCD_STATE_ROOT MCD_GT_ROOT CHECKPOINT_ROOT RUN_SEED
export PYTHON BEAM_CONFIG
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$BASE/src"
if (( seed_mode )); then
  PARTITIONS=$(partition_clip_range "$FIRST_CLIP_POSITION" "$LAST_CLIP_POSITION") || { echo "invalid inclusive full-clip range" >&2; exit 2; }
else
  PARTITIONS=$(partition_range "$FIRST_TASK" "$LAST_TASK") || { echo "invalid inclusive full-task range" >&2; exit 2; }
fi
RANGE_LIST=()
while IFS= read -r partition; do
  RANGE_LIST+=("$partition")
done <<< "$PARTITIONS"
[[ ${#RANGE_LIST[@]} == 8 ]]
mkdir -p "$OUT"

pids=()
for range in "${RANGE_LIST[@]}"; do
  IFS=: read -r start end <<< "$range"
  srun --export=ALL --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 /bin/bash -c '
    set -euo pipefail
    start=$1; end=$2
    for ((position = start; position <= end; position++)); do
      if [[ -n ${RUN_SEED:-} ]]; then
        task=$((3 * position + RUN_SEED))
      else
        task=$position
      fi
      run="$OUT/mcd-repeated-beam-full-$SLURM_JOB_ID-$task"
      [[ ! -e "$run" && ! -L "$run" ]]
      "$PYTHON" "$BASE/scripts/verify_mcd_repeated_correction_beam.py" \
        --output-dir "$run" --source-subsets-dir "$SOURCE_SUBSETS_DIR" \
        --beam-config "$BEAM_CONFIG" --manifest-tree "$MCD_MANIFEST_TREE" \
        --state-root "$MCD_STATE_ROOT" --gt-root "$MCD_GT_ROOT" \
        --plan "$BASE/configs/evaluation/mcd_frozen_models_v1.json" \
        --checkpoint-root "$CHECKPOINT_ROOT" --task-index "$task" --task-count 1599
    done
  ' bash "$start" "$end" &
  pids+=("$!")
done
set +e
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
set -e
exit "$status"
