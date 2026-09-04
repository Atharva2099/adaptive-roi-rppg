#!/bin/bash
# Aggregate completed MCD beam outputs on one cpucluster worker.
#SBATCH --job-name=mcd-beam-analysis
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
set -euo pipefail
: "${BASE:?}" "${RESULTS_ROOT:?}" "${OUT:?}" "${SOURCE_SUBSETS_DIR:?}"
: "${JOB_ID:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="$BASE/src"
ARGS=(--results-root "$RESULTS_ROOT" --output-dir "$OUT" --source-subsets-dir "$SOURCE_SUBSETS_DIR")
IFS=, read -r -a ANALYSIS_JOB_IDS <<< "${JOB_IDS:-$JOB_ID}"
for value in "${ANALYSIS_JOB_IDS[@]}"; do [[ "$value" =~ ^[0-9]+$ ]] || exit 2; ARGS+=(--job-id "$value"); done
IFS=, read -r -a ANALYSIS_SEED_VALUES <<< "${ANALYSIS_SEEDS-0,1,2}"
(( ${#ANALYSIS_SEED_VALUES[@]} > 0 )) || exit 2
ARGS+=(--seeds)
ANALYSIS_SEED_SEEN=,
for value in "${ANALYSIS_SEED_VALUES[@]}"; do
  [[ "$value" =~ ^[012]$ ]] || exit 2
  case "$ANALYSIS_SEED_SEEN" in *,"$value",*) exit 2;; esac
  ANALYSIS_SEED_SEEN+="$value,"
  ARGS+=("$value")
done
[[ -z ${FROZEN_SELECTION_DIR:-} || -d ${FROZEN_SELECTION_DIR:-} ]] || exit 2
[[ -z ${BEAM_CONFIG:-} || -r ${BEAM_CONFIG:-} ]] || exit 2
[[ -z ${FROZEN_SELECTION_DIR:-} ]] || ARGS+=(--selection-dir "$FROZEN_SELECTION_DIR")
[[ -z ${BEAM_CONFIG:-} ]] || ARGS+=(--beam-config "$BEAM_CONFIG")
exec "$PYTHON" "$BASE/scripts/analyze_mcd_repeated_beam_behavior.py" "${ARGS[@]}"
