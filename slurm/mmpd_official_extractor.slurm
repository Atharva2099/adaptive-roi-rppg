#!/bin/bash
# Run only on a compute node; the official source remains outside this repo.
#SBATCH --job-name=mmpd-official-extractor
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
#SBATCH --output=/Users/924254653/mmpd_official_extractor_20260825/outputs/smoke/logs/slurm-%j.out
#SBATCH --error=/Users/924254653/mmpd_official_extractor_20260825/outputs/smoke/logs/slurm-%j.err
set -euo pipefail
: "${OFFICIAL_ROOT:?}" "${RAW_ROOT:?}" "${CACHE_ROOT:?}" "${CONFIG_PATH:?}"
: "${SUMMARY_PATH:?}" "${PYTHON:?}" "${SHEET_PATH:?}" "${INVENTORY_CSV:?}"
: "${INVENTORY_JSON:?}" "${CLIP_LIMIT:?}" "${EXPECTED_PAIRS:?}"
: "${EXPECTED_SUBJECTS:?}"
: "${HARNESS_ROOT:?}"
if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
  echo "submit this launcher through Slurm on a compute node" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
TOOLBOX_PATH=/data/rPPG-Toolbox_MMPD
CACHE_MOUNT_PATH="${CACHE_MOUNT_PATH:-/data/rPPG_dataset/processed_dataset}"
PYTHON_DIR=$(dirname "${PYTHON}")
mkdir -p "${CACHE_ROOT}"
for output_path in "${SUMMARY_PATH}" "${SHEET_PATH}" "${INVENTORY_CSV}" "${INVENTORY_JSON}"; do
  mkdir -p "$(dirname "${output_path}")"
done
if [[ -n "${CLIP_LIST:-}" ]]; then
  : "${CLIP_LIST:?}"
fi

BWRAP_ARGS=(
  --ro-bind /usr /usr --symlink usr/bin /bin --ro-bind /lib /lib
  --ro-bind /lib64 /lib64 --ro-bind /etc /etc --ro-bind /Users /Users
  --proc /proc --dev /dev --tmpfs /tmp --dir /data --dir /data/rPPG_dataset
  --dir "${CACHE_MOUNT_PATH}"
  --setenv PATH "/usr/bin:/bin:${PYTHON_DIR}"
  --ro-bind "${OFFICIAL_ROOT}/rppg-Toolbox_MMPD" "${TOOLBOX_PATH}"
  --ro-bind "${RAW_ROOT}" /data/rPPG_dataset/mat_dataset
  --bind "${CACHE_ROOT}" "${CACHE_MOUNT_PATH}"
  --chdir "${HARNESS_ROOT}"
)
declare -A bound_parents=()
for output_path in "${SUMMARY_PATH}" "${SHEET_PATH}" "${INVENTORY_CSV}" "${INVENTORY_JSON}"; do
  parent=$(dirname "${output_path}")
  if [[ -z "${bound_parents[${parent}]+x}" ]]; then
    BWRAP_ARGS+=(--bind "${parent}" "${parent}")
    bound_parents[${parent}]=1
  fi
done

EXTRACTOR_ARGS=(
  --official-root "${OFFICIAL_ROOT}" --toolbox-path "${TOOLBOX_PATH}" --config-path "${CONFIG_PATH}"
  --summary-path "${SUMMARY_PATH}" --expected-pairs "${EXPECTED_PAIRS}"
  --expected-subjects "${EXPECTED_SUBJECTS}"
)
INSPECTOR_ARGS=(
  --file-list "${FILE_LIST_PATH:-}" --sheet "${SHEET_PATH}"
  --inventory-csv "${INVENTORY_CSV}" --inventory-json "${INVENTORY_JSON}"
  --clip-limit "${CLIP_LIMIT}"
)
if [[ -n "${CLIP_LIST:-}" ]]; then
  EXTRACTOR_ARGS+=(--clip-list "${CLIP_LIST}")
  INSPECTOR_ARGS+=(--clip-list "${CLIP_LIST}")
fi

/usr/bin/bwrap "${BWRAP_ARGS[@]}" \
  "${PYTHON}" "${HARNESS_ROOT}/scripts/run_official_mmpd_extractor.py" \
  "${EXTRACTOR_ARGS[@]}"
FILE_LIST_PATH=$(${PYTHON} -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["file_list_path"])' "${SUMMARY_PATH}")
INSPECTOR_ARGS[1]="${FILE_LIST_PATH}"
/usr/bin/bwrap "${BWRAP_ARGS[@]}" \
  "${PYTHON}" "${HARNESS_ROOT}/scripts/inspect_official_mmpd_cache.py" \
  "${INSPECTOR_ARGS[@]}"
[[ -s "${SHEET_PATH}" ]]
