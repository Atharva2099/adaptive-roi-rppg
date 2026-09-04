#!/bin/bash
# Exact 20-clip diagnostic. Submit this file through Slurm; do not run on login.
# Required environment is the same as mmpd_official_extractor.slurm, with
# HARNESS_ROOT set to this checkout and PYTHON set to the compute-node Python.
# Set output paths to an isolated failure20 directory before submission.
#SBATCH --job-name=mmpd-official-failure20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpucluster
#SBATCH --qos=simple-qos
set -euo pipefail
: "${HARNESS_ROOT:?}"
export CONFIG_PATH="${HARNESS_ROOT}/configs/mmpd_official_extractor_failure20.yaml"
export CLIP_LIST="${HARNESS_ROOT}/configs/mmpd_official_failure20_clip_list.txt"
export CACHE_MOUNT_PATH="/data/rPPG_dataset/processed_dataset_failure20"
: "${OUTPUT_ROOT:?set isolated failure20 output root}"
export CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
export SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.json}"
export SHEET_PATH="${SHEET_PATH:-${OUTPUT_ROOT}/inspection.png}"
export INVENTORY_CSV="${INVENTORY_CSV:-${OUTPUT_ROOT}/inventory.csv}"
export INVENTORY_JSON="${INVENTORY_JSON:-${OUTPUT_ROOT}/inventory.json}"
export CLIP_LIMIT=20 EXPECTED_PAIRS=20 EXPECTED_SUBJECTS=7
exec /bin/bash "${HARNESS_ROOT}/slurm/mmpd_official_extractor.slurm"
