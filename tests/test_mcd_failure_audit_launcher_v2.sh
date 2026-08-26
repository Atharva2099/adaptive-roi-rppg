#!/bin/bash
set -euo pipefail
launcher=$(cd "$(dirname "$0")/.." && pwd)/slurm/mcd_frozen_failure_audit_v2.slurm
bash -n "$launcher"
grep -Fq 'AUDIT_WALLTIME_SECONDS' "$launcher"
grep -Fq 'SLURM_TIMELIMIT' "$launcher"
grep -Fq 'AUDIT_WALLTIME_SECONDS exceeds allocated Slurm limit' "$launcher"
grep -Fq 'WORKERS must satisfy 1 <= W <= SLURM_NTASKS' "$launcher"
grep -Fq 'WORKER_MAP_TEXT' "$launcher"
grep -Fq 'frozen worker assignment differs from requested W/batch' "$launcher"
! grep -Fq 'index % SLURM_NTASKS' "$launcher"
grep -Fq 'SHARD_BATCH' "$launcher"
# Logical shard count may exceed concurrent tasks; the persisted worker map
# is the contract that binds each task to its deterministic shard list.
grep -Fq 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1' "$launcher"
grep -Fq -- '--mode plan' "$launcher"
grep -Fq -- '--mode shards' "$launcher"
grep -Fq -- '--mode merge' "$launcher"
