#!/bin/bash
# CPU-first, MCD-only audit.  Pattern citation: clean Gate 8's independent
# subject shards and separate merge (docs/gates/gate_08_mcd_frozen_models.md).
# GPU is deliberately not requested: it needs a separately verified,
# deterministic resource profile after CPU benchmark evidence warrants it.
#SBATCH --job-name=mcd-failure-audit-v2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
set -euo pipefail
: "${BASE:?}" "${RUN_ID:?stable logical run id required}" "${AUDIT_STAGE:?plan|shards|merge required}"
: "${AUDIT_WALLTIME_SECONDS:?explicit scheduler/requested seconds required}"
: "${SLURM_TIMELIMIT:?allocated Slurm time limit is required}" "${SLURM_NTASKS:?allocated Slurm task count is required}"
PYTHON=${PYTHON:-"$BASE/.venv/bin/python"}; [[ -x "$PYTHON" ]]
[[ "$AUDIT_STAGE" == dispatch-smoke || "$AUDIT_STAGE" == plan || "$AUDIT_STAGE" == shards || "$AUDIT_STAGE" == merge ]] || { echo 'AUDIT_STAGE must be dispatch-smoke, plan, shards, or merge' >&2; exit 2; }
[[ "$AUDIT_WALLTIME_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'AUDIT_WALLTIME_SECONDS must be positive seconds' >&2; exit 2; }
# Slurm exports SLURM_TIMELIMIT in allocated minutes.  Do not accept an
# operator-provided value above that allocation; nonnumeric values fail closed.
[[ "$SLURM_TIMELIMIT" =~ ^[1-9][0-9]*$ && "$SLURM_NTASKS" =~ ^[1-9][0-9]*$ ]] || { echo 'cannot parse allocated Slurm limit/tasks' >&2; exit 2; }
SCHEDULER_LIMIT_SECONDS=$((SLURM_TIMELIMIT * 60))
(( AUDIT_WALLTIME_SECONDS <= SCHEDULER_LIMIT_SECONDS )) || { echo 'AUDIT_WALLTIME_SECONDS exceeds allocated Slurm limit' >&2; exit 2; }
W=${WORKERS:-${SLURM_NTASKS:-1}}
[[ "$W" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be positive' >&2; exit 2; }
(( W <= SLURM_NTASKS )) || { echo 'WORKERS must satisfy 1 <= W <= SLURM_NTASKS' >&2; exit 2; }
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src"
PLAN=${PLAN:-"$BASE/configs/evaluation/mcd_frozen_models_v1.json"}
SCRIPT="$BASE/scripts/run_mcd_frozen_failure_audit_v2.py"
if [[ "$AUDIT_STAGE" == dispatch-smoke ]]; then
  # Scheduler-only proof: no manifest, checkpoint, frame, label, or output path.
  srun --ntasks="$W" --cpus-per-task=1 --exclusive bash -c 'exec "$1" "$2" --mode dispatch-smoke --run-id "$3" --code-snapshot-sha256 "$4" --audit-walltime-seconds "$5" --cpus-per-worker 1 --worker-count "$6" --allocated-task-count "$7" --worker-rank "$SLURM_PROCID"' _ "$PYTHON" "$SCRIPT" "$RUN_ID" "${DECLARED_CODE_SNAPSHOT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}" "$AUDIT_WALLTIME_SECONDS" "$W" "$SLURM_NTASKS"
  exit 0
fi
: "${AUDIT_ROOT:?}"
: "${MANIFEST:?}" "${STATE:?}" "${GT:?}" "${CHECKPOINT_ROOT:?}" "${DECLARED_CODE_SNAPSHOT_SHA256:?}" "${AUDIT_SPLIT:?}"
[[ "$AUDIT_SPLIT" == train || "$AUDIT_SPLIT" == eval ]] || { echo 'AUDIT_SPLIT must be train or eval' >&2; exit 2; }
CODE_HASH=$("$PYTHON" - "$BASE" <<'PY'
import hashlib, sys
from pathlib import Path
base=Path(sys.argv[1]).resolve(); ignored={'.git','.venv','__pycache__'}; files=[]
def walk(path):
 for entry in sorted(path.iterdir(), key=lambda p:p.name.encode()):
  if entry.name in ignored: continue
  if entry.is_symlink(): raise SystemExit('unexpected source symlink')
  if entry.is_dir(): walk(entry)
  elif entry.is_file() and entry.suffix != '.pyc': files.append(entry.relative_to(base))
  else: raise SystemExit('unexpected source entry')
walk(base); digest=hashlib.sha256()
for item in files: digest.update(item.as_posix().encode()+b'\0'); digest.update((base/item).read_bytes()+b'\0')
print(digest.hexdigest())
PY
)
[[ "$DECLARED_CODE_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ && "$CODE_HASH" == "$DECLARED_CODE_SNAPSHOT_SHA256" ]] || { echo 'declared code hash mismatch' >&2; exit 1; }
COMMON=(--audit-root "$AUDIT_ROOT" --manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" --plan "$PLAN" --checkpoint-root "$CHECKPOINT_ROOT" --split "$AUDIT_SPLIT" --run-id "$RUN_ID" --code-snapshot-sha256 "$CODE_HASH" --audit-walltime-seconds "$AUDIT_WALLTIME_SECONDS" --cpus-per-worker 1 --worker-count "$W" --allocated-task-count "$SLURM_NTASKS")
if [[ "$AUDIT_STAGE" == plan ]]; then
  : "${LOGICAL_SHARDS:?plan requires LOGICAL_SHARDS}" "${SHARD_BATCH:?plan requires SHARD_BATCH}"
  "$PYTHON" "$SCRIPT" --mode plan "${COMMON[@]}" --logical-shards "$LOGICAL_SHARDS" --logical-shard-indices "$SHARD_BATCH"
  exit 0
fi
: "${AUDIT_PLAN:?shards/merge requires AUDIT_PLAN}"
COMMON+=(--audit-plan "$AUDIT_PLAN")
if [[ "$AUDIT_STAGE" == merge ]]; then
  if [[ -n "${BENCHMARK_RECORD:-}" ]]; then COMMON+=(--benchmark-record "$BENCHMARK_RECORD"); fi
  if [[ -n "${BENCHMARK_OUTPUT:-}" ]]; then COMMON+=(--benchmark-output "$BENCHMARK_OUTPUT"); fi
  "$PYTHON" "$SCRIPT" --mode merge "${COMMON[@]}"
  exit 0
fi
: "${SHARD_BATCH:?comma-separated logical shard batch required}" "${AUDIT_ATTEMPT_ID:?attempt id required}"
# Read the immutable plan map once.  The launcher does not redistribute a
# local list: each physical rank receives precisely its frozen explicit IDs.
WORKER_MAP_TEXT=$("$PYTHON" - "$AUDIT_PLAN" "$W" "$SHARD_BATCH" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
assignment = plan.get("worker_assignment", {})
batch = [int(value) for value in sys.argv[3].split(",")]
if assignment.get("worker_count") != int(sys.argv[2]) or assignment.get("global_logical_shard_indices") != batch:
    raise SystemExit("frozen worker assignment differs from requested W/batch")
for values in assignment.get("worker_logical_shard_indices", []):
    print(",".join(str(value) for value in values))
PY
)
export WORKER_MAP_TEXT
srun --ntasks="$W" --cpus-per-task=1 --exclusive bash -c '
  chosen=$(printf "%s\n" "$WORKER_MAP_TEXT" | sed -n "$((SLURM_PROCID + 1))p")
  if [[ -n "$chosen" ]]; then
    exec "$1" "$2" --mode shards "${@:3}" --logical-shard-indices "$chosen" --attempt-id "${AUDIT_ATTEMPT_ID}-t${SLURM_PROCID}" --worker-rank "$SLURM_PROCID"
  fi
' _ "$PYTHON" "$SCRIPT" "${COMMON[@]}"
