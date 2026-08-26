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
: "${SLURM_NTASKS:?allocated Slurm task count is required}"
PYTHON=${PYTHON:-"$BASE/.venv/bin/python"}; [[ -x "$PYTHON" ]]
[[ "$AUDIT_STAGE" == dispatch-smoke || "$AUDIT_STAGE" == plan || "$AUDIT_STAGE" == shards || "$AUDIT_STAGE" == merge ]] || { echo 'AUDIT_STAGE must be dispatch-smoke, plan, shards, or merge' >&2; exit 2; }
[[ "$AUDIT_WALLTIME_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'AUDIT_WALLTIME_SECONDS must be positive seconds' >&2; exit 2; }
# Slurm commonly exports a numeric minute limit, but Polaris job 48075 did
# not.  Fall back only to the local authenticated scheduler record, never a
# caller-provided walltime value.
parse_scheduler_limit() {
  local value=$1 days=0 hours=0 minutes=0 seconds=0
  if [[ "$value" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    days=${BASH_REMATCH[1]}; hours=${BASH_REMATCH[2]}; minutes=${BASH_REMATCH[3]}; seconds=${BASH_REMATCH[4]}
  elif [[ "$value" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    hours=${BASH_REMATCH[1]}; minutes=${BASH_REMATCH[2]}; seconds=${BASH_REMATCH[3]}
  elif [[ "$value" =~ ^([0-9]+)$ ]]; then
    minutes=${BASH_REMATCH[1]}
  else
    return 1
  fi
  (( minutes < 60 && seconds < 60 )) || return 1
  printf '%s\n' "$(((((days * 24) + hours) * 60 + minutes) * 60 + seconds))"
}
if [[ -n "${SLURM_TIMELIMIT:-}" ]]; then
  [[ "$SLURM_TIMELIMIT" =~ ^[1-9][0-9]*$ ]] || { echo 'cannot parse SLURM_TIMELIMIT minutes' >&2; exit 2; }
  SCHEDULER_LIMIT_SECONDS=$((SLURM_TIMELIMIT * 60))
else
  : "${SLURM_JOB_ID:?SLURM_JOB_ID is required when SLURM_TIMELIMIT is absent}"
  scheduler_line=$(scontrol show job -o "$SLURM_JOB_ID") || { echo 'cannot read allocated Slurm job record' >&2; exit 2; }
  [[ "$scheduler_line" =~ (^|[[:space:]])TimeLimit=([^[:space:]]+) ]] || { echo 'allocated Slurm job record lacks TimeLimit' >&2; exit 2; }
  SCHEDULER_LIMIT_SECONDS=$(parse_scheduler_limit "${BASH_REMATCH[2]}") || { echo 'cannot parse allocated Slurm TimeLimit' >&2; exit 2; }
fi
[[ "$SLURM_NTASKS" =~ ^[1-9][0-9]*$ ]] || { echo 'cannot parse allocated Slurm tasks' >&2; exit 2; }
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
if [[ -n "${MAX_SUBJECTS:-}" ]]; then
  [[ "$MAX_SUBJECTS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_SUBJECTS must be a positive integer' >&2; exit 2; }
  COMMON+=(--max-subjects "$MAX_SUBJECTS")
fi
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
