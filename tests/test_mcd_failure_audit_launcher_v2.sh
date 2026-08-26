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
grep -Fq 'dispatch-smoke' "$launcher"
grep -Fq 'Scheduler-only proof: no manifest, checkpoint, frame, label, or output path.' "$launcher"

smoke_tmp=$(mktemp -d)
trap 'rm -rf "$smoke_tmp"' EXIT
mkdir -p "$smoke_tmp/base/.venv/bin" "$smoke_tmp/bin"
ln -s "$(command -v python3)" "$smoke_tmp/base/.venv/bin/python"
cat > "$smoke_tmp/bin/srun" <<'EOF'
#!/bin/bash
echo MOCK_SRUN_DISPATCH "$@"
EOF
cat > "$smoke_tmp/bin/scontrol" <<'EOF'
#!/bin/bash
if [[ "${SCONTROL_MALFORMED:-0}" == 1 ]]; then
  echo 'JobId=123 TimeLimit=not-a-limit NumTasks=16'
else
  echo "JobId=123 TimeLimit=${SCONTROL_TIMELIMIT:-00:05:00} NumTasks=16"
fi
EOF
chmod +x "$smoke_tmp/bin/srun"
chmod +x "$smoke_tmp/bin/scontrol"
smoke_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT PATH="$smoke_tmp/bin:$PATH" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-1 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$smoke_output" == *MOCK_SRUN_DISPATCH* ]]
day_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_TIMELIMIT=1-00:00:00 PATH="$smoke_tmp/bin:$PATH" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-2 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$day_output" == *MOCK_SRUN_DISPATCH* ]]
minute_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_TIMELIMIT=5 PATH="$smoke_tmp/bin:$PATH" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-3 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$minute_output" == *MOCK_SRUN_DISPATCH* ]]
if env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_MALFORMED=1 PATH="$smoke_tmp/bin:$PATH" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-1 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher"; then
  echo 'malformed scheduler TimeLimit unexpectedly accepted' >&2
  exit 1
fi
