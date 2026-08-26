#!/bin/bash
set -euo pipefail
launcher=$(cd "$(dirname "$0")/.." && pwd)/slurm/mcd_frozen_failure_audit_v2.slurm
hash_python=$(command -v python3)
shell_bash=$(command -v bash)
system_git=$("$hash_python" -c 'from pathlib import Path; import shutil; print(Path(shutil.which("git")).resolve())')
bash -n "$launcher"
grep -Fq 'AUDIT_WALLTIME_SECONDS' "$launcher"
grep -Fq 'SLURM_TIMELIMIT' "$launcher"
grep -Fq 'AUDIT_WALLTIME_SECONDS exceeds allocated Slurm limit' "$launcher"
grep -Fq 'WORKERS must satisfy 1 <= W <= SLURM_NTASKS' "$launcher"
grep -Fq 'WORKER_MAP_TEXT' "$launcher"
grep -Fq 'frozen worker assignment differs from requested W/batch' "$launcher"
! grep -Fq 'index % SLURM_NTASKS' "$launcher"
grep -Fq 'MAX_SUBJECTS must be a positive integer' "$launcher"
grep -Fq 'COMMON+=(--max-subjects "$MAX_SUBJECTS")' "$launcher"
grep -Fq 'COMMON+=(--benchmark-record "$BENCHMARK_RECORD")' "$launcher"
benchmark_append_line=$(grep -nF 'COMMON+=(--benchmark-record "$BENCHMARK_RECORD")' "$launcher" | cut -d: -f1)
plan_branch_line=$(grep -nF 'if [[ "$AUDIT_STAGE" == plan ]]; then' "$launcher" | cut -d: -f1)
audit_plan_required_line=$(grep -nF ': "${AUDIT_PLAN:?shards/merge requires AUDIT_PLAN}"' "$launcher" | cut -d: -f1)
merge_branch_line=$(grep -nF 'if [[ "$AUDIT_STAGE" == merge ]]; then' "$launcher" | cut -d: -f1)
(( benchmark_append_line < plan_branch_line ))
(( benchmark_append_line < audit_plan_required_line ))
(( benchmark_append_line < merge_branch_line ))
grep -Fq 'GIT_EXECUTABLE=${GIT_EXECUTABLE:-"$(command -v git || true)"}' "$launcher"
grep -Fq '[str(git), "-C", str(base), "ls-files", "-z"]' "$launcher"
grep -Fq 'TRACKED_FILES_MANIFEST was not supplied' "$launcher"
grep -Fq 'NUL tracked-files manifest must end with NUL' "$launcher"
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
smoke_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT PATH="$smoke_tmp/bin:$PATH" GIT_EXECUTABLE="$system_git" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-1 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$smoke_output" == *MOCK_SRUN_DISPATCH* ]]
day_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_TIMELIMIT=1-00:00:00 PATH="$smoke_tmp/bin:$PATH" GIT_EXECUTABLE="$system_git" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-2 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$day_output" == *MOCK_SRUN_DISPATCH* ]]
minute_output=$(env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_TIMELIMIT=5 PATH="$smoke_tmp/bin:$PATH" GIT_EXECUTABLE="$system_git" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-3 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher")
[[ "$minute_output" == *MOCK_SRUN_DISPATCH* ]]
if env -u AUDIT_ROOT -u MANIFEST -u STATE -u GT -u CHECKPOINT_ROOT -u PLAN -u AUDIT_PLAN -u AUDIT_SPLIT -u DECLARED_CODE_SNAPSHOT_SHA256 -u SLURM_TIMELIMIT SCONTROL_MALFORMED=1 PATH="$smoke_tmp/bin:$PATH" GIT_EXECUTABLE="$system_git" BASE="$smoke_tmp/base" RUN_ID=dispatch-smoke-1 AUDIT_STAGE=dispatch-smoke AUDIT_WALLTIME_SECONDS=60 SLURM_JOB_ID=123 SLURM_NTASKS=2 WORKERS=2 bash "$launcher"; then
  echo 'malformed scheduler TimeLimit unexpectedly accepted' >&2
  exit 1
fi

hash_repo="$smoke_tmp/hash-repo"
mkdir -p "$hash_repo"
git -C "$hash_repo" init -q
git -C "$hash_repo" config user.email test@example.invalid
git -C "$hash_repo" config user.name test
printf 'tracked-v1\n' > "$hash_repo/source.py"
git -C "$hash_repo" add source.py
git -C "$hash_repo" commit -qm initial
hash_function=$(awk '/^tracked_worktree_hash\(\)/ {on=1} on {print} /^}\s*$/ && on {exit}' "$launcher")
hash_one=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$system_git" bash -c "$hash_function; tracked_worktree_hash")
printf 'untracked smoke log\n' > "$hash_repo/smoke.log"
hash_two=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$system_git" bash -c "$hash_function; tracked_worktree_hash")
[[ "$hash_one" == "$hash_two" ]]
printf 'tracked-v2\n' > "$hash_repo/source.py"
hash_three=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$system_git" bash -c "$hash_function; tracked_worktree_hash")
[[ "$hash_one" != "$hash_three" ]]
cat > "$smoke_tmp/mock-git" <<'EOF'
#!/bin/bash
printf 'mock git used\n' >> "$GIT_USE_LOG"
exec /usr/bin/git "$@"
EOF
chmod +x "$smoke_tmp/mock-git"
mock_hash=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$smoke_tmp/mock-git" GIT_USE_LOG="$smoke_tmp/git-use.log" PATH=/nonexistent "$shell_bash" -c "$hash_function; tracked_worktree_hash")
[[ "$mock_hash" == "$hash_three" ]]
[[ "$(cat "$smoke_tmp/git-use.log")" == 'mock git used' ]]
if BASE="$hash_repo" PYTHON="$(command -v python3)" GIT_EXECUTABLE="$smoke_tmp/missing-git" bash -c "$hash_function; tracked_worktree_hash"; then
  echo 'missing configured Git executable unexpectedly accepted' >&2
  exit 1
fi
printf 'source.py\n' > "$smoke_tmp/tracked-files.txt"
manifest_hash=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$smoke_tmp/missing-git" TRACKED_FILES_MANIFEST="$smoke_tmp/tracked-files.txt" bash -c "$hash_function; tracked_worktree_hash")
[[ "$manifest_hash" == "$hash_three" ]]
printf 'source.py\0' > "$smoke_tmp/tracked-files-nul"
nul_manifest_hash=$(BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$smoke_tmp/missing-git" TRACKED_FILES_MANIFEST="$smoke_tmp/tracked-files-nul" bash -c "$hash_function; tracked_worktree_hash")
[[ "$nul_manifest_hash" == "$hash_three" ]]
if BASE="$hash_repo" PYTHON="$hash_python" GIT_EXECUTABLE="$smoke_tmp/missing-git" bash -c "$hash_function; tracked_worktree_hash"; then
  echo 'missing tracked-files manifest unexpectedly accepted without Git' >&2
  exit 1
fi
