#!/bin/bash
# One allocation: 16 subject-disjoint frozen-policy shards followed by an
# independent source-authoritative merge.  Set BASE, OUT and DECLARED_CODE_SNAPSHOT_SHA256.
#SBATCH --job-name=gate8-models
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${CHECKPOINT_ROOT:?}" "${DECLARED_CODE_SNAPSHOT_SHA256:?}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}; [[ -x "$PYTHON" ]]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$BASE/src"
MANIFEST=/Users/924254653/adaptive_roi_gate2_reviewed_clean_20260811.fdGR97/mcd_manifest_structured_v2
STATE=/Users/924254653/mcd_stage/precomputed_rgb/semantic_state_vectors_12roi_full
GT=/Users/924254653/mcd_stage/precomputed_rgb/ground_truths_full
PLAN=$BASE/configs/evaluation/mcd_frozen_models_v1.json
CODE_HASH=$("$PYTHON" - "$BASE" <<'PY'
import hashlib, sys
from pathlib import Path
base=Path(sys.argv[1]).resolve(); ignored={".git",".venv","__pycache__"}; files=[]
def walk(path):
 for entry in sorted(path.iterdir(),key=lambda p:p.name.encode()):
  if entry.name in ignored: continue
  if entry.is_symlink(): raise SystemExit(f"unexpected symlink: {entry.relative_to(base)}")
  if entry.is_dir(): walk(entry)
  elif entry.is_file() and entry.suffix != ".pyc": files.append(entry.relative_to(base))
  else: raise SystemExit(f"unexpected entry: {entry.relative_to(base)}")
walk(base); h=hashlib.sha256()
for relative in files: h.update(relative.as_posix().encode()+b"\0"); h.update((base/relative).read_bytes()+b"\0")
print(h.hexdigest())
PY
)
[[ "$DECLARED_CODE_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ && "$CODE_HASH" == "$DECLARED_CODE_SNAPSHOT_SHA256" ]] || { echo "declared code hash mismatch" >&2; exit 1; }
RUN="$OUT/full-$SLURM_JOB_ID"; export RUN; mkdir -p "$RUN/shards"
COMMON=(--manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" --plan "$PLAN" --checkpoint-root "$CHECKPOINT_ROOT" --shard-count 16 --code-snapshot-sha256 "$CODE_HASH")
srun --ntasks=16 --ntasks-per-node=16 --cpus-per-task=2 --exclusive bash -c 'exec "$1" "$2" "${@:3}" --shard-index "$SLURM_PROCID" --output-dir "$RUN/shards/shard-$SLURM_PROCID"' _ "$PYTHON" "$BASE/scripts/verify_gate8_mcd_frozen_models.py" "${COMMON[@]}"
SHARDS=(); for i in $(seq 0 15); do SHARDS+=(--shard-dir "$RUN/shards/shard-$i"); done
"$PYTHON" "$BASE/scripts/verify_gate8_mcd_frozen_models.py" --merge --output-dir "$RUN/merged" "${COMMON[@]}" "${SHARDS[@]}"
