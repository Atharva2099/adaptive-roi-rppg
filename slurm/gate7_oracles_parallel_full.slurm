#!/bin/bash
# One node runs all 16 subject-disjoint train shards, then merges Oracle B/C.
#SBATCH --job-name=gate7-oracles
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00

set -euo pipefail
: "${BASE:?set BASE to the Gate 7 snapshot}"
: "${OUT:?set OUT to a fresh run root}"
: "${DECLARED_CODE_SNAPSHOT_SHA256:?set DECLARED_CODE_SNAPSHOT_SHA256 to the declared clean BASE hash}"
PYTHON=${PYTHON:-$BASE/.venv/bin/python}
[[ "$PYTHON" == "$BASE/"* && -x "$PYTHON" ]] || { echo "Gate 7 requires an executable Python environment under BASE" >&2; exit 1; }
unset PYTHONPATH
CODE_HASH=$("$PYTHON" - "$BASE" <<'PY'
import hashlib
import sys
from pathlib import Path

base = Path(sys.argv[1]).resolve()
excluded_dirs = {".git", ".venv", "__pycache__"}

if not base.is_dir() or base.is_symlink():
    raise SystemExit("BASE must be a real directory")

paths = []
def walk(directory: Path) -> None:
    for entry in sorted(directory.iterdir(), key=lambda path: path.name.encode("utf-8")):
        # These package-local trees are intentionally outside the snapshot hash.
        if entry.name in excluded_dirs:
            continue
        relative = entry.relative_to(base)
        if entry.is_symlink():
            raise SystemExit(f"unexpected symlink in BASE: {relative.as_posix()}")
        if entry.is_dir():
            walk(entry)
        elif entry.is_file():
            if entry.suffix != ".pyc":
                paths.append(relative)
        else:
            raise SystemExit(f"unexpected non-regular entry in BASE: {relative.as_posix()}")

walk(base)
h = hashlib.sha256()
for path in sorted(paths, key=lambda path: path.as_posix().encode("utf-8")):
    data = (base / path).read_bytes()
    h.update(path.as_posix().encode("utf-8")); h.update(b"\0"); h.update(data); h.update(b"\0")
print(h.hexdigest())
PY
)
[[ "$DECLARED_CODE_SNAPSHOT_SHA256" =~ ^[0-9a-fA-F]{64}$ && "$CODE_HASH" == "$DECLARED_CODE_SNAPSHOT_SHA256" ]] || { echo "declared code snapshot hash does not match BASE" >&2; exit 1; }
MANIFEST=/Users/924254653/adaptive_roi_gate2_reviewed_clean_20260811.fdGR97/mcd_manifest_structured_v2
STATE=/Users/924254653/mcd_stage/precomputed_rgb/semantic_state_vectors_12roi_full
GT=/Users/924254653/mcd_stage/precomputed_rgb/ground_truths_full
RUN_ROOT="$OUT/full-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/shards"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHON RUN_ROOT
cd "$BASE"
export PYTHONPATH="$BASE/src"
export PYTHONNOUSERSITE=1
COMMON=(--shard-count 16 --manifest-tree "$MANIFEST" --state-root "$STATE" --gt-root "$GT" --code-snapshot-sha256 "$CODE_HASH" --beam-width 8)
srun --ntasks=16 --ntasks-per-node=16 --cpus-per-task=2 --exclusive bash -c 'exec "$1" scripts/verify_gate7_mcd_oracles.py "${@:2}" --shard-index "$SLURM_PROCID" --output-dir "$RUN_ROOT/shards/shard-$SLURM_PROCID"' _ "$PYTHON" "${COMMON[@]}"
SHARDS=(); for i in $(seq 0 15); do SHARDS+=(--shard-dir "$RUN_ROOT/shards/shard-$i"); done
# The merge reloads and recomputes the 16 validated bindings with at most 16 workers.
"$PYTHON" scripts/verify_gate7_mcd_oracles.py --merge --merge-workers 16 --shard-count 16 --output-dir "$RUN_ROOT/merged" "${COMMON[@]}" "${SHARDS[@]}"
echo "Gate 7 Oracle B/C complete: $RUN_ROOT/merged"
