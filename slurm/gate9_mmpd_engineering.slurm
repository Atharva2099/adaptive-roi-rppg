#!/bin/bash
# Engineering-only launcher. It does not download data or run extraction by itself.
#SBATCH --job-name=gate9-mmpd-engineering
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
set -euo pipefail
: "${BASE:?}" "${PLAN:?}" "${RAW_ROOT:?}" "${CHECKPOINT_ROOT:?}" "${RULE_ROOT:?}" "${RAW_INVENTORY:?}" "${RULE_MANIFEST:?}" "${CHECKPOINT_MANIFEST:?}" "${CODE_SNAPSHOT:?}" "${EXTRACTOR:?}" "${EXTRACTION_CODE_SHA256:?}" "${OUT:?}"
export PYTHONPATH="$BASE/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON:-$BASE/.venv/bin/python}" "$BASE/scripts/evaluate_mmpd_gate9.py" --plan "$PLAN" --raw-root "$RAW_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" --rule-root "$RULE_ROOT" --raw-inventory "$RAW_INVENTORY" --rule-manifest "$RULE_MANIFEST" --checkpoint-manifest "$CHECKPOINT_MANIFEST" --code-snapshot "$CODE_SNAPSHOT" --extractor "$EXTRACTOR" --extraction-code-sha256 "$EXTRACTION_CODE_SHA256" --output "$OUT"
