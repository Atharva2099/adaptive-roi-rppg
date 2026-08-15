#!/bin/bash
set -euo pipefail
launcher=$(cd "$(dirname "$0")/.." && pwd)/slurm/gate8_mcd_frozen_models_full.slurm
grep -Fq 'RUN="$OUT/full-$SLURM_JOB_ID"; export RUN' "$launcher"
RUN=/tmp/gate8-launcher-propagation; SLURM_PROCID=3; export RUN SLURM_PROCID
test "$(bash -c 'printf %s "$RUN/shards/shard-$SLURM_PROCID"')" = "/tmp/gate8-launcher-propagation/shards/shard-3"
