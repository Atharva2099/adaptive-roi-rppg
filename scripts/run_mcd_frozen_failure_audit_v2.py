#!/usr/bin/env python3
"""Create, shard, or merge a restartable frozen MCD failure audit.

GPU is intentionally unsupported.  A GPU profile can be added only after a
separate deterministic CPU-versus-GPU benchmark establishes that it is needed.
"""
from __future__ import annotations

import argparse

from adaptive_roi_rppg.evaluation.mcd_failure_runner_v2 import merge_shards, publish_shard, write_audit_plan


def _indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("logical shard indices must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("logical shard indices must be non-negative")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "shards", "merge"), required=True)
    parser.add_argument("--audit-root", required=True); parser.add_argument("--audit-plan")
    parser.add_argument("--manifest-tree", required=True); parser.add_argument("--state-root", required=True); parser.add_argument("--gt-root", required=True)
    parser.add_argument("--plan", required=True); parser.add_argument("--checkpoint-root", required=True); parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--logical-shards", type=int); parser.add_argument("--logical-shard-indices", type=_indices)
    parser.add_argument("--attempt-id"); parser.add_argument("--audit-walltime-seconds", type=int); parser.add_argument("--cpus-per-worker", type=int, default=1)
    parser.add_argument("--worker-count", type=int, default=1); parser.add_argument("--allocated-task-count", type=int, default=1); parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--benchmark-record"); parser.add_argument("--benchmark-output"); parser.add_argument("--max-subjects", type=int)
    value = parser.parse_args()
    if value.mode == "plan" and (value.logical_shards is None or value.logical_shard_indices is None): parser.error("plan requires --logical-shards and --logical-shard-indices")
    if value.mode != "plan" and not value.audit_plan: parser.error("shards/merge require --audit-plan")
    if value.mode == "shards" and (value.logical_shard_indices is None or not value.attempt_id): parser.error("shards requires --logical-shard-indices and --attempt-id")
    if value.mode != "plan" and value.audit_walltime_seconds is None: parser.error("shards/merge require --audit-walltime-seconds")
    return value


def main() -> int:
    args = _args()
    if args.mode == "plan": print(write_audit_plan(args))
    elif args.mode == "shards": publish_shard(args)
    else: merge_shards(args)
    return 0


if __name__ == "__main__": raise SystemExit(main())
