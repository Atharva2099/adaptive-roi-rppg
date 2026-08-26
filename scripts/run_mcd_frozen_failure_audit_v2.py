#!/usr/bin/env python3
"""Create, shard, or merge a restartable frozen MCD failure audit.

GPU is intentionally unsupported.  A GPU profile can be added only after a
separate deterministic CPU-versus-GPU benchmark establishes that it is needed.
"""
from __future__ import annotations

import argparse
import json

from adaptive_roi_rppg.evaluation.mcd_failure_runner_v2 import _resource, _safe_token, merge_shards, publish_shard, write_audit_plan


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
    parser.add_argument("--mode", choices=("dispatch-smoke", "plan", "shards", "merge"), required=True)
    parser.add_argument("--audit-root"); parser.add_argument("--audit-plan")
    parser.add_argument("--manifest-tree"); parser.add_argument("--state-root"); parser.add_argument("--gt-root")
    parser.add_argument("--plan"); parser.add_argument("--checkpoint-root"); parser.add_argument("--split", choices=("train", "eval"))
    parser.add_argument("--run-id", required=True); parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--logical-shards", type=int); parser.add_argument("--logical-shard-indices", type=_indices)
    parser.add_argument("--attempt-id"); parser.add_argument("--audit-walltime-seconds", type=int); parser.add_argument("--cpus-per-worker", type=int, default=1)
    parser.add_argument("--worker-count", type=int, default=1); parser.add_argument("--allocated-task-count", type=int, default=1); parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--benchmark-record"); parser.add_argument("--benchmark-output"); parser.add_argument("--max-subjects", type=int)
    value = parser.parse_args()
    if value.mode == "dispatch-smoke":
        if any(getattr(value, key) is not None for key in ("manifest_tree", "state_root", "gt_root", "plan", "checkpoint_root", "audit_plan", "audit_root", "benchmark_record", "benchmark_output", "split")): parser.error("dispatch-smoke accepts no data, benchmark, or output paths")
        if value.audit_walltime_seconds is None: parser.error("dispatch-smoke requires --audit-walltime-seconds")
        return value
    if any(getattr(value, key) is None for key in ("audit_root", "manifest_tree", "state_root", "gt_root", "plan", "checkpoint_root", "split")): parser.error("plan/shards/merge require all MCD paths and --split")
    if value.mode == "plan" and (value.logical_shards is None or value.logical_shard_indices is None): parser.error("plan requires --logical-shards and --logical-shard-indices")
    if value.mode != "plan" and not value.audit_plan: parser.error("shards/merge require --audit-plan")
    if value.mode == "shards" and (value.logical_shard_indices is None or not value.attempt_id): parser.error("shards requires --logical-shard-indices and --attempt-id")
    if value.mode != "plan" and value.audit_walltime_seconds is None: parser.error("shards/merge require --audit-walltime-seconds")
    return value


def main() -> int:
    args = _args()
    if args.mode == "dispatch-smoke":
        _safe_token(args.run_id, "run ID"); resource = _resource(args)
        if args.worker_rank not in range(resource["worker_count"]): raise SystemExit("dispatch-smoke worker rank is outside W")
        print(json.dumps({"schema": "mcd-frozen-failure-audit-dispatch-smoke-v1", "proof": "NO_DATA_ACCESS", "run_id": args.run_id, "worker_rank": args.worker_rank, "worker_count": resource["worker_count"], "placeholder_logical_shard_ids": [f"dispatch-placeholder-{args.worker_rank}"], "walltime_seconds": resource["walltime_seconds"]}, sort_keys=True))
    elif args.mode == "plan": print(write_audit_plan(args))
    elif args.mode == "shards": publish_shard(args)
    else: merge_shards(args)
    return 0


if __name__ == "__main__": raise SystemExit(main())
