#!/usr/bin/env python3
"""Validate authenticated MCD source inputs and stream worker subsets."""
import argparse
from adaptive_roi_rppg.evaluation.source_subsets import prepare_full_task_subsets, prepare_task_subsets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-csv", required=True)
parser.add_argument("--plan-json", required=True)
parser.add_argument("--complete-json", required=True)
parser.add_argument("--selection-csv")
parser.add_argument("--full-cohort", action="store_true", help="make 533 x 3 tasks in source plan clip order")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
if args.full_cohort:
    if args.selection_csv:
        parser.error("--selection-csv is incompatible with --full-cohort")
    prepare_full_task_subsets(args.source_csv, args.plan_json, args.complete_json, args.output_dir)
else:
    if not args.selection_csv:
        parser.error("--selection-csv is required unless --full-cohort is set")
    prepare_task_subsets(args.source_csv, args.plan_json, args.complete_json, args.selection_csv, args.output_dir)
