#!/usr/bin/env python3
"""Validate authenticated MCD source inputs and stream worker subsets."""
import argparse
from adaptive_roi_rppg.evaluation.source_subsets import prepare_full_task_subsets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-csv", required=True)
parser.add_argument("--plan-json", required=True)
parser.add_argument("--complete-json", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
prepare_full_task_subsets(args.source_csv, args.plan_json, args.complete_json, args.output_dir)
