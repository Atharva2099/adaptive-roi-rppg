#!/usr/bin/env python3
"""Write the single engineering-only Gate 9 cohort manifest."""
import argparse, json
from pathlib import Path
from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.evaluation.adapters.mmpd.plan import build_engineering_plan

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(canonical_json_bytes(build_engineering_plan()))
if __name__ == "__main__": main()
