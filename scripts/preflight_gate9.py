#!/usr/bin/env python3
import argparse, json
from adaptive_roi_rppg.evaluation.adapters.mmpd.preflight import preflight_gate9_inputs
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--plan", required=True); p.add_argument("--raw-root", required=True); p.add_argument("--checkpoint-root", required=True); p.add_argument("--rule-root", required=True); p.add_argument("--raw-inventory", required=True); p.add_argument("--rule-manifest", required=True); p.add_argument("--checkpoint-manifest", required=True); p.add_argument("--code-snapshot", required=True); p.add_argument("--extractor", required=True); p.add_argument("--extraction-code-sha256", required=True); a=p.parse_args(); print(json.dumps(preflight_gate9_inputs(a.plan, raw_root=a.raw_root, checkpoint_root=a.checkpoint_root, rule_root=a.rule_root, raw_inventory=a.raw_inventory, rule_manifest=a.rule_manifest, checkpoint_manifest=a.checkpoint_manifest, code_snapshot=a.code_snapshot, extractor_id=a.extractor, extraction_code_sha256=a.extraction_code_sha256), sort_keys=True))
if __name__ == "__main__": main()
