#!/usr/bin/env python3
"""Production Gate 9-local runner; extraction is an explicit callable boundary."""
import argparse
from pathlib import Path
import hashlib
from adaptive_roi_rppg.evaluation.adapters.mmpd.plan import load_gate9_plan
from adaptive_roi_rppg.evaluation.adapters.mmpd.preflight import preflight_gate9_inputs
from adaptive_roi_rppg.evaluation.adapters.mmpd.extraction import load_extractor, execute_extractor
from adaptive_roi_rppg.evaluation.adapters.mmpd.publication import start_gate9_evaluation, publish_gate9_evaluation, fail_gate9_evaluation, rebuild_gate9_artifacts
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--plan", required=True); p.add_argument("--raw-root", required=True); p.add_argument("--checkpoint-root", required=True); p.add_argument("--rule-root", required=True); p.add_argument("--raw-inventory", required=True); p.add_argument("--rule-manifest", required=True); p.add_argument("--checkpoint-manifest", required=True); p.add_argument("--code-snapshot", required=True); p.add_argument("--extractor", required=True); p.add_argument("--extraction-code-sha256", required=True); p.add_argument("--face-landmarker-model", required=True); p.add_argument("--face-landmarker-model-sha256", required=True); p.add_argument("--output", required=True); a=p.parse_args()
    model=Path(a.face_landmarker_model)
    if model.is_symlink() or not model.is_file(): raise SystemExit("--face-landmarker-model must be a regular file")
    model_bytes=model.stat().st_size
    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()
    if model_sha256 != a.face_landmarker_model_sha256: raise SystemExit("face-landmarker model SHA-256 mismatch")
    plan=load_gate9_plan(a.plan); provenance=preflight_gate9_inputs(a.plan, raw_root=a.raw_root, checkpoint_root=a.checkpoint_root, rule_root=a.rule_root, raw_inventory=a.raw_inventory, rule_manifest=a.rule_manifest, checkpoint_manifest=a.checkpoint_manifest, code_snapshot=a.code_snapshot, extractor_id=a.extractor, extraction_code_sha256=a.extraction_code_sha256, face_landmarker_model=model, face_landmarker_model_sha256=model_sha256, face_landmarker_model_bytes=model_bytes)
    provenance["face_landmarker_model_path"] = str(model.resolve())
    root=start_gate9_evaluation(a.output, provenance)
    try:
        extracted=execute_extractor(load_extractor(a.extractor), plan, provenance)
        artifacts=rebuild_gate9_artifacts(extracted["per_hop"], plan, oracle_b=extracted["oracle_b"], oracle_c=extracted["oracle_c"], provenance=provenance)
        publish_gate9_evaluation(root, artifacts, provenance)
    except BaseException as exc: fail_gate9_evaluation(root, exc); raise
if __name__ == "__main__": main()
