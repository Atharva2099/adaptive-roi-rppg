#!/usr/bin/env python3
"""Freeze the canonical worst-54 MCD selection once from replay rows."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from adaptive_roi_rppg.contracts import canonical_json_bytes
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.evaluation.long_term_regret import CLIP_SELECTION_FIELDS, SELECTION_SCHEMA, csv_bytes
from adaptive_roi_rppg.evaluation.source_subsets import stream_source_selection
from verify_mcd_long_term_regret import _source_manifest, _load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-per-hop", required=True)
    parser.add_argument("--source-manifest", required=True)
    args = parser.parse_args()
    config, source = _load_config(), Path(args.source_per_hop)
    source_info = _source_manifest(source, args.source_manifest)
    if source.is_symlink() or not source.is_file():
        raise ContractValidationError("replay per-hop source must be a regular file")
    root = Path(args.output_dir)
    if root.exists() or root.is_symlink() or not root.parent.is_dir() or root.parent.is_symlink():
        raise ContractValidationError("selection output must be a fresh path below a real parent")
    rows, source_sha, _source_bytes = stream_source_selection(source, expected_sha256=source_info["source_sha256"], expected_bytes=source_info["source_bytes"], expected_checkpoint_sha256=source_info["advantage_checkpoint_sha256"])
    raw = csv_bytes(rows, CLIP_SELECTION_FIELDS)
    root.mkdir()
    (root / "clip_selection.csv").write_bytes(raw)
    provenance = {"schema": SELECTION_SCHEMA, "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "source_sha256": source_sha, "source_manifest_sha256": source_info.get("audit_plan_sha256"),
        "selection_sha256": hashlib.sha256(raw).hexdigest(), "clip_count": len(rows),
        "selection_rule": "mean over Advantage PPO seeds 0,1,2; descending MAE then clip_id tie-break"}
    (root / "selection_provenance.json").write_bytes(canonical_json_bytes(provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
