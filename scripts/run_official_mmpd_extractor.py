#!/usr/bin/env python3
"""Run the pinned official MMPD preprocessing code through an external driver.

The official checkout is supplied at runtime. This module deliberately does
not reimplement or import any face-processing helper from this repository.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


CLIP_ID_PATTERN = re.compile(r"^p(?P<subject>\d+)_(?P<index>\d+)$")


def _config_args(config_path: Path) -> argparse.Namespace:
    """Build the small argparse namespace expected by the official loader."""
    return argparse.Namespace(config_file=str(config_path))


def input_to_label(input_path: str | Path) -> Path:
    """Replace only the cache filename's ``_input`` marker."""
    path = Path(input_path)
    if "_input" not in path.name:
        raise RuntimeError(f"official cache input has no _input marker: {path}")
    return path.with_name(path.name.replace("_input", "_label", 1))


def clip_key(input_path: str | Path) -> str:
    name = Path(input_path).name
    if "_input" not in name:
        raise RuntimeError(f"official cache input has no _input marker: {input_path}")
    return name.split("_input", 1)[0]


def subject_key(input_path: str | Path) -> str:
    key = clip_key(input_path)
    match = re.match(r"^(subject\d+)_", key)
    if not match:
        raise RuntimeError(f"official cache filename has no subject prefix: {input_path}")
    return match.group(1)


def read_clip_list(path: Path) -> list[str]:
    """Read and validate the exact MMPD ``p<subject>_<index>`` list."""
    requested: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw_line.split("#", 1)[0].strip()
        if not value:
            continue
        if not CLIP_ID_PATTERN.fullmatch(value):
            raise ValueError(f"invalid MMPD clip ID at {path}:{line_number}: {value!r}")
        requested.append(value)
    if not requested:
        raise ValueError(f"MMPD clip list is empty: {path}")
    duplicates = sorted({value for value in requested if requested.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate MMPD clip IDs in {path}: {duplicates}")
    return sorted(requested, key=lambda value: tuple(int(part) for part in re.findall(r"\d+", value)))


def raw_clip_id(entry: dict[str, Any]) -> str:
    return f"p{int(entry['subject'])}_{int(entry['index'])}"


def select_raw_entries(raw_entries: list[dict[str, Any]], requested_ids: list[str]) -> list[dict[str, Any]]:
    """Select exact official raw entries and reject missing or duplicate IDs."""
    available: dict[str, list[dict[str, Any]]] = {}
    for entry in raw_entries:
        available.setdefault(raw_clip_id(entry), []).append(entry)
    requested_set = set(requested_ids)
    missing = sorted(requested_set - set(available))
    duplicates = sorted(clip_id for clip_id, entries in available.items()
                        if clip_id in requested_set and len(entries) != 1)
    if missing:
        raise ValueError(f"requested MMPD clips are missing from official loader: {missing}")
    if duplicates:
        raise ValueError(f"requested MMPD clips are duplicated by official loader: {duplicates}")
    selected = [available[clip_id][0] for clip_id in requested_ids]
    selected_ids = [raw_clip_id(entry) for entry in selected]
    if selected_ids != requested_ids:
        raise RuntimeError(f"official raw selection mismatch: {selected_ids} != {requested_ids}")
    return selected


def validate_selected_raw_ids(raw_entries: list[dict[str, Any]], expected_ids: list[str]) -> None:
    actual_ids = [raw_clip_id(entry) for entry in raw_entries]
    if actual_ids != expected_ids:
        raise RuntimeError(f"official raw clip IDs mismatch: {actual_ids} != {expected_ids}")


def validate_cache_selection(inputs: list[str], expected_ids: list[str]) -> None:
    """Validate the one-output-per-raw-entry boundary.

    Official cache filenames contain metadata but not the source clip index.
    The exact ID validation therefore happens on ``loader.raw_data_dirs``;
    this post-preprocess check validates the corresponding output cardinality.
    """
    if len(inputs) != len(expected_ids):
        raise RuntimeError(f"official cache selection count mismatch: {len(inputs)} != {len(expected_ids)}")


def apply_scipy_compatibility(scipy_module: Any, scipy_config: Any) -> dict[str, Any]:
    """Apply only the inert legacy import shim required by pinned MMPDLoader."""
    shim_applied = not hasattr(scipy_config, "get_info")
    if shim_applied:
        scipy_config.get_info = lambda *args, **kwargs: {}
    return {
        "version": scipy_module.__version__,
        "shim_applied": shim_applied,
        "shim_reason": (
            "inert scipy.__config__.get_info added because pinned MMPDLoader "
            "imports but never uses it" if shim_applied else None
        ),
    }


def _load_official(official_root: Path, toolbox_path: Path, config_path: Path):
    """Load the official config and MMPDLoader without changing its source."""
    import scipy
    import scipy.__config__ as scipy_config

    scipy_info = apply_scipy_compatibility(scipy, scipy_config)
    sys.path.insert(0, str(toolbox_path))
    try:
        from config import get_config
        from dataset.data_loader.MMPDLoader import MMPDLoader
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import official MMPD toolbox from {official_root}"
        ) from exc
    return get_config(_config_args(config_path)), MMPDLoader, scipy_info


def _pair_summary(inputs: list[str], labels: list[str], expected_pairs: int,
                  expected_subjects: int) -> dict[str, Any]:
    """Inspect official cache pairs and return compact, JSON-safe facts."""
    import numpy as np

    if len(inputs) != len(labels):
        raise RuntimeError(f"official input/label count mismatch: {len(inputs)} != {len(labels)}")
    if len(inputs) != expected_pairs:
        raise RuntimeError(f"official pair count mismatch: {len(inputs)} != {expected_pairs}")
    if len(set(inputs)) != len(inputs):
        raise RuntimeError("official cache contains duplicate input paths")
    input_shapes: set[tuple[int, ...]] = set()
    label_shapes: set[tuple[int, ...]] = set()
    clip_keys = set()
    subjects = set()
    finite = True
    for input_path, label_path in zip(inputs, labels):
        input_file = Path(input_path)
        label_file = Path(label_path)
        if not input_file.is_file() or not label_file.is_file():
            raise RuntimeError(f"official cache pair file is missing: {input_file}, {label_file}")
        if label_file != input_to_label(input_file):
            raise RuntimeError(f"official cache pair does not follow input/label naming: {input_path}")
        clip_keys.add(clip_key(input_file))
        subjects.add(subject_key(input_file))
        input_array = np.load(input_file, allow_pickle=False)
        label_array = np.load(label_file, allow_pickle=False)
        if input_array.ndim != 4 or tuple(input_array.shape[1:]) != (72, 72, 3):
            raise RuntimeError(f"official input shape is not [T,72,72,3]: {input_file} {input_array.shape}")
        if label_array.ndim != 1 or label_array.shape[0] != input_array.shape[0]:
            raise RuntimeError(f"official label shape does not match input length: {input_file}")
        input_shapes.add(tuple(int(value) for value in input_array.shape))
        label_shapes.add(tuple(int(value) for value in label_array.shape))
        finite = finite and bool(np.isfinite(input_array).all()) and bool(np.isfinite(label_array).all())
    if len(clip_keys) != expected_pairs:
        raise RuntimeError(f"official clip-key count mismatch: {len(clip_keys)} != {expected_pairs}")
    if len(subjects) != expected_subjects:
        raise RuntimeError(f"official subject count mismatch: {len(subjects)} != {expected_subjects}")
    if not finite:
        raise RuntimeError("official cache contains non-finite values")
    return {
        "pair_count": len(inputs),
        "subject_count": len(subjects),
        "input_shapes": [list(shape) for shape in sorted(input_shapes)],
        "label_shapes": [list(shape) for shape in sorted(label_shapes)],
        "all_values_finite": True,
    }


def run(official_root: Path, toolbox_path: Path, config_path: Path,
        summary_path: Path, expected_pairs: int, expected_subjects: int,
        clip_list_path: Path | None = None) -> dict[str, Any]:
    """Run official preprocessing and write its cache/file-list summary."""
    config, loader_type, scipy_info = _load_official(official_root, toolbox_path, config_path)
    data_config = config.UNSUPERVISED.DATA
    requested_ids = read_clip_list(clip_list_path) if clip_list_path else None
    selected_loader_type = loader_type
    if requested_ids is not None:
        class SelectedMMPDLoader(loader_type):
            def get_raw_data(self, raw_data_path):
                return select_raw_entries(super().get_raw_data(raw_data_path), requested_ids)
        selected_loader_type = SelectedMMPDLoader
    loader = selected_loader_type(
        name="unsupervised",
        data_path=data_config.DATA_PATH,
        config_data=data_config,
    )
    if requested_ids is not None:
        validate_selected_raw_ids(loader.raw_data_dirs, requested_ids)
        validate_cache_selection(list(loader.inputs), requested_ids)
    summary = {
        "official_source_root": str(official_root),
        "config_path": str(config_path),
        "cache_root": str(data_config.CACHED_PATH),
        "file_list_path": str(data_config.FILE_LIST_PATH),
        "data_format": str(data_config.DATA_FORMAT),
        "requested_clip_ids": requested_ids,
        "scipy": scipy_info,
        **_pair_summary(list(loader.inputs), list(loader.labels), expected_pairs, expected_subjects),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolbox-path", type=Path, required=True,
                        help="official rPPG-Toolbox_MMPD checkout")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True,
                        help="path to an execution config readable by official config.py")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--clip-list", type=Path,
                        help="optional exact p<subject>_<index> list filtered after official get_raw_data")
    args = parser.parse_args(argv)
    run(args.official_root, args.toolbox_path, args.config_path, args.summary_path,
        args.expected_pairs, args.expected_subjects, args.clip_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
