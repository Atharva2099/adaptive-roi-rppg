#!/usr/bin/env python3
"""Make a read-only visual and tabular inventory of official MMPD cache files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_official_mmpd_extractor import (
    clip_key, input_to_label, read_clip_list, subject_key, validate_cache_selection,
)


def _read_file_list(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["input_files"] for row in csv.DictReader(handle) if row.get("input_files")]


def _clip_key(path: str) -> str:
    return clip_key(path)


def _selected_inputs(paths: list[str], limit: int) -> list[tuple[str, str]]:
    grouped: dict[str, str] = {}
    for path in sorted(paths):
        grouped.setdefault(_clip_key(path), path)
    by_subject: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, path in sorted(grouped.items()):
        by_subject[subject_key(path)].append((key, path))
    selected = []
    subject_lists = [by_subject[key] for key in sorted(by_subject)]
    while len(selected) < limit and any(subject_lists):
        for subject_list in subject_lists:
            if subject_list and len(selected) < limit:
                selected.append(subject_list.pop(0))
    return selected


def _inventory_record(input_path: str) -> dict[str, Any]:
    import numpy as np

    input_file = Path(input_path)
    label_file = input_to_label(input_path)
    data = np.load(input_file, allow_pickle=False)
    label = np.load(label_file, allow_pickle=False)
    frame_indices = [0, int(data.shape[0] // 2), int(data.shape[0] - 1)]
    return {
        "clip_key": _clip_key(input_path),
        "source_clip_id": None,
        "source_mapping": "unmapped_cache_collection",
        "input_path": str(input_file),
        "label_path": str(label_file),
        "input_shape": list(data.shape),
        "label_shape": list(label.shape),
        "input_dtype": str(data.dtype),
        "label_dtype": str(label.dtype),
        "input_finite": bool(np.isfinite(data).all()),
        "label_finite": bool(np.isfinite(label).all()),
        "frame_indices": frame_indices,
    }


def _to_uint8(frame):
    import numpy as np

    values = frame.astype(np.float32, copy=False)
    if not np.isfinite(values).any():
        return np.zeros(values.shape, dtype=np.uint8)
    if values.size and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def inspect(file_list_path: Path, sheet_path: Path, inventory_csv: Path,
            inventory_json: Path, limit: int, clip_list_path: Path | None = None) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw
    import numpy as np

    paths = _read_file_list(file_list_path)
    if clip_list_path:
        requested_ids = read_clip_list(clip_list_path)
        validate_cache_selection(paths, requested_ids)
        # The list establishes the requested cohort and output cardinality;
        # official cache filenames do not preserve a traceable source index.
        selected = [(None, path) for path in paths]
    else:
        selected = _selected_inputs(paths, limit)
    if not selected:
        raise RuntimeError("official cache file list produced no records")
    records = [_inventory_record(path) for _, path in selected]
    panels: list[Image.Image] = []
    for record in records:
        data = np.load(record["input_path"], allow_pickle=False)
        for frame_index in record["frame_indices"]:
            image = Image.fromarray(_to_uint8(data[frame_index])).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="yellow", width=2)
            draw.text((3, 3), f"{record['clip_key']} f{frame_index}", fill="yellow")
            panels.append(image)
    if panels:
        width = max(image.width for image in panels)
        height = max(image.height for image in panels)
        sheet = Image.new("RGB", (width * 3, height * ((len(panels) + 2) // 3)), "black")
        for index, image in enumerate(panels):
            sheet.paste(image, ((index % 3) * width, (index // 3) * height))
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path)
    if not sheet_path.is_file():
        raise RuntimeError(f"official inspection sheet was not created: {sheet_path}")
    fields = sorted({key for record in records for key in record})
    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    with inventory_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                for key, value in record.items()
            })
    inventory_json.parent.mkdir(parents=True, exist_ok=True)
    inventory_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--clip-limit", type=int, default=12)
    parser.add_argument("--clip-list", type=Path,
                        help="optional requested cohort; validate its size and render the resulting cache items without source-ID mapping")
    args = parser.parse_args(argv)
    if args.clip_limit < 1:
        parser.error("--clip-limit must be positive")
    inspect(args.file_list, args.sheet, args.inventory_csv, args.inventory_json,
            args.clip_limit, args.clip_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
