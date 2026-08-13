#!/usr/bin/env python3
"""Generate the deterministic, patient-free HistCFM public smoke-test data.

The generator uses only Python's standard library. It accepts no source-data
path and never calls an image encoder or another model. The repository already
contains its generated outputs; users do not run this script before the demo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import sys
from array import array
from pathlib import Path
from typing import Iterable, Sequence


SEED = 20260813
HEIGHT = 768
WIDTH = 768
PATCH_HEIGHT = 256
PATCH_WIDTH = 256
OVERLAP = 32
VALIDATION_STOP = 256
CELL_COUNT = 96
GENE_COUNT = 24
FEATURE_DIM = 1024
GENES = tuple(f"synthetic_gene_{index:02d}" for index in range(1, GENE_COUNT + 1))
CELL_TYPES = (
    "synthetic_type_a",
    "synthetic_type_b",
    "synthetic_type_c",
    "synthetic_type_d",
)


class GenerationError(RuntimeError):
    """Raised when deterministic demo generation violates its own contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/demo/data"),
        help="Destination data directory (default: examples/demo/data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the known synthetic output files.",
    )
    return parser.parse_args(argv)


def _clamp(value: int) -> int:
    return max(0, min(255, int(value)))


def _tiff_entry(tag: int, field_type: int, count: int, value: int) -> bytes:
    if field_type == 3 and count == 1:
        encoded = struct.pack("<H", value) + b"\x00\x00"
    else:
        encoded = struct.pack("<I", value)
    return struct.pack("<HHI", tag, field_type, count) + encoded


def _write_rgb_tiff(path: Path, pixels: bytes) -> None:
    expected = HEIGHT * WIDTH * 3
    if len(pixels) != expected:
        raise GenerationError(f"RGB byte count must be {expected}")
    entry_count = 11
    ifd_end = 8 + 2 + entry_count * 12 + 4
    bits_offset = ifd_end
    sample_format_offset = bits_offset + 6
    data_offset = sample_format_offset + 6
    entries = [
        _tiff_entry(256, 4, 1, WIDTH),
        _tiff_entry(257, 4, 1, HEIGHT),
        _tiff_entry(258, 3, 3, bits_offset),
        _tiff_entry(259, 3, 1, 1),
        _tiff_entry(262, 3, 1, 2),
        _tiff_entry(273, 4, 1, data_offset),
        _tiff_entry(277, 3, 1, 3),
        _tiff_entry(278, 4, 1, HEIGHT),
        _tiff_entry(279, 4, 1, expected),
        _tiff_entry(284, 3, 1, 1),
        _tiff_entry(339, 3, 3, sample_format_offset),
    ]
    with path.open("wb") as handle:
        handle.write(b"II")
        handle.write(struct.pack("<H", 42))
        handle.write(struct.pack("<I", 8))
        handle.write(struct.pack("<H", entry_count))
        handle.writelines(entries)
        handle.write(struct.pack("<I", 0))
        handle.write(struct.pack("<HHH", 8, 8, 8))
        handle.write(struct.pack("<HHH", 1, 1, 1))
        handle.write(pixels)


def _write_mask_tiff(path: Path, labels: array) -> None:
    expected_count = HEIGHT * WIDTH
    if labels.typecode != "I" or len(labels) != expected_count:
        raise GenerationError("Mask must be a uint32-compatible array of the expected size")
    values = array("I", labels)
    if values.itemsize != 4:
        raise GenerationError("This generator requires a four-byte unsigned integer array")
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    entry_count = 10
    data_offset = 8 + 2 + entry_count * 12 + 4
    entries = [
        _tiff_entry(256, 4, 1, WIDTH),
        _tiff_entry(257, 4, 1, HEIGHT),
        _tiff_entry(258, 3, 1, 32),
        _tiff_entry(259, 3, 1, 1),
        _tiff_entry(262, 3, 1, 1),
        _tiff_entry(273, 4, 1, data_offset),
        _tiff_entry(277, 3, 1, 1),
        _tiff_entry(278, 4, 1, HEIGHT),
        _tiff_entry(279, 4, 1, len(payload)),
        _tiff_entry(339, 3, 1, 1),
    ]
    with path.open("wb") as handle:
        handle.write(b"II")
        handle.write(struct.pack("<H", 42))
        handle.write(struct.pack("<I", 8))
        handle.write(struct.pack("<H", entry_count))
        handle.writelines(entries)
        handle.write(struct.pack("<I", 0))
        handle.write(payload)


def _write_float16_npy(path: Path, rows: Sequence[Sequence[float]]) -> None:
    if not rows or any(len(row) != FEATURE_DIM for row in rows):
        raise GenerationError("Synthetic feature rows must be non-empty and 1024-dimensional")
    header_text = (
        "{'descr': '<f2', 'fortran_order': False, "
        f"'shape': ({len(rows)}, {FEATURE_DIM}), }}"
    )
    padding = (-(10 + len(header_text) + 1)) % 16
    header = (header_text + " " * padding + "\n").encode("latin-1")
    if len(header) > 65535:
        raise GenerationError("NPY header is unexpectedly large")
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(b"\x01\x00")
        handle.write(struct.pack("<H", len(header)))
        handle.write(header)
        for row in rows:
            for value in row:
                handle.write(struct.pack("<e", value))


def _cell_records() -> list[dict[str, int | str]]:
    rng = random.Random(SEED)
    records: list[dict[str, int | str]] = []
    for row in range(8):
        for column in range(12):
            index = row * 12 + column
            records.append(
                {
                    "cell_id": index + 1,
                    "x": 45 + column * 61 + rng.randint(-5, 5),
                    "y": 45 + row * 93 + rng.randint(-6, 6),
                    "radius_x": rng.randint(8, 13),
                    "radius_y": rng.randint(7, 12),
                    "cell_type": CELL_TYPES[(row + column) % len(CELL_TYPES)],
                    "type_index": (row + column) % len(CELL_TYPES),
                }
            )
    if len(records) != CELL_COUNT:
        raise GenerationError("Unexpected synthetic cell count")
    return records


def _draw_mask(records: Sequence[dict[str, int | str]]) -> tuple[array, dict[int, int]]:
    labels = array("I", [0]) * (HEIGHT * WIDTH)
    for record in records:
        cell_id = int(record["cell_id"])
        center_x = int(record["x"])
        center_y = int(record["y"])
        radius_x = int(record["radius_x"])
        radius_y = int(record["radius_y"])
        for y in range(center_y - radius_y, center_y + radius_y + 1):
            for x in range(center_x - radius_x, center_x + radius_x + 1):
                normalized = ((x - center_x) / radius_x) ** 2 + (
                    (y - center_y) / radius_y
                ) ** 2
                if normalized <= 1.0:
                    labels[y * WIDTH + x] = cell_id
    areas = {int(record["cell_id"]): 0 for record in records}
    for value in labels:
        if value:
            areas[int(value)] += 1
    if set(areas.values()) == {0} or any(area < 10 for area in areas.values()):
        raise GenerationError("Synthetic cells must all exceed the formal minimum area")
    return labels, areas


def _render_histology(labels: array, records: Sequence[dict[str, int | str]]) -> bytes:
    rng = random.Random(SEED + 1)
    type_by_id = {int(record["cell_id"]): int(record["type_index"]) for record in records}
    pixels = bytearray(HEIGHT * WIDTH * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            label = int(labels[y * WIDTH + x])
            noise = rng.randint(-5, 5)
            if label == 0:
                red = 225 + ((x // 32 + y // 48) % 9) + noise
                green = 188 + ((x // 53 + y // 29) % 11) + noise
                blue = 205 + ((x // 41 + y // 37) % 13) + noise
            else:
                type_index = type_by_id[label]
                red = 105 + 9 * type_index + noise
                green = 52 + 7 * type_index + noise
                blue = 132 + 8 * type_index + noise
            offset = (y * WIDTH + x) * 3
            pixels[offset : offset + 3] = bytes(
                (_clamp(red), _clamp(green), _clamp(blue))
            )
    return bytes(pixels)


def _write_tables(
    output: Path,
    records: Sequence[dict[str, int | str]],
    areas: dict[int, int],
) -> None:
    with (output / "matched_nuclei.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id_histology", "id_xenium", "overlap", "size_pix_histology"])
        for record in records:
            cell_id = int(record["cell_id"])
            writer.writerow([cell_id, cell_id, "1.0", areas[cell_id]])

    expression_rng = random.Random(SEED + 2)
    with (output / "expression.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell_id", *GENES])
        for record in records:
            type_index = int(record["type_index"])
            x_bin = int(record["x"]) // 192
            y_bin = int(record["y"]) // 192
            counts = []
            for gene_index in range(GENE_COUNT):
                baseline = (gene_index % 4) + expression_rng.randint(0, 3)
                marker = 7 if gene_index % len(CELL_TYPES) == type_index else 0
                spatial = (x_bin + 2 * y_bin + gene_index) % 4
                counts.append(baseline + marker + spatial)
            writer.writerow([record["cell_id"], *counts])

    with (output / "cell_types.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["c_id", "ct"])
        for record in records:
            writer.writerow([record["cell_id"], record["cell_type"]])


def _width_starts(overlap: int) -> list[int]:
    starts = list(range(0, WIDTH - PATCH_WIDTH, PATCH_WIDTH - overlap))
    starts.append(WIDTH - PATCH_WIDTH)
    return starts


def _patch_keys(labels: array) -> tuple[list[str], list[str]]:
    eligible = set(range(1, CELL_COUNT + 1))
    train_coordinates = [
        (row, column)
        for row in (VALIDATION_STOP, VALIDATION_STOP + PATCH_HEIGHT)
        for column in _width_starts(0)
    ]
    validation_coordinates = [
        (0, column)
        for column in _width_starts(OVERLAP)
    ]

    def select(coordinates: Iterable[tuple[int, int]]) -> list[str]:
        keys = []
        for row, column in coordinates:
            found = set()
            for y in range(row, row + PATCH_HEIGHT):
                start = y * WIDTH + column
                found.update(labels[start : start + PATCH_WIDTH])
            found.discard(0)
            if found & eligible:
                keys.append(
                    f"histology|{row}|{column}|{PATCH_HEIGHT}|{PATCH_WIDTH}|0"
                )
        return keys

    train = select(train_coordinates)
    validation = select(validation_coordinates)
    if len(train) < 2 or len(validation) < 2:
        raise GenerationError("Demo must contain multiple train and validation patches")
    if len(set(train + validation)) != len(train) + len(validation):
        raise GenerationError("Canonical demo patch keys must be unique")
    return train, validation


def _synthetic_features(count: int) -> list[list[float]]:
    rng = random.Random(SEED + 3)
    rows = []
    fingerprints = set()
    for row_index in range(count):
        row = []
        for column_index in range(FEATURE_DIM):
            periodic = 0.2 * math.sin((row_index + 1) * (column_index + 3) / 37.0)
            value = rng.uniform(-1.0, 1.0) + periodic + row_index * 0.002
            if not math.isfinite(value):
                raise GenerationError("Synthetic feature generation produced a non-finite value")
            row.append(value)
        packed = b"".join(struct.pack("<e", value) for value in row)
        norm_squared = sum(struct.unpack("<e", packed[index : index + 2])[0] ** 2 for index in range(0, len(packed), 2))
        if norm_squared <= 1e-24 or packed in fingerprints:
            raise GenerationError("Synthetic feature rows must be nonzero and unique")
        fingerprints.add(packed)
        rows.append(row)
    return rows


def generate(output: Path, force: bool) -> dict[str, int]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_dir = output / "uni"
    feature_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        output / "histology.tif",
        output / "nucleus_mask.tif",
        output / "matched_nuclei.csv",
        output / "expression.csv",
        output / "cell_types.csv",
        feature_dir / "uni_index.json",
        feature_dir / "uni_features.npy",
    )
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing and not force:
        raise GenerationError(
            "Refusing to replace existing outputs without --force: "
            + ", ".join(path.name for path in existing)
        )
    if any(path.is_symlink() for path in existing):
        raise GenerationError("Refusing to replace a symlinked output")

    records = _cell_records()
    labels, areas = _draw_mask(records)
    pixels = _render_histology(labels, records)
    train_keys, validation_keys = _patch_keys(labels)
    keys = train_keys + validation_keys
    features = _synthetic_features(len(keys))

    _write_rgb_tiff(output / "histology.tif", pixels)
    _write_mask_tiff(output / "nucleus_mask.tif", labels)
    _write_tables(output, records, areas)
    with (feature_dir / "uni_index.json").open("w", encoding="utf-8") as handle:
        json.dump({key: index for index, key in enumerate(keys)}, handle, indent=2)
        handle.write("\n")
    _write_float16_npy(feature_dir / "uni_features.npy", features)
    return {
        "seed": SEED,
        "cells": len(records),
        "genes": len(GENES),
        "cell_types": len(CELL_TYPES),
        "train_patches": len(train_keys),
        "validation_patches": len(validation_keys),
        "feature_rows": len(keys),
        "feature_dim": FEATURE_DIM,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate(args.output_dir, args.force)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
