"""Shared strict spatial split and patch-coordinate generation."""

import math
from typing import List, Sequence, Tuple


Interval = Tuple[int, int]
Coordinate = Tuple[int, int]


def validation_pixel_interval(
    image_height: int,
    validation_division: Sequence[float],
) -> Interval:
    """Convert two height fractions to one clamped half-open pixel interval."""

    height = int(image_height)
    if height <= 0:
        raise ValueError("image_height must be positive")
    if len(validation_division) != 2:
        raise ValueError("validation_division must contain exactly two fractions")
    fractions = []
    for index, value in enumerate(validation_division):
        try:
            fraction = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"validation_division[{index}] must be numeric"
            ) from error
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"validation_division[{index}] must be finite and in [0, 1]"
            )
        fractions.append(fraction)
    start = int(round(fractions[0] * height))
    stop = int(round(fractions[1] * height))
    if stop < start:
        start, stop = stop, start
    start = max(0, min(height, start))
    stop = max(0, min(height, stop))
    if start == stop:
        raise ValueError("validation interval must contain at least one image row")
    return start, stop


def split_row_intervals(
    image_height: int,
    validation_division: Sequence[float],
    mode: str,
    test_division=None,
) -> Tuple[Interval, ...]:
    """Return contiguous row intervals for one strict train/validation split."""

    start, stop = validation_pixel_interval(image_height, validation_division)
    if test_division is not None:
        test_start, test_stop = validation_pixel_interval(image_height, test_division)
        if max(start, test_start) < min(stop, test_stop):
            raise ValueError("Validation and test intervals overlap")
        if mode == "test":
            return ((test_start, test_stop),)
        if mode == "train":
            boundaries = sorted([(start, stop), (test_start, test_stop)])
            intervals, cursor = [], 0
            for lo, hi in boundaries:
                if cursor < lo:
                    intervals.append((cursor, lo))
                cursor = hi
            if cursor < image_height:
                intervals.append((cursor, image_height))
            if not intervals:
                raise ValueError("Validation and test leave no training rows")
            return tuple(intervals)
    elif mode == "test":
        raise ValueError("Test inference requires data.test_split")
    if mode in {"validation", "val", "prediction", "predict"}:
        return ((start, stop),)
    if mode != "train":
        raise ValueError("mode must be train, validation/val, or prediction/predict")
    intervals = []
    if start > 0:
        intervals.append((0, start))
    if stop < int(image_height):
        intervals.append((stop, int(image_height)))
    if not intervals:
        raise ValueError("validation interval leaves no training rows")
    return tuple(intervals)


def _axis_starts(
    start: int,
    stop: int,
    patch_size: int,
    overlap: int,
) -> List[int]:
    """Tile one contiguous interval without clipping or crossing its boundary."""

    size = int(patch_size)
    overlap = int(overlap)
    if size <= 0:
        raise ValueError("patch dimensions must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("patch overlap must be non-negative and smaller than patch size")
    if int(stop) - int(start) < size:
        return []
    last = int(stop) - size
    step = size - overlap
    starts = list(range(int(start), last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_split_patch_coordinates(
    *,
    image_height: int,
    image_width: int,
    validation_division: Sequence[float],
    patch_height: int,
    patch_width: int,
    overlap: int,
    mode: str,
    test_division=None,
) -> Tuple[Coordinate, ...]:
    """Build fixed-size patches wholly inside one spatial split.

    Training tiles each contiguous complement interval independently with
    zero configured overlap. Validation/prediction tiles only the validation
    interval and uses the requested overlap. Terminal patches retain their
    full size and remain inside their interval.
    """

    height = int(image_height)
    width = int(image_width)
    patch_height = int(patch_height)
    patch_width = int(patch_width)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if patch_height > height or patch_width > width:
        raise ValueError("Patch dimensions must not exceed histology dimensions")
    if mode not in {"train", "validation", "val", "prediction", "predict", "test"}:
        raise ValueError("mode must be train, validation/val, or prediction/predict")
    validation_start, validation_stop = validation_pixel_interval(
        height, validation_division
    )
    if validation_stop - validation_start < patch_height:
        raise ValueError(
            "Validation interval is too small for one full-height patch: "
            f"interval={validation_stop - validation_start}, patch_height={patch_height}"
        )
    train_intervals = ()
    if mode not in {"prediction", "predict"}:
        train_intervals = split_row_intervals(height, validation_division, "train", test_division)
        if not any(stop - start >= patch_height for start, stop in train_intervals):
            raise ValueError(
                "Training complement is too small for one full-height patch: "
                f"patch_height={patch_height}"
            )

    normalized_mode = "train" if mode == "train" else "validation"
    effective_overlap = 0 if normalized_mode == "train" else int(overlap)
    if effective_overlap < 0 or effective_overlap >= min(patch_height, patch_width):
        raise ValueError("Patch overlap must be smaller than each patch dimension")
    row_intervals = (
        train_intervals
        if normalized_mode == "train"
        else split_row_intervals(height, validation_division, mode, test_division)
    )
    row_starts = []
    for start, stop in row_intervals:
        row_starts.extend(
            _axis_starts(start, stop, patch_height, effective_overlap)
        )
    column_starts = _axis_starts(0, width, patch_width, effective_overlap)
    if not row_starts:
        raise ValueError(f"{normalized_mode} split cannot contain a full-size patch")
    if not column_starts:
        raise ValueError("Image width cannot contain a full-size patch")
    return tuple(
        (row_start, column_start)
        for row_start in row_starts
        for column_start in column_starts
    )


def partition_cell_ids(mask, intervals):
    """Assign each nucleus once by its whole-mask row centroid (half-open strips)."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    labels, inverse = np.unique(mask[ys, xs], return_inverse=True)
    centroids = np.bincount(inverse, weights=ys) / np.bincount(inverse)
    selected = np.zeros(len(labels), dtype=bool)
    for lo, hi in intervals:
        selected |= (centroids >= lo) & (centroids < hi)
    return set(labels[selected].tolist())
