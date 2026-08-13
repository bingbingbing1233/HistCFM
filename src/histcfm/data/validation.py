"""Read-only preflight validation for HistCFM-ready inputs.

Validation rejects incompatible or ambiguous inputs before a loader can
silently alter model dimensions. It does not modify input arrays or files.
Distributed under GNU GPL version 3 only; see ``LICENSE``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .image_io import load_image
from .splitting import (
    build_split_patch_coordinates,
    split_row_intervals,
    validation_pixel_interval,
)


PathLike = Union[str, Path]
UNI_MIN_L2_NORM = 1e-12


@dataclass(frozen=True)
class PreflightSummary:
    """Counts established by a successful HistCFM-ready input preflight."""

    image_shape: Tuple[int, ...]
    mask_shape: Tuple[int, ...]
    mask_cell_count: int
    expression_cell_count: int
    gene_count: int
    patch_count: int
    uni_feature_count: int
    uni_feature_dim: int


@dataclass(frozen=True)
class PatchInventory:
    """Read-only inventory of eligible cells in formal patch coordinates."""

    patch_keys: Tuple[str, ...]
    largest_cell_count: int


def build_patch_key(
    slide_id: str,
    row_start: int,
    column_start: int,
    patch_height: int,
    patch_width: int,
    level: int = 0,
) -> str:
    """Build the canonical key used by the precomputed UNI feature store."""

    slide_id = str(slide_id)
    if (
        not slide_id
        or slide_id in {".", ".."}
        or any(marker in slide_id for marker in ("/", "\\", "|", ":"))
    ):
        raise ValueError("slide_id must be a non-empty path-free identifier")
    values = (row_start, column_start, patch_height, patch_width, level)
    if any(int(value) < 0 for value in values):
        raise ValueError("patch coordinates, dimensions, and level must be non-negative")
    if int(patch_height) == 0 or int(patch_width) == 0:
        raise ValueError("patch dimensions must be positive")
    return (
        f"{slide_id}|{int(row_start)}|{int(column_start)}|"
        f"{int(patch_height)}|{int(patch_width)}|{int(level)}"
    )


def validate_required_files(paths: Mapping[str, Optional[PathLike]]) -> None:
    """Require each non-optional path to identify an existing regular file."""

    missing = []
    for label, raw_path in paths.items():
        if raw_path is None:
            missing.append(f"{label}=<not provided>")
            continue
        path = Path(raw_path)
        if not path.is_file():
            missing.append(f"{label}={path}")
    if missing:
        raise FileNotFoundError("Missing required HistCFM input files: " + ", ".join(missing))


def validate_split_nonempty(
    image_height: int,
    validation_division: Sequence[float],
) -> Tuple[int, int]:
    """Validate the audited row-wise train/validation split without fallback."""

    start, stop = validation_pixel_interval(image_height, validation_division)
    validation_count = stop - start
    training_count = int(image_height) - validation_count
    if training_count == 0:
        raise ValueError("The configured split must leave non-empty training rows")
    return training_count, validation_count


def inspect_patch_inventory(
    *,
    histology_path: PathLike,
    nucleus_mask_path: PathLike,
    matched_nuclei_path: PathLike,
    expression_path: PathLike,
    cell_type_path: Optional[PathLike],
    validation_division: Sequence[float],
    patch_height: int,
    patch_width: int,
    overlap: int,
    min_nucleus_area: float,
    mode: str,
    require_cell_types: bool,
) -> PatchInventory:
    """Derive train/validation patch keys without normalization or file writes."""

    import numpy as np
    import pandas as pd

    if mode not in {"train", "validation"}:
        raise ValueError("Patch inventory mode must be 'train' or 'validation'")
    histology = load_image(histology_path)
    mask = load_image(nucleus_mask_path)
    height, width = histology.shape[:2]
    if int(patch_height) > height or int(patch_width) > width:
        raise ValueError("Patch dimensions must not exceed histology dimensions")
    if int(overlap) < 0 or int(overlap) >= min(int(patch_height), int(patch_width)):
        raise ValueError("Patch overlap must be smaller than each patch dimension")
    validate_split_nonempty(height, validation_division)
    row_intervals = split_row_intervals(height, validation_division, mode)
    coordinates = build_split_patch_coordinates(
        image_height=height,
        image_width=width,
        validation_division=validation_division,
        patch_height=patch_height,
        patch_width=patch_width,
        overlap=overlap,
        mode=mode,
    )

    matched = pd.read_csv(matched_nuclei_path, index_col=0)
    if "size_pix_histology" not in matched.columns:
        raise ValueError("Matched-nuclei table must contain 'size_pix_histology'")
    sizes = matched["size_pix_histology"].to_numpy(dtype=np.float64)
    matched_ids = _as_integer_ids(matched.index, "matched-nuclei index")
    eligible = set(
        matched_ids[sizes >= float(min_nucleus_area)].astype(np.int64).tolist()
    )
    expression = pd.read_csv(expression_path, index_col=0)
    eligible &= set(_as_integer_ids(expression.index, "expression index").tolist())
    if require_cell_types:
        cell_types = pd.read_csv(cell_type_path)
        if "c_id" not in cell_types.columns:
            raise ValueError("Cell-type table must contain 'c_id'")
        eligible &= set(_as_integer_ids(cell_types["c_id"], "cell-type c_id").tolist())
    selected_ids = set()
    for start, stop in row_intervals:
        selected_ids.update(
            np.unique(mask[start:stop, :]).astype(np.int64).tolist()
        )
    selected_ids.discard(0)
    eligible &= selected_ids
    slide_id = Path(histology_path).stem
    keys = []
    largest = 0
    for row_start, column_start in coordinates:
        patch = mask[
            int(row_start) : int(row_start) + int(patch_height),
            int(column_start) : int(column_start) + int(patch_width),
        ]
        patch_ids = set(np.unique(patch).astype(np.int64).tolist())
        patch_ids.discard(0)
        count = len(patch_ids & eligible)
        if count:
            keys.append(
                build_patch_key(
                    slide_id,
                    int(row_start),
                    int(column_start),
                    int(patch_height),
                    int(patch_width),
                )
            )
            largest = max(largest, count)
    if not keys:
        raise ValueError(f"{mode} split contains no patch with an eligible cell")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{mode} patch keys are not unique")
    return PatchInventory(tuple(keys), largest)


def _read_indexed_csv(path: PathLike, index_column):
    import pandas as pd

    table = pd.read_csv(path)
    if isinstance(index_column, str):
        if index_column not in table.columns:
            raise ValueError(f"{path!s} must contain index column {index_column!r}")
        table = table.set_index(index_column)
    else:
        table = table.set_index(table.columns[int(index_column)])
    if not table.index.is_unique:
        raise ValueError(f"{path!s} contains duplicate cell IDs")
    return table


def _as_integer_ids(values: Iterable, label: str):
    import numpy as np

    array = np.asarray(list(values))
    try:
        numeric = array.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain integer-compatible cell IDs") from error
    if array.size and not np.all(array.astype(str) == numeric.astype(str)):
        try:
            if not np.all(array.astype(float) == numeric):
                raise ValueError(f"{label} contains non-integer cell IDs")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains non-integer cell IDs") from error
    return numeric


def load_uni_index(path: PathLike) -> Dict[str, int]:
    """Load the formal JSON key-to-row index and reject ambiguous content."""

    index_path = Path(path)
    if index_path.name != "uni_index.json":
        raise ValueError("Formal UNI index filename must be 'uni_index.json'")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"UNI JSON index contains duplicate key: {key!r}")
            result[key] = value
        return result

    with index_path.open("r", encoding="utf-8") as handle:
        raw_index = json.load(handle, object_pairs_hook=unique_object)
    if not isinstance(raw_index, dict):
        raise ValueError("UNI JSON index must be a key-to-row object")
    if not raw_index:
        raise ValueError("UNI JSON index must not be empty")
    mapping: Dict[str, int] = {}
    for key, row in raw_index.items():
        if not isinstance(key, str) or not key:
            raise ValueError("UNI feature keys must be non-empty strings")
        parts = key.split("|")
        if len(parts) != 6:
            raise ValueError(f"UNI feature key is not a canonical patch key: {key!r}")
        try:
            rebuilt = build_patch_key(parts[0], *(int(value) for value in parts[1:]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"UNI feature key is not a canonical patch key: {key!r}") from error
        if rebuilt != key:
            raise ValueError(f"UNI feature key is not canonical: {key!r}")
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError("UNI feature rows must be non-negative integers")
        mapping[key] = row
    rows = list(mapping.values())
    if len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("UNI feature rows must be unique non-negative integers")
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("UNI feature rows must continuously cover 0..N-1")
    return mapping


def load_uni_feature_store(
    index_path: PathLike,
    feature_path: PathLike,
    expected_dim: int = 1024,
    minimum_l2_norm: float = UNI_MIN_L2_NORM,
):
    """Load and validate the formal JSON+NPY store without changing values."""

    import numpy as np

    validate_required_files({"uni_index": index_path, "uni_features": feature_path})
    if Path(feature_path).name != "uni_features.npy":
        raise ValueError("Formal UNI feature filename must be 'uni_features.npy'")
    index = load_uni_index(index_path)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or int(features.shape[1]) != int(expected_dim):
        raise ValueError(
            f"UNI features must have shape (N, {int(expected_dim)}); got {features.shape}"
        )
    if features.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("UNI features must use float16 or float32 dtype")
    if len(index) != int(features.shape[0]):
        raise ValueError("UNI index and feature matrix row counts must match")
    threshold = float(minimum_l2_norm)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("UNI minimum L2 norm must be a finite positive value")
    chunk_size = 4096
    for start in range(0, int(features.shape[0]), chunk_size):
        chunk = np.asarray(features[start : start + chunk_size], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError("UNI features contain NaN or infinite values")
        norms = np.linalg.norm(chunk, axis=1)
        if np.any(norms <= threshold):
            raise ValueError(
                f"UNI feature rows must have L2 norm > {threshold:g}"
            )
    return index, features


def validate_uni_feature_store(
    patch_keys: Sequence[str],
    index_path: PathLike,
    feature_path: PathLike,
    expected_dim: int = 1024,
    minimum_l2_norm: float = UNI_MIN_L2_NORM,
) -> Tuple[int, int]:
    """Validate formal store schema, values, and complete patch-key coverage."""

    if len(patch_keys) != len(set(patch_keys)):
        raise ValueError("Patch keys must be unique")
    index, features = load_uni_feature_store(
        index_path,
        feature_path,
        expected_dim=expected_dim,
        minimum_l2_norm=minimum_l2_norm,
    )
    missing = [key for key in patch_keys if key not in index]
    if missing:
        preview = ", ".join(repr(key) for key in missing[:5])
        raise ValueError(f"UNI features are missing {len(missing)} patch keys: {preview}")
    return int(features.shape[0]), int(features.shape[1])


def validate_histcfm_ready_inputs(
    *,
    histology_path: PathLike,
    nucleus_mask_path: PathLike,
    matched_nuclei_path: PathLike,
    expression_path: PathLike,
    cell_type_path: Optional[PathLike],
    average_expression_path: Optional[PathLike],
    gene_names: Sequence[str],
    validation_division: Sequence[float],
    patch_keys: Sequence[str],
    uni_index_path: Optional[PathLike] = None,
    uni_feature_path: Optional[PathLike] = None,
    expected_uni_dim: int = 1024,
    require_cell_types: bool = True,
    require_average_expression: bool = True,
    require_uni: bool = True,
    minimum_uni_l2_norm: float = UNI_MIN_L2_NORM,
) -> PreflightSummary:
    """Validate the audited HistCFM-ready schema without changing any data."""

    import numpy as np

    paths = {
        "histology": histology_path,
        "nucleus_mask": nucleus_mask_path,
        "matched_nuclei": matched_nuclei_path,
        "expression": expression_path,
    }
    if require_cell_types:
        paths["cell_type"] = cell_type_path
    if require_average_expression:
        paths["average_expression"] = average_expression_path
    validate_required_files(paths)

    histology = load_image(histology_path)
    mask = load_image(nucleus_mask_path)
    if histology.ndim != 3 or int(histology.shape[-1]) != 3:
        raise ValueError(f"Histology must be an H x W x 3 image; got {histology.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Nucleus mask must be two-dimensional; got {mask.shape}")
    if tuple(histology.shape[:2]) != tuple(mask.shape):
        raise ValueError("Histology and nucleus mask spatial dimensions must match")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Nucleus mask must use an integer dtype")
    if np.any(mask < 0):
        raise ValueError("Nucleus mask IDs must be non-negative; ID 0 is background")
    mask_ids = np.unique(mask)
    if 0 not in mask_ids:
        raise ValueError("Nucleus mask must reserve ID 0 for background")
    mask_ids = mask_ids[mask_ids != 0].astype(np.int64)

    matched = _read_indexed_csv(matched_nuclei_path, 0)
    if "size_pix_histology" not in matched.columns:
        raise ValueError("Matched-nuclei table must contain 'size_pix_histology'")
    matched_ids = _as_integer_ids(matched.index, "matched-nuclei index")
    try:
        matched_sizes = matched["size_pix_histology"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Matched-nuclei size_pix_histology must be numeric") from error
    if not np.isfinite(matched_sizes).all() or np.any(matched_sizes < 0):
        raise ValueError("Matched-nuclei sizes must be finite and non-negative")

    expression = _read_indexed_csv(expression_path, 0)
    if not expression.columns.is_unique:
        raise ValueError("Expression gene names must be unique")
    if len(gene_names) != len(set(gene_names)):
        raise ValueError("Configured gene names must be unique")
    if list(expression.columns) != list(gene_names):
        raise ValueError(
            "Expression columns must exactly match the configured gene set and order; "
            "silent intersection or reordering is not allowed"
        )
    try:
        expression_values = expression.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Expression values must be numeric") from error
    if not np.isfinite(expression_values).all() or np.any(expression_values < 0):
        raise ValueError("Expression counts must be finite and non-negative")
    expression_ids = _as_integer_ids(expression.index, "expression index")

    mask_set = set(mask_ids.tolist())
    matched_set = set(matched_ids.tolist())
    expression_set = set(expression_ids.tolist())
    if not expression_set.issubset(mask_set):
        raise ValueError("Every expression cell ID must be present in the nucleus mask")
    if not expression_set.issubset(matched_set):
        raise ValueError("Every expression cell ID must be present in matched nuclei")

    if require_cell_types:
        cell_types = _read_indexed_csv(cell_type_path, "c_id")
        if "ct" not in cell_types.columns:
            raise ValueError("Cell-type table must contain 'c_id' and 'ct' columns")
        cell_type_ids = set(_as_integer_ids(cell_types.index, "cell-type c_id").tolist())
        if cell_type_ids != expression_set:
            raise ValueError("Cell-type c_id values must exactly match expression cell IDs")

    if require_average_expression:
        average_expression = _read_indexed_csv(average_expression_path, 0)
        if not average_expression.columns.is_unique:
            raise ValueError("Average-expression gene names must be unique")
        if list(average_expression.columns) != list(expression.columns):
            raise ValueError(
                "Average-expression genes must exactly match expression genes and order"
            )

    validate_split_nonempty(int(histology.shape[0]), validation_division)
    if not patch_keys or len(patch_keys) != len(set(patch_keys)):
        raise ValueError("The formal loader patch-key set must be non-empty and unique")

    if require_uni:
        if uni_index_path is None or uni_feature_path is None:
            raise ValueError("UNI validation requires both index and feature paths")
        uni_count, uni_dim = validate_uni_feature_store(
            patch_keys,
            uni_index_path,
            uni_feature_path,
            expected_dim=expected_uni_dim,
            minimum_l2_norm=minimum_uni_l2_norm,
        )
    else:
        uni_count, uni_dim = 0, 0

    return PreflightSummary(
        image_shape=tuple(int(value) for value in histology.shape),
        mask_shape=tuple(int(value) for value in mask.shape),
        mask_cell_count=int(mask_ids.size),
        expression_cell_count=int(expression.shape[0]),
        gene_count=int(expression.shape[1]),
        patch_count=len(patch_keys),
        uni_feature_count=uni_count,
        uni_feature_dim=uni_dim,
    )
