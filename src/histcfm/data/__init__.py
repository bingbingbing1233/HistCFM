"""Public data interfaces for HistCFM-ready cell-level inputs."""

from .dataset import HistCFMDataset
from .splitting import (
    build_split_patch_coordinates,
    split_row_intervals,
    validation_pixel_interval,
)
from .validation import (
    PreflightSummary,
    PatchInventory,
    UNI_MIN_L2_NORM,
    build_patch_key,
    inspect_patch_inventory,
    load_uni_feature_store,
    load_uni_index,
    validate_histcfm_ready_inputs,
    validate_required_files,
    validate_split_nonempty,
    validate_uni_feature_store,
)

__all__ = [
    "HistCFMDataset",
    "build_split_patch_coordinates",
    "split_row_intervals",
    "validation_pixel_interval",
    "PreflightSummary",
    "PatchInventory",
    "UNI_MIN_L2_NORM",
    "build_patch_key",
    "inspect_patch_inventory",
    "load_uni_feature_store",
    "load_uni_index",
    "validate_histcfm_ready_inputs",
    "validate_required_files",
    "validate_split_nonempty",
    "validate_uni_feature_store",
]
