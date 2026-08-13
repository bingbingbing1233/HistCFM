"""Regression coverage for strict spatial train/validation isolation."""

from pathlib import Path

import numpy as np
import pytest

import histcfm.data.dataset as dataset_module
from histcfm.config import load_config
from histcfm.data.dataset import HistCFMDataset
from histcfm.data.splitting import (
    build_split_patch_coordinates,
    validation_pixel_interval,
)
from histcfm.data.validation import inspect_patch_inventory


ROOT = Path(__file__).parents[1]


def _rows(coordinates, patch_height):
    return {
        row
        for start, _ in coordinates
        for row in range(start, start + patch_height)
    }


@pytest.mark.parametrize("division", ([0.0, 0.2], [0.8, 1.0], [0.4, 0.6]))
def test_edge_and_middle_splits_are_strictly_isolated(division):
    kwargs = {
        "image_height": 1280,
        "image_width": 768,
        "validation_division": division,
        "patch_height": 256,
        "patch_width": 256,
        "overlap": 32,
    }
    train = build_split_patch_coordinates(**kwargs, mode="train")
    validation = build_split_patch_coordinates(**kwargs, mode="validation")
    start, stop = validation_pixel_interval(1280, division)
    validation_rows = set(range(start, stop))

    assert _rows(train, 256).isdisjoint(validation_rows)
    assert _rows(validation, 256) <= validation_rows
    assert train and validation


def test_middle_split_regression_has_no_crossing_training_patch():
    train = build_split_patch_coordinates(
        image_height=2048,
        image_width=512,
        validation_division=[0.4, 0.6],
        patch_height=256,
        patch_width=256,
        overlap=32,
        mode="train",
    )
    # Pixel interval is [819, 1229). The former discontiguous-row algorithm
    # emitted row 768, whose [768, 1024) patch crossed into validation.
    assert all(row + 256 <= 819 or row >= 1229 for row, _ in train)
    assert not any(row == 768 for row, _ in train)


def test_split_too_small_for_full_patch_fails_explicitly():
    with pytest.raises(ValueError, match="Validation interval is too small"):
        build_split_patch_coordinates(
            image_height=1000,
            image_width=512,
            validation_division=[0.4, 0.5],
            patch_height=256,
            patch_width=256,
            overlap=32,
            mode="validation",
        )


def _dataset_kwargs(config):
    return {
        "nucleus_mask_path": config.data.nucleus_mask_path,
        "histology_path": config.data.histology_path,
        "matched_nuclei_path": config.data.matched_nuclei_path,
        "expression_path": config.data.expression_path,
        "cell_type_path": config.data.cell_type_path,
        "gene_names": config.data.genes,
        "cell_types": config.data.cell_types,
        "patch_height": config.data.patch_height,
        "patch_width": config.data.patch_width,
        "overlap": config.data.overlap,
        "max_cells_per_patch": config.data.max_cells_per_patch,
        "min_nucleus_area": config.data.min_nucleus_area,
        "expression_scale": config.data.expression_scale,
        "validation_division": config.data.validation_split,
        "use_cell_types": config.model.use_cell_types,
    }


def _inventory(config, mode):
    return inspect_patch_inventory(
        histology_path=config.data.histology_path,
        nucleus_mask_path=config.data.nucleus_mask_path,
        matched_nuclei_path=config.data.matched_nuclei_path,
        expression_path=config.data.expression_path,
        cell_type_path=config.data.cell_type_path,
        validation_division=config.data.validation_split,
        patch_height=config.data.patch_height,
        patch_width=config.data.patch_width,
        overlap=config.data.overlap,
        min_nucleus_area=config.data.min_nucleus_area,
        mode=mode,
        require_cell_types=config.model.use_cell_types,
    )


def test_dataset_and_preflight_use_identical_demo_coordinates(tmp_path):
    config = load_config(ROOT / "configs" / "demo.yaml")
    kwargs = _dataset_kwargs(config)
    training = HistCFMDataset(
        **kwargs,
        mode="train",
        stain_augmentation=False,
        normalization_path=None,
        normalization_output_dir=tmp_path,
    )
    validation = HistCFMDataset(
        **kwargs,
        mode="val",
        stain_augmentation=False,
        normalization_path=training.normalization_path,
        normalization_output_dir=None,
    )
    assert training.patch_keys == list(_inventory(config, "train").patch_keys)
    assert validation.patch_keys == list(_inventory(config, "validation").patch_keys)


def test_training_normalization_receives_only_training_coordinates(
    tmp_path, monkeypatch
):
    config = load_config(ROOT / "configs" / "demo.yaml")
    captured = []

    def fake_normalization(histology, patch_coordinates, patch_height, patch_width):
        del histology, patch_height, patch_width
        captured.extend(patch_coordinates)
        return np.ones((2, 3), dtype=np.float64)

    monkeypatch.setattr(
        dataset_module, "compute_histology_normalization", fake_normalization
    )
    HistCFMDataset(
        **_dataset_kwargs(config),
        mode="train",
        stain_augmentation=False,
        normalization_path=None,
        normalization_output_dir=tmp_path,
    )
    expected = build_split_patch_coordinates(
        image_height=768,
        image_width=768,
        validation_division=config.data.validation_split,
        patch_height=config.data.patch_height,
        patch_width=config.data.patch_width,
        overlap=config.data.overlap,
        mode="train",
    )
    assert tuple(captured) == expected
    start, stop = validation_pixel_interval(768, config.data.validation_split)
    assert all(row + config.data.patch_height <= start or row >= stop for row, _ in captured)
