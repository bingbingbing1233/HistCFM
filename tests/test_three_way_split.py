from pathlib import Path

import numpy as np
import pytest

from histcfm.config import load_config, DataConfig
from histcfm.data.splitting import (
    build_split_patch_coordinates, partition_cell_ids, split_row_intervals,
)
from histcfm.data.dataset import HistCFMDataset, compute_histology_normalization
from histcfm.inference import _prepare_output


ROOT = Path(__file__).parents[1]


def test_three_strips_are_disjoint_and_cover_every_row():
    assignments = []
    for mode in ('train', 'val', 'test'):
        intervals = split_row_intervals(768, [1/3, 2/3], mode, [2/3, 1])
        rows = {i for lo, hi in intervals for i in range(lo, hi)}
        assignments.append(rows)
        coordinates = build_split_patch_coordinates(
            image_height=768, image_width=768, validation_division=[1/3, 2/3],
            test_division=[2/3, 1], patch_height=256, patch_width=256,
            overlap=32, mode=mode,
        )
        assert all(set(range(y, y + 256)) <= rows for y, x in coordinates)
    assert set.union(*assignments) == set(range(768))
    assert sum(map(len, assignments)) == 768


def test_centroid_assigns_crossing_nucleus_to_only_one_strip():
    mask = np.zeros((12, 4), dtype=int)
    mask[3:6, 1] = 7  # centroid exactly 4 -> second strip
    mask[7:10, 2] = 91  # centroid exactly 8 -> third strip
    ids = [partition_cell_ids(mask, [(i, i+4)]) for i in (0, 4, 8)]
    assert ids == [set(), {7}, {91}]


def test_overlapping_or_missing_test_intervals_fail():
    with pytest.raises(ValueError, match='overlap'):
        split_row_intervals(768, [1/3, 2/3], 'train', [0.5, 1])
    with pytest.raises(ValueError, match='requires'):
        split_row_intervals(768, [0, 1/3], 'test')
    with pytest.raises(ValueError, match='full-size patch'):
        build_split_patch_coordinates(
            image_height=768, image_width=768, validation_division=[0, 1/3],
            test_division=[0.9, 1], patch_height=256, patch_width=256,
            overlap=32, mode='test',
        )


def test_train_complement_does_not_bridge_held_out_regions():
    assert split_row_intervals(1000, [400/1000, 500/1000], 'train', [0, .2]) == (
        (200, 400), (500, 1000),
    )


def test_real_demo_cell_sets_and_train_only_normalization(tmp_path):
    cfg = load_config(ROOT / 'configs/demo.yaml')
    d = cfg.data
    common = dict(
        nucleus_mask_path=d.nucleus_mask_path, histology_path=d.histology_path,
        matched_nuclei_path=d.matched_nuclei_path, expression_path=d.expression_path,
        cell_type_path=d.cell_type_path, gene_names=d.genes, cell_types=d.cell_types,
        patch_height=d.patch_height, patch_width=d.patch_width, overlap=d.overlap,
        max_cells_per_patch=d.max_cells_per_patch, min_nucleus_area=d.min_nucleus_area,
        expression_scale=d.expression_scale, validation_division=d.validation_split,
        test_division=d.test_split,
    )
    train = HistCFMDataset(**common, normalization_output_dir=tmp_path / 'artifacts')
    val = HistCFMDataset(**common, mode='val', normalization_path=train.normalization_path)
    test = HistCFMDataset(**common, mode='test', normalization_path=train.normalization_path)
    groups = [set(x.all_intersect) for x in (train, val, test)]
    assert set.union(*groups) == set(range(1, 97))
    assert sum(map(len, groups)) == 96
    assert np.array_equal(train.norms_hist, val.norms_hist)
    assert np.array_equal(train.norms_hist, test.norms_hist)
    changed = train.hist.copy()
    changed[256:] = 0
    recomputed = compute_histology_normalization(changed, train.coords_starts, 256, 256)
    assert np.array_equal(recomputed, train.norms_hist)


def test_internal_validation_cannot_overwrite_checkpoints(tmp_path):
    cfg = load_config(ROOT / 'configs/demo.yaml')
    checkpoint = tmp_path / 'run/checkpoints/last.pth'
    for destination in (checkpoint.parent, checkpoint.parent.parent):
        with pytest.raises(ValueError, match='outside'):
            _prepare_output(destination, cfg, checkpoint, training_validation=True)
    with pytest.raises(ValueError, match='outside'):
        _prepare_output(tmp_path / 'run/validation/epoch_1', cfg, checkpoint)


def test_config_rejects_reversed_test_interval():
    import dataclasses
    cfg = load_config(ROOT / 'configs/demo.yaml')
    data = dataclasses.asdict(cfg.data)
    data['test_split'] = [1, 2/3]
    with pytest.raises(ValueError, match='increasing'):
        DataConfig.from_mapping(data)


def test_legacy_checkpoint_cannot_be_relabelled_as_three_way():
    import copy
    from histcfm.checkpoint import CheckpointError
    from histcfm.inference import _compare_checkpoint_config
    runtime = load_config(ROOT / 'configs/demo.yaml')
    trained = copy.deepcopy(runtime)
    trained.data.test_split = None
    checkpoint = dict(
        genes=trained.data.genes,
        cell_type_mapping={c: i for i, c in enumerate(trained.data.cell_types)},
        model_metadata=dict(n_classes=len(trained.data.cell_types),
                            n_genes=len(trained.data.genes),
                            fusion_method=trained.uni.fusion_method),
    )
    with pytest.raises(CheckpointError, match='spatial partitions'):
        _compare_checkpoint_config(runtime, trained, checkpoint)
