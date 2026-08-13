import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from histcfm.config import load_config
from histcfm.data.validation import (
    inspect_patch_inventory,
    load_uni_feature_store,
    validate_uni_feature_store,
)
from histcfm.data.image_io import load_image


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "demo.yaml"
DEMO_ROOT = ROOT / "examples" / "demo"
DATA_ROOT = DEMO_ROOT / "data"
CORE_FILES = (
    DATA_ROOT / "histology.tif",
    DATA_ROOT / "nucleus_mask.tif",
    DATA_ROOT / "matched_nuclei.csv",
    DATA_ROOT / "expression.csv",
    DATA_ROOT / "cell_types.csv",
    DATA_ROOT / "uni" / "uni_index.json",
    DATA_ROOT / "uni" / "uni_features.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventories(config):
    common = {
        "histology_path": config.data.histology_path,
        "nucleus_mask_path": config.data.nucleus_mask_path,
        "matched_nuclei_path": config.data.matched_nuclei_path,
        "expression_path": config.data.expression_path,
        "cell_type_path": config.data.cell_type_path,
        "validation_division": config.data.validation_split,
        "patch_height": config.data.patch_height,
        "patch_width": config.data.patch_width,
        "overlap": config.data.overlap,
        "min_nucleus_area": config.data.min_nucleus_area,
        "require_cell_types": config.model.use_cell_types,
    }
    return (
        inspect_patch_inventory(**common, mode="train"),
        inspect_patch_inventory(**common, mode="validation"),
    )


def test_demo_configuration_is_strict_and_minimal():
    config = load_config(CONFIG_PATH)
    assert config.training.epochs == 1
    assert config.training.workers == 0
    assert config.model.average_expression_compatibility is False
    assert config.data.average_expression_path is None
    assert config.data.normalization_path is None
    assert config.uni.enabled is True
    assert config.uni.mode == "precomputed"


def test_all_committed_demo_inputs_exist():
    assert all(path.is_file() and not path.is_symlink() for path in CORE_FILES)
    assert not (DATA_ROOT / "average_expression.csv").exists()


def test_mask_matched_expression_and_cell_type_ids_are_identical():
    mask = load_image(DATA_ROOT / "nucleus_mask.tif")
    mask_ids = set(np.unique(mask).astype(np.int64).tolist())
    mask_ids.discard(0)
    matched = pd.read_csv(DATA_ROOT / "matched_nuclei.csv", index_col=0)
    expression = pd.read_csv(DATA_ROOT / "expression.csv", index_col=0)
    cell_types = pd.read_csv(DATA_ROOT / "cell_types.csv")
    expected = set(range(1, 97))
    assert mask_ids == set(matched.index.astype(int))
    assert mask_ids == set(expression.index.astype(int))
    assert mask_ids == set(cell_types["c_id"].astype(int))
    assert mask_ids == expected


def test_expression_gene_and_cell_type_order_matches_configuration():
    config = load_config(CONFIG_PATH)
    expression = pd.read_csv(DATA_ROOT / "expression.csv")
    cell_types = pd.read_csv(DATA_ROOT / "cell_types.csv")
    assert expression.columns[0] == "cell_id"
    assert expression.columns[1:].tolist() == config.data.genes
    assert list(dict.fromkeys(cell_types["ct"].tolist())) == config.data.cell_types


def test_patch_inventory_and_feature_store_have_exact_coverage():
    config = load_config(CONFIG_PATH)
    train, validation = _inventories(config)
    assert len(train.patch_keys) == 6
    assert len(validation.patch_keys) == 4
    assert train.largest_cell_count <= config.data.max_cells_per_patch
    assert validation.largest_cell_count <= config.data.max_cells_per_patch
    keys = list(train.patch_keys) + list(validation.patch_keys)
    with (DATA_ROOT / "uni" / "uni_index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    assert list(index) == keys
    assert list(index.values()) == list(range(10))
    assert validate_uni_feature_store(
        keys,
        config.uni.index_path,
        config.uni.features_path,
        expected_dim=1024,
    ) == (10, 1024)


def test_synthetic_features_are_finite_nonzero_and_unique():
    index, features = load_uni_feature_store(
        DATA_ROOT / "uni" / "uni_index.json",
        DATA_ROOT / "uni" / "uni_features.npy",
        expected_dim=1024,
    )
    values = np.asarray(features, dtype=np.float32)
    assert len(index) == 10
    assert features.shape == (10, 1024)
    assert features.dtype == np.dtype("float16")
    assert np.isfinite(values).all()
    assert np.all(np.linalg.norm(values, axis=1) > 1e-12)
    assert np.unique(features, axis=0).shape[0] == features.shape[0]


def test_demo_data_contains_no_private_paths_or_source_markers():
    forbidden = (
        b"candidate_01",
        b"BreastCancer",
        b"Breast_cancer",
    )
    for path in CORE_FILES:
        content = path.read_bytes()
        assert all(marker not in content for marker in forbidden), path
        assert re.search(rb"/(?:mnt|root|home)/", content) is None, path
        assert re.search(rb"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", content) is None, path
    generator = (ROOT / "scripts" / "generate_synthetic_demo.py").read_text(
        encoding="utf-8"
    )
    assert "PRIVATE_BC1_STAGING" not in generator
    assert "MahmoodLab" not in generator


def test_optional_private_reference_has_no_shared_hash_or_cell_id():
    """Enable author-side leakage checking without committing private identifiers."""

    raw_reference = os.environ.get("HISTCFM_PRIVATE_REFERENCE_DIR")
    if not raw_reference:
        return
    private_root = Path(raw_reference)
    assert private_root.is_dir()
    private_hashes = {
        _sha256(path)
        for path in private_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert private_hashes.isdisjoint({_sha256(path) for path in CORE_FILES})

    private_ids = set()
    for name, column in (
        ("matched_nuclei.csv", 0),
        ("expression.csv", 0),
        ("cell_types.csv", "c_id"),
    ):
        with (private_root / name).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            key = reader.fieldnames[column] if isinstance(column, int) else column
            private_ids.update(int(row[key]) for row in reader)
    assert private_ids.isdisjoint(set(range(1, 97)))


def test_demo_checksums_cover_all_public_demo_artifacts():
    checksum_path = DEMO_ROOT / "checksums.sha256"
    entries = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert relative not in entries
        entries[relative] = digest
        assert _sha256(DEMO_ROOT / relative) == digest
    expected = {
        "README.md",
        "DATA_PROVENANCE.md",
        "../../configs/demo.yaml",
        "../../scripts/generate_synthetic_demo.py",
        *(path.relative_to(DEMO_ROOT).as_posix() for path in CORE_FILES),
    }
    assert set(entries) == expected
