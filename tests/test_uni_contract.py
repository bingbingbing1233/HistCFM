import json
from pathlib import Path

import numpy as np
import pytest

from histcfm.data.validation import (
    UNI_MIN_L2_NORM,
    load_uni_feature_store,
    load_uni_index,
    validate_uni_feature_store,
)
from histcfm.features.uni import UniFeatureProvider


KEYS = ["slide|0|0|16|16|0", "slide|0|16|16|16|0"]


def _write_index(path, mapping):
    path.write_text(json.dumps(mapping), encoding="utf-8")


def _write_store(tmp_path, mapping=None, features=None):
    index_path = tmp_path / "uni_index.json"
    features_path = tmp_path / "uni_features.npy"
    _write_index(index_path, mapping or {KEYS[0]: 0, KEYS[1]: 1})
    if features is None:
        features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(features_path, features)
    return index_path, features_path


def test_json_index_and_runtime_use_the_same_mapping(tmp_path):
    index_path, features_path = _write_store(tmp_path)
    parsed = load_uni_index(index_path)
    provider = UniFeatureProvider(
        enable=True,
        mode="precomputed",
        index_path=index_path,
        features_path=features_path,
        uni_dim=2,
    )
    assert parsed == {KEYS[0]: 0, KEYS[1]: 1}
    assert provider.index_map == parsed


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({KEYS[0]: "0"}, "non-negative integers"),
        ({KEYS[0]: 0, KEYS[1]: 0}, "unique"),
        ({KEYS[0]: 0, KEYS[1]: 2}, "continuously cover"),
    ],
)
def test_json_index_rejects_invalid_rows(tmp_path, mapping, message):
    index_path = tmp_path / "uni_index.json"
    _write_index(index_path, mapping)
    with pytest.raises(ValueError, match=message):
        load_uni_index(index_path)


def test_json_index_rejects_duplicate_keys(tmp_path):
    index_path = tmp_path / "uni_index.json"
    index_path.write_text(
        '{"slide|0|0|16|16|0": 0, "slide|0|0|16|16|0": 1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_uni_index(index_path)


def test_feature_row_count_must_match_index(tmp_path):
    index_path, features_path = _write_store(
        tmp_path, features=np.array([[1.0, 0.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="row counts must match"):
        load_uni_feature_store(index_path, features_path, expected_dim=2)


def test_feature_dimension_is_exact(tmp_path):
    index_path, features_path = _write_store(
        tmp_path, features=np.ones((2, 3), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="shape"):
        load_uni_feature_store(index_path, features_path, expected_dim=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_features_are_rejected(tmp_path, bad_value):
    features = np.array([[1.0, 0.0], [bad_value, 1.0]], dtype=np.float32)
    index_path, features_path = _write_store(tmp_path, features=features)
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_uni_feature_store(index_path, features_path, expected_dim=2)


def test_zero_norm_features_are_rejected(tmp_path):
    features = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    index_path, features_path = _write_store(tmp_path, features=features)
    with pytest.raises(ValueError, match="L2 norm"):
        load_uni_feature_store(index_path, features_path, expected_dim=2)
    assert UNI_MIN_L2_NORM == 1e-12


def test_missing_patch_key_is_rejected_by_preflight(tmp_path):
    index_path, features_path = _write_store(tmp_path)
    with pytest.raises(ValueError, match="missing 1 patch keys"):
        validate_uni_feature_store(
            KEYS + ["slide|16|0|16|16|0"],
            index_path,
            features_path,
            expected_dim=2,
        )


def test_formal_uni_path_never_enables_pickle():
    release_root = Path(__file__).parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in release_root.joinpath("src").rglob("*.py")
    )
    assert "allow_pickle" + "=True" not in source
