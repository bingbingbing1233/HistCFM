"""Strict inference metadata JSON contract."""

import json

import pytest

from histcfm.inference import _write_metadata_json


def _reject_constant(value):
    raise ValueError(f"non-standard constant: {value}")


def test_inference_metadata_is_standard_json(tmp_path):
    path = tmp_path / "metadata.json"
    payload = {"metric": 1.25, "optional": None, "values": [0.0, -2.0]}
    _write_metadata_json(payload, path)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text, parse_constant=_reject_constant) == payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_inference_metadata_rejects_nonfinite_values(tmp_path, value):
    with pytest.raises(ValueError, match="JSON compliant"):
        _write_metadata_json({"bad": value}, tmp_path / "metadata.json")
