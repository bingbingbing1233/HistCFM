import pytest

from histcfm.config import HistCFMConfig


def _minimal_config():
    return {
        "data": {
            "histology_path": "inputs/histology.tif",
            "nucleus_mask_path": "inputs/nucleus_mask.tif",
            "matched_nuclei_path": "inputs/matched_nuclei.csv",
            "expression_path": "inputs/expression.csv",
            "genes": ["gene-1"],
        },
        "model": {
            "use_cell_types": False,
            "use_neighborhood": False,
        },
        "flow": {
            "num_layers": 2,
            "inference_steps": 2,
        },
        "uni": {
            "enabled": False,
        },
        "sonrm": {},
        "loss": {},
        "training": {},
        "runtime": {},
    }


def test_config_rejects_unknown_fields():
    raw = _minimal_config()
    raw["model"]["unexpected_field"] = 1
    with pytest.raises(ValueError, match="Unknown model field"):
        HistCFMConfig.from_mapping(raw)


def test_minimal_known_config_is_accepted():
    config = HistCFMConfig.from_mapping(_minimal_config())
    assert config.flow.num_layers == 2
    assert config.flow.inference_steps == 2


def test_uni_accepts_only_precomputed_mode():
    raw = _minimal_config()
    raw["uni"]["mode"] = "online"
    with pytest.raises(ValueError, match="precomputed"):
        HistCFMConfig.from_mapping(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_float_with_field_name(value):
    raw = _minimal_config()
    raw["flow"]["train_noise_std"] = value
    with pytest.raises(ValueError, match="flow.train_noise_std must be finite"):
        HistCFMConfig.from_mapping(raw)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_config_rejects_string_nonfinite_values(value):
    raw = _minimal_config()
    raw["flow"]["train_noise_std"] = value
    with pytest.raises(TypeError, match="flow.train_noise_std must be numeric"):
        HistCFMConfig.from_mapping(raw)


def test_config_accepts_normal_finite_float():
    raw = _minimal_config()
    raw["flow"]["train_noise_std"] = 0.125
    assert HistCFMConfig.from_mapping(raw).flow.train_noise_std == 0.125
