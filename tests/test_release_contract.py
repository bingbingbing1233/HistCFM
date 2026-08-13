import ast
from pathlib import Path

import pytest
import torch

from histcfm import __version__
from histcfm.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    __version__ as checkpoint_version,
    validate_checkpoint,
)


def test_package_and_checkpoint_share_one_version():
    assert checkpoint_version == __version__


def test_train_explicitly_passes_mode_paths_and_fusion_method():
    root = Path(__file__).parents[1] / "src" / "histcfm"
    tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    histcfm_call = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "HistCFM"
    )
    keywords = {item.arg: ast.unparse(item.value) for item in histcfm_call.keywords}
    assert keywords["uni_mode"] == "config.uni.mode"
    assert keywords["uni_index_path"] == "config.uni.index_path"
    assert keywords["uni_features_path"] == "config.uni.features_path"
    assert keywords["fusion_type"] == "config.uni.fusion_method"


def test_inference_binds_fusion_to_checkpoint_and_metadata():
    root = Path(__file__).parents[1] / "src" / "histcfm"
    source = (root / "inference.py").read_text(encoding="utf-8")
    assert '"uni.fusion_method"' in source
    assert 'metadata["fusion_method"] != trained.uni.fusion_method' in source


def test_checkpoint_rejects_missing_fusion_method():
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": 1,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "resolved_config": {
            name: {}
            for name in (
                "data",
                "model",
                "flow",
                "uni",
                "sonrm",
                "loss",
                "training",
                "runtime",
            )
        },
        "genes": ["gene"],
        "cell_type_mapping": {},
        "normalization_artifact": "artifacts/histology_normalization.npy",
        "seed": 1,
        "histcfm_version": __version__,
        "torch_version": str(torch.__version__),
        "cuda_version": None,
        "model_metadata": {
            "n_classes": 0,
            "n_genes": 1,
            "reference_profile_count": 0,
        },
    }
    with pytest.raises(CheckpointError, match="model_metadata has unexpected fields"):
        validate_checkpoint(payload)
