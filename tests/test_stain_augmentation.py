"""Optional stain augmentation must never be silently disabled."""

import builtins

import pytest

from histcfm.config import load_config
from histcfm.data import dataset as dataset_module


def test_public_demo_keeps_stain_augmentation_disabled():
    config = load_config("configs/demo.yaml")
    assert config.data.stain_augmentation is False


def test_disabled_augmentation_does_not_touch_stainlib(monkeypatch):
    monkeypatch.setattr(
        dataset_module,
        "_create_stain_augmenter",
        lambda: (_ for _ in ()).throw(AssertionError("stainlib path was touched")),
    )
    assert dataset_module._optional_stain_augmenter(False) is None


def test_missing_stainlib_fails_with_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def import_without_stainlib(name, *args, **kwargs):
        if name.startswith("stainlib"):
            raise ImportError("stainlib deliberately unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_stainlib)
    with pytest.raises(RuntimeError, match="stain.*extra"):
        dataset_module.validate_stain_augmentation_dependency()


def test_stainlib_initialization_error_is_not_silenced(monkeypatch):
    class BrokenAugmenter:
        def __init__(self):
            raise ValueError("initialization failed")

    monkeypatch.setattr(
        dataset_module,
        "validate_stain_augmentation_dependency",
        lambda: BrokenAugmenter,
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        dataset_module._create_stain_augmenter()
