import ast
import re
from pathlib import Path

import yaml

from histcfm.cli import build_parser
from histcfm.config import load_config


ROOT = Path(__file__).parents[1]


def _conda_names(dependencies):
    names = set()
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.add(re.split(r"[<>=!~ ]", dependency, maxsplit=1)[0].lower())
    return names


def _absolute_config_paths(config):
    candidates = (
        config.data.histology_path,
        config.data.nucleus_mask_path,
        config.data.matched_nuclei_path,
        config.data.expression_path,
        config.data.cell_type_path,
        config.data.average_expression_path,
        config.data.normalization_path,
        config.uni.index_path,
        config.uni.features_path,
    )
    return [value for value in candidates if value and Path(value).is_absolute()]


def test_environment_name_channels_and_validated_stack():
    environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    assert environment["name"] == "histcfm"
    assert environment["channels"] == ["pytorch", "nvidia", "conda-forge"]
    dependencies = environment["dependencies"]
    assert "python=3.10" in dependencies
    assert "pytorch=2.1.1" in dependencies
    assert "torchvision=0.16.1" in dependencies
    assert "pytorch-cuda=12.1" in dependencies
    assert "numpy=1.26.4" in dependencies


def test_environment_and_pyproject_dependency_contract():
    environment = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    conda_names = _conda_names(environment["dependencies"])
    required_conda = {
        "python",
        "pip",
        "setuptools",
        "wheel",
        "pytorch",
        "torchvision",
        "pytorch-cuda",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "imageio",
        "tifffile",
        "natsort",
        "tqdm",
        "pillow",
        "pytest",
    }
    assert required_conda <= conda_names
    assert {"timm", "stainlib", "uni"}.isdisjoint(conda_names)

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for distribution in (
        "imageio",
        "natsort",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "tifffile",
        "torch",
        "torchvision",
        "tqdm",
    ):
        assert f'"{distribution}"' in pyproject
    assert 'stain = ["stainlib"]' in pyproject
    assert 'test = ["pytest"]' in pyproject
    assert 'histcfm = "histcfm.cli:main"' in pyproject
    assert '"timm"' not in pyproject


def test_public_demo_is_independent_of_online_feature_software():
    config = load_config(ROOT / "configs" / "demo.yaml")
    assert config.uni.enabled is True
    assert config.uni.mode == "precomputed"
    assert config.data.stain_augmentation is False
    assert not _absolute_config_paths(config)

    public_runtime_files = [
        ROOT / "src" / "histcfm" / "features" / "uni.py",
        ROOT / "src" / "histcfm" / "data" / "dataset.py",
        ROOT / "scripts" / "generate_synthetic_demo.py",
    ]
    imports = set()
    for path in public_runtime_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    assert "timm" not in imports
    dataset_tree = ast.parse(
        (ROOT / "src" / "histcfm" / "data" / "dataset.py").read_text(
            encoding="utf-8"
        )
    )
    top_level_imports = set()
    for node in dataset_tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "stainlib" not in top_level_imports
    generator = (ROOT / "scripts" / "generate_synthetic_demo.py").read_text(
        encoding="utf-8"
    )
    assert "stainlib" not in generator
    assert "checkpoint" not in generator.lower()


def test_environment_checker_has_no_install_download_or_data_access():
    path = ROOT / "scripts" / "check_environment.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert {"requests", "urllib", "subprocess", "socket"}.isdisjoint(imported)
    assert "PRIVATE_BC1" not in source
    assert "pip install" not in source
    assert "conda install" not in source


def test_public_environment_and_run_docs_match_cli_names():
    public_files = [
        ROOT / "README.md",
        ROOT / "examples" / "demo" / "README.md",
        ROOT / "docs" / "environment.md",
        ROOT / "docs" / "server_validation.md",
    ]
    assert not (ROOT / "docs" / "server_run_histcfm.md").exists()
    text = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
    assert "conda activate ghist" not in text
    assert "conda activate histcfm" in text
    for command in ("train", "infer", "evaluate", "validate-data"):
        assert f"histcfm {command}" in text

    parser = build_parser()
    subparser_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparser_action.choices) == {
        "train",
        "infer",
        "evaluate",
        "validate-data",
    }


def test_real_data_guides_preserve_external_boundaries():
    data_guide = (ROOT / "docs" / "data_preparation.md").read_text(
        encoding="utf-8"
    )
    uni_guide = (ROOT / "docs" / "uni_features.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://github.com/SydneyBioX/GHIST" in data_guide
    for upstream_path in (
        "tutorials/1_data_preprocessing.ipynb",
        "data_processing/1_get_xenium_nuclei_seg_image.py",
        "data_processing/2_get_xenium_cell_gene_matrix.py",
        "data_processing/3_segment_nuclei_he_image.py",
        "data_processing/4_get_corresponding_cells.py",
    ):
        assert upstream_path in data_guide
    for field in (
        "data.histology_path",
        "data.nucleus_mask_path",
        "data.matched_nuclei_path",
        "data.expression_path",
        "data.cell_type_path",
    ):
        assert field in data_guide

    assert "https://github.com/mahmoodlab/UNI" in uni_guide
    assert "https://huggingface.co/MahmoodLab/uni" in uni_guide
    assert "vit_large_patch16_224" in uni_guide
    assert "1024-dimensional output" in uni_guide
    assert "uni_index.json" in uni_guide
    assert "uni_features.npy" in uni_guide
    assert "not a claim of byte identity" in uni_guide

    assert "Preparing real data" in readme
    assert "Quick smoke test" in readme
    assert "does not vendor the GHIST preprocessing workflow" in readme
    assert "does not redistribute UNI source code or model weights" in readme


def test_default_tests_do_not_run_complete_training():
    for path in (ROOT / "tests").glob("test_*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                assert name != "train", f"default pytest invokes full training in {path}"
