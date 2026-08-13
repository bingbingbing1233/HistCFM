import ast
from pathlib import Path

import pytest

from histcfm.cli import _validate_data_arguments, build_parser


def _subcommands(parser):
    action = next(
        action
        for action in parser._actions
        if isinstance(action, __import__("argparse")._SubParsersAction)
    )
    return action.choices


def test_cli_exposes_only_formal_commands():
    commands = _subcommands(build_parser())
    assert set(commands) == {"train", "infer", "evaluate", "validate-data"}


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["train", "--config", "config.yaml", "--output-dir", "run"], "train"),
        (
            [
                "infer",
                "--config",
                "config.yaml",
                "--checkpoint",
                "model.pth",
                "--output-dir",
                "prediction",
            ],
            "infer",
        ),
        (
            [
                "evaluate",
                "--predictions",
                "predictions.csv",
                "--targets",
                "targets.csv",
                "--output-dir",
                "metrics",
            ],
            "evaluate",
        ),
        (
            [
                "validate-data",
                "--config",
                "config.yaml",
                "--mode",
                "train",
            ],
            "validate-data",
        ),
    ],
)
def test_cli_argument_contract(argv, expected):
    arguments = build_parser().parse_args(argv)
    assert arguments.command == expected
    assert callable(arguments.handler)


def test_cli_module_has_no_heavy_top_level_imports():
    source_path = Path(__file__).parents[1] / "src" / "histcfm" / "cli.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module)
    assert set(top_level_imports) == {"argparse", "typing"}


@pytest.mark.parametrize("command", ["train", "infer", "evaluate", "validate-data"])
def test_every_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, "--help"])
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_train_data_validation_does_not_require_normalization_or_checkpoint():
    arguments = build_parser().parse_args(
        ["validate-data", "--mode", "train", "--config", "config.yaml"]
    )
    _validate_data_arguments(arguments)
    assert arguments.checkpoint is None
    assert not hasattr(arguments, "normalization")


def test_inference_data_validation_requires_checkpoint():
    arguments = build_parser().parse_args(
        ["validate-data", "--mode", "infer", "--config", "config.yaml"]
    )
    with pytest.raises(ValueError, match="requires --checkpoint"):
        _validate_data_arguments(arguments)
