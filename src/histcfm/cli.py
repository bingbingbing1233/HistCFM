"""Thin command-line wrappers for the formal HistCFM package."""

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="histcfm", description="HistCFM workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train HistCFM")
    train_parser.add_argument("--config", required=True, help="Formal YAML configuration")
    train_parser.add_argument("--output-dir", required=True, help="New training output directory")
    train_parser.set_defaults(handler=_run_train)

    infer_parser = subparsers.add_parser("infer", help="Run one checkpoint inference")
    infer_parser.add_argument("--config", required=True, help="Formal YAML configuration")
    infer_parser.add_argument("--checkpoint", required=True, help="Trusted formal checkpoint")
    infer_parser.add_argument("--output-dir", required=True, help="New inference output directory")
    infer_parser.add_argument(
        "--split",
        choices=("validation", "prediction"),
        default="validation",
        help="Whether supervised targets are available",
    )
    infer_parser.set_defaults(handler=_run_infer)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate aligned inference tables"
    )
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--targets", required=True)
    evaluate_parser.add_argument("--cell-types")
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--overwrite", action="store_true")
    evaluate_parser.set_defaults(handler=_run_evaluate)

    validate_parser = subparsers.add_parser(
        "validate-data", help="Read-only validation for training or inference data"
    )
    validate_parser.add_argument("--config", required=True, help="Formal YAML configuration")
    validate_parser.add_argument(
        "--mode", choices=("train", "infer"), required=True
    )
    validate_parser.add_argument("--checkpoint", help="Trusted formal inference checkpoint")
    validate_parser.add_argument(
        "--split", choices=("validation", "prediction"), default="validation"
    )
    validate_parser.set_defaults(handler=_run_validate_data)
    return parser


def _run_train(args: argparse.Namespace) -> int:
    from .train import train

    train(args.config, args.output_dir)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    from .inference import infer

    infer(args.config, args.checkpoint, args.output_dir, split=args.split)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate

    evaluate(
        args.predictions,
        args.targets,
        args.output_dir,
        cell_types_path=args.cell_types,
        overwrite=args.overwrite,
    )
    return 0


def _run_validate_data(args: argparse.Namespace) -> int:
    _validate_data_arguments(args)
    if args.mode == "train":
        from .train import validate_training_data

        validate_training_data(args.config)
        return 0
    from .inference import validate_inference_data

    validate_inference_data(args.config, args.checkpoint, split=args.split)
    return 0


def _validate_data_arguments(args: argparse.Namespace) -> None:
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("--checkpoint is not accepted with validate-data --mode train")
    if args.mode == "infer" and args.checkpoint is None:
        raise ValueError("validate-data --mode infer requires --checkpoint")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
