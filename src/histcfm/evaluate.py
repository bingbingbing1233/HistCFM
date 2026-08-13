"""Evaluate formal HistCFM inference tables without running the model.

This release-side evaluator was written on 2026-08-13. It is intentionally
independent of paper-specific epoch selection, fold aggregation, spatial
statistics, and plotting. Distributed under GNU GPL version 3 only; see
``LICENSE``.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def _read_expression_table(path: PathLike, label: str) -> Tuple[np.ndarray, list, list]:
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"{label} must be an existing regular file: {table_path}")
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{label} column names must be unique")
    table = pd.read_csv(table_path)
    if not len(table.columns) or table.columns[0] != "cell_id":
        raise ValueError(f"{label} first column must be 'cell_id'")
    if table.empty:
        raise ValueError(f"{label} must contain at least one cell row")
    if len(table.columns) < 2:
        raise ValueError(f"{label} must contain at least one gene column")
    if table["cell_id"].isna().any() or not table["cell_id"].is_unique:
        raise ValueError(f"{label} cell_id values must be non-null and unique")
    genes = table.columns[1:].tolist()
    if len(genes) != len(set(genes)):
        raise ValueError(f"{label} gene columns must be unique")
    try:
        values = table.iloc[:, 1:].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} gene values must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label} gene values must be finite")
    return values, table["cell_id"].tolist(), genes


def _pearson_by_gene(
    predictions: np.ndarray,
    targets: np.ndarray,
    genes: list,
) -> Tuple[list, list]:
    rows = []
    valid = []
    for index, gene in enumerate(genes):
        predicted = predictions[:, index]
        target = targets[:, index]
        predicted_constant = bool(np.ptp(predicted) == 0)
        target_constant = bool(np.ptp(target) == 0)
        if predicted_constant or target_constant:
            reasons = []
            if target_constant:
                reasons.append("zero_variance_target")
            if predicted_constant:
                reasons.append("zero_variance_prediction")
            pcc = None
            reason = ";".join(reasons)
        else:
            pcc = float(np.corrcoef(target, predicted)[0, 1])
            if not math.isfinite(pcc):
                pcc = None
                reason = "undefined_correlation"
            else:
                valid.append(pcc)
                reason = None
        rows.append(
            {
                "gene": gene,
                "pcc": pcc,
                "valid_pcc": pcc is not None,
                "invalid_reason": reason,
                "mse": float(np.mean((predicted - target) ** 2)),
                "mae": float(np.mean(np.abs(predicted - target))),
            }
        )
    return rows, valid


def _validate_cell_types(path: PathLike, expected_cell_ids: list) -> Tuple[list, list]:
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"cell_types must be an existing regular file: {table_path}")
    table = pd.read_csv(table_path)
    required = ["cell_id", "ground_truth_label", "predicted_label"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            "cell_types.csv cannot be evaluated without column(s): " + ", ".join(missing)
        )
    if table["cell_id"].isna().any() or not table["cell_id"].is_unique:
        raise ValueError("cell_types.csv cell_id values must be non-null and unique")
    if table["cell_id"].tolist() != expected_cell_ids:
        raise ValueError(
            "cell_types.csv cell IDs and order must exactly match expression tables"
        )
    if table[required[1:]].isna().any().any():
        raise ValueError("cell_types.csv labels must be non-null")
    return table["ground_truth_label"].astype(str).tolist(), table[
        "predicted_label"
    ].astype(str).tolist()


def _json_dump(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def evaluate(
    predictions_path: PathLike,
    targets_path: PathLike,
    output_dir: PathLike,
    cell_types_path: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Mapping[str, str]:
    """Compute general metrics from aligned `log1p(count)` inference tables."""

    predictions, prediction_ids, prediction_genes = _read_expression_table(
        predictions_path, "predictions"
    )
    targets, target_ids, target_genes = _read_expression_table(targets_path, "targets")
    if prediction_ids != target_ids:
        raise ValueError(
            "predictions and targets cell IDs and order must exactly match; no join is performed"
        )
    if prediction_genes != target_genes:
        raise ValueError(
            "predictions and targets gene names and order must exactly match"
        )
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets shapes must exactly match")

    per_gene, valid_pcc = _pearson_by_gene(predictions, targets, prediction_genes)
    errors = predictions - targets
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    metrics = {
        "expression_scale": "log1p(count)",
        "num_cells": int(predictions.shape[0]),
        "num_genes": int(predictions.shape[1]),
        "num_valid_pcc_genes": len(valid_pcc),
        "num_invalid_pcc_genes": int(predictions.shape[1] - len(valid_pcc)),
        "mean_gene_pcc": None if not valid_pcc else float(np.mean(valid_pcc)),
        "median_gene_pcc": None if not valid_pcc else float(np.median(valid_pcc)),
        "mse": mse,
        "mae": mae,
        "rmse": float(math.sqrt(mse)),
    }
    cell_metrics = None
    if cell_types_path is not None:
        ground_truth, predicted = _validate_cell_types(cell_types_path, prediction_ids)
        from sklearn.metrics import accuracy_score, f1_score

        cell_metrics = {
            "num_cells": len(ground_truth),
            "accuracy": float(accuracy_score(ground_truth, predicted)),
            "macro_f1": float(f1_score(ground_truth, predicted, average="macro")),
        }

    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing non-empty evaluation output_dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _json_dump(metrics, output / "metrics.json")
    pd.DataFrame(per_gene).to_csv(output / "per_gene_metrics.csv", index=False)
    written: Dict[str, str] = {
        "metrics": str(output / "metrics.json"),
        "per_gene_metrics": str(output / "per_gene_metrics.csv"),
    }

    if cell_metrics is not None:
        _json_dump(cell_metrics, output / "cell_type_metrics.json")
        written["cell_type_metrics"] = str(output / "cell_type_metrics.json")
    return written
