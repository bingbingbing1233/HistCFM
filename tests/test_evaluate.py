import json

import numpy as np
import pandas as pd
import pytest

from histcfm.evaluate import evaluate


def _write_table(path, cell_ids, genes, values):
    table = pd.DataFrame(values, columns=genes)
    table.insert(0, "cell_id", cell_ids)
    table.to_csv(path, index=False)


def test_evaluate_known_matrix_and_zero_variance_gene(tmp_path):
    cell_ids = ["cell-1", "cell-2", "cell-3"]
    genes = ["gene_positive", "gene_negative", "gene_constant"]
    targets = np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0], [3.0, 3.0, 2.0]])
    predictions = np.array(
        [[1.0, 3.0, 1.0], [2.0, 2.0, 2.0], [3.0, 1.0, 3.0]]
    )
    predictions_path = tmp_path / "predictions.csv"
    targets_path = tmp_path / "targets.csv"
    output_dir = tmp_path / "metrics"
    _write_table(predictions_path, cell_ids, genes, predictions)
    _write_table(targets_path, cell_ids, genes, targets)

    written = evaluate(predictions_path, targets_path, output_dir)

    metrics_text = (output_dir / "metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)
    expected_errors = predictions - targets
    expected_mse = float(np.mean(expected_errors**2))
    assert metrics["expression_scale"] == "log1p(count)"
    assert metrics["num_valid_pcc_genes"] == 2
    assert metrics["num_invalid_pcc_genes"] == 1
    assert metrics["mean_gene_pcc"] == pytest.approx(0.0)
    assert metrics["median_gene_pcc"] == pytest.approx(0.0)
    assert metrics["mse"] == pytest.approx(expected_mse)
    assert metrics["mae"] == pytest.approx(float(np.mean(np.abs(expected_errors))))
    assert metrics["rmse"] == pytest.approx(float(np.sqrt(expected_mse)))
    assert "NaN" not in metrics_text

    per_gene = pd.read_csv(output_dir / "per_gene_metrics.csv")
    constant = per_gene.loc[per_gene["gene"] == "gene_constant"].iloc[0]
    assert pd.isna(constant["pcc"])
    assert not bool(constant["valid_pcc"])
    assert constant["invalid_reason"] == "zero_variance_target"
    assert set(written) == {"metrics", "per_gene_metrics"}


@pytest.mark.parametrize("mismatch", ["cells", "genes"])
def test_evaluate_rejects_order_mismatch(tmp_path, mismatch):
    predictions_path = tmp_path / "predictions.csv"
    targets_path = tmp_path / "targets.csv"
    _write_table(predictions_path, ["a", "b"], ["g1", "g2"], [[1, 2], [2, 3]])
    if mismatch == "cells":
        _write_table(targets_path, ["b", "a"], ["g1", "g2"], [[2, 3], [1, 2]])
        expected = "cell IDs and order"
    else:
        _write_table(targets_path, ["a", "b"], ["g2", "g1"], [[2, 1], [3, 2]])
        expected = "gene names and order"

    with pytest.raises(ValueError, match=expected):
        evaluate(predictions_path, targets_path, tmp_path / "metrics")


def test_evaluate_cell_type_metrics_and_json_are_finite(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    targets_path = tmp_path / "targets.csv"
    cell_types_path = tmp_path / "cell_types.csv"
    output_dir = tmp_path / "metrics"
    _write_table(predictions_path, ["a", "b"], ["g1"], [[1], [2]])
    _write_table(targets_path, ["a", "b"], ["g1"], [[1], [2]])
    pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "ground_truth_label": ["x", "y"],
            "predicted_label": ["x", "x"],
        }
    ).to_csv(cell_types_path, index=False)

    evaluate(
        predictions_path,
        targets_path,
        output_dir,
        cell_types_path=cell_types_path,
    )

    text = (output_dir / "cell_type_metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(text)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert "NaN" not in text
