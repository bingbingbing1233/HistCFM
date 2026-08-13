"""Contract tests for supervised cell-type labels."""

from types import SimpleNamespace

import pandas as pd
import pytest

from histcfm.inference import _validate_cell_type_values as validate_inference_labels
from histcfm.train import _validate_cell_type_values as validate_training_labels


@pytest.mark.parametrize(
    "validator", (validate_training_labels, validate_inference_labels)
)
def test_supervised_cell_type_labels_reject_nulls(tmp_path, validator):
    table_path = tmp_path / "cell_types.csv"
    pd.DataFrame(
        {"c_id": [1, 2], "ct": ["synthetic_type_a", None]}
    ).to_csv(table_path, index=False)
    config = SimpleNamespace(
        model=SimpleNamespace(use_cell_types=True),
        data=SimpleNamespace(
            cell_type_path=str(table_path),
            cell_types=["synthetic_type_a", "synthetic_type_b"],
        ),
    )

    with pytest.raises(ValueError, match="must be non-null"):
        validator(config)
