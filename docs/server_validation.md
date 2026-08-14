# Public validation record

HistCFM `0.1.0` completed an end-to-end public synthetic validation on
2026-08-13 in an independently created `histcfm` environment. This page records
the validated environment, scope, expected outputs, and commands needed to
repeat that public software check.

## Validated environment

```text
Linux x86-64
Python 3.10.14
PyTorch 2.1.1
torchvision 0.16.1
PyTorch CUDA runtime 12.1
NumPy 1.26.4
NVIDIA GPU with a compatible driver
```

The environment import and CUDA preflight passed before testing. See
[environment.md](environment.md) for installation and the complete dependency
boundary.

## Validated scope

The release validation confirmed:

- package version `0.1.0` and an available `histcfm` CLI;
- all 75 automated tests;
- training-data preflight;
- six synthetic training patches and one training epoch;
- creation and strict reload of one schema-3 checkpoint carrying version
  `0.1.0`;
- inference-data preflight and four validation patches;
- validation inference for 36 cells and 24 genes;
- evaluation and all 14 expected workflow outputs;
- identical cell and gene order in predictions and targets; and
- standards-compliant JSON metadata and metrics without NaN or Infinity.

This is software-flow evidence for the committed synthetic demo. It is not a
paper benchmark, biological result, real-dataset validation, runtime guarantee,
or claim of bitwise reproducibility on every platform.

## Repeat the validation

Create and activate the environment as described in
[environment.md](environment.md), install HistCFM from the repository root,
and choose new or empty output directories. Runtime outputs must not be
committed.

```bash
python scripts/check_environment.py
python -m pytest

histcfm validate-data \
  --mode train \
  --config configs/demo.yaml

histcfm train \
  --config configs/demo.yaml \
  --output-dir runs/demo_train

histcfm validate-data \
  --mode infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint runs/demo_train/checkpoints/epoch_1.pth

histcfm infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint runs/demo_train/checkpoints/epoch_1.pth \
  --output-dir runs/demo_infer

histcfm evaluate \
  --predictions runs/demo_infer/predictions.csv \
  --targets runs/demo_infer/targets.csv \
  --cell-types runs/demo_infer/cell_types.csv \
  --output-dir runs/demo_metrics
```

With the committed one-epoch demo configuration, training writes
`checkpoints/epoch_1.pth`. Confirm its schema and package version with the
trusted local checkpoint loader:

```bash
python - <<'PY'
from histcfm import __version__
from histcfm.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint

checkpoint = load_checkpoint(
    "runs/demo_train/checkpoints/epoch_1.pth", map_location="cpu"
)
assert __version__ == "0.1.0"
assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 3
assert checkpoint["histcfm_version"] == __version__
print("checkpoint contract: PASSED")
PY
```

## Expected outputs

The complete workflow produces 14 files:

```text
runs/demo_train/artifacts/histology_normalization.npy
runs/demo_train/checkpoints/epoch_1.pth
runs/demo_train/logs/train.log
runs/demo_train/logs/train_losses.csv
runs/demo_train/resolved_config.yaml
runs/demo_infer/predictions.csv
runs/demo_infer/targets.csv
runs/demo_infer/cell_types.csv
runs/demo_infer/cells.csv
runs/demo_infer/metadata.json
runs/demo_infer/resolved_inference_config.yaml
runs/demo_metrics/metrics.json
runs/demo_metrics/per_gene_metrics.csv
runs/demo_metrics/cell_type_metrics.json
```

Predictions and targets must both contain 36 rows and the same ordered 24-gene
header for the committed demo. `metadata.json`, `metrics.json`, and
`cell_type_metrics.json` must parse as standard JSON without non-finite
constants. Generated checkpoints, normalization, logs, predictions, targets,
and metrics are local validation artifacts and are intentionally excluded from
the release repository.
