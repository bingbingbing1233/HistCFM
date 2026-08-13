# Server validation record and sequence

On 2026-08-13, HistCFM `0.1.0` passed preflight in the independent `histcfm`
environment, all 75 tests, train and inference data validation, one-epoch
synthetic training, strict schema-3 checkpoint reload, validation inference,
evaluation, strict JSON parsing, and cell/gene ordering checks. The run used
six training patches and four validation patches; inference produced 36 cells
by 24 genes. The checkpoint and inference metadata both recorded version
`0.1.0`, and all 14 expected output files existed.

This is a software-flow result only. It is not a private-BC1 validation, paper
benchmark rerun, biological result, or model-performance claim.

This document lists commands only; it contains no server address, account,
private source path, or environment-installation command. Run each section in
an already prepared environment. Output directories must be new or empty.

Use the independent `histcfm` Conda environment. The authoritative environment
creation, one-shot validation, checkpoint discovery/schema check, resume rules,
and log return list are in [server_run_histcfm.md](server_run_histcfm.md).

## Repeating the synthetic end-to-end validation

Run from the release repository root:

```bash
pytest

histcfm validate-data --mode train --config configs/demo.yaml

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

Expected paths follow directly from the formal implementation:

```text
runs/demo_train/artifacts/histology_normalization.npy
runs/demo_train/checkpoints/epoch_1.pth
runs/demo_infer/predictions.csv
runs/demo_infer/targets.csv
runs/demo_infer/cell_types.csv
runs/demo_metrics/metrics.json
runs/demo_metrics/per_gene_metrics.csv
runs/demo_metrics/cell_type_metrics.json
```

Do not commit any file under `runs/`.

## Optional private real-data technical validation (not a release gate)

The real-data configuration and instructions live outside this public
repository. They use the same CLI and differ only in input/configuration paths.
Substitute local paths; run the data commands from the private staging
directory because its YAML paths are staging-relative.

```bash
RELEASE_REPO='<RELEASE_REPO>'
PRIVATE_STAGING_DIR='<PRIVATE_STAGING_DIR>'
PRIVATE_RUN_ROOT='<PRIVATE_RUN_ROOT>'

cd "$RELEASE_REPO"
pytest

cd "$PRIVATE_STAGING_DIR"

histcfm validate-data --mode train --config demo_private_bc1.yaml

histcfm train \
  --config demo_private_bc1.yaml \
  --output-dir "$PRIVATE_RUN_ROOT/train"

histcfm validate-data \
  --mode infer \
  --split validation \
  --config demo_private_bc1.yaml \
  --checkpoint "$PRIVATE_RUN_ROOT/train/checkpoints/epoch_1.pth"

histcfm infer \
  --split validation \
  --config demo_private_bc1.yaml \
  --checkpoint "$PRIVATE_RUN_ROOT/train/checkpoints/epoch_1.pth" \
  --output-dir "$PRIVATE_RUN_ROOT/infer"

histcfm evaluate \
  --predictions "$PRIVATE_RUN_ROOT/infer/predictions.csv" \
  --targets "$PRIVATE_RUN_ROOT/infer/targets.csv" \
  --cell-types "$PRIVATE_RUN_ROOT/infer/cell_types.csv" \
  --output-dir "$PRIVATE_RUN_ROOT/metrics"
```

Private inputs, precomputed real features, configurations, checkpoints,
normalization, predictions, metrics, and logs must remain outside the public
repository. This technical run does not reproduce or select paper benchmark
results.
