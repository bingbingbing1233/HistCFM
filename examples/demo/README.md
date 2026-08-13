# Optional synthetic end-to-end smoke test

This optional committed demo is entirely synthetic and exercises the same `histcfm
train`, `histcfm infer`, and `histcfm evaluate` commands used for prepared real
inputs. There is no alternate demo model or training script.

It contains 96 generated cells, 24 synthetic genes, four synthetic cell types,
six training patches, four validation patches, and ten rows of synthetic
precomputed morphology features. The feature matrix matches the formal
1024-dimensional interface but is not an output of UNI or any other model.
See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and `checksums.sha256`.

The smoke test validates software plumbing only. It contains no patient data,
cannot support biological interpretation, does not reproduce paper metrics,
and must not be used to assess model quality. On 2026-08-13, the independent
server environment passed all 75 tests and the full synthetic workflow: six
training patches, four validation patches, checkpoint save/strict reload,
inference for 36 cells and 24 genes, evaluation, strict JSON, and output-order
checks. That run preceded the `0.1.0` metadata update, so it must be repeated
before tagging. No runtime estimate or example metric is claimed.

## Run from the repository root

Use the independent environment defined at the repository root. The committed
feature matrix is already prepared and synthetic; neither UNI, `timm`, a UNI
checkpoint, nor `stainlib` is needed:

```bash
conda env create -f environment.yml
conda activate histcfm
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_environment.py
```

Output directories below must not already contain files.

```bash
pytest

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

Training creates, rather than consumes, these artifacts:

```text
runs/demo_train/artifacts/histology_normalization.npy
runs/demo_train/checkpoints/epoch_1.pth
runs/demo_train/logs/train.log
runs/demo_train/logs/train_losses.csv
runs/demo_train/resolved_config.yaml
```

Inference writes `predictions.csv`, `targets.csv`, `cell_types.csv`,
`cells.csv`, `metadata.json`, and `resolved_inference_config.yaml` under
`runs/demo_infer/`. Evaluation writes `metrics.json`,
`per_gene_metrics.csv`, and `cell_type_metrics.json` under
`runs/demo_metrics/`.

These commands generate checkpoints, predictions, and metrics locally. None of
those runtime outputs is committed as demo input or presented as a paper
result.
