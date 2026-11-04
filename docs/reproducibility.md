# Reproducibility

Use `environment.yml` for the recommended environment. Set `training.seed`
and `runtime.deterministic` in the YAML configuration. Exact numerical results
can still vary across hardware and dependency versions.

## Spatial demo

The included 768 × 768 synthetic image uses three half-open row intervals:
Train [0, 256), Val [256, 512), Test [512, 768). Cell nuclei are assigned by their
whole-mask row centroid. Patches do not cross strip boundaries, and normalization
statistics are fitted on Train only. This is a 1:1:1 demonstration, not the formal
7:1:2 experiment protocol or a benchmark of biological accuracy.

Training evaluates Val after every epoch with a fixed validation seed, saves
`best_val.pth` using the maximum finite mean gene PCC, and keeps the earlier epoch
on ties. Validation restores training random-number states after evaluation.
Test is scored separately after selection. Held-out labels are never model inputs.

## Run outputs

Keep `resolved_config.yaml`, `selection.json`, `logs/validation_metrics.csv`,
`checkpoints/` and `artifacts/histology_normalization.npy` together. Inference
reuses the training normalization and checks checkpoint/configuration consistency.
These files are generated in the requested output directory; do not upload them
as demo inputs.

## Tests and synthetic inputs

Run `python -m pytest tests` to check the package. The demo images, labels and
morphology features are generated with fixed seed 20260813 by
`scripts/generate_synthetic_demo.py`; no real data or UNI encoder is used.
See the [demo guide](../examples/demo/README.md) for the end-to-end commands and
[data provenance](../examples/demo/DATA_PROVENANCE.md) for synthetic-data details.
