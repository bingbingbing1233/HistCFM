# Independent HistCFM server runbook

This runbook contains placeholders, not a server address, account name, or
private absolute path. Substitute paths in the shell only; do not write them
back into the repository. Run on a Linux x86-64 NVIDIA server from the release
repository root.

The public synthetic workflow is the release runtime gate. Private real-data
validation is optional follow-up and is not required before submission.
Neither workflow modifies or clones a historical Conda environment.

## A. One-shot environment and public synthetic validation

Choose a new validation root outside the repository. The block stops if either
the `histcfm` environment or validation root already exists.

```bash
RELEASE_REPO='<RELEASE_REPO>'
VALIDATION_ROOT='<NEW_PUBLIC_VALIDATION_ROOT>'

cd "$RELEASE_REPO"
eval "$(conda shell.bash hook)"
set -euo pipefail

if conda env list | awk '$1 == "histcfm" { found=1 } END { exit(found ? 0 : 1) }'; then
  echo 'STOP: Conda environment histcfm already exists; it was not changed.' >&2
  exit 2
fi
if [ -e "$VALIDATION_ROOT" ]; then
  echo 'STOP: VALIDATION_ROOT already exists; choose a new path.' >&2
  exit 2
fi
mkdir -p "$VALIDATION_ROOT/logs"

conda env create -f environment.yml \
  2>&1 | tee "$VALIDATION_ROOT/logs/conda_env_create.log"
conda activate histcfm

python -m pip install --no-deps --no-build-isolation -e . \
  2>&1 | tee "$VALIDATION_ROOT/logs/package_install.log"

{
  python --version
  command -v python
  command -v histcfm
  python -c "import histcfm; print('HistCFM', histcfm.__version__)"
  conda list
} 2>&1 | tee "$VALIDATION_ROOT/logs/environment_inventory.log"

python scripts/check_environment.py \
  2>&1 | tee "$VALIDATION_ROOT/logs/environment_preflight.log"

python -m pytest \
  2>&1 | tee "$VALIDATION_ROOT/logs/pytest.log"

histcfm validate-data \
  --mode train \
  --config configs/demo.yaml \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_validate_train.log"

histcfm train \
  --config configs/demo.yaml \
  --output-dir "$VALIDATION_ROOT/train" \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_train.log"

CHECKPOINT_COUNT=$(find "$VALIDATION_ROOT/train/checkpoints" \
  -maxdepth 1 -type f -name 'epoch_*.pth' -print | wc -l)
if [ "$CHECKPOINT_COUNT" -ne 1 ]; then
  echo "Expected exactly one one-epoch checkpoint, found $CHECKPOINT_COUNT" >&2
  exit 3
fi
CHECKPOINT=$(find "$VALIDATION_ROOT/train/checkpoints" \
  -maxdepth 1 -type f -name 'epoch_*.pth' -print)

python - "$CHECKPOINT" "$VALIDATION_ROOT/train" <<'PY' \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_checkpoint_schema.log"
from pathlib import Path
import sys
from histcfm import __version__
from histcfm.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint

checkpoint_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
payload = load_checkpoint(checkpoint_path, map_location="cpu")
assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 3
assert payload["epoch"] == 1
assert payload["histcfm_version"] == __version__
assert payload["model_metadata"]["n_genes"] == 24
assert payload["model_metadata"]["n_classes"] == 4
normalization = run_root / payload["normalization_artifact"]
assert normalization.is_file()
print("checkpoint", checkpoint_path.name)
print("schema_version", payload["schema_version"])
print("epoch", payload["epoch"])
print("histcfm_version", payload["histcfm_version"])
print("normalization", payload["normalization_artifact"])
PY

histcfm validate-data \
  --mode infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_validate_infer.log"

histcfm infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$VALIDATION_ROOT/infer" \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_infer.log"

histcfm evaluate \
  --predictions "$VALIDATION_ROOT/infer/predictions.csv" \
  --targets "$VALIDATION_ROOT/infer/targets.csv" \
  --cell-types "$VALIDATION_ROOT/infer/cell_types.csv" \
  --output-dir "$VALIDATION_ROOT/metrics" \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_evaluate.log"

python - "$VALIDATION_ROOT/infer" "$VALIDATION_ROOT/metrics" \
  configs/demo.yaml <<'PY' \
  2>&1 | tee "$VALIDATION_ROOT/logs/public_output_contract.log"
import csv
import json
import math
from pathlib import Path
import sys
import yaml

infer_dir = Path(sys.argv[1])
metrics_dir = Path(sys.argv[2])
config = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader), list(reader)

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")

def assert_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)

required_infer = {
    "predictions.csv", "targets.csv", "cell_types.csv", "cells.csv",
    "metadata.json", "resolved_inference_config.yaml",
}
required_metrics = {
    "metrics.json", "per_gene_metrics.csv", "cell_type_metrics.json",
}
assert all((infer_dir / name).is_file() for name in required_infer)
assert all((metrics_dir / name).is_file() for name in required_metrics)

pred_header, pred_rows = read_csv(infer_dir / "predictions.csv")
target_header, target_rows = read_csv(infer_dir / "targets.csv")
type_header, type_rows = read_csv(infer_dir / "cell_types.csv")
expected_header = ["cell_id", *config["data"]["genes"]]
assert pred_header == target_header == expected_header
assert [row[0] for row in pred_rows] == [row[0] for row in target_rows]
assert type_header == ["cell_id", "ground_truth_label", "predicted_label"]
assert [row[0] for row in pred_rows] == [row[0] for row in type_rows]

for name in ("metrics.json", "cell_type_metrics.json"):
    payload = json.loads(
        (metrics_dir / name).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    assert_finite(payload)
print("public output contract: PASSED")
print("cells", len(pred_rows), "genes", len(expected_header) - 1)
PY
```

With the committed `epochs: 1` and `checkpoint_frequency: 1`, the training code
constructs `epoch_1.pth`. The runbook still discovers the checkpoint and
requires exactly one file rather than silently selecting a filename.

## B. The same public run as separate steps

After environment creation and activation, the resumable workflow boundaries
are:

```bash
cd '<RELEASE_REPO>'
conda activate histcfm

python scripts/check_environment.py
python -m pytest
histcfm validate-data --mode train --config configs/demo.yaml
histcfm train --config configs/demo.yaml --output-dir '<NEW_RUN_ROOT>/train'

CHECKPOINT=$(find '<NEW_RUN_ROOT>/train/checkpoints' \
  -maxdepth 1 -type f -name 'epoch_*.pth' -print)

histcfm validate-data --mode infer --split validation \
  --config configs/demo.yaml --checkpoint "$CHECKPOINT"

histcfm infer --split validation --config configs/demo.yaml \
  --checkpoint "$CHECKPOINT" --output-dir '<NEW_RUN_ROOT>/infer'

histcfm evaluate \
  --predictions '<NEW_RUN_ROOT>/infer/predictions.csv' \
  --targets '<NEW_RUN_ROOT>/infer/targets.csv' \
  --cell-types '<NEW_RUN_ROOT>/infer/cell_types.csv' \
  --output-dir '<NEW_RUN_ROOT>/metrics'
```

Resume rules:

- Environment creation failure: stop; do not install the package. Return the
  complete `conda_env_create.log`, `conda info`, and `conda config --show-sources`.
- Installation/import/pytest/preflight failure: fix the diagnosed problem and
  rerun from that read-only step. Do not start training.
- Training failure: the training directory may be partial and must not be
  reused. Preserve it for diagnosis and choose a new run root.
- Once training and checkpoint-schema validation pass, inference preflight can
  be rerun safely from that step.
- Inference or evaluation failure may leave a non-empty output directory.
  Preserve it and use a new `infer` or `metrics` directory for a retry; the
  formal commands deliberately refuse silent overwrite.

## C. Optional private real-data technical validation

Run this only after every public synthetic step above passes. The private YAML
uses staging-relative paths, so execute data commands from the private staging
directory. Choose a new private output root outside both the release and
staging directories.

```bash
RELEASE_REPO='<RELEASE_REPO>'
PRIVATE_STAGING_DIR='<PRIVATE_BC1_STAGING>'
PRIVATE_RUN_ROOT='<NEW_PRIVATE_VALIDATION_ROOT>'

eval "$(conda shell.bash hook)"
conda activate histcfm
set -euo pipefail

if [ -e "$PRIVATE_RUN_ROOT" ]; then
  echo 'STOP: PRIVATE_RUN_ROOT already exists; choose a new path.' >&2
  exit 2
fi
mkdir -p "$PRIVATE_RUN_ROOT/logs"

cd "$RELEASE_REPO"
python scripts/check_environment.py \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/environment_preflight.log"

HISTCFM_PRIVATE_REFERENCE_DIR="$PRIVATE_STAGING_DIR" \
  python -m pytest tests/test_demo_contract.py \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_separation_test.log"

cd "$PRIVATE_STAGING_DIR"
histcfm validate-data --mode train --config demo_private_bc1.yaml \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_validate_train.log"

histcfm train --config demo_private_bc1.yaml \
  --output-dir "$PRIVATE_RUN_ROOT/train" \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_train.log"

PRIVATE_CHECKPOINT_COUNT=$(find "$PRIVATE_RUN_ROOT/train/checkpoints" \
  -maxdepth 1 -type f -name 'epoch_*.pth' -print | wc -l)
if [ "$PRIVATE_CHECKPOINT_COUNT" -ne 1 ]; then
  echo "Expected one private checkpoint, found $PRIVATE_CHECKPOINT_COUNT" >&2
  exit 3
fi
PRIVATE_CHECKPOINT=$(find "$PRIVATE_RUN_ROOT/train/checkpoints" \
  -maxdepth 1 -type f -name 'epoch_*.pth' -print)

python - "$PRIVATE_CHECKPOINT" <<'PY' \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_checkpoint_schema.log"
import sys
from histcfm.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint
payload = load_checkpoint(sys.argv[1], map_location="cpu")
assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 3
assert payload["epoch"] == 1
print("checkpoint schema", payload["schema_version"], "epoch", payload["epoch"])
PY

histcfm validate-data --mode infer --split validation \
  --config demo_private_bc1.yaml \
  --checkpoint "$PRIVATE_CHECKPOINT" \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_validate_infer.log"

histcfm infer --split validation --config demo_private_bc1.yaml \
  --checkpoint "$PRIVATE_CHECKPOINT" \
  --output-dir "$PRIVATE_RUN_ROOT/infer" \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_infer.log"

histcfm evaluate \
  --predictions "$PRIVATE_RUN_ROOT/infer/predictions.csv" \
  --targets "$PRIVATE_RUN_ROOT/infer/targets.csv" \
  --cell-types "$PRIVATE_RUN_ROOT/infer/cell_types.csv" \
  --output-dir "$PRIVATE_RUN_ROOT/metrics" \
  2>&1 | tee "$PRIVATE_RUN_ROOT/logs/private_evaluate.log"
```

Private data, features, configuration, normalization, checkpoints, logs,
predictions, and metrics must not be copied into or uploaded with the release.
This one-epoch run is a technical validation, not a paper benchmark.

## D. Return for review

For the public run, return these text files and small metadata outputs; do not
return the checkpoint or normalization array unless separately requested:

```text
logs/conda_env_create.log
logs/package_install.log
logs/environment_inventory.log
logs/environment_preflight.log
logs/pytest.log
logs/public_validate_train.log
logs/public_train.log
logs/public_checkpoint_schema.log
logs/public_validate_infer.log
logs/public_infer.log
logs/public_evaluate.log
logs/public_output_contract.log
train/resolved_config.yaml
train/logs/train.log
train/logs/train_losses.csv
infer/metadata.json
infer/resolved_inference_config.yaml
metrics/metrics.json
metrics/per_gene_metrics.csv
metrics/cell_type_metrics.json
```

If Conda resolution fails, also return the full terminal error plus:

```bash
conda info
conda config --show-sources
conda env list
```

Do not delete or replace an existing environment to make a failed solve pass.
