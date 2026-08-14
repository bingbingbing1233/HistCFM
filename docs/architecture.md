# HistCFM architecture

HistCFM predicts cell-level spatial gene expression from registered histology,
nucleus instances, spatial context, and precomputed patch-level morphology
features. The public package exposes one model and one set of training,
inference, validation, and evaluation entry points.

## System overview

```text
Histology + nucleus mask
  -> histology backbone
  -> nucleus-aware cell representations
  -> conditional flow-matching expression prediction

Precomputed patch morphology feature
  -> feature lookup and fusion
  -> UNI-guided expression hint

Cell representations + centroid coordinates
  -> SONRM spatial regularization during training
```

HistCFM starts from processed, spatially aligned cell-level inputs. Raw Xenium
processing, image registration, nucleus segmentation, cell matching, and
external foundation-model feature extraction are preparation steps outside the
runtime package.

## Public interfaces

The model is available as:

```python
from histcfm import HistCFM
```

The command-line interface delegates to the same formal Python entries:

```bash
histcfm validate-data --mode train --config CONFIG.yaml
histcfm train --config CONFIG.yaml --output-dir OUTPUT_DIR
histcfm validate-data --mode infer --config CONFIG.yaml --checkpoint CHECKPOINT
histcfm infer --config CONFIG.yaml --checkpoint CHECKPOINT --output-dir OUTPUT_DIR
histcfm evaluate --predictions PREDICTIONS.csv --targets TARGETS.csv --output-dir OUTPUT_DIR
```

Importing the CLI does not read data, select a device, create output
directories, or start a workflow.

## Data and patch construction

The data layer reads aligned histology, a nucleus-instance mask, matched-nucleus
metadata, a cell-by-gene count matrix, optional supervised cell types, and a
precomputed morphology-feature store. It constructs fixed-size image patches
and uses the mask labels as cell identifiers.

One shared coordinate generator defines both dataset patches and validation
inventory. Validation patches lie fully inside the configured validation row
interval; training patches lie fully in its complement and cannot cross that
interval. Histology normalization is estimated from training patches only and
stored with the training run for reuse during inference.

The complete file, identifier, coordinate, gene-order, split, and canonical-key
contracts are documented in [input_format.md](input_format.md).

## Model components

### Histology backbone and cell representation

The U-Net 3+-style backbone produces a segmentation map and two feature maps.
HistCFM pools these maps over each nucleus instance, combines cell-pooled and
patch-global information, and embeds the resulting representation into the
configured cell-embedding dimension. Optional auxiliary heads predict cell
types and patch composition.

### Conditional flow matching

The expression-flow path receives the original histology-derived cell
embedding and normalized cell-centroid coordinates. During training it samples
a configured prior, time, and noise, constructs an intermediate expression
state, and predicts endpoint expression. The primary expression objective is
MSE against the configured noisy endpoint target; the implementation does not
directly supervise a velocity vector.

### Precomputed morphology-feature integration

The public runtime supports only precomputed patch-level features. A canonical
patch key selects one feature row, which is normalized at runtime and broadcast
to the cells in that patch. Fusion with the cell representation produces an
auxiliary semantic hint. The flow path continues to use the original
histology-derived cell representation; the fused hint is supervised through
its dedicated consistency objective.

HistCFM does not include an online UNI encoder, UNI source code, model weights,
automatic downloads, or feature extraction. See [uni_features.md](uni_features.md)
for the external feature-store boundary.

### SONRM and auxiliary objectives

SONRM receives the cached histology-derived cell embeddings and cell-centroid
coordinates and builds its own spatial neighborhood graph. It regularizes the
embedding path separately from the precomputed-feature hint branch. Additional
configured objectives cover segmentation, cell type, composition, and
expression-derived auxiliary representations.

## Training behavior

The formal training entry performs strict configuration and input validation,
creates training-only normalization, constructs `HistCFM`, and runs the
endpoint-expression, semantic-hint, SONRM, and configured auxiliary losses.
It writes a resolved configuration, normalization artifact, loss records, and
schema-versioned checkpoints to a new or empty output directory.

The validation split is enumerated and checked before training so held-out
patches, labels, genes, and feature keys are valid. Version `0.1.0` does not run
a validation forward pass, select a best epoch, or choose a checkpoint from
validation metrics. Evaluation is a separate explicit workflow.

## S-step inference

Inference loads one explicit trusted checkpoint, reuses its training
normalization, samples the configured prior and inference noise, and performs
the configured number of endpoint-based steps. At each non-final step:

```text
Y_t <- Y_t + (Y_hat - Y_t) / (1 - t1) * (t2 - t1)
```

The final step returns the endpoint prediction directly. Cells seen in
overlapping patches are de-duplicated deterministically using the observation
with the largest nucleus area. Validation inference emits aligned predictions
and targets; prediction mode does not fabricate unavailable targets.

## Evaluation and outputs

The evaluator reads inference tables rather than a model or checkpoint. It
requires identical cell and gene identity and order, reports gene-wise PCC,
valid-gene PCC summaries, MSE, MAE, RMSE, and optional cell-type accuracy and
macro-F1, and writes strict JSON without NaN or Infinity. It does not select an
epoch or implement paper-specific figure and fold aggregation workflows.

## Checkpoint and compatibility considerations

- Schema-3 checkpoints record the resolved model configuration, ordered genes,
  cell-type mapping, normalization provenance, seed, package version, and
  runtime metadata. Model state is loaded strictly.
- The internal `Framework = HistCFM` alias is retained for compatibility, but
  `HistCFM` is the only documented public class.
- Some initialized refinement modules and output aliases are retained because
  removing them would change state-dict keys, parameter initialization order,
  or the preserved training objective.
- The optional average-expression compatibility input is disabled by default;
  the current model stores its related settings but does not consume the
  reference in the forward computation.
- The flow `SpatialBlock` accepts a neighborhood graph for compatibility, but
  its current attention operation does not index or aggregate that graph;
  centroid coordinates still enter the learned key representation.

These behaviors are part of the documented `0.1.0` implementation. Changing
them would require an explicit model revision and renewed numerical validation.

## Validation status and scope

HistCFM `0.1.0` passed all 75 automated tests and the complete committed
synthetic validate/train/checkpoint/infer/evaluate workflow in the independent
Linux/CUDA environment documented in [server_validation.md](server_validation.md).
That run covered six training patches, four validation patches, strict
schema-3 checkpoint reload, inference for 36 cells and 24 genes, evaluation,
strict JSON, and output-order checks.

The synthetic workflow verifies software integration only. It is not a paper
benchmark, biological result, validation on every supported dataset, or claim
of bitwise identity across hardware and dependency versions.
