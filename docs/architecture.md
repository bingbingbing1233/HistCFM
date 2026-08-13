# Audited architecture behavior

This page records behavior observed during the read-only audit of the existing research implementation. The formal model, data, configuration, training, checkpoint, inference, evaluation, packaging, CLI, and optional synthetic demo are present. Static contracts have been checked, while full server runtime and numerical-equivalence validation remain outstanding.

- The target implementation is the audited cell-level/nucleus HistCFM model identified in the provenance record.
- It includes a conditional flow-matching (CFM) expression component.
- The precomputed UNI feature is fused into the hint branch.
- The flow component receives the original cell embedding rather than the UNI-fused hint representation.
- Training uses an endpoint expression prediction objective.
- SONRM uses spatial neighborhoods separately for its regularization objective.
- The current `SpatialBlock` accepts a `knn_graph` argument but does not consume that graph in its attention computation.

These statements describe the audited research behavior behind the implemented
public API. They are statically verified but do not constitute a completed
server runtime or numerical-reproducibility claim.

## Full framework call graph

The audited cell-level model follows this static call graph:

```text
HistCFM.forward
├── Backbone(H&E)
│   ├── UNet3+-style encoder and decoder
│   └── segmentation map, hd1 (320 channels), h1 (64 channels)
├── nucleus-mask pooling of hd1, h1, and patch-global features
├── Embed(768 → embedding dimension)
├── optional cell-type and patch-composition heads
├── offline UNI patch feature → broadcast to cells → FusionGate/FiLM
│   └── uni_hint_head only
├── flow_forward(original cell embedding, normalized centroid coordinates)
│   └── FlowExpressionModel → CellFlowDenoiser → endpoint expression
└── expression-to-cell-type auxiliary head

Training entry
└── SONRMLoss(HistCFM.last_embeddings, HistCFM.last_coords)
    └── builds a separate KNN graph
```

The backbone receives three H&E channels. Its segmentation output has `n_classes + 1` channels when the cell-type component is enabled, otherwise two channels. The framework concatenates cell-pooled and patch-global `hd1` and `h1` features, producing 768 input features for `Embed`.

## Training flow path

Training is selected only when the module is in training mode and ground-truth expression is supplied.

- During an optional pretraining stage, `Y_t` is the clean expression and `t=0`.
- Otherwise, the model samples `Y0` from its configured prior, samples one uniform `t` per cell, adds Gaussian training noise to `Y0`, and forms `Y_t = t * Y_clean + (1-t) * (Y0 + noise)`.
- A KNN graph is constructed and passed to the denoiser, which predicts endpoint expression `Y_hat` in one call.
- The training entry applies MSE between the predicted endpoint and a separately noise-augmented expression target. It does not directly supervise a velocity vector.

## Inference flow path

Inference starts from a prior sample plus inference noise. It constructs one KNN graph, then performs `S` endpoint-based steps. At each non-final step it updates the state as:

```text
Y_t ← Y_t + (Y_hat - Y_t) / (1 - t1) * (t2 - t1)
```

The final step returns `Y_hat` directly. The trajectory is accumulated internally but is discarded by `HistCFM.forward` and is not part of its returned tuple.

## UNI, CFM, and SONRM separation

One precomputed UNI vector is fetched per patch and repeated for every cell in that patch. Fusion produces a hint representation and `uni_hint_head` maps it to expression space. The flow receives the original image-derived cell embedding, not the fused representation. The training hint loss compares the flow expression prediction with the UNI hint; inference currently discards the hint output.

SONRM receives the original image-derived embeddings cached as `last_embeddings`, together with cell centroids cached as `last_coords`. It builds its own KNN graph and does not reuse the graph passed through the flow denoiser. Its gradient therefore regularizes the image embedding path rather than directly consuming a flow or UNI-fused representation.

## Known inactive or compatibility-only structure

- Three `CrossAttention` refinement modules are instantiated when neighborhood components are enabled, but are not called by the current forward path. They still contribute parameters and state-dict keys.
- The average-expression reference is read by the entries and passed as `ref_orig`, but `HistCFM.forward` does not use that argument. `use_avgexp` and `n_ref` are stored only.
- `out_expr_immune` and `out_expr_invasive` are aliases of `out_expr`; the training entry nevertheless applies separate expression losses to all three when neighborhood components are enabled.
- `use_flow_expr` is stored but does not gate the call to `flow_forward`.
- `safe_shape`, the framework's functional alias `F`, and a UNI broadcast-loop index are unused in the cell-level main path.

These behaviors are recorded, not corrected.

## SpatialBlock and graph behavior

The flow graph is built from normalized cell centroids and passed through `FlowExpressionModel` and `CellFlowDenoiser` into every `SpatialBlock`. No operation in `SpatialBlock` indexes, masks, aggregates, or otherwise consumes `knn_graph`. Its multi-head attention is full attention across all cells concatenated in the current batch. Coordinates still affect the key representation through a learned projection.

Changing this behavior would alter the model and is outside the migration plan.

## Checkpoint compatibility boundary

Existing state dictionaries depend on framework attribute prefixes including `cnn`, `embed_hist`, conditional `estimate_comp` and refinement modules, conditional cell-type MLPs, `flow_expr`, `fusion`, and `uni_hint_head`. The offline UNI cache is not stored in the state dictionary.

Renaming Python source files and changing import paths do not by themselves alter these keys. Renaming model attributes, removing the inactive refinement modules, or restructuring backbone layers would break strict loading. Removing inactive initialized modules would also change the random-number sequence for later parameter initialization during training from scratch.

## Formal public package design

The release presents one formal software identity, **HistCFM**, rather than a collection of research-stage variants. The implemented public API is:

```python
from histcfm import HistCFM
```

The implemented command surface is:

```bash
histcfm train --config CONFIG.yaml --output-dir OUTPUT_DIR
histcfm infer --config CONFIG.yaml --checkpoint PATH --output-dir OUTPUT_DIR
histcfm evaluate --predictions predictions.csv --targets targets.csv --output-dir OUTPUT_DIR
histcfm validate-data --mode train --config CONFIG.yaml
histcfm validate-data --mode infer --config CONFIG.yaml --checkpoint PATH
```

The `HistCFM` class and public import are implemented. The callable training,
single-checkpoint inference, and evaluator layers are present as
`histcfm.train.train(config, output_dir)`,
`histcfm.inference.infer(config, checkpoint_path, output_dir, split)`, and
`histcfm.evaluate.evaluate(predictions, targets, output_dir, ...)`. The CLI
delegates to these functions. Training-mode data validation uses a read-only
patch inventory and the formal schema/UNI preflight without creating
normalization. Inference-mode validation reuses checkpoint loading,
configuration comparison, checkpoint-relative normalization validation and
the formal inference preflight. CLI imports delay model/data dependencies
until command execution.

The intended first-release organization is:

```text
HistCFM package
├── one formal HistCFM model
├── one responsibility-accurate data interface
├── one training entry
├── one inference entry
├── one evaluation entry
└── one demo configuration
```

Research variants, ablations, spot-level branches, dataset-specific launchers, fold-specific configurations, and paper-result/figure scripts are outside the first release. Their old names remain only in `docs/provenance.md` where needed to make the source chain auditable.

## Formal model boundary

The formal `HistCFM` class encapsulates the audited cell-level behavior:

- histology backbone and cell/nucleus representation;
- expression flow and endpoint prediction;
- configured multi-step inference;
- precomputed UNI feature integration and its guided hint branch;
- auxiliary heads;
- intermediate representation needed by SONRM.

UNI is a feature source and auxiliary semantic branch, not a model version. The first release supports only an author-provided precomputed feature store. It does not include an online encoder, UNI weights, automatic download, or feature computation.

The internal compatibility alias `Framework = HistCFM` is retained for later research-script and checkpoint comparison, but public examples and documentation use only `HistCFM`. The committed optional synthetic demo trains from scratch; nevertheless, source/class renaming preserves computation independently of any later mathematical cleanup.

The migrated model package consists of `histcfm.py`, `components.py`, `backbone.py`, `layers.py`, `initialization.py`, `flow.py`, `flow_denoiser.py`, `priors.py`, and `fusion.py`. No constructor call, data read, weight load, device selection, or network access occurs merely by declaring these modules; runtime verification remains pending the approved static checks and later server-side numerical validation.

## Data and configuration design

The data class name must reflect its verified responsibility:

- `HistCFMDataset` denotes a direct PyTorch dataset implementation;
- `HistCFMDataModule` denotes a component that reads data, manages splits, and creates multiple data loaders.

The source-responsibility audit selected and implemented `HistCFMDataset`.
It directly implements patch-indexed PyTorch dataset access while also loading
one slide and selecting its configured row split. It does not construct
DataLoaders or jointly manage train and validation datasets, so no
`HistCFMDataModule` exists in the first release.

The data layer exposes one canonical patch-key builder and a separate read-only
preflight. Normalization computation remains the audited mean of patch means
and mean of patch standard deviations, but writes are restricted to an
explicit training output directory. Validation and inference must reuse the
training-generated `(2, 3)` statistic.

Formal configurations will be named `configs/demo.yaml` and `configs/histcfm.yaml`. Fields must describe behavior, for example embedding dimension, flow hidden dimension, number of flow layers, inference steps, precomputed UNI feature dimension, and SONRM neighborhood parameters. Every renamed field must have an explicit old-to-new mapping and preserve its previous default and numerical effect.

## Formal training entry (2026-08-12)

The release training call is deliberately separate from configuration parsing and import:

```text
explicit train(config, output_dir)
├── validate strict configuration and output ownership
├── enumerate train and validation datasets
├── validate schema, split, exact genes, patch keys, UNI coverage and finite features
├── fix seeds and select the configured device without changing environment variables
├── instantiate HistCFM and AdamW
├── run the audited endpoint-training loop
└── write resolved config, normalization, loss log and periodic checkpoints
```

Importing `histcfm.train` does not parse configuration, select a GPU, create a directory, read data, or begin training. The validation dataset is constructed only to enumerate and preflight held-out patch keys; no validation forward pass, metric, model selection, or best-PCC behavior is included.

The Dataset tuple remains, in order: nucleus mask, cell-type raster, H&E tensor, padded expression, cell count, padded cell-type target, padded cell IDs, and patch key. The formal entry passes `patch_ids=None` to the model, as the audited source training entry did, while passing the patch keys required by precomputed UNI integration.

The 15 model outputs retain their source order and meaning:

| Position | Output | Training use |
| --- | --- | --- |
| 0 | histology-derived cell-type logits | cell-type CE and histology-composition calculation |
| 1 | segmentation/map logits | map CE against the cell-type raster |
| 2 | flattened cell-type target | CE, cosine target sizing, and composition target |
| 3 | endpoint expression | endpoint MSE and UNI-hint consistency input |
| 4 | immune expression alias | separate neighborhood MSE against the same noisy target |
| 5 | invasive expression alias | separate neighborhood MSE against the same noisy target |
| 6 | UNI expression hint | consistency target for output 3, not ground truth |
| 7 | expression-derived cell-type logits | auxiliary CE and logits MSE |
| 8 | expression-derived cell-type embedding | cosine embedding input |
| 9 | ground-truth-expression cell-type logits | logits-MSE target |
| 10 | ground-truth-expression embedding | cosine-embedding target |
| 11 | flattened ground-truth expression | independently noise-augmented expression target |
| 12 | per-patch estimated composition | KL term against cell-type composition |
| 13 | cell area | returned but unused by formal training |
| 14 | flattened patch cell IDs | absent because training intentionally passes `patch_ids=None` |

Outputs 4 and 5 are aliases of output 3 in the audited model. Their duplicate losses are retained rather than merged. Average expression is an optional compatibility input: when enabled it is read, scaled and passed as `ref_orig`, but the current model does not consume `ref_orig`; `use_avgexp` and `n_ref` remain stored attributes only. The formal default disables that compatibility input.

The source uses a manual epoch-level learning-rate update rather than a PyTorch scheduler object. The formal default preserves its linear rule and lower bound. Optional cosine and two-stage branches remain explicitly configurable because they exist in the source entry. Checkpoint selection preserves the source zero-based frequency condition (`epoch_index % frequency == 0`), while the new checkpoint combines model, optional optimizer state, and reproducibility metadata in one file.

The strict configuration implementation uses standard-library dataclasses and loads/dumps YAML through PyYAML only when explicitly requested. PyYAML is declared in `pyproject.toml`; no package was installed or imported for runtime validation in this organization environment.

## Behavior-preservation rule

Later formalization may change filenames, class names, imports, configuration field names, path handling, validation, type hints, documentation, and clearly separated module boundaries. It must not simultaneously change:

- backbone structure or feature extraction;
- endpoint training objective, flow prior, or noise construction;
- multi-step inference update;
- UNI hint connection or fusion computation;
- SONRM formula or auxiliary losses;
- tensor dimensions or default numerical parameters.

Apparently inactive modules remain until the complete demo and behavior comparison pass. Cleanup is a separate decision after equivalence evidence exists.

## Model-package migration validation (2026-08-12)

Static validation, without importing or instantiating the model, established:

- all 16 Python files in the release parse successfully with Python's AST parser;
- after normalizing only the approved class/debug-label renames, the complete `HistCFM` class AST equals the audited research class AST;
- constructor `self.*` assignment order is identical, including conditional refinement modules and both fusion branches;
- the formal forward tuple retains 15 elements in the same order;
- the backbone still returns `seg, hd1, h1`;
- `components.py`, `backbone.py`, `layers.py`, and `initialization.py` have ASTs identical to their sources after removing the added module provenance strings and normalizing the initialization import spelling;
- no dataset, configuration, checkpoint, UNI feature, weight, result, or cache artifact was introduced.

The optional public-import check could not complete in the audit environment because neither PyTorch nor NumPy is installed. It stopped at `ModuleNotFoundError: torch`; no dependency was installed and no model was instantiated. Import and numerical equivalence therefore remain server-side validation tasks.

## Dataset-to-model boundary

The dataset returns nucleus mask, cell-type raster, normalized H&E, padded
expression, cell count, padded cell-type targets, padded cell IDs and the
canonical patch key, in that order. The model receives H&E, nucleus masks,
cell counts, optional targets and patch keys. It computes cell centroids from
the mask; the dataset does not load coordinate tables. Average-expression
profiles remain entry-point responsibilities rather than dataset
responsibilities.

The caller—not the dataset—sets DataLoader behavior: training uses configured
shuffle and drop-last; formal validation/prediction inference forces both to
false; both take configured worker counts.

## Formal inference entry (2026-08-13)

Formal inference accepts one explicit schema-3 checkpoint and one explicit output directory:

```text
infer(config, checkpoint_path, output_dir, split)
├── restricted checkpoint load and schema validation
├── checkpoint/runtime structural compatibility comparison
├── normalization resolution relative to the training run root
├── validation or prediction Dataset construction and complete UNI preflight
├── strict HistCFM state-dict load
├── one seeded stochastic CFM sample under inference mode
├── largest-area de-duplication of cells observed in overlapping patches
└── prediction, cell, optional target/type, and metadata outputs
```

The checkpoint must remain at `<run>/checkpoints/<file>` and its safe relative `normalization_artifact` is resolved against `<run>`, not the current working directory. Shape `(2, 3)`, finite statistics, and positive standard deviations are required. Inference never estimates normalization from validation or prediction inputs.

Model-defining values come from checkpoint `resolved_config`. The runtime configuration may supply input paths, UNI store location, device, worker/batch settings and a sampling seed, but must exactly match checkpoint genes/classes, patch dimensions and capacity, expression scale, cell/neighborhood structure, flow architecture/steps/prior/noise, and UNI/fusion structure. `load_state_dict(..., strict=True)` is mandatory; there is no filename inference, state-key rewrite, or legacy fallback.

Both formal splits use `shuffle=False`, `drop_last=False`, no stain augmentation, and actual per-batch lengths. `validation` requires expression and, when configured, cell-type labels; it emits targets. `prediction` intentionally supplies neither target to the model and emits no fabricated target file. Both call the unchanged model in evaluation mode, so CFM sampling still draws from the configured prior, adds inference noise, and executes the unchanged configured S-step update.

The source inference's cell-type `adjust_pr` is excluded. Static tracing shows that it affects only an additional adjusted-label output through random composition-based reassignment; raw expression prediction, raw argmax labels, historical F1, and epoch ranking do not depend on it. The formal first inference therefore emits raw argmax labels and leaves optional postprocessing policy for a separate reviewed stage.

The source multi-checkpoint scan, per-gene PCC, F1, rank-sum selection, max-PCC selection, best-expression copy, and result summary are evaluator/model-selection concerns and are not present. One invocation performs one checkpoint and one stochastic sample.

## Formal evaluation, CLI, and package boundary (2026-08-13)

`histcfm.evaluate` is a new release-side general evaluator rather than a copy
of dataset-bound paper scripts. It accepts only already-produced expression
tables and optional labels, requires exact cell/gene identity and order, and
never reads a checkpoint or chooses an epoch. Gene-wise PCC is undefined for
zero-variance genes and is represented as missing with an explicit reason;
only valid genes enter mean and median PCC. MSE, MAE, and RMSE are computed on
the complete aligned matrix on the documented `log1p(count)` scale.

`histcfm.cli` is an argument/delegation layer for four formal commands:
training, one-checkpoint inference, table evaluation, and inference-ready data
validation. Parser construction performs no data read, directory creation,
device choice, or workflow execution. The package top-level uses a lazy model
export so CLI/help discovery does not eagerly import PyTorch model modules.

`pyproject.toml` declares a single `src/` layout and console entry point. It is
the Python package dependency source of truth; no separate `requirements.txt`
was added. `environment.yml` defines the independent Linux/CUDA validation
environment. The packaging and runtime matrix remain unverified on the target
server.

## UNI store and explicit fusion configuration (2026-08-13)

The public UNI mode is exactly `precomputed`. `UniFeatureProvider` receives
explicit `index_path` and `features_path`; it does not infer filenames or
translate an internal mode string. Both provider and preflight call the same
JSON+NPY loader. The loader validates canonical unique keys, contiguous rows,
shape/dimension, float16/float32 dtype, finiteness, and per-row L2 norm above
`1e-12`. Coverage is checked before training/inference, and a missing runtime
key is also an error. Valid rows still undergo the existing L2 normalization
before fusion.

The research `HistCFM` constructor keeps its `fusion_type="gate"` default.
Formal model construction explicitly supplies `config.uni.fusion_method` for
both training and inference; inference reconstructs the model from checkpoint
configuration. Schema-3 checkpoint metadata records `fusion_method`, and a
missing or inconsistent value fails schema/compatibility validation. No
backbone, flow, UNI fusion math, or default numerical configuration was
changed by this interface repair.

Patch overflow behavior remains strict: every valid patch must contain at
most `max_cells_per_patch` eligible cells. Validation never truncates or drops
cells and never increases the limit.
