# HistCFM-ready input format

HistCFM begins from aligned, preprocessed cell-level inputs. Raw Xenium
processing, H&E registration, nucleus segmentation, and cell matching are
outside this release. The preprocessing boundary remains:

```text
Raw data -> GHIST-referenced preprocessing -> HistCFM-ready inputs
HistCFM-ready inputs -> HistCFM dataset/model/entry points
```

For upstream entry points and the exact processed-data mapping, see
[data_preparation.md](data_preparation.md). For the external original-UNI
boundary and feature-store preparation protocol, see
[uni_features.md](uni_features.md).

The schema below is derived from the audited cell-level loader. The committed
synthetic smoke inputs implement this schema and are covered by contract tests;
the complete synthetic validate/train/checkpoint/infer/evaluate workflow has
also passed in the documented independent Linux/CUDA environment.

## Required files

| Input | Format and required fields | Loader use |
| --- | --- | --- |
| Histology image | `.tif`/TIFF through `tifffile`, or another image readable by `imageio`; H x W x 3 after audited channel handling | Normalized H&E patch tensor |
| Nucleus mask | Image aligned to the histology H x W dimensions; non-negative integer IDs; 0 is background | Patch nucleus labels, cell filtering, areas and model-side centroid coordinates |
| Matched-nuclei table | CSV; first column is the unique cell/nucleus ID; required `size_pix_histology` column | Minimum nucleus-area filtering and ID alignment |
| Expression matrix | CSV; first column is unique cell ID; remaining columns are unique genes in the exact configured order; finite non-negative counts | `expression_scale * log1p(count)` targets |
| Cell-type table | CSV with unique `c_id` and `ct` columns when cell-type supervision is enabled | Cell-type raster and padded per-cell class targets |
| Average-expression table | CSV; first column is row identity; gene columns must exactly match expression genes and order when compatibility is explicitly enabled | Optional compatibility input; disabled in the public demo |
| Precomputed-feature index | `uni_index.json`; JSON object from canonical patch-key strings to unique contiguous integer rows `0..N-1` | Maps canonical patch keys to feature rows; the filename is retained by the formal interface |
| Precomputed features | `uni_features.npy`; non-pickle float16/float32 array with shape N x configured feature dimension (1024 by default) | Externally prepared real features or the public demo's explicitly synthetic interface-compatible matrix |

The matched-nuclei file is the source loader's nucleus/cell matching input.
There is no additional matching table in the formal loader. Cells are retained
by intersecting non-background mask IDs, nucleus-size-filtered IDs, expression
IDs, and—when enabled—cell-type IDs. The preflight requires the supervised
tables to align instead of silently shrinking the model gene dimension.
When supervision is enabled, `ct` values must also be non-null and must resolve
to the configured class list (or valid zero-based numeric class indices).

## Gene and expression rules

- Gene names must be unique.
- The expression matrix must contain the exact configured output gene list and
  order. Missing genes are errors; silent intersection is forbidden.
- The preflight requires the average-expression gene set and order to equal
  the expression matrix.
- Raw expression counts must be numeric, finite, and non-negative.
- The retained training transform is:

```text
expression_scale * log1p(count)
```

- Average-expression values follow the audited entry behavior and are scaled
  by `expression_scale` without `log1p`; this remains an entry-point concern.

The synthetic demo has 24 ordered synthetic gene columns that exactly match
`configs/demo.yaml`. Real prepared inputs must establish their own exact order;
the loader never infers or intersects it.

## Patch and split semantics

- `patch_height` and `patch_width` retain the source `hsize` and `wsize`
  meanings.
- Training uses zero overlap.
- Validation and prediction use the configured overlap.
- The validation division is a pair of image-height fractions. Validation
  patches must fit wholly inside that contiguous row interval. Training tiles
  each contiguous part of its complement independently, so no training patch
  shares a row with or crosses the validation interval.
- Horizontal patching spans the whole slide and keeps the audited terminal
  patch behavior.
- A patch is retained when its filtered nucleus mask has a nonzero sum. A split
  with no eligible full-size patch fails explicitly; no `(0, 0)` fallback or
  patch clipping is used.

The dataset does not create a DataLoader. The audited callers use:

| Purpose | Shuffle | Drop last | Workers |
| --- | --- | --- | --- |
| Training | Configured | Configured | Configured `workers` |
| Validation/prediction | `False` | `False` | Configured `workers` |

These policies are now carried by the formal training and inference entries.

## Canonical patch key

All consumers use `build_patch_key` from `histcfm.data`:

```text
{slide_id}|{row_start}|{column_start}|{patch_height}|{patch_width}|0
```

`slide_id` is the basename stem of the histology path. It cannot contain a
path separator or `|`, so absolute paths never enter the key. Patch keys must
be unique, and preflight rejects any missing UNI feature key. The existing
runtime also raises for any missing key; it never silently substitutes a zero
vector for an enabled feature store.

## Dataset sample and batch tuple

`HistCFMDataset[index]` returns exactly eight fields:

| Position | Meaning | Single-sample shape | dtype |
| --- | --- | --- | --- |
| 0 | Filtered nucleus-ID mask | H x W | `torch.int64` |
| 1 | Cell-type raster, with 0 background | H x W | `torch.int64` |
| 2 | Normalized H&E | 3 x H x W | `torch.float32` |
| 3 | Padded expression | max_cells x genes | `torch.float32` |
| 4 | Number of cells | 1 | `torch.int64` |
| 5 | Padded ground-truth cell types | max_cells | `torch.int64` |
| 6 | Padded cell IDs | max_cells | `torch.int64` |
| 7 | Canonical patch key | scalar string | `str` |

Default PyTorch collation adds a leading batch dimension to tensor fields and
produces a sequence of patch-key strings. Prediction returns zero-filled
expression and cell-type targets because ground truth is not read.

Cell coordinates are not loaded from a table and are not part of this tuple.
The formal model computes normalized centroids directly from the nucleus mask.
The nucleus-size table filters cells; per-patch pixel area is recomputed by the
model.

## Histology normalization

The retained statistic is the average of per-patch channel means and the
average of per-patch channel standard deviations over all candidate patches.
It is not a single global-pixel mean/std. The saved NumPy array is float64 with
shape `(2, 3)`:

```text
row 0: channel means
row 1: channel standard deviations
```

Training may compute the statistic only when given an explicit
`normalization_output_dir`; it writes
`histology_normalization.npy` there and refuses to place it in an input-data
directory. Validation and prediction require an explicit training-generated
`normalization_path` and never recompute or write statistics. The formal train
entry records `artifacts/histology_normalization.npy` in checkpoint metadata;
inference resolves it relative to the checkpoint's training run root and
verifies shape, finiteness, and positive standard deviations before use.

## Preflight coverage

`validate_histcfm_ready_inputs` checks required files, image/mask geometry,
mask IDs, table fields and unique IDs, cross-file cell alignment, finite
non-negative expression, gene identity/order, non-empty splits, unique patch
keys, complete UNI key coverage, feature row bounds, 1024-dimensional feature
shape, float16/float32 dtype, finite feature values, contiguous index rows, and
an L2 norm greater than `1e-12` for every feature row. Validation does not
normalize, rewrite, or repair supplied features. Runtime opens the same NPY
with `mmap_mode="r"` and `allow_pickle=False`, then retains the audited L2
normalization before fusion. There is no public legacy pickle-index path.

`histcfm validate-data --mode train` derives the train and validation patch
inventories and performs this preflight without requiring, estimating, or
writing normalization. `--mode infer` requires a formal checkpoint and checks
checkpoint-bound genes/classes/structure, its training-generated normalization
artifact, the selected inference split, and UNI coverage. Both modes are
read-only.

## Public synthetic smoke inputs

`examples/demo/data/` contains a generated 768 x 768 RGB image and aligned
integer mask, 96 IDs shared exactly by the three cell tables, 24 raw-count
columns, four string cell types, and ten finite float16 feature rows. The
configured first-third validation interval produces six training and four
validation canonical keys. `average_expression_compatibility` is disabled, so
no `average_expression.csv` or `genes.txt` is present.

The demo feature values are deterministic synthetic precomputed morphology
features matching the interface. They are not embeddings from UNI or another
model. Real-data use of externally generated features remains governed by the
external model and source-data terms; the same JSON+NPY schema does not imply
the same provenance or redistribution rights.

## Formal inference inputs and outputs

The `validation` split requires the supervised expression table and, when the checkpoint enables cell types, the cell-type table. The `prediction` split reads no expression or cell-type ground truth. Both require histology, nucleus mask, matched-nuclei table, training normalization, and a complete precomputed UNI store when UNI is enabled. Neither applies training or stain augmentation.

Runtime input paths may differ from training paths, but checkpoint-bound schema and structure may not: ordered genes, ordered cell types, patch dimensions, maximum cells, expression scale, model/flow structure, inference steps, prior/noise, and UNI/fusion dimensions must match metadata exactly.

Each valid patch must contain no more than `data.max_cells_per_patch` eligible
cells. Preflight rejects an overflow; it does not truncate cells, discard them,
or increase the configured limit. No coordinate CSV is needed because the
model derives normalized cell centroids from the nucleus mask.

The formal output tables are:

| File | Presence | Columns and scale |
| --- | --- | --- |
| `predictions.csv` | Always | `cell_id` then checkpoint genes in exact order; values are `log1p(count)`, obtained by dividing the model-space endpoint by `expression_scale` |
| `targets.csv` | Validation only | `cell_id` then the same ordered genes; transformed ground truth in `log1p(count)`, never fabricated for prediction |
| `cell_types.csv` | When the checkpoint enables cell types | `cell_id`, optional validation `ground_truth_label`, and raw-argmax `predicted_label`; no `adjust_pr` postprocessing |
| `cells.csv` | Always | `cell_id`, stable hashed `slide_id`, numeric patch row/column/height/width/level, and `nucleus_area_pixels`; the internal key's original filename stem is not published |
| `metadata.json` | Always | checkpoint/schema/epoch, seed/split, genes/classes, relative normalization provenance, model/sampling parameters, path-free input identifiers, output scale, and runtime versions |
| `resolved_inference_config.yaml` | Always | Runtime configuration with external-input placeholders rather than input names/absolute paths and the checkpoint-relative normalization artifact |

Overlapping-patch duplicates are resolved with the source rule: retain the observation with the largest nucleus pixel area, then restore original observation order. A single dataset instance represents one slide, so `cell_id` is stable within that input; `cells.csv` carries a deterministic SHA-256-derived slide pseudonym and numeric patch provenance without exposing an input filename or absolute path.

## Formal evaluator inputs and outputs

Evaluation starts from the formal inference tables and never loads a model or
checkpoint. Both expression inputs must have `cell_id` as their first column,
non-null unique cell IDs in exactly the same order, and identical unique gene
columns in exactly the same order. All expression values must be numeric and
finite. No row or gene is joined, intersected, reordered, or silently removed.

Both `predictions.csv` and `targets.csv` are interpreted on the
`log1p(count)` scale. The evaluator writes:

| Output | Contents |
| --- | --- |
| `metrics.json` | valid-gene mean/median PCC plus matrix-wide MSE, MAE, and RMSE |
| `per_gene_metrics.csv` | PCC validity/reason and per-gene MSE/MAE |
| `cell_type_metrics.json` | optional accuracy and macro-F1 |

A gene with zero variance in either target or prediction has no defined PCC.
Its CSV PCC is missing, its invalid reason identifies the constant side, and
it is excluded from the mean/median PCC denominator. JSON uses `null`, never
NaN. Optional cell-type evaluation requires an exactly aligned
`cell_types.csv` containing `cell_id`, `ground_truth_label`, and
`predicted_label`.
