# Preparing precomputed morphology features

HistCFM consumes a precomputed patch-feature store. It does not contain or run
an online UNI encoder, distribute UNI source or checkpoints, authenticate to a
model host, or download weights. For real-data use, each user must obtain
authorized access through the official
[MahmoodLab/UNI repository](https://github.com/mahmoodlab/UNI) and the
[original UNI model page](https://huggingface.co/MahmoodLab/uni), then comply
with the current license and access terms.

The optional public synthetic smoke test is different: its 1024-dimensional
matrix was produced directly by the repository generator and is **not** UNI,
UNI2, or any other model's output.

## Model used by the historical HistCFM feature path

The audited HistCFM research feature script selected the original UNI model,
not UNI2, and strictly loaded its checkpoint into:

```text
architecture: vit_large_patch16_224 (ViT-L/16)
model input: 224 x 224 RGB
model output: 1024 values per patch
mode: eval, with gradient computation disabled
```

These values agree with UNI's original README: a
`vit_large_patch16_224` model with `img_size=224`, `patch_size=16`,
`num_classes=0`, and a `[1, 1024]` feature output. The public HistCFM package
does not reproduce that encoder implementation.

## Recommended deterministic extraction protocol

For new real data, the HistCFM recommended deterministic extraction protocol
is:

```text
256 x 256 raw RGB HistCFM patch
  -> resize to 224 x 224 with antialiasing
  -> convert to tensor
  -> ImageNet normalization
       mean = (0.485, 0.456, 0.406)
       std  = (0.229, 0.224, 0.225)
  -> original UNI ViT-L/16 in eval/no_grad mode
  -> 1024-dimensional output
  -> save as float16 or float32
```

This protocol is a deterministic recommendation for new data and for the
published HistCFM feature interface. Use raw RGB patches from the same aligned
histology and the same canonical patch coordinates as the HistCFM loader. Do
not add stain augmentation, random flips, random rotations, or random crops.

It is not a claim of byte identity with the historical paper cache. The
historical source constructed its dataset in training mode; that path could
apply random spatial/stain augmentation and supplied an already normalized
HistCFM patch to the UNI transform. Those research-cache semantics were not a
deterministic portable protocol. This distinction must be considered when
comparing newly extracted features with historical checkpoints or results.

HistCFM runtime converts stored rows to float32 and applies L2 normalization
immediately before fusion. Do not pre-normalize merely to satisfy validation;
if an extraction workflow performs any additional normalization, record it as
part of feature provenance.

## Canonical patch keys

For every eligible loader patch, use exactly:

```text
{slide_id}|{row_start}|{column_start}|{patch_height}|{patch_width}|0
```

- `slide_id` is the basename stem of `data.histology_path`, not an absolute
  path. It must not contain `/`, `\`, `|`, or `:`.
- `row_start` and `column_start` are zero-based coordinates in the prepared
  histology/mask coordinate frame.
- `patch_height` and `patch_width` must match the YAML values.
- The final `0` is the current image level.

Use the patch inventory produced by the formal loader; do not infer a
different tiling grid. `histcfm validate-data --mode train` checks both train
and validation inventories, including their distinct overlap behavior.

## Feature-store files

Configure:

```yaml
uni:
  enabled: true
  mode: precomputed
  index_path: path/to/uni_index.json
  features_path: path/to/uni_features.npy
  feature_dim: 1024
```

`uni_index.json` must be a JSON object whose keys are canonical patch keys and
whose values are row indices:

```json
{
  "slide|0|0|256|256|0": 0,
  "slide|0|256|256|256|0": 1
}
```

The keys are unique. Row indices are unique non-negative JSON integers and,
over the whole object, continuously cover `0..N-1`. The row named by each
index value is the corresponding row in `uni_features.npy`; JSON object order
is not used as an implicit mapping.

`uni_features.npy` must be a non-pickle, two-dimensional NumPy array with:

- shape `(N, uni.feature_dim)`; the original UNI interface uses 1024;
- dtype `float16` or `float32`;
- the same `N` as the JSON index;
- finite values only;
- L2 norm greater than `1e-12` for every row.

The index must cover every canonical key required by the selected split.
Missing keys fail preflight and runtime; HistCFM does not substitute zero
features. The shared loader opens the NPY read-only with `allow_pickle=False`
and checks contiguous row numbers, dimensions, dtype, finiteness, norms, and
coverage.

```bash
histcfm validate-data --mode train --config path/to/config.yaml
```

## Access, citation, and redistribution boundary

UNI's official README directs users to request access on Hugging Face and to
accept its current terms. HistCFM does not grant rights to UNI code, weights,
model outputs, or source images, and attribution does not replace permission.
Do not commit a checkpoint or another user's copy of the model. Before sharing
real feature arrays, separately review the source-image terms and UNI access
terms and obtain written clarification where needed.

The official UNI citation is:

> Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. Towards a
> general-purpose foundation model for computational pathology. *Nature
> Medicine* (2024). <https://doi.org/10.1038/s41591-024-02857-3>

This release includes only the offline HistCFM loader and the clearly labeled
synthetic smoke-test feature matrix. It includes no UNI encoder, checkpoint,
weights, or real UNI-generated feature file.
