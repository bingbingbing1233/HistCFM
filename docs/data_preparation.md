# Preparing real data for HistCFM

HistCFM starts from aligned, preprocessed cell-level inputs. It does not
implement the complete conversion from raw Xenium exports and whole-slide
histology. The intended boundary is:

```text
Raw Xenium data + registered H&E
  -> GHIST-referenced preprocessing
  -> HistCFM-ready files
  -> histcfm validate-data / train / infer / evaluate
```

GHIST preprocessing and HistCFM modeling are consecutive but independently
maintained steps. Users should consult the current
[SydneyBioX/GHIST repository](https://github.com/SydneyBioX/GHIST) rather than
expecting a copied GHIST pipeline here. A prepared HistCFM dataset can be used
without installing GHIST.

## Official GHIST entry points

The following paths existed on the audited GHIST `main` branch at commit
`917456be305fc82e92293ea272812e79675e821c`:

| Official GHIST path | Purpose |
| --- | --- |
| [`tutorials/1_data_preprocessing.ipynb`](https://github.com/SydneyBioX/GHIST/blob/main/tutorials/1_data_preprocessing.ipynb) | Worked sequence from selected Xenium exports and an aligned H&E image to GHIST inputs |
| [`data_processing/1_get_xenium_nuclei_seg_image.py`](https://github.com/SydneyBioX/GHIST/blob/main/data_processing/1_get_xenium_nuclei_seg_image.py) | Rasterizes Xenium nucleus boundaries in the registered histology frame |
| [`data_processing/2_get_xenium_cell_gene_matrix.py`](https://github.com/SydneyBioX/GHIST/blob/main/data_processing/2_get_xenium_cell_gene_matrix.py) | Converts a Xenium cell-feature matrix to a cell-by-gene CSV and gene list |
| [`data_processing/3_segment_nuclei_he_image.py`](https://github.com/SydneyBioX/GHIST/blob/main/data_processing/3_segment_nuclei_he_image.py) | Splits H&E, invokes an external Hover-Net workflow, and combines nucleus masks |
| [`data_processing/4_get_corresponding_cells.py`](https://github.com/SydneyBioX/GHIST/blob/main/data_processing/4_get_corresponding_cells.py) | Matches H&E nuclei to Xenium cells and filters correspondences |
| [`data_demo/`](https://github.com/SydneyBioX/GHIST/tree/main/data_demo) | Example of GHIST's processed-file layout; it is not copied into HistCFM |

The official tutorial requires an already aligned H&E image. It explicitly
does not supply H&E-to-Xenium registration code and links to the 10x Genomics
registration guide as one option. Its H&E segmentation step also relies on an
external Hover-Net repository/environment/checkpoint. Consequently, GHIST does
not provide a single self-contained command that converts arbitrary raw
Xenium data into every required input.

Follow the current upstream documentation and the terms of every source
dataset and external component. This release has not validated every dataset
or every version of the upstream preprocessing stack.

## Mapping GHIST concepts to HistCFM

GHIST's tutorial identifies `cell_gene_matrix_filtered.csv`,
`he_image_nuclei_seg.tif`, and `matched_nuclei_filtered.csv` as the files used
by GHIST. HistCFM gives those concepts stable configuration fields rather than
requiring the historical filenames:

| GHIST/processed-data concept | HistCFM configuration field | HistCFM file role |
| --- | --- | --- |
| Registered/aligned H&E histology | `data.histology_path` | Histology TIFF or supported image |
| H&E nucleus segmentation | `data.nucleus_mask_path` | Integer nucleus-label mask |
| Filtered H&E/Xenium nucleus correspondence | `data.matched_nuclei_path` | Matched-nuclei CSV |
| Filtered cell-gene matrix | `data.expression_path` | Cell-by-gene raw-count CSV |
| Optional cell-type annotation | `data.cell_type_path` | `c_id,ct` CSV when cell-type supervision is enabled |
| Ordered gene panel | `data.genes` | YAML list matching expression columns exactly |

The conversion from GHIST names is a schema operation, not a new biological
preprocessing algorithm. A user may rename files, select/reorder columns, and
set YAML fields, but must preserve the aligned pixels, cell identities, raw
counts, and gene meaning. In particular:

- the first column of the HistCFM expression CSV is the cell ID; the remaining
  columns are genes in exactly the same order as `data.genes`;
- the first column of the matched-nuclei CSV is the H&E mask label used as the
  HistCFM cell ID, and it must include `size_pix_histology`;
- if GHIST correspondence output still contains both `id_histology` and
  `id_xenium`, construct the final cell-by-gene table with the H&E mask ID as
  its row ID; do not merely rename a Xenium ID column and assume it matches the
  mask;
- `genes.txt` is a GHIST preprocessing output, but HistCFM does not read that
  file directly. Transfer its verified order to `data.genes` and confirm that
  the expression header is identical;
- GHIST describes cell types and average expression as optional. This release
  requires cell types only when `model.use_cell_types: true` and requires an
  average-expression table only when
  `model.average_expression_compatibility: true` (false by default).

## Enforced HistCFM-ready contract

`histcfm validate-data` enforces the following current code behavior:

- Histology loads as an `H x W x 3` array. The validator enforces three
  channels and geometry but does not impose a histology dtype; an ordinary
  numeric RGB TIFF is the intended input.
- The nucleus mask is two-dimensional, has the same `H x W` extent, uses a
  non-negative integer dtype, and reserves label `0` for background.
- The first columns of matched nuclei and expression contain unique,
  integer-compatible cell IDs. `c_id` is unique in the cell-type file.
- `size_pix_histology` is required, numeric, finite, and non-negative. Cells
  below `data.min_nucleus_area` are excluded when patch eligibility is built.
- Every expression ID must occur in both the mask and matched-nuclei table.
  The mask and matched table may contain additional cells. When cell types are
  required, their `c_id` set must equal the expression ID set exactly.
- Cell-type `ct` values must be non-null. They may be strings from the ordered
  `data.cell_types` list or integer class indices in
  `0..len(data.cell_types)-1`; unknown or out-of-range values are rejected.
- Expression is cell by gene: rows are cells and columns after the first are
  unique genes. Columns must equal `data.genes` in both identity and order.
  Values must be numeric, finite, non-negative raw counts. At runtime the
  target is `data.expression_scale * log1p(count)`.
- The validation split is a pair of image-height fractions. Validation uses
  that row interval and training uses its complement. Both must contain at
  least one eligible patch. Training uses zero overlap; validation uses
  `data.overlap`.
- Patch dimensions cannot exceed the image; overlap must be smaller than both
  dimensions. If any eligible patch exceeds `data.max_cells_per_patch`,
  preflight fails instead of truncating cells.
- Training creates `artifacts/histology_normalization.npy` in its new output
  directory. Validation inference and prediction reuse the checkpoint-bound
  training artifact; they do not recompute it from inference data.

The full CSV and feature-store schemas are specified in
[input_format.md](input_format.md). Run the read-only preflight before starting
training:

```bash
histcfm validate-data --mode train --config path/to/config.yaml
```

For a checkpoint-bound validation or prediction input:

```bash
histcfm validate-data --mode infer --split validation \
  --config path/to/config.yaml --checkpoint path/to/checkpoint.pth
```

## Citation

The upstream repository asks users to cite:

> Fu et al. Spatial gene expression at single-cell resolution from histology
> using deep learning with GHIST. *Nature Methods* 22, 1900–1910 (2025).
> <https://doi.org/10.1038/s41592-025-02795-z>

GHIST code is distributed under GNU GPL version 3 in its official repository.
Dataset, registration-tool, Hover-Net, and model terms remain separate and
must be checked at their respective sources.
