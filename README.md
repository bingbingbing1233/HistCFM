# HistCFM

## Paper

**HistCFM: Semantics-Guided Conditional Flow Matching for Single-Cell Spatial Gene Expression Prediction from Histology**

HistCFM is a cell-level histology-to-gene-expression framework that integrates tissue morphology, spatial-neighborhood information, and conditional flow matching. It operates on processed cell-level inputs and precomputed morphology features.

The repository provides installation instructions, documented command-line workflows for training, inference, and evaluation, and a fully synthetic end-to-end example for verifying the software pipeline.

## Overview

HistCFM integrates cell/nucleus representations, spatial context, precomputed morphology features, conditional flow-matching inference, and optional cell-type and SONRM objectives.

```python
from histcfm import HistCFM
```

## Installation

Create the recommended independent environment and install the package from the repository root:

```bash
conda env create -f environment.yml
conda activate histcfm

python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e .
```

For a first-install check, optionally run:

```bash
python scripts/check_environment.py
```

This check is not required before every run. Dependency and validated-stack details are in the [environment guide](docs/environment.md).

## Quick smoke test

The demo is completely synthetic and exercises the formal software workflow. It contains no patient data or real UNI output and does not reproduce or estimate paper performance. See the [demo guide](examples/demo/README.md).

```bash
histcfm validate-data --mode train --config configs/demo.yaml

histcfm train \
  --config configs/demo.yaml \
  --output-dir outputs/demo/train

histcfm validate-data \
  --mode infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint outputs/demo/train/checkpoints/epoch_1.pth

histcfm infer \
  --split validation \
  --config configs/demo.yaml \
  --checkpoint outputs/demo/train/checkpoints/epoch_1.pth \
  --output-dir outputs/demo/infer

histcfm evaluate \
  --predictions outputs/demo/infer/predictions.csv \
  --targets outputs/demo/infer/targets.csv \
  --cell-types outputs/demo/infer/cell_types.csv \
  --output-dir outputs/demo/metrics
```

## Preparing real data

```text
Raw Xenium data and registered H&E
  -> GHIST-referenced preprocessing
  -> HistCFM-ready cell-level inputs
  -> precomputed morphology features
  -> HistCFM
```

- **GHIST preprocessing workflow:** [https://github.com/SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST)
- **UNI foundation model:** [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)

Users starting from raw Xenium data should follow the GHIST preprocessing workflow and then organize the resulting files according to the [HistCFM input specification](docs/input_format.md). Users with HistCFM-ready inputs do not need to install or rerun GHIST.

Histology embeddings should be generated using an authorized local installation of UNI and converted to the precomputed feature format described in the [UNI feature guide](docs/uni_features.md). HistCFM does not vendor the GHIST preprocessing workflow or download or redistribute UNI source code, UNI model weights, or real-data UNI features. These external resources are not required for the included synthetic smoke demo.

See [real-data preparation](docs/data_preparation.md) and the [input format](docs/input_format.md) for the complete preparation and data contracts.

| Input | Purpose |
| --- | --- |
| Histology TIFF | Registered RGB histology |
| Nucleus mask | Cell/nucleus instance labels; `0` is background |
| Matched nuclei CSV | Cell matching and nucleus-size metadata |
| Expression CSV | Cell-by-gene non-negative raw counts |
| Cell types CSV | Required with cell-type supervision; omitted for unlabeled prediction |
| `uni_index.json` | Canonical patch-key-to-row mapping |
| `uni_features.npy` | Precomputed morphology-feature matrix |

## Datasets used in this study

All source datasets analyzed in the HistCFM study are publicly available from their original providers. The datasets themselves are not redistributed in this repository.

| Dataset in the HistCFM study | Platform and source | Public access |
|---|---|---|
| **Breast Sample 1** | 10x Genomics Xenium FFPE Human Breast Cancer, Replicate 1 | [Dataset explorer](https://www.10xgenomics.com/products/xenium-in-situ/human-breast-dataset-explorer) · [Xenium output bundle](https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip) |
| **Breast Sample 2** | 10x Genomics Xenium V1 FFPE Preview Human Breast Cancer, Sample 2 | [Dataset explorer](https://www.10xgenomics.com/products/xenium-in-situ/human-breast-dataset-explorer) · [Xenium output bundle](https://cf.10xgenomics.com/samples/xenium/1.4.0/Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2/Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2_outs.zip) |
| **Melanoma** | 10x Genomics Xenium Human Skin Preview dataset with the Human Skin Gene Expression Panel add-on | [Dataset page](https://www.10xgenomics.com/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard) |
| **Human Breast Cancer (Visium)** | Constructed by combining Human Breast Cancer Block A, Sections 1 and 2 | [Section 1](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) · [Section 2](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0) |

For the Xenium datasets, HistCFM follows the cell-level preprocessing workflow used by [GHIST](https://github.com/SydneyBioX/GHIST). After preprocessing, the resulting files must be organized according to the [HistCFM input specification](docs/input_format.md). Precomputed histology features can be prepared using an authorized installation of [UNI](https://github.com/mahmoodlab/UNI), following the [UNI feature guide](docs/uni_features.md).

The included synthetic demo is provided solely to verify the HistCFM software workflow and does not contain material derived from these biological datasets.

## Training

Train from a strict YAML configuration and write artifacts to a new or empty directory:

```bash
histcfm train \
  --config path/to/config.yaml \
  --output-dir path/to/training_run
```

The standalone `validate-data` command is a recommended preflight; the formal runtime also enforces the relevant contracts. See the [input contract](docs/input_format.md) and [reproducibility guide](docs/reproducibility.md).

## Inference

Run one explicit trusted checkpoint. Use `--split validation` with targets or `--split prediction` for unlabeled inputs:

```bash
histcfm infer \
  --split validation \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint.pth \
  --output-dir path/to/inference_run
```

Inference reuses the training normalization bound to the checkpoint. The checkpoint-aware preflight is documented in the [server validation guide](docs/server_validation.md).

## Evaluation

Evaluate aligned tables produced by validation inference:

```bash
histcfm evaluate \
  --predictions path/to/inference_run/predictions.csv \
  --targets path/to/inference_run/targets.csv \
  --cell-types path/to/inference_run/cell_types.csv \
  --output-dir path/to/metrics
```

The evaluator reports gene-wise PCC, mean/median valid-gene PCC, MSE, MAE, RMSE, and optional cell-type accuracy and macro-F1 on the `log1p(count)` expression scale. See the [input format](docs/input_format.md) for alignment requirements.

## Repository structure

```text
src/histcfm/       Core package
configs/           Training and demo configurations
examples/demo/     Synthetic smoke-test inputs
tests/             Automated software tests
docs/              Detailed preparation and reproducibility guides
```

## Reproducibility and scope

- HistCFM `0.1.1` passed all 75 tests and the complete synthetic validate/train/checkpoint/infer/evaluate workflow in the declared independent Linux/CUDA environment.
- The synthetic demo validates software plumbing only; it is not a paper benchmark or biological result.
- Real datasets and paper checkpoints are not distributed here.
- Users prepare real morphology features independently through authorized external-model access.
- See [environment](docs/environment.md), [server validation](docs/server_validation.md), and [reproducibility](docs/reproducibility.md) for detailed scope.

## Citation and archival record

Zenodo DOI for this software release:

[https://doi.org/10.5281/zenodo.21936408](https://doi.org/10.5281/zenodo.21936408)

## License and third-party components

HistCFM is released under `GPL-3.0-only` and includes code derived from [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST). Third-party attribution, provenance, licensing information, and modification notices are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [provenance](docs/provenance.md), and [licensing](docs/licensing.md). HistCFM does not redistribute UNI source code or model weights.

## Acknowledgements

HistCFM builds upon [GHIST](https://github.com/SydneyBioX/GHIST). We thank the GHIST authors for making their implementation publicly available.

The histology backbone follows the U-Net 3+ implementation included in GHIST, which references [avBuffer/UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth). We acknowledge the authors of [U-Net 3+](https://github.com/ZJUGiveLab/UNet-Version) and its PyTorch implementation.

We also thank the authors of [UNI](https://github.com/mahmoodlab/UNI) for the pretrained foundation model used to derive histology features in the HistCFM experiments.

Users should obtain all external resources from their official repositories and comply with the corresponding access and licensing terms.
