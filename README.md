# HistCFM

## Paper

**HistCFM: Semantics-Guided Conditional Flow Matching for Single-Cell Spatial Gene Expression Prediction from Histology**

**Authors:** Yanbing Xiao, Chunyang Meng, Anping Xiong<sup>*</sup>, Yi Jiang, Zipeng Wang, Wei Cheng, Xiang Ao, and Yuansong Zeng<sup>**</sup>

**Corresponding author:** Yuansong Zeng ([zengys@cqu.edu.cn](mailto:zengys@cqu.edu.cn))

HistCFM is a cell-level histology-to-gene-expression model that combines tissue morphology, spatial neighborhoods, and conditional flow matching. It operates on processed cell-level inputs and precomputed morphology features.

The package provides one model and consistent command-line workflows for training, inference, and evaluation. See [provenance](docs/provenance.md) for the complete research-to-release source map.

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

HistCFM links to the official [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST) preprocessing resources rather than copying the complete pipeline. Users with HistCFM-ready inputs do not need to install or rerun GHIST.

HistCFM does not include or download UNI code or weights. Users preparing real features must obtain access through the official [MahmoodLab/UNI](https://github.com/mahmoodlab/UNI) channel and follow its terms.

See [real-data preparation](docs/data_preparation.md), the [input format](docs/input_format.md), and [precomputed morphology features](docs/uni_features.md).

| Input | Purpose |
| --- | --- |
| Histology TIFF | Registered RGB histology |
| Nucleus mask | Cell/nucleus instance labels; `0` is background |
| Matched nuclei CSV | Cell matching and nucleus-size metadata |
| Expression CSV | Cell-by-gene non-negative raw counts |
| Cell types CSV | Required with cell-type supervision; omitted for unlabeled prediction |
| `uni_index.json` | Canonical patch-key-to-row mapping |
| `uni_features.npy` | Precomputed morphology-feature matrix |

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

- HistCFM `0.1.0` passed all 75 tests and the complete synthetic validate/train/checkpoint/infer/evaluate workflow in the declared independent Linux/CUDA environment.
- The synthetic demo validates software plumbing only; it is not a paper benchmark or biological result.
- Real datasets and paper checkpoints are not distributed here.
- Users prepare real morphology features independently through authorized external-model access.
- See [environment](docs/environment.md), [server validation](docs/server_validation.md), and [reproducibility](docs/reproducibility.md) for detailed scope.

## Citation

The HistCFM citation will be updated when the accompanying manuscript is published. Please also cite the upstream methods used in your workflow:

- Fu et al. “Spatial gene expression at single-cell resolution from histology using deep learning with GHIST.” *Nature Methods* 22, 1900–1910 (2025). <https://doi.org/10.1038/s41592-025-02795-z>
- Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. “Towards a general-purpose foundation model for computational pathology.” *Nature Medicine* (2024). <https://doi.org/10.1038/s41591-024-02857-3>

## License and third-party components

HistCFM is released under `GPL-3.0-only` and is derived in part from [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST). The backbone provenance includes the earlier avBuffer/U-Net 3+ lineage; attribution, license context, modification notices, and source-chain uncertainty are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [provenance](docs/provenance.md), and [licensing](docs/licensing.md). HistCFM does not distribute the UNI model or weights.
