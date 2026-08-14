# HistCFM provenance

This document records how the public HistCFM package relates to the research
implementation, official GHIST, and external projects. It preserves technical
source attribution without exposing internal paths, experiment outputs, or
private data. Licensing conclusions are summarized in
[licensing.md](licensing.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Source references

The paper experiments used a non-Git research working tree containing GHIST
code, HistCFM additions, and several research iterations. A post-study
extracted repository was used as a cross-checking reference. The public release
was assembled selectively and does not copy either research tree wholesale or
inherit its Git history.

| Source | Reference used for provenance | Role |
| --- | --- | --- |
| [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST) | Default branch `main`; audited commit `917456be305fc82e92293ea272812e79675e821c` | Direct upstream for the GHIST-derived model, backbone, components, data, image-I/O, training, and inference code |
| [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST) | Closest matching commit `14bf60f92fadab6985e5c3f9649760f63798cd75`; relevant paths unchanged through the audited commit | Closest official content range for the priority files in the research working tree |
| [avBuffer/UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth) | Default branch `master`; audited commit `263534e4a48964e907324622b14b90f1c3b4270d` | Upstream implementation identified by GHIST's backbone source comment |
| [ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version) | Paper-linked history including `4a3568c1` (`UNet_3Plus.py`), `4c18d2e9` (`layers.py`), and `c32741b9` (`init_weights.py`) | Earlier paper-linked U-Net 3+ implementation retained for attribution |
| [MahmoodLab/UNI](https://github.com/mahmoodlab/UNI) | External project; no source snapshot is vendored | Authorized external model access used to prepare real morphology features |

Official GHIST had no release tag for the matched source range. The commit IDs
above therefore provide the reproducible code references.

## GHIST content comparison

Static file comparison established the following relationship between the
research working tree and official GHIST:

| Research path | Official GHIST relationship |
| --- | --- |
| `model/model.py` | Byte-identical from `14bf60f9` through the audited GHIST commit |
| `model/modules.py`, `model/backbone.py`, `model/layers.py`, `model/intialisation.py` | Byte-identical to the audited GHIST files; earlier history differs only in line endings where noted during comparison |
| `dataio/utils.py` | Byte-identical to official GHIST |
| `dataio/dataset_input.py` | GHIST-derived local variant with defensive data handling and normalization changes |
| `utils/utils.py` | GHIST-derived local variant with a CPU device fallback |

The formal HistCFM framework evolved from the GHIST `Framework` structure.
The research `model/model_new_stflow_uniB.py` retained GHIST backbone pooling,
cell embeddings, composition/cell-type heads, and return conventions while
adding HistCFM conditional flow matching, morphology-feature fusion and hint
logic, spatial state, numerical guards, and SONRM-facing representations.

## Research-to-release file mapping

Research-stage names appear here only as provenance identifiers. Public users
interact with the formal `histcfm` package and `HistCFM` class.

| Research source | Public release file | Classification | Release treatment |
| --- | --- | --- | --- |
| `model/model_new_stflow_uniB.py` | `src/histcfm/models/histcfm.py` | GHIST-derived framework with HistCFM additions | Formal `HistCFM` class, package imports, provenance header, and internal `Framework` compatibility alias; model computation retained |
| `model/modules.py` | `src/histcfm/models/components.py` | Direct GHIST-derived component code | Package organization and provenance header; layer definitions retained |
| `model/backbone.py` | `src/histcfm/models/backbone.py` | Direct GHIST-derived backbone | Package import and provenance header; GHIST's avBuffer reference retained |
| `model/layers.py` | `src/histcfm/models/layers.py` | Direct GHIST-derived backbone helpers | Package-relative initialization import and provenance header |
| `model/intialisation.py` | `src/histcfm/models/initialization.py` | Direct GHIST-derived initialization helpers | Corrected filename/import and provenance header; initialization functions retained |
| `dataio/dataset_input_uniA.py` | `src/histcfm/data/dataset.py` | GHIST-derived cell-level dataset with HistCFM research changes | Formal `HistCFMDataset`, explicit paths, strict identifiers/genes, shared patch keys, controlled normalization, and delayed optional augmentation import |
| `dataio/utils.py` | `src/histcfm/data/image_io.py` | Direct GHIST-derived image loading | Delayed TIFF/image backends, typing, and function-level provenance |
| `model/flow_expression_new_stflow.py` | `src/histcfm/models/flow.py` | HistCFM-specific | Package imports and documentation; endpoint-flow behavior retained |
| `model/flow_denoiser_new_stflow.py` | `src/histcfm/models/flow_denoiser.py` | HistCFM-specific | Documentation and package organization |
| `model/priors_new_stflow.py` | `src/histcfm/models/priors.py` | HistCFM-specific | Documentation and package organization |
| `fusion_block_new_stflow_uniA.py` | `src/histcfm/models/fusion.py` | HistCFM-specific UNI-feature integration | Formal package organization; fusion computation retained |
| `uni_feature_provider_new_stflow_uniA.py` | `src/histcfm/features/uni.py` | HistCFM-specific external-feature integration | Restricted to explicit precomputed JSON+NPY input; encoder loading, path mutation, downloads, and pickle index removed |
| `loss_sonrm_new_stflow_uniB.py` | `src/histcfm/losses/sonrm.py` | HistCFM-specific | Package-relative imports and documentation; SONRM computation retained |
| `utils/graph_new_stflow.py` | `src/histcfm/graph.py` | HistCFM-specific | Package organization and documentation |
| `train_new_stflow_uniB.py` | `src/histcfm/train.py` | GHIST-derived research entry with HistCFM changes | Strict YAML configuration, preflight, caller-owned outputs, metadata checkpoint, formal names, and removal of internal path/fold discovery; training math and loss order retained |
| `inference_new_stflow_uniB.py` | `src/histcfm/inference.py` | GHIST-derived research entry with HistCFM changes | One explicit schema-3 checkpoint, strict structural checks, normalization reuse, stable outputs, and no embedded epoch selection or evaluation |
| Dataset-specific evaluation and figure scripts | `src/histcfm/evaluate.py` | HistCFM-specific release implementation | New aligned-table evaluator; no dataset-bound script was copied |

## Release-authored public components

The following components were written for the formal release rather than
copied from an upstream package or paper experiment script:

| Public component | Purpose |
| --- | --- |
| `src/histcfm/config.py` | Strict configuration dataclasses, parsing, finite-value checks, and resolved configuration output |
| `src/histcfm/checkpoint.py` | Schema-3 checkpoint metadata, trusted local loading boundary, and version/configuration binding |
| `src/histcfm/data/splitting.py` | Shared strict train/validation patch-coordinate generation |
| `src/histcfm/data/validation.py` | Input, identifier, feature-store, split, and checkpoint-aware preflight validation |
| `src/histcfm/cli.py`, `src/histcfm/__main__.py` | Thin delayed-import CLI and module entry point |
| `pyproject.toml`, `environment.yml` | Package metadata and the independent Linux/CUDA environment definition |
| `tests/` | Synthetic and static software-contract tests |
| `docs/` | Public input, environment, architecture, reproducibility, licensing, and provenance documentation |

These HistCFM-specific and release-authored components are covered by the
author team's approval to publish the HistCFM-specific contributions under
`GPL-3.0-only`.

## U-Net 3+ attribution chain

The direct source of the public `backbone.py`, `layers.py`, and
`initialization.py` files is official GHIST. GHIST identifies
[avBuffer/UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth) in its
backbone. The U-Net 3+ paper links
[ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version), whose
earlier corresponding files have substantial structural and line-level
similarity to the avBuffer implementation.

The resulting attribution path is recorded as:

```text
ZJUGiveLab/UNet-Version (paper-linked implementation)
  -> avBuffer/UNet3plus_pth (implementation identified by GHIST)
  -> SydneyBioX/GHIST (direct source used by HistCFM)
  -> HistCFM backbone-family files
```

HistCFM obtained the relevant backbone code from the official GHIST repository
and redistributes the GHIST-derived files, including HistCFM modifications,
under GPL-3.0-only, consistent with the license published by GHIST. GHIST
identifies the avBuffer U-Net 3+ implementation as an upstream source for its
backbone. HistCFM preserves attribution to GHIST and the referenced upstream
projects and does not claim original authorship of upstream code.

HistCFM does not claim separate written authorization from avBuffer,
ZJUGiveLab, or another earlier U-Net 3+ implementation. See
[licensing.md](licensing.md) for the public redistribution statement.

## Formal release scope

The public repository contains:

- one formal cell-level `HistCFM` model;
- one formal dataset and strict input/preflight implementation;
- training, single-checkpoint inference, aligned-table evaluation, and CLI
  entries;
- schema-versioned checkpoint support without a distributed checkpoint;
- configuration templates and an independently defined environment;
- deterministic synthetic smoke-demo inputs and interface-compatible synthetic
  morphology features; and
- automated tests and public documentation.

The formal configuration preserves the named fields and numerical defaults
documented by `configs/histcfm.yaml`. Research dataset/fold names, experiment
timestamps, absolute paths, multi-checkpoint ranking, paper-figure scripts,
and dataset-specific result aggregation are not public runtime interfaces.

## External and excluded material

The release does not contain:

- the complete GHIST repository, its Git history, preprocessing scripts,
  tutorial data, processed-data bundles, checkpoints, or results;
- Hover-Net code or weights;
- UNI source code, encoder code, weights, checkpoints, download code, or real
  UNI-generated features;
- Breast Sample 1, Breast Sample 2, Melanoma, Visium breast cancer, or another
  real biological dataset;
- patient images, real expression/mask/annotation tables, or sample identifiers;
- paper HistCFM checkpoints, private checkpoints, predictions, metrics, result
  archives, or server logs; or
- research ablations, spot-level branches, dataset-specific launchers, and
  figure-generation workflows.

Raw-data preparation is referenced through official GHIST resources, while
real morphology features are prepared separately through authorized UNI
access. External links identify provenance and access locations; linked
material remains subject to its provider's terms.

The committed demo was generated with fixed seed `20260813` without input from
the research datasets, GHIST, UNI, or another model. Its detailed provenance
and separate data dedication are in
[examples/demo/DATA_PROVENANCE.md](../examples/demo/DATA_PROVENANCE.md).

## Cross-references

- [Licensing statement](licensing.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [Architecture](architecture.md)
- [Real-data preparation](data_preparation.md)
- [Input format](input_format.md)
- [Precomputed morphology features](uni_features.md)
- [Reproducibility](reproducibility.md)
