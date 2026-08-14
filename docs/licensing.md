# Licensing statement

This page summarizes the licensing and external-resource boundaries of the
public HistCFM repository. It is documentation rather than legal advice or an
independent legal review. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
for third-party attribution and [provenance.md](provenance.md) for the detailed
research-to-release source map.

## Project license

HistCFM is released under `GPL-3.0-only`. The repository
[LICENSE](../LICENSE) contains the GNU General Public License, version 3. The
license applies to the HistCFM software as distributed, including authorized
HistCFM-specific contributions and modified GHIST-derived components.

Recipients who copy, modify, or redistribute covered software must comply with
the GPL's full terms, including applicable source, license, notice, change-
marking, and warranty-disclaimer requirements. The repository does not add a
non-commercial restriction to the HistCFM software license.

The official GHIST source used for comparison was
[SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST), default branch `main`,
at commit `917456be305fc82e92293ea272812e79675e821c`. The closest official
content match for the priority local GHIST files begins at commit
`14bf60f92fadab6985e5c3f9649760f63798cd75`; those paths were unchanged through
the audited commit. GHIST publishes the GNU GPL version 3 in its root
`LICENSE`, which is the redistribution basis recorded for GHIST-derived code
in HistCFM.

## HistCFM-specific contributions

The HistCFM author team confirmed:

> The HistCFM author team confirms that the HistCFM-specific additions were
> created by the author team and approves their public release under
> GPL-3.0-only.

HistCFM-specific contributions include the conditional flow-matching modules,
priors, feature-fusion integration, offline precomputed-feature provider,
SONRM loss, graph support, strict configuration and validation additions,
checkpoint metadata, general evaluator, CLI/package integration, tests, and
documentation. These components are identified more precisely in
[provenance.md](provenance.md).

The synthetic smoke-demo image, mask, tables, counts, and precomputed
morphology-feature matrix are independently generated release material. Their
separate CC0-1.0 dedication and generation details are recorded in
[examples/demo/DATA_PROVENANCE.md](../examples/demo/DATA_PROVENANCE.md).

## GHIST-derived components

HistCFM contains selected code obtained from the official GHIST repository,
not a wholesale copy of GHIST. The following public component groups are
GHIST-derived and include HistCFM packaging, interface, validation, or model
changes where noted:

| HistCFM component | GHIST relationship | HistCFM treatment |
| --- | --- | --- |
| `src/histcfm/models/histcfm.py` | Derived from the GHIST cell-level framework | Adds the formal `HistCFM` API and HistCFM CFM, feature-hint, spatial, and SONRM behavior while retaining a compatibility alias |
| `src/histcfm/models/components.py` | Direct GHIST neural-network components | Package organization and provenance header |
| `src/histcfm/models/backbone.py`, `layers.py`, `initialization.py` | Directly obtained from the GHIST backbone family | Package-relative imports, corrected filename, provenance and modification notices; numerical structure retained |
| `src/histcfm/data/dataset.py`, `image_io.py` | Derived from GHIST cell-level data and image-loading code | Formal paths, strict alignment/preflight, shared patch keys, controlled normalization output, and delayed image backends |
| `src/histcfm/train.py`, `inference.py` | Derived from GHIST-based cell-level research entries | Formal configuration and output ownership, strict checkpoint handling, stable single-run interfaces, and removal of dataset/fold/server-path discovery |

HistCFM obtained the relevant backbone code from the official GHIST repository
and redistributes the GHIST-derived files, including HistCFM modifications,
under GPL-3.0-only, consistent with the license published by GHIST. GHIST
identifies the avBuffer U-Net 3+ implementation as an upstream source for its
backbone. HistCFM preserves attribution to GHIST and the referenced upstream
projects and does not claim original authorship of upstream code.

The HistCFM author team approved publication of these GHIST-derived components
in reliance on the GPL version 3 published by official GHIST. HistCFM does not
claim separate written authorization from avBuffer, ZJUGiveLab, or another
earlier U-Net 3+ implementation.

## U-Net 3+ attribution

The direct source used by HistCFM for `backbone.py`, `layers.py`, and
`initialization.py` is official GHIST. GHIST's backbone cites
[avBuffer/UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth). The
U-Net 3+ paper-linked implementation is available from
[ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version).

HistCFM retains these project links, the GHIST source identification, and
modification notices in the relevant files and public documentation.
Attribution records origin; it is not represented as additional authorization
from those earlier projects.

## UNI and external resources

[MahmoodLab/UNI](https://github.com/mahmoodlab/UNI) is an external foundation
model used to prepare morphology features for the HistCFM experiments. Users
must obtain UNI code and model access through the official upstream channels
and comply with the terms that apply to their access and use.

The HistCFM runtime consumes only a user-provided `uni_index.json` and
`uni_features.npy`. This repository contains no UNI encoder implementation,
UNI source repository, model weights, checkpoint, download utility, or real
UNI-generated feature matrix. The demo feature matrix is deterministic
synthetic data and is not output from UNI or another model.

GHIST preprocessing, Hover-Net resources, optional stain augmentation
software, and public biological datasets are also external. Links are provided
for reproducibility, but those external projects and datasets remain subject
to their own terms.

## Datasets, features, and checkpoints not redistributed

The repository does not redistribute:

- Breast Sample 1, Breast Sample 2, Melanoma, Visium breast cancer, or another
  real biological dataset;
- patient images, real expression matrices, masks, cell annotations, or sample
  identifiers;
- real UNI-derived feature arrays or a UNI checkpoint;
- a paper HistCFM checkpoint, private training checkpoint, prediction, metric,
  result archive, or server log; or
- GHIST tutorial data, processed-data bundles, checkpoints, or the complete
  GHIST preprocessing implementation.

Public dataset locations and preparation boundaries are documented in the
root [README](../README.md) and [data_preparation.md](data_preparation.md).
Providing an external link does not place the linked material under the
HistCFM software license.
