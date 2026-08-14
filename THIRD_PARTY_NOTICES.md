# Third-party notices

This file records third-party attribution and external-resource boundaries for
HistCFM. The repository-level software license is `GPL-3.0-only`; see
[LICENSE](LICENSE). Detailed file provenance is in
[docs/provenance.md](docs/provenance.md), and the licensing statement is in
[docs/licensing.md](docs/licensing.md).

## SydneyBioX/GHIST

- Official repository: <https://github.com/SydneyBioX/GHIST>
- Reference commit: `917456be305fc82e92293ea272812e79675e821c`
- Published repository license: GNU General Public License, version 3

HistCFM contains selected model, data, image-I/O, training, and inference code
derived from GHIST. The relevant HistCFM files identify GHIST and their
modifications; the complete GHIST repository, Git history, preprocessing
scripts, tutorial data, checkpoints, and results are not copied here.

HistCFM obtained the relevant backbone code from the official GHIST repository
and redistributes the GHIST-derived files, including HistCFM modifications,
under GPL-3.0-only, consistent with the license published by GHIST. GHIST
identifies the avBuffer U-Net 3+ implementation as an upstream source for its
backbone. HistCFM preserves attribution to GHIST and the referenced upstream
projects and does not claim original authorship of upstream code.

Please cite:

Fu et al., “Spatial gene expression at single-cell resolution from histology
using deep learning with GHIST,” *Nature Methods* 22, 1900–1910 (2025),
<https://doi.org/10.1038/s41592-025-02795-z>.

## U-Net 3+ lineage

- GHIST-referenced PyTorch implementation:
  <https://github.com/avBuffer/UNet3plus_pth>
- Paper-linked U-Net 3+ implementation:
  <https://github.com/ZJUGiveLab/UNet-Version>

The direct source used by HistCFM for the public backbone-family files is
official GHIST. HistCFM retains attribution to GHIST, avBuffer, and U-Net 3+,
and does not claim original authorship of the upstream code. HistCFM does not
claim separate written authorization from avBuffer, ZJUGiveLab, or another
earlier U-Net 3+ implementation. These notices do not constitute an independent
legal review or legal guarantee.

## MahmoodLab/UNI

- Official repository: <https://github.com/mahmoodlab/UNI>
- Official model-access page: <https://huggingface.co/MahmoodLab/uni>

UNI is an external resource used to prepare morphology features for the
HistCFM experiments. Users must obtain authorized access from the official
provider and comply with the applicable upstream terms. HistCFM does not
redistribute UNI source code, encoder code, weights, checkpoints, download
utilities, or real UNI-generated features.

The committed synthetic demo contains interface-compatible synthetic
morphology features generated without UNI or another model.

Please cite:

Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al., “Towards a
general-purpose foundation model for computational pathology,” *Nature
Medicine* (2024), <https://doi.org/10.1038/s41591-024-02857-3>.

## Public datasets

The HistCFM study used publicly accessible 10x Genomics Xenium and Visium data.
Their source pages are linked in the root [README](README.md#datasets-used-in-this-study).
Those datasets are not copied into this repository and remain governed by the
terms of their original provider.

## Excluded external artifacts

This repository does not contain real patient data, real expression matrices,
real UNI-generated feature arrays, UNI or Hover-Net weights, paper HistCFM
checkpoints, GHIST tutorial artifacts, private checkpoints, predictions,
metrics, results, or server logs. External links are supplied for attribution
and reproducibility and do not change the terms of the linked material.
