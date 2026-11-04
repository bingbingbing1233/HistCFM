# Code and data sources

HistCFM includes code derived from [SydneyBioX/GHIST](https://github.com/SydneyBioX/GHIST),
with HistCFM additions for conditional flow matching, morphology-feature fusion,
SONRM, configuration, training and evaluation. The relevant source files preserve
upstream attribution and modification notices. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [licensing.md](licensing.md).

The backbone was obtained through GHIST, whose implementation references
[avBuffer/UNet3plus_pth](https://github.com/avBuffer/UNet3plus_pth).
The paper-linked U-Net 3+ project is
[ZJUGiveLab/UNet-Version](https://github.com/ZJUGiveLab/UNet-Version).
These upstream components are not original HistCFM contributions.

[UNI](https://github.com/mahmoodlab/UNI) is an external model used to prepare
real morphology features. Its source and weights are not distributed here.
Users obtain authorized access separately.

The included demo image, mask, expression values, cell types and morphology
features are synthetic. They were generated without research data or a UNI
encoder. See [demo data provenance](../examples/demo/DATA_PROVENANCE.md).
No real datasets, trained weights, formal experiment package or ablation code
is included in this update.
