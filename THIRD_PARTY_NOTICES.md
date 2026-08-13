# Third-party notices

This inventory identifies material and external dependencies relevant to
HistCFM. Attribution is not a substitute for a license or permission.

## GHIST

- Official source: <https://github.com/SydneyBioX/GHIST>
- Upstream license found: GNU General Public License, version 3
- Relationship: HistCFM model, data, image-I/O, training, and inference paths
  contain selected code modified/derived from GHIST. Source files identify the
  upstream and release-side modifications.
- Redistribution basis: the HistCFM author team elected to redistribute these
  GHIST-derived components in reliance on the GNU GPL v3 published by the
  official SydneyBioX/GHIST repository, while retaining attribution,
  provenance, modification, license, and warranty notices.
- Not included: GHIST's full repository, preprocessing scripts, tutorial data,
  checkpoint, results, or Git history.
- Citation: Fu et al., “Spatial gene expression at single-cell resolution from
  histology using deep learning with GHIST,” *Nature Methods* 22, 1900–1910
  (2025), <https://doi.org/10.1038/s41592-025-02795-z>.

## U-Net 3+ / UNet3+

- Paper-linked implementation: <https://github.com/ZJUGiveLab/UNet-Version>
- Direct source cited by GHIST's backbone:
  <https://github.com/avBuffer/UNet3plus_pth>
- Relationship: the direct source used for
  `src/histcfm/models/backbone.py`, `layers.py`, and `initialization.py` is
  official GHIST. GHIST cites avBuffer in its backbone; file comparison also
  supports the documented probable relationship to the paper-linked U-Net 3+
  implementation.
- License finding: no explicit redistribution license was found in the
  audited ZJUGiveLab or avBuffer repositories. GHIST's GPL file does not by
  itself prove permission to relicense separately sourced code.
- Status: the HistCFM author team has made the publication decision to rely on
  official GHIST's GPL v3 for its direct GHIST copies. HistCFM has not obtained
  and does not claim separate written authorization from avBuffer,
  ZJUGiveLab, or another earlier UNet3+ implementation. The earlier source-
  chain uncertainty remains disclosed; attribution is not represented as an
  independent license or permission.

## UNI

- Official source: <https://github.com/mahmoodlab/UNI>
- Original UNI model access: <https://huggingface.co/MahmoodLab/uni>
- Upstream terms: the official repository states CC BY-NC-ND 4.0 and gated
  model-access conditions; users must review the current upstream text.
- Relationship: real-data HistCFM can consume separately prepared original
  UNI patch features through an offline JSON+NPY interface.
- Not included: UNI source, encoder, weights, checkpoint, download code, or
  real UNI-generated features.
- Citation: Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al.,
  “Towards a general-purpose foundation model for computational pathology,”
  *Nature Medicine* (2024), <https://doi.org/10.1038/s41591-024-02857-3>.

The synthetic morphology-feature matrix under `examples/demo/data/` was
generated without UNI or another model. Its formal filenames describe the
HistCFM interface, not its provenance.
