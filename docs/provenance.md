# Provenance audit

The formal research implementation came from the non-Git working source directory used for the experiments. The post-study extracted repository is a cross-checking reference, not automatic evidence of the executed version. This release repository does not copy either old repository wholesale or inherit their Git history.

This record distinguishes local HistCFM additions, code derived from GHIST, and third-party material. It records technical source relationships only; redistribution decisions are tracked in `docs/licensing.md`.

## Upstream audit snapshot (2026-08-12)

### GHIST

- official repository: <https://github.com/SydneyBioX/GHIST>
- audited default branch: `main`
- audited HEAD: `917456be305fc82e92293ea272812e79675e821c`
- tags found: none
- remote branches found by the audit clone: `origin/main` only
- commits reachable from `main`: 22
- root license: GNU General Public License, version 3
- root `LICENSE` first appeared at `4bba2a6a0165aeda225a07aa3195c1457e34bf02` on 2024-07-02; it was removed and re-added during the 2025-03-15 repository replacement/restructure sequence

The nine priority local files are collectively closest to official commit `14bf60f92fadab6985e5c3f9649760f63798cd75` (2025-05-09, `restructured`) and the unchanged target-file range through audited HEAD `917456be...` (2025-09-16). No commit after `14bf60f...` changes any of those nine paths. There is no tag with which to name this range.

### UNet3+

The source comment in both the local and official-GHIST backbone identifies:

- repository: <https://github.com/avBuffer/UNet3plus_pth>
- audited default branch: `master`
- audited HEAD: `263534e4a48964e907324622b14b90f1c3b4270d`
- tags found: none
- commits on `master` at audit time: six, from `d9ca06dc...` through `263534e4...`

No `LICENSE`, `COPYING`, `NOTICE`, `CITATION`, or `AUTHORS` file was found at that upstream HEAD or in the available commit trees. The README's academic-communication wording is not an explicit software copy, modification, or redistribution grant. GHIST's inclusion of these files does not by itself establish that GHIST can relicense the upstream material.

## Local-to-official GHIST comparison

Hashes below refer to the audited working-tree bytes. Historical comparisons also normalized CRLF/LF where noted.

| Local working-source file | Official GHIST path | Relationship | Closest official commit/range | Local modification and source marker |
| --- | --- | --- | --- | --- |
| `model/model.py` | `model/model.py` | Byte-for-byte identical | First exact at `14bf60f...`; unchanged through `917456b...` | No file-level copyright/license header |
| `model/modules.py` | `model/modules.py` | Byte-for-byte identical | Exact from `982f833...` through HEAD; same code content was present at the initial commit with different line endings | No third-party source marker found; defines `CrossAttention`, `Embed`, `MLP`, and `MLPSoftmax` |
| `model/backbone.py` | `model/backbone.py` | Byte-for-byte identical | Exact from `982f833...` through HEAD; same normalized content was present initially | First line attributes the implementation to `avBuffer/UNet3plus_pth`; adapted third-party code |
| `model/layers.py` | `model/layers.py` | Byte-for-byte identical | Exact from `982f833...` through HEAD; same normalized content was present initially | No independent header; comparison ties it to UNet3+ |
| `model/intialisation.py` | `model/intialisation.py` | Byte-for-byte identical | Exact from `982f833...` through HEAD; same normalized content was present initially | Renamed exact UNet3+ initialization implementation; no independent header |
| `model/adjustments.py` | `model/adjustments.py` | Content-equivalent; local file only adds a final newline | All available official versions | No substantive local modification identified |
| `dataio/dataset_input.py` | `dataio/dataset_input.py` | Local modified version; closest to `14bf60f...` through HEAD | `14bf60f...`–`917456b...` | Adds defensive fold-range behavior, missing-gene/cell-type handling, finite/nonnegative expression cleanup, normalization fallback, standard-deviation clipping, and safer cell-type conversion |
| `dataio/utils.py` | `dataio/utils.py` | Byte-for-byte identical | Exact from `982f833...` through HEAD; same normalized content was present initially | No file-level copyright/license header |
| `utils/utils.py` | `utils/utils.py` | Local modified version | `14bf60f...`–`917456b...` | Adds CPU fallback to `get_device`; otherwise only removes a blank line |

The closest-version conclusion is based on content, not timestamps: six files are exact at HEAD, one differs only at EOF, and the two substantively modified files have their smallest diffs against the `14bf60f...`–HEAD form. The official repository has no tag, so no release tag can be claimed.

## HistCFM framework lineage

Official GHIST does not contain `model/model_new_stflow_uniB.py` at HEAD or in any inspected commit tree. Static comparison establishes this chain:

| Local file | Evidence-based classification | Material relationship |
| --- | --- | --- |
| `model/model.py` | Direct GHIST inheritance | Byte-identical to official GHIST from `14bf60f...` through HEAD; supplies the Framework skeleton, backbone pooling, cell embeddings, composition/cell-type heads, and return conventions |
| `model/model_new_stflow.py` | Modified/derived from GHIST | Replaces original expression behavior with the flow path |
| `model/model_new_stflow_uniA.py` | Modified/derived from GHIST plus HistCFM additions | Adds UNI provision, fusion, hint conditioning, coordinates, and KNN behavior |
| `model/model_new_stflow_uniB.py` | HistCFM-modified GHIST derivative | Retains the GHIST Framework structure and adds CFM, UNI fusion/hints, spatial graph state, NaN guards, and SONRM-facing embeddings/coordinates |

Compared directly with official `model/model.py`, `model_new_stflow_uniB.py` retains the `Framework`, `Backbone`, `Embed`, composition and cell-type structures but replaces substantial expression-path logic (163 inserted and 200 deleted lines in the static diff). It must be labeled “modified/derived from GHIST,” with the modification date and description required by the applicable GPL conditions.

## UNet3+ file-level comparison

Comparisons ignored CRLF/LF-only differences where applicable.

| Local working-source file | UNet3+ file | Relationship |
| --- | --- | --- |
| `model/backbone.py` | `unet/UNet3Plus.py` | Modified copy of the first `UNet3Plus` implementation: class renamed to `Backbone`; defaults and imports changed; output head renamed; unused imports/attributes removed; sigmoid output replaced by raw `seg, hd1, h1` tuple. Encoder/decoder structure and calculations remain substantially the same. |
| `model/layers.py` | `unet/layers.py` | Same implementation after newline normalization except the import changes from `.init_weights` to `.intialisation`; retains `unetConv2`, `unetUp`, and `unetUp_origin`. |
| `model/intialisation.py` | `unet/init_weights.py` | Identical after newline normalization; local file is renamed/misspelled. |

The local files are also byte-identical to the corresponding current GHIST files, but this does not cure the separate UNet3+ permission problem. Technical compatibility requires preserving the local raw outputs, module hierarchy, parameter shapes, state-dict keys, and initialization order.

## Core migration batch 1

All seven working-source/extracted-reference pairs were byte-for-byte identical before migration. No same-named file, earlier version, unique class/function name, or comparable CFM/UNI/SONRM/graph implementation was found in the official GHIST HEAD or inspected commit trees. They are therefore HistCFM additions relative to official GHIST. This comparison does not by itself prove that every line is independently authored or clear all other third-party rights.

| Release file | Working-source file | Release-side changes | Revised origin classification |
| --- | --- | --- | --- |
| `src/histcfm/models/flow.py` | `model/flow_expression_new_stflow.py` | Package imports, documentation, formatting | HistCFM addition relative to GHIST |
| `src/histcfm/models/flow_denoiser.py` | `model/flow_denoiser_new_stflow.py` | Documentation and formatting | HistCFM addition relative to GHIST |
| `src/histcfm/models/priors.py` | `model/priors_new_stflow.py` | Documentation and formatting | HistCFM addition relative to GHIST |
| `src/histcfm/models/fusion.py` | `fusion_block_new_stflow_uniA.py` | Documentation and formatting | HistCFM UNI-integration addition relative to GHIST |
| `src/histcfm/features/uni.py` | `uni_feature_provider_new_stflow_uniA.py` | Restricted to precomputed features; removed encoder/weight-loading, path mutation and pickle index; uses shared strict JSON+NPY validation and explicit paths | HistCFM UNI-integration addition; no UNI encoder code or weights copied |
| `src/histcfm/losses/sonrm.py` | `loss_sonrm_new_stflow_uniB.py` | Package-relative import, documentation, formatting | HistCFM addition relative to GHIST |
| `src/histcfm/graph.py` | `utils/graph_new_stflow.py` | Documentation and formatting | HistCFM addition relative to GHIST |

The HistCFM author team has confirmed that it created these HistCFM-specific additions and has approved their public release under GPL-3.0-only. UNI code, weights, and real feature data remain outside the release.

## Candidate batch 2

| Local file | Official GHIST relationship | HistCFM modification | Third-party source | Technical status | License status |
| --- | --- | --- | --- | --- | --- |
| `model/model_new_stflow_uniB.py` | Absent officially; derived from byte-identical local/official `model/model.py` | CFM, UNI hint/fusion, coordinates/KNN, NaN guards, SONRM state | No additional copied source identified inside this file | Migrated as the formal `HistCFM` class with its model attributes and forward contract retained | GPL-3.0-only with GHIST attribution and modification/date notice; HistCFM authorship and release authorization confirmed |
| `model/modules.py` | Byte-identical to official GHIST | None | No separate source marker found | Ready | GHIST GPL v3 permits migration with license, source, notice, and whole-work obligations |
| `model/backbone.py` | Byte-identical to official GHIST; direct release source is GHIST | GHIST/local adaptation is already present | GHIST cites `avBuffer/UNet3plus_pth`; probable earlier ZJUGiveLab relationship disclosed | Migrated | Author team relies on official GHIST GPL v3; no separate earlier authorization claimed |
| `model/layers.py` | Byte-identical to official GHIST; direct release source is GHIST | Import rename relative to UNet3+ | Near-copy in the disclosed earlier source chain | Migrated | Same GHIST GPL v3 reliance and no-additional-authorization disclosure |
| `model/intialisation.py` | Byte-identical to official GHIST; direct release source is GHIST | Filename rename relative to UNet3+ | Exact content in the disclosed earlier source chain | Migrated | Same GHIST GPL v3 reliance and no-additional-authorization disclosure |

No file in this batch was migrated during that earlier audit. All five files
were subsequently migrated in the approved complete-model stage recorded
below. The author team later approved publication of the GHIST-derived files
in reliance on official GHIST's GPL v3. That decision does not claim separate
permission from avBuffer, ZJUGiveLab, or another earlier implementation.

## GHIST preprocessing boundary

The official reference path is:

- overview/tutorial: <https://github.com/SydneyBioX/GHIST/blob/main/tutorials/1_data_preprocessing.ipynb>
- preprocessing scripts: <https://github.com/SydneyBioX/GHIST/tree/main/data_processing>
- official demo layout: <https://github.com/SydneyBioX/GHIST/tree/main/data_demo>

The official pipeline covers Xenium nucleus-mask preparation, Xenium cell-by-gene matrix preparation, H&E patch creation and external Hover-Net segmentation, nucleus/cell correspondence, and filtered expression/matching outputs. The tutorial uses a subset of 10x Genomics Breast Cancer In Situ Sample 2 and refers to external Hover-Net code/checkpoints. It does not provide H&E-to-Xenium alignment code.

HistCFM documentation should describe the boundary as:

```text
Raw data -> GHIST-referenced preprocessing
HistCFM-ready inputs -> HistCFM demo/training/inference/evaluation
```

Future HistCFM documentation should link to the official GHIST tutorial rather than copy its preprocessing scripts. The tutorial contains saved execution outputs and environment-specific paths, and the demo data/checkpoint have no separately identified redistribution terms in the GHIST repository; none should be copied without a separate data/model audit.

## UNet3+ upstream lineage audit (2026-08-12)

This section originated as a source-only audit. The HistCFM author team later approved public release of the GHIST-distributed files under GPL-3.0-only in reliance on the license published by official GHIST. The unresolved earlier source chain remains disclosed; the decision is not evidence of, and does not claim, separate permission from avBuffer or ZJUGiveLab.

### Target-file history

The UNet 3+ paper links its implementation to <https://github.com/ZJUGiveLab/UNet-Version>. That repository describes itself as the code for the ICASSP 2020 paper. Its target files predate the corresponding files in `avBuffer/UNet3plus_pth`:

| Repository and path | Earliest path history found | Evidence at introduction |
| --- | --- | --- |
| ZJUGiveLab `models/UNet_3Plus.py` | `4a3568c1...`, 2020-01-29, `Create UNet_3Plus` (later renamed to `.py`) | No source, copyright, or license header |
| ZJUGiveLab `models/layers.py` | `4c18d2e9...`, 2020-04-22, `Create layers` (later renamed and updated) | No source, copyright, or license header |
| ZJUGiveLab `models/init_weights.py` | `c32741b9...`, 2020-04-22, `Create init_weights` (later renamed) | No source, copyright, or license header |
| avBuffer `unet/UNet3Plus.py` | `f06bd4c8...`, 2020-07-09, `upload new code` | Encoding header only; no source/copyright statement |
| avBuffer `unet/layers.py` | Same `f06bd4c8...` bulk upload | Imports only; no source/copyright statement |
| avBuffer `unet/init_weights.py` | Same `f06bd4c8...` bulk upload | Imports only; no source/copyright statement |

The three avBuffer targets entered together in a bulk commit. The avBuffer README cites the U-Net, UNet++, and UNet 3+ papers but does not identify ZJUGiveLab as the code source. The repositories have separate Git histories and avBuffer is not presented as a GitHub fork. Commit metadata therefore does not prove a copy operation. The dates and file-level comparisons below nevertheless make ZJUGiveLab the high-confidence probable immediate code source; this remains an evidence-based inference rather than an explicit author statement.

### File-level comparison and probable chain

Comparisons below normalize CRLF/LF and ignore trailing whitespace. “Aligned unchanged” is the number of lines retained by a sequence diff, not a copyright or authorship test.

| Link | Comparison evidence | Material changes |
| --- | --- | --- |
| ZJUGiveLab `UNet_3Plus.py` -> avBuffer `UNet3Plus.py` | 722 aligned unchanged lines; 61 ZJU-only and 54 avBuffer-only lines at audited HEADs | Package-relative imports, class/argument renames, formatting, stored attributes, and a NumPy import; the base, DeepSup, and DeepSup-CGM classes and full-scale decoder structure remain aligned |
| ZJUGiveLab `layers.py` -> avBuffer `layers.py` | 77 aligned unchanged; 11 ZJU-only and 10 avBuffer-only | Initialization import made package-relative plus formatting/comment changes; `unetConv2`, `unetUp`, and `unetUp_origin` remain aligned |
| ZJUGiveLab `init_weights.py` -> avBuffer `init_weights.py` | 59 aligned unchanged; 5 ZJU-only and 1 avBuffer-only | Blank/comment differences; initialization function bodies remain aligned |
| avBuffer base `UNet3Plus` path -> GHIST `model/backbone.py` | 182 aligned unchanged lines against the first avBuffer class; 60 avBuffer-only and 135 GHIST-only | Renamed to `Backbone`, defaults/imports changed, deep-supervision/CGM variants omitted, output head and return contract changed to raw `seg, hd1, h1` |
| avBuffer `layers.py` -> GHIST `model/layers.py` | 86 of 87 lines aligned unchanged | Only functional difference is `.init_weights` -> `.intialisation` import |
| avBuffer `init_weights.py` -> GHIST `model/intialisation.py` | All 60 normalized lines unchanged | Filename renamed/misspelled; function content unchanged |

The most likely current chain is:

```text
ZJUGiveLab/UNet-Version (paper-linked implementation; no license found)
  -> avBuffer/UNet3plus_pth adaptation (no license found; no explicit ZJU attribution)
  -> SydneyBioX/GHIST adaptation (root GPL v3; backbone retains the avBuffer URL)
  -> HistCFM working/extracted copies (no further changes to these three files)
```

The working source, extracted reference, and audited official GHIST copies of `backbone.py`, `layers.py`, and `intialisation.py` match at the file level. HistCFM therefore made no identified change to those three GHIST files. GHIST preserves the avBuffer URL only in `backbone.py`; it does not identify the probable ZJUGiveLab source in these files.

No earlier repository was proven to be the exact source of `unetConv2` or `init_weights`. Similar names and generic initialization patterns are insufficient to complete that part of the chain.

### Candidate upstream and replacement implementations

| Candidate | URL | File/code relationship | First relevant public time found | Explicit repository license | Similarity and upstream assessment |
| --- | --- | --- | --- | --- | --- |
| ZJUGiveLab/UNet-Version | <https://github.com/ZJUGiveLab/UNet-Version> | Corresponding `UNet_3Plus.py`, `layers.py`, and `init_weights.py` | 2020-01-29 / 2020-04-22 | None found at HEAD or in inspected history | Very high file-level alignment and earlier dates; high-confidence probable upstream of avBuffer, but not explicitly acknowledged there |
| avBuffer/UNet3plus_pth | <https://github.com/avBuffer/UNet3plus_pth> | Direct corresponding files named by GHIST | All three added 2020-07-09 | None found at HEAD or in inspected history | Direct upstream of the GHIST adaptation; provenance before avBuffer is undocumented |
| SydneyBioX/GHIST | <https://github.com/SydneyBioX/GHIST> | Local three files are byte-identical to audited GHIST files | Present in GHIST history; closest local source range documented above | GNU GPL v3 root license | Direct local source, but its root license does not establish rights in separately sourced no-license code |
| hamidriasat/UNet-3-Plus | <https://github.com/hamidriasat/UNet-3-Plus> | TensorFlow 2 UNet 3+ implementation; not a file/state-dict match | README identifies an initial February 2023 release | MIT license in the repository | Later independently packaged implementation; not a plausible upstream of the 2020 files and not a drop-in PyTorch replacement |
| nikhilroxtomar/UNET-3-plus-Implementation-in-TensorFlow-and-PyTorch | <https://github.com/nikhilroxtomar/UNET-3-plus-Implementation-in-TensorFlow-and-PyTorch> | Later PyTorch and TensorFlow implementations; no exact-source mapping established | Public history inspected begins in 2024 | Apache License 2.0 in the repository | Potential implementation reference only; not a proven upstream or state-dict-compatible replacement |

The later MIT and Apache-2.0 repositories do not cure the provenance of the files used by the paper experiments. Before considering either as a replacement, a separate static interface and implementation audit would have to establish tensor shapes, parameter hierarchy, initialization, output semantics, and license scope. A replacement would likely require retraining and new validation.

### Current technical disposition

- The probable ZJUGiveLab -> avBuffer link is strong but lacks an explicit author statement.
- Neither probable upstream contains an explicit redistribution license in its inspected current or historical trees.
- The three local files are technically identified and have been organized locally through the audited GHIST source under the selected GPL-3.0-only strategy.
- The direct source of the three release files is official GHIST, and the author team has elected to redistribute them in reliance on official GHIST's GPL v3.
- HistCFM asserts no separate avBuffer or ZJUGiveLab permission. The probable earlier chain and its missing explicit licenses remain disclosed rather than represented as resolved.
- HistCFM-specific authorship and GPL-3.0-only release authorization have been confirmed by the author team.

## Formal release naming and source map (2026-08-12)

The public project identity is **HistCFM**. Research-stage labels are provenance metadata only: they must not become public module names, class names, configuration names, commands, or README usage. In particular, `uniA`, `uniB`, `uniC`, `new_stflow`, `add_stflow`, `noflow`, `L8S5`, fold labels, dataset abbreviations, experiment timestamps, internal experiment directories, and server paths are not release-version names. Parameter values such as `num_layers: 8` and `inference_steps: 5` remain valid when expressed by their actual meaning.

The intended public import is:

```python
from histcfm import HistCFM
```

The formal command family is `histcfm train`, `histcfm infer`,
`histcfm evaluate`, and `histcfm validate-data`. It was recorded first as a
design requirement and was subsequently implemented by the thin release-side
CLI documented below.

### Research-to-release file mapping

| Research source | Formal release target | Origin and release-side treatment |
| --- | --- | --- |
| `model/model_new_stflow_uniB.py` | `src/histcfm/models/histcfm.py` | Migrated HistCFM-modified GHIST Framework derivative; exposes `HistCFM`, retains all audited model attributes and adds an internal `Framework` alias |
| `dataio/dataset_input_uniA.py` | `src/histcfm/data/dataset.py` | Migrated GHIST-derived cell-level PyTorch Dataset as `HistCFMDataset`; retained patch/filter/tuple formulas while formalizing paths, patch keys and normalization ownership |
| `dataio/utils.py` | `src/histcfm/data/image_io.py` | Migrated the GHIST `load_image` behavior with delayed TIFF/image backends and explicit per-function source documentation |
| `model/modules.py` | `src/histcfm/models/components.py` | Migrated direct GHIST inheritance; package/provenance header added without changing layer definitions |
| `model/backbone.py` | `src/histcfm/models/backbone.py` | Migrated direct GHIST source; import/provenance header changed; retains GHIST's avBuffer reference, probable U-Net 3+ chain, and no-additional-authorization disclosure |
| `model/layers.py` | `src/histcfm/models/layers.py` | Migrated near-copy in the unresolved UNet3+ chain; initialization import and provenance header changed |
| `model/intialisation.py` | `src/histcfm/models/initialization.py` | Migrated exact implementation under the corrected filename; functions unchanged and provenance header added |
| `model/flow_expression_new_stflow.py` | `src/histcfm/models/flow.py` | HistCFM addition relative to official GHIST; author-team GPL-3.0-only release approved |
| `model/flow_denoiser_new_stflow.py` | `src/histcfm/models/flow_denoiser.py` | HistCFM addition relative to official GHIST; author-team GPL-3.0-only release approved |
| `model/priors_new_stflow.py` | `src/histcfm/models/priors.py` | HistCFM addition relative to official GHIST; author-team GPL-3.0-only release approved |
| `fusion_block_new_stflow_uniA.py` | `src/histcfm/models/fusion.py` | HistCFM UNI-feature integration addition; author-team GPL-3.0-only release approved |
| `uni_feature_provider_new_stflow_uniA.py` | `src/histcfm/features/uni.py` | Release copy is restricted to precomputed feature loading; no encoder, weights, download, or online extraction |
| `loss_sonrm_new_stflow_uniB.py` | `src/histcfm/losses/sonrm.py` | HistCFM addition relative to official GHIST; author-team GPL-3.0-only release approved |
| `utils/graph_new_stflow.py` | `src/histcfm/graph.py` | HistCFM addition relative to official GHIST; author-team GPL-3.0-only release approved |
| `train_new_stflow_uniB.py` | `src/histcfm/train.py` | Migrated as a substantially reorganized GHIST-derived entry: formal config/path/model names, preflight, output ownership and checkpoint metadata added; audited loss math and order retained |
| `inference_new_stflow_uniB.py` | `src/histcfm/inference.py` | Migrated as a substantially reorganized GHIST-derived entry: one explicit formal checkpoint, restricted schema load, strict structural checks, training-normalization reuse, evaluation-free outputs and formal naming |
| Dataset-bound evaluation and paper-figure scripts | `src/histcfm/evaluate.py` | No direct script copy: newly written general evaluator for formal aligned inference tables; excludes absolute paths, checkpoint/epoch selection, dataset binding, SVG/spatial/image metrics, fold aggregation, and figure generation |

### Class and compatibility mapping

The formal model class will be `HistCFM`. The research class name `Framework` may be retained temporarily inside the implementation as `Framework = HistCFM` only if checkpoint or behavior comparison requires it; it is not a supported public name and must not appear in README usage. Source-file and class renaming must be separated from mathematical changes.

The formal data abstraction was selected from observed responsibility rather
than aesthetics. The research `DataProcessing` class directly implements the
PyTorch dataset contract and creates patch samples, but it does not construct
multiple DataLoaders. It was therefore migrated as `HistCFMDataset`. A
`HistCFMDataModule` was not created.

### Attribution remains visible after renaming

Formal names do not obscure origin. Every migrated GHIST-derived file identifies <https://github.com/SydneyBioX/GHIST>, states that it is modified/derived, records the relevant modification date and description, retains applicable notices, and is placed under the GPL-3.0-only whole-work strategy. The direct source of the three backbone-family files is GHIST. Their notices preserve the avBuffer reference and the audited probable ZJUGiveLab/U-Net 3+ chain. The author team relies on official GHIST's GPL v3 for redistribution and claims no separate authorization from those earlier implementations. HistCFM-specific additions are covered by the author team's confirmed GPL-3.0-only authorization.

### Complete-model migration record (2026-08-12)

The approved migration added the five previously held files and completed the formal model directory. Source-to-target changes were limited to provenance headers, package-relative imports, the `HistCFM` public class rename, a non-public `Framework = HistCFM` compatibility alias, the corrected `initialization.py` filename/import, and the debug-only environment variable rename from its research label to `HISTCFM_TRAIN_DEBUG`. The model constructor parameters, model attribute names, module construction order, forward tuple, flow branches, prior/noise logic, iterative update, UNI hint path, cached SONRM representations, backbone tensor outputs, and two-stage Kaiming initialization behavior were not intentionally changed.

### Data-interface migration record (2026-08-12)

The source `DataProcessing` class is a one-slide, patch-indexed PyTorch
`Dataset`. It also reads the slide-level input tables, applies the configured
row split, filters/matches cells, enumerates patches and manages image
normalization. It is not a DataLoader factory and does not construct or own
multiple train/validation loaders. The formal class is therefore
`HistCFMDataset`; no `HistCFMDataModule` was created.

The migration retained the row split, patch enumeration, nucleus-ID
intersection, minimum-area filter, cell-type offset/raster behavior,
`expression_scale * log1p(count)`, H&E normalization formula, augmentations,
padding, eight-field tuple, tensor dtypes and patch-key values. Formal changes
are explicit keyword paths/options, exception-based path errors, exact missing
gene rejection, one shared `build_patch_key`, delayed optional stainlib import,
and controlled training-only normalization output. Validation/prediction now
require training statistics instead of silently estimating them.

Unused source imports (`h5py`, `cv2`, `torchstain`, the old transforms alias,
direct image backends and functional torch operations) were not migrated.
Unused source state (`classes`, `device`, `uni_disable_color_aug`, and computed
patch-bound extrema) was not retained because static inspection found no read
in the dataset or audited callers. `torchvision`, PyTorch, NumPy, pandas,
natsort and tqdm remain required for the dataset; tifffile/imageio are delayed
image backends; stainlib is delayed and optional when stain augmentation is
disabled.

## Formal training/config/checkpoint migration (2026-08-12)

The formal training entry was derived from the working-source cell-level training script, not from the later extracted-repository training extension. The extracted extension's per-epoch inference, PCC/F1 calculation, validation CSVs, best-PCC checkpoints, copied expression outputs, and tuning behavior were inspected only to define the exclusion boundary; none entered `src/histcfm/train.py`.

`src/histcfm/config.py` and `src/histcfm/checkpoint.py` are new release-side implementations. They must not be described as verbatim historical research files. The training entry is GHIST-derived with substantial HistCFM modifications: formal package imports, strict YAML, explicit output ownership, preflight validation, formal names, one metadata checkpoint, and removal of timestamp/fold/resume path discovery. Its mathematical training path retains the audited endpoint target, target-noise draw, loss reductions, repeated alias losses, addition order, finite-value guards, optimizer, manual learning-rate rules, clipping, and zero-based checkpoint-frequency condition.

### Old-to-formal configuration map

“Default” lists the audited entry fallback or the value in the formal paper-parameter configuration where no code fallback exists. “Effective” refers only to the audited cell-level training call chain.

| Old field | Formal field | Source consumer | Default | Effective status |
| --- | --- | --- | --- | --- |
| `data_sources_train_val.fp_hist` | `data.histology_path` | Dataset image load | required path | Active |
| `data_sources_train_val.fp_nuc_seg` | `data.nucleus_mask_path` | Dataset mask load | required path | Active |
| `data_sources_train_val.fp_nuc_sizes` | `data.matched_nuclei_path` | nucleus-size filtering | required path | Active |
| `data_sources_train_val.fp_expr` | `data.expression_path` | gene derivation and Dataset expression | required path | Active |
| `data_sources_train_val.fp_cell_type` | `data.cell_type_path` | Dataset cell labels | required when cell-type branch is enabled | Conditionally active |
| `data_sources_train_val.fp_avgexp` | `data.average_expression_path` | entry reads/scales and passes `ref_orig` | source configured | Compatibility only; model does not read `ref_orig` |
| implicit expression columns | `data.genes` | model gene dimension and preflight | source CSV order | Active; now explicit and exact |
| `data.cell_types` | `data.cell_types` | class count/mapping | nine source-config classes | Active when cell types are enabled; no dataset-specific default published |
| `regions_val.divisions[fold_id-1]` | `data.validation_split` | Dataset row selection | selected fold interval | Active; fold indirection removed |
| `data.hsize` / `data.wsize` | `data.patch_height` / `data.patch_width` | Dataset patching | 256 / 256 | Active |
| `data.overlap` | `data.overlap` | validation patching; training forces zero | 30 | Active with source mode behavior |
| `data.max_cells_per_patch` | `data.max_cells_per_patch` | tuple padding | 200 | Active; formal preflight rejects overflow |
| `data.min_nuc_area` | `data.min_nucleus_area` | matched-nucleus filter | 10 | Active |
| `data.expr_scale` | `data.expression_scale` | cell expression transform; legacy average-expression scale | 5.0 | Active |
| generated `norms_hist.npy` | `data.normalization_path` in resolved config | Dataset normalization | generated | Active; fixed relative artifact path |
| `training.stain_aug` | `data.stain_augmentation` | Dataset augmentation | source true; formal template false | Active; deliberate safer template difference |
| `data.num_workers` | `training.workers` | DataLoader | 1 | Active |
| hard-coded `shuffle=True` | `training.shuffle` | DataLoader | true | Active |
| hard-coded `drop_last=True` | `training.drop_last` | DataLoader | true | Active |
| `comps.celltype` | `model.use_cell_types` | model heads and conditional losses | true | Active |
| `comps.neighb` | `model.use_neighborhood` | composition and duplicate expression losses | true | Active only with cell types |
| `comps.avgexp` | `model.average_expression_compatibility` | reads reference, sets `use_avgexp/n_ref` | source true; formal false | Stored/compatibility only; no forward/loss effect |
| `model.emb_dim` | `model.embedding_dim` | `HistCFM` constructor | 256 | Active |
| `stflow.flow_hidden_dim` | `flow.hidden_dim` | flow denoiser constructor | 512 | Active |
| `stflow.flow_layers` | `flow.num_layers` | flow denoiser depth | class fallback 4; formal paper template 8 | Active; template explicitly selects 8 |
| `stflow.flow_k_neighbors` | `flow.k_neighbors` | flow model and graph construction | 16 | Active, although current `SpatialBlock` does not consume the graph argument |
| `stflow.flow_steps` | `flow.inference_steps` | iterative inference | 5 | Inactive during training forward; retained for later inference metadata |
| `stflow.flow_use_zinb` | `flow.prior` | prior construction | false / Gaussian | Active during non-warmup training sampling |
| `stflow.flow_noise_train_sigma` | `flow.train_noise_std` | prior-side training corruption | 0.02 | Active |
| `stflow.flow_noise_infer_sigma` | `flow.inference_noise_std` | inference initialization | 0.05 | Not used by training; retained for later inference |
| training/stflow `expr_noise_train_sigma` fallback | `flow.target_noise_std` | separate target corruption before MSE | 0.001 | Active |
| top-level `flow_warmup_epochs` | `flow.warmup_epochs` | epoch-stage flag | 0 | Active optional branch |
| `uni.uni_enable` | `uni.enabled` | feature provider, fusion hint and hint loss | true | Active |
| `uni.uni_mode` | `uni.mode` | feature provider | `offline_raw` | Formalized to the only public value, `precomputed` |
| `uni.uni_feature_dir` plus fixed legacy filenames | `uni.index_path` / `uni.features_path` | explicit safe precomputed feature files | required when enabled | Active; no runtime filename inference |
| `uni.uni_dim` | `uni.feature_dim` | provider/fusion/hint dimensions | 1024 | Active |
| `uni.fusion_type` | `uni.fusion_method` | fusion constructor | source entry fallback gate; paper config film | Active; template selects film |
| `uni.fusion_hidden` / `fusion_dropout` | `uni.fusion_hidden_dim` / `fusion_dropout` | fusion constructor | 256 / 0.0 | Active |
| implicit pickle-backed object index | no public equivalent | legacy cache lookup | source implicitly enabled | Excluded; formal store is JSON plus non-pickle NPY |
| `sonrm.sonrm_enable` | `sonrm.enabled` | SONRM branch | true | Active |
| `sonrm.sonrm_k` / `sonrm_hops` | `sonrm.neighbors` / `sonrm.hops` | `SONRMLoss` | 16 / 3 | Active |
| `sonrm.sonrm_m12` / `sonrm_m23` | `sonrm.margin_12` / `margin_23` | `SONRMLoss` | 0.1 / 0.1 | Active |
| fixed map/cell/auxiliary coefficients | corresponding `loss.*` fields | loss addition | 1.0 | Active and now explicit |
| `training.w_expr` | three expression-related `loss.*` fields | endpoint and two alias MSE additions | 10.0 in paper config; code fallback 5.0 | Active; three fields retain 10.0 in template |
| fixed `100 * CosineEmbeddingLoss` | `loss.expression_embedding` | expression embedding auxiliary | 100.0 | Active |
| `uni.uni_hint_weight` fallback | `loss.uni_hint` | MSE between flow output and UNI hint | 0.1 | Active when UNI is enabled |
| `sonrm.sonrm_lambda` | `loss.sonrm` | SONRM addition | 0.05 | Active when SONRM is enabled |
| `training.total_epochs` | `training.epochs` | epoch loop | 150 | Active |
| `training.batch_size` | `training.batch_size` | DataLoader and composition | 8 | Active |
| `training.learning_rate` | `training.learning_rate` | AdamW and linear rule | 0.001 | Active |
| `training.beta1/beta2/weight_decay/eps` | same formal optimizer fields | AdamW | 0.9 / 0.999 / 0.0001 / 1e-8 | Active |
| `training.lr_sched` and fallbacks | `training.learning_rate_schedule` | manual epoch update | linear | Active; no scheduler object exists |
| `lr_initial/lr_final/min_lr/lr_switch_ratio` fallbacks | formal schedule fields | manual optional schedules/clamp | 0.001 / 1e-5 / 1e-5 / 0.5 | Branch-dependent |
| `training.grad_clip` or CLI fallback | `training.gradient_clip` | gradient clipping | 0.5 | Active |
| `save_freqs.model_freq` | `training.checkpoint_frequency` | checkpoint condition | 1 | Active |
| always-saved optimizer file | `training.save_optimizer` | checkpoint payload | source true | Active and configurable |
| `training.seed` | `training.seed` | Python/NumPy/PyTorch/loader seed | 42 | Active; null permits nondeterministic run |
| CLI `gpu_id` plus `CUDA_VISIBLE_DEVICES` | `runtime.device` / `runtime.gpu_index` | explicit `torch.device` selection | source GPU 0/auto fallback | Active without environment mutation |
| source deterministic CuDNN settings | `runtime.deterministic` | seeding helper | true | Active |

### Excluded or inactive old fields

| Old field/group | Evidence and disposition |
| --- | --- |
| `stflow.fp_expr_coords` | Not read by the source training entry, Dataset, or formal model call; excluded and documented rather than presented as functional |
| `stflow.expr_std_fallback_threshold` / `enable_old_head_fallback` | Present in configurations but not read by the audited entry/model; excluded |
| `use_flow_expr` | Entry always constructs/passes true and resets `model.use_flow_expr=True`, while `HistCFM.forward` calls `flow_forward` unconditionally; not exposed as a false capability |
| `data.high_conf_prob` | Not read by the audited cell-level Dataset or training entry; excluded |
| `data_sources_predict`, `regions_predict` | Inference-only; deferred with inference migration |
| `experiment_dirs.*`, CLI fold, timestamp and resume fields | Research path management; excluded from the formal caller-owned output contract. No checkpoint loading is implemented in this stage |
| `uni_ckpt_dir`, `uni_ckpt_file`, `uni_unfreeze_layers`, `uni_lr_factor` | Online encoder/training controls; excluded because the first release supports precomputed features only and the provider contributes no optimizer parameters |
| `n_ref`, `use_avgexp`, `ref_orig` | Compatibility state remains constructible, but the current forward path does not consume the reference tensor |
| initialized `refine_expr*` modules | Present in model/state dict but not called; retained to preserve initialization and state structure |
| extracted-repository `validation.*` and tuning fields | Added after the primary working-source entry; intentionally excluded from formal training |

### Loss targets and default weights

| Formal loss | Source tensor(s) and target | Reduction | Template weight | Default branch |
| --- | --- | --- | --- | --- |
| segmentation | output 1 vs Dataset cell-type raster | cross entropy mean | 1.0 | always |
| histology cell type | output 0 vs output 2 target | cross entropy mean | 1.0 | cell types |
| expression cell type | output 7 vs output 2 target | cross entropy mean | 1.0 | cell types |
| endpoint expression | output 3 vs `output 11 + sigma * randn_like` | MSE mean | 10.0 | always |
| neighborhood immune expression | output 4 alias vs same noisy target | MSE mean | 10.0 | neighborhood |
| neighborhood invasive expression | output 5 alias vs same noisy target | MSE mean | 10.0 | neighborhood |
| expression embedding | output 8 vs output 10, target all `+1` | cosine embedding mean | 100.0 | cell types |
| expression logits | output 7 vs output 9 | MSE mean | 1.0 | cell types |
| estimated composition | softmax/log of output 12 aggregate vs softmax GT composition | KL batchmean | 1.0 | neighborhood |
| histology composition | argmax/one-hot output 0 composition vs softmax GT composition | KL batchmean | 1.0 | neighborhood |
| UNI hint | output 3 vs output 6, never directly vs expression truth | MSE mean | 0.1 | UNI |
| SONRM | cached original embeddings and normalized centroids | source SONRM formula | 0.05 | SONRM enabled |

The source finite guard is retained for the ten base losses through composition. The UNI-hint and SONRM values, and the final total, are not silently zeroed when non-finite because the source did not guard them. The addition order is the order shown above, with SONRM added last. No velocity target was introduced.

### Formal checkpoint/output boundary

Each schema-3 checkpoint contains one-based epoch, unchanged model state-dict keys, optional optimizer state, resolved configuration, ordered genes, zero-based cell-type mapping, explicit model dimensions and fusion method, relative normalization artifact, seed, package version, and runtime PyTorch/CUDA versions. It is a PyTorch pickle-backed artifact and is safe only under the documented trusted-source boundary. No paper checkpoint was copied or loaded.

The formal run owns only `resolved_config.yaml`, `artifacts/histology_normalization.npy`, `checkpoints/epoch_<N>.pth`, `logs/train.log`, and `logs/train_losses.csv` below the explicit output directory. It creates no timestamp/fold directory, result figure, validation metric, or best-metric checkpoint.

## Formal inference migration (2026-08-13)

The primary source was the working-source cell-level inference script. Its call chain was statically compared with the extracted copy; the latter differs only in its Dataset import and adds no separate inference behavior. The source performs:

1. old JSON/CLI/fold/GPU parsing and experiment-directory discovery;
2. genes-file loading or reconstruction from training expression;
3. optional average-expression loading/intersection;
4. `Framework` construction;
5. scan/selection of one, last, or all epoch checkpoints;
6. val/train/predict Dataset routing and non-shuffled, non-drop-last DataLoader;
7. repeated unrestricted `torch.load`, model state load, `eval`, and `no_grad`;
8. unchanged stochastic prior/noise and S-step model inference;
9. model-output truncation through `n_cells`, inverse expression scaling, ID/area collection, and largest-area cell de-duplication;
10. raw and randomly adjusted cell-type labels;
11. epoch-named expression/target/type/cell CSVs;
12. embedded gene PCC, cell-type F1, rank-sum epoch selection, PCC/F1 maxima, summary CSV, and best-expression copy.

Only items needed for one explicit checkpoint and prediction output were migrated. The formal entry uses checkpoint genes/config rather than filenames or experiment directories, `HistCFMDataset`, strict `HistCFM` state loading, one seeded stochastic pass, inverse expression scaling and the same largest-area de-duplication rule. Output 3 remains the expression endpoint; outputs 0, 2, 11, 13, and 14 supply optional labels/targets, area and IDs.

### Source behavior classification

| Source behavior | Classification and release disposition |
| --- | --- |
| one checkpoint load, `eval`, `no_grad`, Dataset/DataLoader, model forward | Formal inference essential; migrated with `torch.inference_mode` and strict schema/state checks |
| Gaussian/ZINB prior, inference noise and configured S-step update | Formal inference essential; unchanged inside `HistCFM` |
| `/ expression_scale` output conversion | Formal inference essential; preserved and documented as `log1p(count)`, not raw count |
| `n_cells` flattening and largest-area duplicate removal | Formal inference essential; preserved using actual batch lengths |
| raw cell-type argmax | Optional formal prediction output; retained when the checkpoint has cell-type heads |
| `adjust_pr` / `create_tensors` | Historical randomized adjusted-label output only; does not affect expression, raw labels, source F1, or epoch ranking; not migrated |
| checkpoint glob, `all`/`last`, experiment/fold lookup | Old multi-epoch experiment selection; excluded |
| PCC, F1, rankdata, best epoch, PCC/F1 maxima | Embedded evaluation/model selection; deferred to evaluator or later model-selection policy |
| epoch-prefixed CSVs, best-expression copy, result summary | Paper experiment output convention; replaced by stable single-run filenames |
| uncertainty, repeated sampling, robustness, genome-wide/figures | Later or separate experimental additions; absent from the primary script and excluded |
| unused `natsort`, old utility wildcard, CSV helper/imports | Not required after experiment/evaluation removal; excluded |

### Checkpoint and normalization boundary

Schema version 3 carries explicit `model_metadata` (`n_classes`, `n_genes`, compatibility reference-profile count, and `fusion_method`) with the required model state, resolved configuration, genes, cell mapping, normalization, seed, epoch and software metadata. No release checkpoint was generated under the earlier organization schemas. The loader accepts only schema 3, calls `torch.load(..., weights_only=True)`, warns that pickle-backed untrusted files remain unsafe, and deliberately fails rather than falling back on older unrestricted loading.

The normalization locator accepts only `<run>/checkpoints/<checkpoint>` plus a safe relative artifact path recorded in metadata. It rejects absolute/parent-traversing metadata, resolves against `<run>`, and validates `(2, 3)`, finite values and positive standard deviations. It does not use current-working-directory assumptions or old experiment naming.

### Formal inference differences from the source

- exactly one caller-specified release checkpoint; no old/paper checkpoint fallback;
- checkpoint schema/config/genes/classes and structure are authoritative;
- runtime paths/device/workers/batch size and sampling seed are explicit overrides, while structural overrides are rejected;
- no environment-variable GPU mutation or import-time path insertion;
- `prediction` no longer dereferences absent `batch_expr_pc` and therefore emits no fabricated target;
- output configuration uses external-input placeholders, cells use a stable hashed slide pseudonym, and checkpoint/normalization provenance is limited to a filename or safe relative path;
- no metric, epoch ranking, best-result copy, randomized label adjustment, or timestamp directory;
- a formal single-checkpoint seed resets the RNG for one sample, so it is not claimed byte-identical to a historical multi-checkpoint loop that continuously consumed RNG.

## Evaluation, CLI, packaging, and test provenance (2026-08-13)

The following files were newly authored for the release organization; no old
paper evaluator, launcher, package manifest, or test was copied:

| Release file | Source classification | Scope |
| --- | --- | --- |
| `src/histcfm/evaluate.py` | New release-side implementation | Strict aligned-table validation; gene PCC; valid-gene summaries; MSE/MAE/RMSE; optional accuracy/macro-F1; finite JSON |
| `src/histcfm/cli.py` | New release-side implementation | Thin delayed-import delegation to formal train, infer, evaluate, and data-validation functions |
| `src/histcfm/__main__.py` | New release-side implementation | `python -m histcfm` delegation only |
| `pyproject.toml` | New release-side packaging metadata | One `src/` package, console script, static-import dependency inventory, optional stain/test extras |
| `tests/test_evaluate.py` | New synthetic contract test | In-memory/tabular toy arithmetic and validation; not biological demo evidence |
| `tests/test_cli_contract.py` | New static/interface contract test | Command/argument/help surface and absence of heavy top-level CLI imports |
| `tests/test_config_contract.py` | New strict-schema contract test | Unknown field rejection and one minimal known mapping |
| `tests/test_uni_contract.py` | New technical-array contract test | JSON rows/duplicates/continuity, NPY shape/dtype/finiteness/norm, coverage, shared runtime parser, and no pickle path |
| `tests/test_release_contract.py` | New static release contract test | Single package/checkpoint version and explicit formal UNI/fusion construction |
| `tests/test_demo_contract.py` | New synthetic/public-private separation test | Strict demo config, files/IDs/genes/patch keys/features/checksums, path markers, and optional author-side private hash/ID disjointness |

The evaluator deliberately replaces neither paper-result reproduction scripts
nor their reported metrics. It does not read archived experiment results or
reuse dataset-specific selection rules. The CLI introduces no research-stage
version, dataset, fold, or server-path name in its public command surface.
These files are included under the GPL-3.0-only release strategy confirmed by
the HistCFM author team.

## Demo-interface repair record (2026-08-13)

The research cache interface inferred an object-array index and raw feature
filename from one directory and enabled pickle loading. The public release now
uses explicit `uni.index_path` and `uni.features_path` values identifying
`uni_index.json` and `uni_features.npy`. `data.validation.load_uni_index` and
`load_uni_feature_store` are new release-side validation code; the runtime
provider imports the same implementation rather than maintaining a second
parser. The supplied feature values are not rewritten. For valid feature
rows, the provider retains the research L2-normalization step before fusion.

The old internal mode label is recorded above only to preserve source
provenance. The public configuration, provider, model construction and
metadata all use `precomputed`. No legacy pickle interface, online encoder,
UNI weights, download path, or feature-generation implementation remains in
the public code.

`validate_training_data` and the patch-inventory helper are new release-side
read-only checks. They reproduce formal train/validation patch-key and maximum
cell-count inspection without instantiating a Dataset or generating
normalization. `validate_inference_data` composes the existing restricted
checkpoint loader, structural comparison, checkpoint-relative normalization
lookup, and Dataset preflight without creating output. The training and
inference workflows themselves remain the authoritative execution entries.

The version string moved from inconsistent packaging/checkpoint literals to
`src/histcfm/_version.py`; setuptools reads that attribute and checkpoint and
inference metadata import it. Schema 3 additionally binds the explicit formal
fusion method. These are release-interface changes, not claims of runtime or
numerical validation.

## Synthetic public demo provenance (2026-08-13)

The public demo is newly generated release material, not a migration from
either research source. `scripts/generate_synthetic_demo.py` uses Python's
standard library, fixed seed `20260813`, and no input-data argument to create
the image, mask, aligned tables, raw counts, canonical-key index and synthetic
float16 morphology-feature matrix. It neither imports nor executes HistCFM,
GHIST, UNI, or another model.

`configs/demo.yaml` is a one-epoch smoke configuration for the same formal CLI
used by prepared real inputs. It does not encode a research dataset, fold,
paper hyperparameter claim, checkpoint, or result. Runtime normalization,
checkpoint, prediction and metric files are created by the normal train,
infer, and evaluate workflows and remain uncommitted.

The separately prepared BC1 staging data and its real UNI feature store are
used only for private technical validation. No file was copied from that
staging area into the release. Its data/feature rights remain unresolved and
its execution is not paper-benchmark reproduction.

## Public preparation guidance and independent environment (2026-08-13)

`docs/data_preparation.md` was written from the audited official GHIST README,
preprocessing notebook, and `data_processing/1_...` through `4_...` scripts.
It links to those upstream materials and maps their processed concepts onto
the formal HistCFM configuration; it copies no GHIST preprocessing code. It
also records that GHIST's tutorial does not implement H&E/Xenium registration
and uses external Hover-Net resources.

`docs/uni_features.md` was written from the official current and original UNI
README files plus the local historical HistCFM feature-generation call chain.
It documents original UNI `vit_large_patch16_224`, the 224-pixel input and
1024-value output, the offline store, and a deterministic recommendation for
new data. It explicitly distinguishes that recommendation from the historical
training-mode/random-augmentation cache and does not copy an encoder,
checkpoint, weight, or extraction implementation.

`environment.yml` remains an independently curated `histcfm` environment. It
is not a clone or export of the historical research environment. The public
synthetic path imports no `timm` or UNI package and does not require a UNI
checkpoint or `stainlib`; optional stain augmentation is isolated as a Python
extra. Server execution of this environment and the synthetic end-to-end
workflow remains the runtime release gate.
