# Licensing audit

This is a technical licensing inventory, not legal advice. The HistCFM author team selected `GPL-3.0-only` and has confirmed its authority to release the HistCFM-specific additions under that license. The repository `LICENSE` is the complete GNU GPL version 3 text obtained from the official GHIST repository. No separate `NOTICE` file has been created; third-party provenance and the absence of additional independent UNet3+ authorization are disclosed below.

Audit date: 2026-08-12.

## GHIST license

Audited source: <https://github.com/SydneyBioX/GHIST>, `main` at `917456be305fc82e92293ea272812e79675e821c`.

The root `LICENSE` contains the GNU General Public License, version 3. No source file contains an “or later” program notice, so this audit does not assume an option to use a later GPL version. The license text was first added at commit `4bba2a6a0165aeda225a07aa3195c1457e34bf02` on 2024-07-02 and is present throughout the closest local-match range (`14bf60f...` through audited HEAD).

The stock license text names the Free Software Foundation as copyright holder of the license document itself. It does not fill in a program copyright-holder notice for GHIST, and the inspected source files have no copyright headers. Repository ownership and commit authorship are not substitutes for an express copyright-holder statement, so the GHIST program copyright holder remains unstated in the audited files.

### Permissions and conditions

Subject to the GPL's full terms, the root license permits copying, modification, source redistribution, publication of modified source, and commercial distribution. It does not impose a noncommercial-use restriction.

For copied or modified covered code, the relevant obligations include:

- provide recipients a copy of the GPL and keep applicable copyright, license, and warranty notices intact;
- place prominent notices on modified versions stating that they were modified and giving a relevant date;
- license the entire covered work, as a whole, under GNU GPL v3 when conveying a modified source version;
- provide Corresponding Source through an allowed method when conveying object code;
- comply with the GPL's patent provisions; the repository adds no separate patent grant, trademark policy, or additional terms.

The README does not state a conflicting code license. It asks users to cite the GHIST paper, which is an important scholarly attribution request but is not presented as an additional GPL condition.

### Scope and third-party limits

The repository has one root license and no `NOTICE`, `COPYING`, `AUTHORS`, `CITATION`, submodule, or nested third-party license file. No separate data, demo checkpoint, or model-weight license was found. The absence of separate terms does not prove that bundled demo data or externally hosted checkpoints are redistributable under the software license.

Known external boundaries are:

- `model/backbone.py` explicitly cites `avBuffer/UNet3plus_pth`, and comparison identifies associated `layers.py` and `intialisation.py` as upstream-derived;
- `stainlib` is referenced as a separately installed dependency rather than bundled code;
- preprocessing invokes an external Hover-Net repository/checkpoint rather than bundling that project;
- demo/tutorial material cites a 10x Genomics dataset and externally hosted processed data/checkpoints, whose separate terms were not established here.

GHIST's root GPL is the stated redistribution basis for the GHIST-derived components in this release. That project decision does not prove that GHIST had permission to relicense any separately sourced code whose earlier repository published no license. The earlier UNet3+ chain therefore remains a disclosed source-chain uncertainty, not a claim of separate authorization.

## Author-team release decision (2026-08-13)

The HistCFM author team supplied and confirmed the following statements:

> The HistCFM author team confirms that the HistCFM-specific additions were created by the author team and approves their public release under GPL-3.0-only.

> The HistCFM author team approves publication of the HistCFM repository under GPL-3.0-only and elects to redistribute the GHIST-derived components in reliance on the GPL v3 license published with the official SydneyBioX/GHIST repository, while retaining complete attribution, provenance, and modification notices. The team acknowledges that no additional independent authorization from the earlier UNet3+ implementations is claimed.

This resolves the prior HistCFM-specific authorship/authority review item and records the team's publication decision for GHIST-derived components. The record explicitly rejects four interpretations: HistCFM-team authorship of the UNet3+ code; any direct authorization by avBuffer or ZJUGiveLab; attribution as a license; or certainty throughout the earlier upstream chain.

## GHIST-derived HistCFM conclusions

`model/model.py` is byte-identical to official GHIST over the closest range. The formal `models/histcfm.py` preserves substantial GHIST Framework structure while adding HistCFM flow, UNI, spatial-graph, and SONRM behavior. It is a modified/derived GHIST work, not a clean-room implementation.

On the source-license evidence now available:

- `models/histcfm.py` has been migrated under the selected GPL-3.0-only strategy, with GHIST identification and prominent HistCFM change/date notices;
- `modules.py` has been migrated as `components.py` under the same GPL obligations; it is byte-identical to official GHIST apart from release-side header/line-ending treatment;
- the GHIST-distributed backbone family has been organized locally with explicit GHIST, avBuffer, and U-Net 3+ provenance, without claiming separate upstream permission;
- the HistCFM author team has confirmed authorship of the HistCFM-specific additions and approved their GPL-3.0-only release.

## UNet3+ findings

The local and official-GHIST backbone are byte-identical, but both contain a direct source URL to <https://github.com/avBuffer/UNet3plus_pth>. Static comparison shows that the backbone is a modified copy, `layers.py` is effectively the same except for its import, and `intialisation.py` is an exact renamed copy of upstream files.

No license-like file was found in the audited UNet3+ HEAD or available commit trees. The upstream README's academic-communication wording is not an explicit redistribution grant. Adding attribution alone cannot resolve this absence, and GHIST's public inclusion of the files does not demonstrate upstream permission.

The author team has elected to publish these files as GHIST-derived components in reliance on the GPL v3 license distributed by official GHIST. No separate avBuffer or ZJUGiveLab authorization is claimed. If additional permission or license evidence is later obtained, the record should retain at minimum:

- upstream author/project identification and repository URL;
- the applicable license or permission text;
- a description of class, import, output, default, and filename changes;
- any copyright, notice, citation, patent, or trademark terms supplied by the rights holder.

## UNI boundary

The real-data UNI decision is unchanged:

- do not copy the UNI repository, encoder, or weights;
- treat UNI as a separately obtained external dependency under its own terms;
- do not migrate real UNI feature data in the current state;
- use synthetic, non-model feature data for the public software smoke demo.

The release-side `features/uni.py` is local offline-loading integration code and contains no copied UNI encoder or weights. That separation does not decide the redistribution status of precomputed features.

## Batch status

### Batch 1

Official GHIST contains no same-named, historical, or comparable implementation of the seven migrated CFM/fusion/UNI-provider/SONRM/graph files. They are HistCFM additions relative to GHIST. The HistCFM author team has confirmed that its team created these additions and approved their public release under GPL-3.0-only.

### Batch 2

All five batch-2 files have been migrated through the audited GHIST source under the selected GPL-3.0-only strategy. `histcfm.py` and `components.py` carry GHIST-derived modification notices. The direct source used for `backbone.py`, `layers.py`, and `initialization.py` is official GHIST; their notices also retain GHIST's avBuffer reference and the audited U-Net 3+ source-chain disclosure. The author team has made the publication decision to rely on GHIST's GPL v3 for these GHIST-derived files. No additional independent authorization from avBuffer, ZJUGiveLab, or another earlier implementation is claimed.

## Detailed release matrix

Statuses describe the present audit evidence, not a final legal guarantee. “Planned copy” does not itself authorize publication.

| Component | Source | License | Modified? | Planned copy? | Publication condition | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| GHIST core | SydneyBioX/GHIST, closest range `14bf60f...`–`917456b...` | GNU GPL v3 | Selected local files modified | Selective only | Retain GPL/notices, mark changes/date, distribute covered whole under GPL v3, provide source as required | 可迁移但必须 attribution/遵守 GPL |
| HistCFM Framework | Formal `models/histcfm.py`, derived from official GHIST `model.py` | GPL-3.0-only strategy: GHIST GPL v3 plus authorized HistCFM modifications | Yes | Migrated locally | Same GPL whole-work duties; identify GHIST and changes | 已确认作者权属并批准 GPL-3.0-only 发布 |
| GHIST components | Official `model/modules.py` | GNU GPL v3 / selected GPL-3.0-only release | Header/package only | Migrated locally | Retain GPL/attribution and satisfy covered-work/source duties | 已迁移并保留 attribution |
| HistCFM data loader | GHIST-derived cell-level loader from the working source | GNU GPL v3 / selected GPL-3.0-only release | Formal naming, paths, preflight, patch-key reuse and normalization-output control | Migrated locally | Identify GHIST, mark changes/date, retain GPL and source obligations | 已迁移并保留 attribution |
| HistCFM image I/O | GHIST `dataio/utils.py::load_image` | GNU GPL v3 / selected GPL-3.0-only release | Delayed imports and documentation | Migrated locally | Same GHIST/GPL duties | 已迁移并逐函数记录来源 |
| HistCFM input validation | New release-side validation around the GHIST-derived schema | GPL-3.0-only, authorized by HistCFM author team | New | Present | Retain release provenance | 作者团队已批准发布 |
| CFM | HistCFM flow/denoiser/prior additions; absent from official GHIST | GPL-3.0-only, authorized by HistCFM author team | Release packaging/documentation | Present | Retain release provenance | 作者团队已批准发布 |
| Fusion | HistCFM UNI integration; absent from official GHIST | GPL-3.0-only, authorized by HistCFM author team | Yes | Present | Keep UNI code/weights external | 作者团队已批准集成代码发布 |
| UNI provider | HistCFM offline loader; no UNI encoder copied | GPL-3.0-only for integration code; feature terms separate | Yes | Provider only | Keep UNI code/weights external | 集成代码已获批准；真实 feature 许可独立 |
| SONRM | HistCFM addition; absent from official GHIST | GPL-3.0-only, authorized by HistCFM author team | Yes | Present | Retain release provenance | 作者团队已批准发布 |
| Graph | HistCFM addition relative to official GHIST | GPL-3.0-only, authorized by HistCFM author team | Yes | Present | Retain release provenance | 作者团队已批准发布 |
| UNet3+ backbone | Direct release source: official GHIST; GHIST cites `avBuffer/UNet3plus_pth` | GHIST GPL v3 is the author team's redistribution basis; no separate earlier license found | Yes | Migrated through GHIST | Retain complete chain and no-additional-authorization disclosure | 作者团队已决定依赖 GHIST GPL v3 发布；上游不确定性披露 |
| UNet3+ layers | Direct release source: official GHIST; near-copy in earlier source chain | Same GHIST GPL v3 reliance; no separate earlier license found | Import changed | Migrated through GHIST | Same disclosure as backbone | 作者团队已作发布决定；未声称额外授权 |
| UNet3+ initialization | Direct release source: official GHIST; renamed implementation in earlier chain | Same GHIST GPL v3 reliance; no separate earlier license found | Filename only | Migrated through GHIST | Same disclosure as backbone | 作者团队已作发布决定；未声称额外授权 |
| UNI code | MahmoodLab/UNI | CC BY-NC-ND 4.0 plus upstream conditions | Not copied | No | User obtains separately and follows upstream terms | 可作为依赖但不复制 |
| UNI weights | MahmoodLab gated model access | Upstream gated/no-copy conditions | No | No | Users obtain authorized access directly | 暂不可迁移 |
| Public demo features | Newly generated deterministic synthetic matrix; not a model output | CC0-1.0 dedication recorded in demo provenance | Generated to the formal 1024-dimensional interface | Included | Keep clearly labeled synthetic and separate from the software license | 可公开；非 UNI 输出 |
| Private real UNI features | UNI-derived output held outside the release | Exact redistribution permission unresolved | Generated locally from authorized checkpoint access | Not included | Obtain written feature-redistribution permission before any publication | 许可未明确；保持私有 |
| GHIST preprocessing code | Official GHIST `data_processing` and tutorial | GNU GPL v3 for code; external Hover-Net boundary | Not copied | No; link only | Link and cite official GHIST preprocessing; users follow third-party terms | 与代码发布无关 |
| GHIST demo/preprocessed data | 10x-derived subset and externally hosted processed artifacts | No separate data license identified in GHIST | No | No | Separate dataset/privacy/redistribution audit | 许可未明确 |
| GHIST demo checkpoint | Externally hosted official tutorial artifact | No separate model-weight terms identified | No | No | Separate checkpoint/model audit | 许可未明确 |
| HistCFM public demo data | Newly generated synthetic image, mask, tables, counts and features | CC0-1.0 dedication recorded in `examples/demo/DATA_PROVENANCE.md` | Generated from fixed seed without research inputs | Included | Preserve provenance and do not describe as biological or paper-result data | 可公开；纯合成 |

## Required attribution and decisions

Before public release:

1. retain the confirmed GPL-3.0-only whole-work decision and ensure all distributed source follows it;
2. carry the GHIST license text when authorized to create the final release license/notice materials;
3. identify the official GHIST repository and closest source range;
4. mark modified/derived files prominently with a change description and relevant date;
5. retain all existing source notices, including the UNet3+ URL, without treating attribution as permission;
6. record the author team's decision to rely on official GHIST's GPL v3 for the GHIST-derived files while disclosing that no separate earlier UNet3+ authorization is claimed;
7. retain the completed HistCFM-specific authorship and GPL-3.0-only authorization record;
8. keep UNI code, weights, and real features outside the repository under the current strategy;
9. audit demo data and checkpoints separately from the software license.

The complete GHIST GNU GPL version 3 text is now present as `LICENSE`. No `NOTICE` file was created.

The release `LICENSE` was compared byte-for-byte with <https://raw.githubusercontent.com/SydneyBioX/GHIST/main/LICENSE> on 2026-08-12 after preserving the upstream CRLF line endings. The comparison succeeded; its SHA-256 is `230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809`.

## UNet3+ source-chain permission review (2026-08-12)

The UNet 3+ paper points to <https://github.com/ZJUGiveLab/UNet-Version>. Its three corresponding files predate and closely align with the avBuffer targets. Neither ZJUGiveLab nor avBuffer contains a `LICENSE`, `COPYING`, `NOTICE`, copyright license header, or another explicit software grant at the audited HEAD or in the inspected commit trees.

For each of these two repositories, the present conclusion is:

> 未发现明确的公开复制、修改和再分发许可。

The probable source chain is documented in `docs/provenance.md`. Because avBuffer does not state whether it authored the files or derived them from ZJUGiveLab, later permission from avBuffer alone might not resolve the whole chain unless the maintainer confirms ownership or authority to sublicense the relevant code. GHIST's root GNU GPL v3 cannot retroactively supply rights that a third-party contributor did not possess. The author team nevertheless has made and documented its publication decision to redistribute its direct GHIST copies in reliance on official GHIST's GPL v3; this decision does not eliminate the disclosed upstream uncertainty.

### License compatibility is not source proof

MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, GPL-3.0-only, and GPL-3.0-or-later are examples of terms that can generally be incorporated into a GPL v3 combined source work, subject to their full notices and conditions. Apache-2.0 is compatible with GPL v3, not GPL v2-only. This inventory does not choose a license for an upstream author or provide a legal guarantee.

Any new license or written permission must unambiguously cover the existing target files and relevant historical versions, and must allow copying, modification, public source redistribution, and inclusion of modified derivatives in the planned GPL v3 HistCFM work. It must also identify required attribution, notices, change marking, and any other conditions. If ZJUGiveLab is the actual source, the applicable rights must cover that link as well.

### Resolution routes

| Route | What would be required | Scientific/technical effect | Current assessment |
| --- | --- | --- | --- |
| A — Obtain explicit permission | Ask avBuffer for copy, modification, public redistribution, and GPL v3 inclusion rights; retain attribution/link/change notices; ask whether avBuffer authored the targets or has authority to sublicense them. If ZJUGiveLab rights are implicated, obtain applicable permission there too. | Preserves the paper implementation and is least likely to disturb interfaces/checkpoints | Recommended first route, but not complete until the source/authority question is answered in writing |
| B — Upstream adds a formal license | The rights holder chooses a GPL-v3-compatible license and states that it covers the three existing files, their relevant history, and downstream modified copies | Preserves implementation if scope is explicit | Valid in principle; a new avBuffer license alone does not resolve probable ZJUGiveLab rights unless avBuffer has authority |
| C — Use a clearly licensed same-purpose implementation | Prove the candidate's own source and license, then audit architecture, output tensors, parameter/module names, initialization, numerical behavior, and state-dict mapping | Existing checkpoints are unlikely to load directly; adapter work, retraining, and result revalidation may be necessary | No proven drop-in, same-source licensed implementation was found; later MIT/Apache-2.0 implementations are candidates for a separate engineering audit only |
| D — Use another clearly licensed backbone | Select and integrate a permitted backbone after a separate decision and audit | Changes the model; paper checkpoints are incompatible and demo results require retraining/revalidation | Feasible for a from-scratch public demo, but must be labeled an alternative-backbone HistCFM variant, not the exact paper implementation |

Route A is recommended because it minimizes scientific divergence. The permission request should include the probable ZJUGiveLab relationship rather than assume avBuffer can grant all necessary rights. If permission cannot be secured, route D is the clearest public-software fallback; it would not reproduce the original backbone/checkpoint and must be disclosed as such. Route C should be used only after a candidate's license, provenance, and compatibility have been positively established.

### Draft permission request (not sent)

**Subject:** Permission request for UNet3+ source files used in HistCFM

> Hello,
>
> We are preparing a public reproducibility repository for the academic HistCFM software, distributed under GNU GPL v3. GHIST adapted the following files from your `avBuffer/UNet3plus_pth` repository: `unet/UNet3Plus.py`, `unet/layers.py`, and `unet/init_weights.py`.
>
> Would you please confirm whether you permit us to copy, modify, and publicly redistribute these files and our modified derivatives as part of the GPL v3 HistCFM source repository? We will retain attribution to you, link to your repository, identify our modifications, and preserve any notices or additional wording you require.
>
> We also found closely aligned earlier files in `ZJUGiveLab/UNet-Version`, the repository linked by the UNet 3+ paper. Could you confirm whether you authored the three files independently or derived them from that or another source, and whether you hold authority to grant the requested permission? If they are derived, please identify the upstream license or permission that applies.
>
> An explicit reply granting copying, modification, public source redistribution, and inclusion of modified versions under GPL v3—or a repository license that clearly covers these files and their existing versions—would help us document the permission accurately.
>
> Thank you. Any permission or source-license clarification will be retained with the repository's provenance record.

This text is only a draft; no message or issue was sent.

### Revised batch-2 matrix

| Planned release file | Immediate source | Deeper-source issue | Technical status | Permission status | Present decision |
| --- | --- | --- | --- | --- | --- |
| `histcfm.py` | HistCFM modification of GHIST `model.py` | None additionally identified | Migrated with model attributes and forward contract retained | GHIST GPL conditions plus confirmed HistCFM author authorization | 已确认并批准 GPL-3.0-only 发布 |
| `components.py` | GHIST `modules.py` | No separate third-party marker found | Migrated with definitions and construction behavior retained | GHIST GPL conditions and GPL-3.0-only whole-work decision | 已本地迁移并标记来源 |
| `backbone.py` | Direct copy from GHIST's adaptation; GHIST cites avBuffer `UNet3Plus.py` | Probable ZJUGiveLab source also has no license | Migrated; interface and changes mapped | Relies on official GHIST GPL v3; no proven separate upstream grant | 已决定依赖 GHIST GPL v3 发布；保留不确定性披露 |
| `layers.py` | Direct copy from GHIST; near-copy of avBuffer `layers.py` in earlier chain | Probable ZJUGiveLab source also has no license | Migrated; initialization import renamed | Relies on official GHIST GPL v3; no proven separate upstream grant | 已决定发布；未声称额外授权 |
| `initialization.py` | Direct copy from GHIST; renamed avBuffer `init_weights.py` in earlier chain | Probable ZJUGiveLab source also has no license | Migrated; functions unchanged | Relies on official GHIST GPL v3; no proven separate upstream grant | 已决定发布；未声称额外授权 |

The full second batch has been migrated under the author team's GHIST/GPL-3.0-only strategy. The team has approved public release in reliance on official GHIST's GPL v3. The upstream uncertainty is unchanged and no separate avBuffer or ZJUGiveLab permission is claimed. A formal `LICENSE` exists; no separate `NOTICE` was created.

## Training/config/checkpoint files (2026-08-12)

| Release file | Source classification | License treatment and remaining condition |
| --- | --- | --- |
| `src/histcfm/train.py` | GHIST-derived training entry with substantial HistCFM release modifications | GPL-3.0-only; retains GHIST attribution and a 2026-08-12 modification notice; HistCFM modifications are covered by the confirmed author-team authorization |
| `src/histcfm/config.py` | New release-side strict configuration implementation | GPL-3.0-only under the confirmed HistCFM author-team authorization; not represented as historical GHIST code |
| `src/histcfm/checkpoint.py` | New release-side checkpoint metadata/save implementation | Same confirmed GPL-3.0-only authorization |
| `configs/histcfm.yaml` | New release-side formal template based on audited source values | GPL-3.0-only repository material; contains no source paths, data, credentials, features, or weights |

The checkpoint format is a generated runtime artifact, not a redistributable paper checkpoint. PyTorch serialization is pickle-backed; the repository does not claim arbitrary external checkpoints are safe. The formal loader accepts only the new schema with `weights_only=True` support and has no unrestricted or legacy fallback. Precomputed UNI feature files remain external data whose redistribution rights must be reviewed separately. This training migration does not change the disclosed UNet3+ source-chain uncertainty or the author team's documented GHIST-GPL reliance decision.

## Formal inference entry (2026-08-13)

`src/histcfm/inference.py` is a GHIST-derived inference entry with substantial HistCFM modifications, distributed under the selected GPL-3.0-only strategy. Its header identifies SydneyBioX/GHIST, the official URL, the formalization date, single-checkpoint behavior, output/path changes, and retained iterative CFM sampling. The author team has confirmed and authorized the HistCFM-specific modifications under GPL-3.0-only.

No upstream checkpoint, result, UNI feature, dataset, adjusted-label helper, evaluator, or third-party scoring implementation was copied. The excluded `adjust_pr`/`create_tensors` implementation remains in the readonly working source and was not needed for formal expression or raw label output. The checkpoint-loading additions in `checkpoint.py` are new release-side organization code, not represented as historical GHIST code. These additions do not resolve or change the existing UNet3+ or UNI-feature redistribution questions.

## Evaluation, CLI, package metadata, and tests (2026-08-13)

`src/histcfm/evaluate.py`, `src/histcfm/cli.py`,
`src/histcfm/__main__.py`, `pyproject.toml`, and the release contract-test files
were newly written during release organization. No source from the archived
paper evaluators or launchers, and no third-party metric implementation, was
copied into them. NumPy/pandas are used for expression metrics and
scikit-learn is called for optional accuracy and macro-F1; these libraries are
declared as dependencies and are not vendored.

The package metadata identifies the distribution as `GPL-3.0-only` and uses
the repository's existing `LICENSE`. The HistCFM author team has confirmed its
authority to license its additions. This does not remove the disclosed UNet3+
provenance uncertainty or create separate upstream authorization. No new
`NOTICE`, dependency lockfile, downloaded source,
model, feature, dataset, or checkpoint was added in this stage.

The default dependencies declared by `pyproject.toml` follow the current
static imports: `imageio`, `natsort`, `numpy`, `pandas`, `PyYAML`,
`scikit-learn`, `tifffile`, `torch`, `torchvision`, and `tqdm`. `stainlib` is
an optional extra because it is imported only for enabled stain augmentation.
No online UNI encoder, UNI weight package, `timm`, private index, or CUDA wheel
URL is declared. Compatibility and installation remain runtime-validation
questions rather than license conclusions.

## Safe UNI interface and validation additions (2026-08-13)

The JSON index parser, NPY validation, read-only patch inventory, train/infer
validation orchestration, static version module, and accompanying contract
tests were newly authored for this release. They are included under the
confirmed HistCFM author-team GPL-3.0-only authorization. No implementation
from a UNI encoder repository
was copied.

The public interface names `uni_index.json` and `uni_features.npy`. For real
prepared inputs these remain external, author-provided artifacts whose
provenance and redistribution terms must be reviewed. The committed public
demo uses the same filenames for an independently generated synthetic matrix;
its provenance expressly states that it is not UNI-derived. HistCFM does not
bundle or download UNI code or weights and does not compute real UNI features.

Removal of the legacy pickle-index path reduces an input security risk but
does not change the disclosed UNet3+ source-license uncertainty or GHIST
attribution duties. HistCFM-specific ownership is confirmed. The synthetic demo
adds no checkpoint, dependency source, LICENSE replacement, or NOTICE.

## Synthetic demo and private-validation separation (2026-08-13)

The public demo image, mask, matched table, raw-count matrix, cell labels and
feature matrix are generated from fixed seed `20260813` by repository-local
standard-library code. The generator has no private-data input and invokes no
model. The generated data are dedicated under CC0-1.0 in
`examples/demo/DATA_PROVENANCE.md`; generator and test source remain governed
by the software license.

Real BC1 inputs and real UNI-derived feature outputs remain in an external
private staging directory. Their local technical validation configuration is
not part of this repository and neither its presence nor successful execution
establishes public redistribution rights. No real-data hash, cell ID, image,
table, feature vector, checkpoint, prediction, or result is deliberately
included in the release.

## Publication guidance update (2026-08-13)

The release now links to official GHIST preprocessing and UNI access materials
through `docs/data_preparation.md`, `docs/uni_features.md`, and
`THIRD_PARTY_NOTICES.md`. No GHIST preprocessing implementation, GHIST demo
data, Hover-Net component, UNI source, UNI checkpoint, or real UNI output was
added. The public synthetic feature matrix remains independently generated
CC0-1.0 demo data, not an output licensed by or derived from UNI.

The HistCFM author team has confirmed its ownership of HistCFM-specific
contributions and authorized them under GPL-3.0-only. The team has also made
the publication decision to redistribute GHIST-derived code in reliance on
official GHIST's GNU GPL version 3 while retaining attribution, provenance,
and modification notices. The included UNet3+ source chain still lacks a
proven separate public-redistribution grant from the earlier repositories;
the team does not claim one, and attribution alone is not described as such a
grant. The source-chain uncertainty remains disclosed even though the team's
release decision has been made.
