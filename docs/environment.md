# HistCFM environment

Use a separate Conda environment for HistCFM. The recommended stack in
`environment.yml` is:

```text
Python 3.10.14
PyTorch 2.1.1
torchvision 0.16.1
PyTorch CUDA runtime 12.1
NumPy 1.26.4
Linux x86-64 with an NVIDIA GPU and compatible driver
```

A compatible NVIDIA driver is required for GPU execution. CPU execution is
also supported by setting `runtime.device: cpu`.

## Dependencies

| Package | Role | Public synthetic demo | Declaration |
| --- | --- | --- | --- |
| `pytorch` / Python import `torch` | Model, training, checkpoint, inference | Required | Conda `pytorch`; PyPI metadata `torch` |
| `torchvision` | Image tensor transforms | Required | Core |
| `numpy` | Arrays, features, data validation, metrics | Required | Core |
| `pandas` | Aligned tables and outputs | Required | Core |
| `PyYAML` / Python import `yaml` | Strict configuration and resolved configs | Required | Core |
| `imageio`, `tifffile` | Image input backends | Required | Core |
| `natsort` | Deterministic source-compatible patch ordering | Required | Core |
| `tqdm` | Training and inference progress | Required | Core |
| `scikit-learn` / Python import `sklearn` | Optional-label evaluator accuracy and macro-F1 | Required by the formal evaluator path | Core |
| `Pillow` / Python import `PIL` | Explicit image-stack support; also required transitively by torchvision | Required by the environment, though not directly imported by HistCFM source | Environment |
| `pytest` | Contract tests | Required for release validation, not package runtime | Environment and `test` extra |
| `pip`, `setuptools`, `wheel` | Local editable package installation | Required for installation only | Environment/build |
| `stainlib` | Optional stain augmentation, imported only when enabled | Not required; demo sets augmentation to false | `stain` extra only |

Training and inference use the shared core data/model packages. Evaluation
adds scikit-learn only when cell-type metrics are requested. The committed
synthetic-data generator uses only the Python standard library; users do not
need to rerun it before the demo.

The formal repository does **not** import or require `timm`, an online UNI
encoder, a UNI repository, or a UNI checkpoint. Historical GHIST-only imports
such as `h5py`, OpenCV, `torchstain`, scientific plotting packages, and notebook
packages were not migrated because the formal package does not use them.

## Create and install

First verify that an environment with this name does not already exist. Never
overwrite or clone another environment:

```bash
conda env list
conda env create -f environment.yml
conda activate histcfm
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_environment.py
```

The editable installation is supported by the setuptools `src/` layout in
`pyproject.toml`. Dependencies are resolved once by Conda; `--no-deps` and
`--no-build-isolation` prevent pip from performing a second dependency
resolution or creating an isolated build environment. Do not use the install
command until `environment.yml` has completed successfully.

No `requirements.txt` is maintained: `pyproject.toml` is the Python package
metadata, while `environment.yml` is the Linux/CUDA validation environment.
Maintaining a third dependency list would create another unsynchronized source
of truth.

## External UNI boundary

The public smoke demo reads committed, deterministic synthetic morphology
features. It requires no UNI code, weights, checkpoint, `timm`, download, or
network access. Prepared real-data users must separately obtain authorized
access through UNI's official channel and generate real features under the
applicable terms. HistCFM neither downloads nor redistributes that software or
its weights.
