#!/usr/bin/env python3
"""Read-only dependency and CUDA preflight for an installed HistCFM package.

This utility performs imports and prints environment facts.  It does not
install packages, contact a network, read project data, select a CUDA device,
or run any HistCFM workflow.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from typing import Callable, Optional


def _version(module: object) -> str:
    value = getattr(module, "__version__", None)
    return "unknown" if value is None else str(value)


def _load(
    import_name: str,
    display_name: str,
    failures: list[str],
    version_getter: Optional[Callable[[object], str]] = None,
) -> Optional[object]:
    try:
        module = importlib.import_module(import_name)
    except Exception as error:  # report the exact import failure without hiding it
        failures.append(f"{display_name}: {type(error).__name__}: {error}")
        print(f"{display_name}: IMPORT FAILED ({type(error).__name__}: {error})")
        return None
    getter = version_getter or _version
    print(f"{display_name}: {getter(module)}")
    return module


def main() -> int:
    failures: list[str] = []
    print(f"Python: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    torch = _load("torch", "PyTorch", failures)
    _load("torchvision", "torchvision", failures)
    _load("numpy", "NumPy", failures)
    _load("pandas", "pandas", failures)
    _load("sklearn", "scikit-learn", failures)
    _load("yaml", "PyYAML", failures)
    _load("tifffile", "tifffile", failures)
    _load("imageio", "imageio", failures)
    _load("natsort", "natsort", failures)
    _load("tqdm", "tqdm", failures)
    _load("PIL", "Pillow", failures)
    _load("pytest", "pytest", failures)
    histcfm = _load("histcfm", "HistCFM", failures)

    cli = shutil.which("histcfm")
    print(f"HistCFM CLI: {cli if cli is not None else 'NOT FOUND'}")
    if cli is None:
        failures.append("histcfm console command was not found on PATH")

    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        print(f"CUDA available: {cuda}")
        print(f"PyTorch CUDA runtime: {getattr(torch.version, 'cuda', None)}")
        print(f"Visible CUDA devices: {torch.cuda.device_count() if cuda else 0}")
        print(f"GPU 0: {torch.cuda.get_device_name(0) if cuda else None}")
        if not cuda:
            failures.append("CUDA is not available to PyTorch")

    if histcfm is not None and not getattr(histcfm, "__version__", None):
        failures.append("histcfm.__version__ is missing or empty")

    if failures:
        print("Environment preflight: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Environment preflight: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
