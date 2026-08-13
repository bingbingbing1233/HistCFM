"""Checkpoint creation for trusted, locally generated HistCFM runs.

This release-side module was written during the 2026-08-12 reorganization; it
is not a verbatim research file. PyTorch checkpoints use pickle internally and
must not be loaded from untrusted sources. This stage intentionally provides
a restricted ``weights_only=True`` loader with no unsafe fallback. Distributed
under GNU GPL version 3 only; see ``LICENSE``.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import warnings

import torch

from ._version import __version__


PathLike = Union[str, Path]
CHECKPOINT_SCHEMA_VERSION = 3


class CheckpointError(ValueError):
    """Raised when a checkpoint does not satisfy the formal release schema."""


def build_checkpoint(
    *,
    epoch: int,
    model,
    optimizer,
    include_optimizer: bool,
    resolved_config: Mapping[str, Any],
    genes: Sequence[str],
    cell_types: Sequence[str],
    normalization_artifact: str,
    seed: Optional[int],
) -> Mapping[str, Any]:
    """Build the stable metadata envelope without changing state-dict keys."""

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if include_optimizer and optimizer is not None else None
        ),
        "resolved_config": dict(resolved_config),
        "genes": list(genes),
        "cell_type_mapping": {
            name: index for index, name in enumerate(cell_types)
        },
        "normalization_artifact": str(normalization_artifact),
        "seed": seed,
        "histcfm_version": __version__,
        "torch_version": str(torch.__version__),
        "cuda_version": None if cuda_version is None else str(cuda_version),
        "model_metadata": {
            "n_classes": int(model.n_classes),
            "n_genes": int(model.n_genes),
            "reference_profile_count": int(model.n_ref),
            "fusion_method": str(model.fusion_type),
        },
    }
    return checkpoint


def save_checkpoint(
    path: PathLike,
    *,
    epoch: int,
    model,
    optimizer,
    include_optimizer: bool,
    resolved_config: Mapping[str, Any],
    genes: Sequence[str],
    cell_types: Sequence[str],
    normalization_artifact: str,
    seed: Optional[int],
) -> Path:
    """Save one trusted run checkpoint to a caller-created output directory."""

    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory must already exist: {destination.parent}"
        )
    payload = build_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        include_optimizer=include_optimizer,
        resolved_config=resolved_config,
        genes=genes,
        cell_types=cell_types,
        normalization_artifact=normalization_artifact,
        seed=seed,
    )
    torch.save(payload, destination)
    return destination


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"Checkpoint {label} must be a mapping")
    return value


def validate_checkpoint(payload: Any) -> Dict[str, Any]:
    """Validate a restricted-load payload without modifying its state dict."""

    checkpoint = dict(_require_mapping(payload, "payload"))
    required = {
        "schema_version",
        "epoch",
        "model_state_dict",
        "resolved_config",
        "genes",
        "cell_type_mapping",
        "normalization_artifact",
        "seed",
        "histcfm_version",
        "torch_version",
        "cuda_version",
        "model_metadata",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise CheckpointError(
            "Checkpoint is missing required field(s): " + ", ".join(missing)
        )
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "Unsupported checkpoint schema version: "
            f"{checkpoint['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if isinstance(checkpoint["epoch"], bool) or not isinstance(checkpoint["epoch"], int):
        raise CheckpointError("Checkpoint epoch must be an integer")
    if checkpoint["epoch"] < 1:
        raise CheckpointError("Checkpoint epoch must be >= 1")
    state_dict = _require_mapping(checkpoint["model_state_dict"], "model_state_dict")
    if not state_dict or any(not isinstance(key, str) for key in state_dict):
        raise CheckpointError("Checkpoint model_state_dict must have string keys and not be empty")
    resolved = _require_mapping(checkpoint["resolved_config"], "resolved_config")
    expected_groups = {"data", "model", "flow", "uni", "sonrm", "loss", "training", "runtime"}
    if set(resolved) != expected_groups:
        raise CheckpointError(
            "Checkpoint resolved_config groups do not match the formal schema"
        )
    genes = checkpoint["genes"]
    if (
        not isinstance(genes, list)
        or not genes
        or any(not isinstance(gene, str) or not gene for gene in genes)
        or len(genes) != len(set(genes))
    ):
        raise CheckpointError("Checkpoint genes must be a non-empty ordered unique string list")
    mapping = _require_mapping(checkpoint["cell_type_mapping"], "cell_type_mapping")
    if any(not isinstance(name, str) for name in mapping):
        raise CheckpointError("Checkpoint cell_type_mapping keys must be strings")
    indices = list(mapping.values())
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise CheckpointError("Checkpoint cell_type_mapping values must be integers")
    if sorted(indices) != list(range(len(indices))):
        raise CheckpointError("Checkpoint cell_type_mapping must be contiguous and zero-based")
    artifact = checkpoint["normalization_artifact"]
    if not isinstance(artifact, str) or not artifact:
        raise CheckpointError("Checkpoint normalization_artifact must be a relative path")
    artifact_path = Path(artifact)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise CheckpointError("Checkpoint normalization_artifact must be a safe relative path")
    seed = checkpoint["seed"]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise CheckpointError("Checkpoint seed must be null or a non-negative integer")
    for key in ("histcfm_version", "torch_version"):
        if not isinstance(checkpoint[key], str) or not checkpoint[key]:
            raise CheckpointError(f"Checkpoint {key} must be a non-empty string")
    if checkpoint["cuda_version"] is not None and not isinstance(
        checkpoint["cuda_version"], str
    ):
        raise CheckpointError("Checkpoint cuda_version must be null or a string")
    model_metadata = _require_mapping(checkpoint["model_metadata"], "model_metadata")
    if set(model_metadata) != {
        "n_classes",
        "n_genes",
        "reference_profile_count",
        "fusion_method",
    }:
        raise CheckpointError("Checkpoint model_metadata has unexpected fields")
    for key in ("n_classes", "n_genes", "reference_profile_count"):
        value = model_metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointError(f"Checkpoint model_metadata.{key} must be non-negative")
    if model_metadata["n_genes"] != len(genes):
        raise CheckpointError("Checkpoint gene metadata is internally inconsistent")
    if model_metadata["n_classes"] != len(mapping):
        raise CheckpointError("Checkpoint cell-type metadata is internally inconsistent")
    if model_metadata["fusion_method"] not in {"film", "gate"}:
        raise CheckpointError("Checkpoint model_metadata.fusion_method is invalid")
    return checkpoint


def load_checkpoint(path: PathLike, map_location: Any = "cpu") -> Dict[str, Any]:
    """Restricted-load one trusted formal checkpoint and validate its schema.

    PyTorch checkpoint files are pickle-backed containers. ``weights_only=True``
    narrows the accepted object set but does not make an untrusted file harmless.
    This function deliberately has no unsafe fallback for older PyTorch releases.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint must be an existing regular file: {checkpoint_path}")
    warnings.warn(
        "Load only a trusted HistCFM checkpoint. PyTorch serialization is pickle-backed; "
        "weights_only=True reduces but does not eliminate untrusted-file risk.",
        UserWarning,
        stacklevel=2,
    )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "This PyTorch version lacks the required weights_only checkpoint loader; "
            "upgrade PyTorch rather than falling back to unrestricted pickle loading"
        ) from error
    return validate_checkpoint(payload)
