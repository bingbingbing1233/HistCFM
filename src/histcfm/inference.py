"""Formal single-checkpoint HistCFM inference.

Derived from the cell-level inference entry in SydneyBioX/GHIST:
https://github.com/SydneyBioX/GHIST

Substantially modified for HistCFM on 2026-08-13 to use an explicit formal
checkpoint, strict configuration compatibility, caller-owned paths, reusable
training normalization, prediction-only inputs, and evaluation-free outputs.
The audited CFM prior/noise and iterative sampling remain in ``HistCFM``.
Distributed under GNU GPL version 3 only; see ``LICENSE``.
"""

import json
import hashlib
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import CheckpointError, load_checkpoint
from ._version import __version__
from .config import (
    DataConfig,
    FlowConfig,
    HistCFMConfig,
    LossConfig,
    ModelConfig,
    PathLike,
    RuntimeConfig,
    SonrmConfig,
    TrainingConfig,
    UniConfig,
    write_config,
)
from .data import (
    HistCFMDataset,
    validate_histcfm_ready_inputs,
    validate_required_files,
    validate_uni_feature_store,
)
from .data.image_io import load_image
from .train import build_model, seed_worker, set_seed


ConfigInput = Union[HistCFMConfig, PathLike]
SUPPORTED_SPLITS = {"validation": "val", "prediction": "predict"}


def _config_from_mapping(raw: Any) -> HistCFMConfig:
    """Parse strict groups without applying training-only path requirements."""

    if not isinstance(raw, Mapping):
        raise TypeError("Inference configuration must be a mapping")
    groups = {"data", "model", "flow", "uni", "sonrm", "loss", "training", "runtime"}
    unknown = sorted(set(raw) - groups)
    missing = sorted(groups - set(raw))
    if unknown:
        raise ValueError("Unknown inference configuration group(s): " + ", ".join(unknown))
    if missing:
        raise ValueError("Missing inference configuration group(s): " + ", ".join(missing))
    return HistCFMConfig(
        data=DataConfig.from_mapping(raw["data"]),
        model=ModelConfig.from_mapping(raw["model"]),
        flow=FlowConfig.from_mapping(raw["flow"]),
        uni=UniConfig.from_mapping(raw["uni"]),
        sonrm=SonrmConfig.from_mapping(raw["sonrm"]),
        loss=LossConfig.from_mapping(raw["loss"]),
        training=TrainingConfig.from_mapping(raw["training"]),
        runtime=RuntimeConfig.from_mapping(raw["runtime"]),
    )


def _load_inference_config(config: ConfigInput) -> HistCFMConfig:
    if isinstance(config, HistCFMConfig):
        return config
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read HistCFM YAML configuration") from error
    with Path(config).open("r", encoding="utf-8") as handle:
        return _config_from_mapping(yaml.safe_load(handle))


def _ordered_cell_types(mapping: Mapping[str, int]) -> List[str]:
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> HistCFMConfig:
    config = _config_from_mapping(checkpoint["resolved_config"])
    if list(config.data.genes) != list(checkpoint["genes"]):
        raise CheckpointError("Checkpoint genes disagree with its resolved_config")
    classes = _ordered_cell_types(checkpoint["cell_type_mapping"])
    if list(config.data.cell_types) != classes:
        raise CheckpointError("Checkpoint cell-type mapping disagrees with resolved_config")
    if config.model.use_cell_types and not classes:
        raise CheckpointError("Checkpoint enables cell types but has no class mapping")
    if config.data.normalization_path != checkpoint["normalization_artifact"]:
        raise CheckpointError(
            "Checkpoint normalization metadata disagrees with resolved_config"
        )
    return config


def _compare_checkpoint_config(
    runtime: HistCFMConfig,
    trained: HistCFMConfig,
    checkpoint: Mapping[str, Any],
) -> None:
    comparisons = {
        "data.genes": (runtime.data.genes, checkpoint["genes"]),
        "data.cell_types": (
            runtime.data.cell_types,
            _ordered_cell_types(checkpoint["cell_type_mapping"]),
        ),
        "data.patch_height": (runtime.data.patch_height, trained.data.patch_height),
        "data.patch_width": (runtime.data.patch_width, trained.data.patch_width),
        "data.max_cells_per_patch": (
            runtime.data.max_cells_per_patch,
            trained.data.max_cells_per_patch,
        ),
        "data.expression_scale": (
            runtime.data.expression_scale,
            trained.data.expression_scale,
        ),
        "model.embedding_dim": (
            runtime.model.embedding_dim,
            trained.model.embedding_dim,
        ),
        "model.use_cell_types": (
            runtime.model.use_cell_types,
            trained.model.use_cell_types,
        ),
        "model.use_neighborhood": (
            runtime.model.use_neighborhood,
            trained.model.use_neighborhood,
        ),
        "model.average_expression_compatibility": (
            runtime.model.average_expression_compatibility,
            trained.model.average_expression_compatibility,
        ),
        "flow.hidden_dim": (runtime.flow.hidden_dim, trained.flow.hidden_dim),
        "flow.num_layers": (runtime.flow.num_layers, trained.flow.num_layers),
        "flow.k_neighbors": (runtime.flow.k_neighbors, trained.flow.k_neighbors),
        "flow.inference_steps": (
            runtime.flow.inference_steps,
            trained.flow.inference_steps,
        ),
        "flow.prior": (runtime.flow.prior, trained.flow.prior),
        "flow.train_noise_std": (
            runtime.flow.train_noise_std,
            trained.flow.train_noise_std,
        ),
        "flow.inference_noise_std": (
            runtime.flow.inference_noise_std,
            trained.flow.inference_noise_std,
        ),
        "uni.enabled": (runtime.uni.enabled, trained.uni.enabled),
        "uni.mode": (runtime.uni.mode, trained.uni.mode),
        "uni.feature_dim": (runtime.uni.feature_dim, trained.uni.feature_dim),
        "uni.fusion_method": (
            runtime.uni.fusion_method,
            trained.uni.fusion_method,
        ),
        "uni.fusion_hidden_dim": (
            runtime.uni.fusion_hidden_dim,
            trained.uni.fusion_hidden_dim,
        ),
        "uni.fusion_dropout": (
            runtime.uni.fusion_dropout,
            trained.uni.fusion_dropout,
        ),
    }
    mismatches = [
        f"{name}: runtime={actual!r}, checkpoint={expected!r}"
        for name, (actual, expected) in comparisons.items()
        if actual != expected
    ]
    metadata = checkpoint["model_metadata"]
    expected_classes = len(checkpoint["cell_type_mapping"])
    if metadata["n_classes"] != expected_classes:
        mismatches.append("checkpoint model_metadata.n_classes is inconsistent")
    if metadata["n_genes"] != len(runtime.data.genes):
        mismatches.append("checkpoint model_metadata.n_genes is inconsistent")
    if metadata["fusion_method"] != trained.uni.fusion_method:
        mismatches.append("checkpoint model_metadata.fusion_method is inconsistent")
    if mismatches:
        raise CheckpointError(
            "Inference configuration changes checkpoint-bound behavior:\n- "
            + "\n- ".join(mismatches)
        )


def _validate_runtime_paths(config: HistCFMConfig, split: str) -> None:
    required = {
        "histology": config.data.histology_path,
        "nucleus_mask": config.data.nucleus_mask_path,
        "matched_nuclei": config.data.matched_nuclei_path,
    }
    if split == "validation":
        required["expression"] = config.data.expression_path
        if config.model.use_cell_types:
            required["cell_type"] = config.data.cell_type_path
    if config.model.average_expression_compatibility:
        required["average_expression"] = config.data.average_expression_path
    if config.uni.enabled:
        required["uni_index"] = config.uni.index_path
        required["uni_features"] = config.uni.features_path
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Missing required inference path(s): " + ", ".join(missing))
    validate_required_files(required)


def _normalization_path(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> Path:
    if checkpoint_path.parent.name != "checkpoints":
        raise CheckpointError(
            "Formal checkpoint must remain inside its run's checkpoints directory "
            "so the normalization artifact can be located safely"
        )
    run_root = checkpoint_path.parent.parent.resolve()
    relative = Path(checkpoint["normalization_artifact"])
    resolved = (run_root / relative).resolve()
    if resolved != run_root and run_root not in resolved.parents:
        raise CheckpointError("Normalization artifact escapes the checkpoint run directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Training normalization artifact is missing: {resolved}")
    _validate_normalization_file(resolved)
    return resolved


def _validate_normalization_file(path: PathLike) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Training normalization artifact is missing: {resolved}"
        )
    values = np.load(resolved, allow_pickle=False)
    if values.shape != (2, 3):
        raise ValueError(f"Histology normalization must have shape (2, 3); got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Histology normalization contains NaN or infinite values")
    if np.any(values[1, :] <= 0):
        raise ValueError("Histology normalization standard deviations must be positive")
    return resolved


def _uni_paths(config: HistCFMConfig) -> Tuple[Optional[Path], Optional[Path]]:
    if not config.uni.enabled:
        return None, None
    return Path(config.uni.index_path), Path(config.uni.features_path)


def _patch_statistics(dataset: HistCFMDataset) -> Tuple[int, int]:
    valid_ids = set(dataset.all_intersect)
    nonempty = 0
    largest = 0
    for row_start, column_start in dataset.coords_starts:
        patch = dataset.nuclei[
            row_start : row_start + dataset.hsize,
            column_start : column_start + dataset.wsize,
        ]
        ids = set(np.unique(patch).tolist())
        ids.discard(0)
        eligible = ids & valid_ids
        if eligible:
            nonempty += 1
            largest = max(largest, len(eligible))
    return nonempty, largest


def _validate_prediction_inputs(
    config: HistCFMConfig,
    dataset: HistCFMDataset,
) -> None:
    histology = load_image(config.data.histology_path)
    mask = load_image(config.data.nucleus_mask_path)
    if histology.ndim != 3 or int(histology.shape[-1]) != 3:
        raise ValueError(f"Histology must be H x W x 3; got {histology.shape}")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(histology.shape[:2]):
        raise ValueError("Nucleus mask must be 2D and match histology spatial dimensions")
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("Nucleus mask must contain non-negative integer IDs")
    mask_ids = set(np.unique(mask).tolist())
    if 0 not in mask_ids:
        raise ValueError("Nucleus mask must reserve ID 0 for background")
    mask_ids.discard(0)
    matched = pd.read_csv(config.data.matched_nuclei_path, index_col=0)
    if not matched.index.is_unique or "size_pix_histology" not in matched.columns:
        raise ValueError("Matched nuclei must have unique IDs and size_pix_histology")
    try:
        matched_ids = {int(value) for value in matched.index.tolist()}
        sizes = matched["size_pix_histology"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Matched-nuclei IDs and sizes must be numeric") from error
    if not np.isfinite(sizes).all() or np.any(sizes < 0):
        raise ValueError("Matched-nuclei sizes must be finite and non-negative")
    if not matched_ids.issubset(mask_ids):
        raise ValueError("Every matched-nuclei ID must be present in the nucleus mask")
    if not dataset.patch_keys or len(dataset.patch_keys) != len(set(dataset.patch_keys)):
        raise ValueError("Prediction patch keys must be non-empty and unique")
    uni_index, uni_features = _uni_paths(config)
    if config.uni.enabled:
        validate_uni_feature_store(
            dataset.patch_keys,
            uni_index,
            uni_features,
            expected_dim=config.uni.feature_dim,
        )


def _validate_cell_type_values(config: HistCFMConfig) -> None:
    if not config.model.use_cell_types:
        return
    table = pd.read_csv(config.data.cell_type_path)
    if "ct" not in table.columns:
        raise ValueError("Cell-type table must contain a 'ct' column")
    if table["ct"].isna().any():
        raise ValueError("Cell-type labels must be non-null")
    numeric = pd.to_numeric(table["ct"], errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Numeric cell-type labels must be finite integers")
        if np.any(values < 0) or np.any(values >= len(config.data.cell_types)):
            raise ValueError("Numeric cell-type labels must index checkpoint cell types")
        return
    labels = set(table["ct"].dropna().astype(str))
    unknown = sorted(labels - set(config.data.cell_types))
    if unknown:
        raise ValueError(f"Cell-type table contains unknown labels: {unknown[:10]}")


def _build_dataset_and_preflight(
    config: HistCFMConfig,
    split: str,
    normalization_path: Path,
) -> HistCFMDataset:
    mode = SUPPORTED_SPLITS[split]
    dataset = HistCFMDataset(
        nucleus_mask_path=config.data.nucleus_mask_path,
        histology_path=config.data.histology_path,
        matched_nuclei_path=config.data.matched_nuclei_path,
        expression_path=(config.data.expression_path if split == "validation" else None),
        cell_type_path=(config.data.cell_type_path if split == "validation" else None),
        gene_names=config.data.genes,
        cell_types=config.data.cell_types,
        patch_height=config.data.patch_height,
        patch_width=config.data.patch_width,
        overlap=config.data.overlap,
        max_cells_per_patch=config.data.max_cells_per_patch,
        min_nucleus_area=config.data.min_nucleus_area,
        expression_scale=config.data.expression_scale,
        validation_division=config.data.validation_split,
        mode=mode,
        use_cell_types=(config.model.use_cell_types and split == "validation"),
        stain_augmentation=False,
        normalization_path=normalization_path,
        normalization_output_dir=None,
    )
    if split == "validation":
        uni_index, uni_features = _uni_paths(config)
        validate_histcfm_ready_inputs(
            histology_path=config.data.histology_path,
            nucleus_mask_path=config.data.nucleus_mask_path,
            matched_nuclei_path=config.data.matched_nuclei_path,
            expression_path=config.data.expression_path,
            cell_type_path=config.data.cell_type_path,
            average_expression_path=config.data.average_expression_path,
            gene_names=config.data.genes,
            validation_division=config.data.validation_split,
            patch_keys=dataset.patch_keys,
            uni_index_path=uni_index,
            uni_feature_path=uni_features,
            expected_uni_dim=config.uni.feature_dim,
            require_cell_types=config.model.use_cell_types,
            require_average_expression=config.model.average_expression_compatibility,
            require_uni=config.uni.enabled,
        )
        _validate_cell_type_values(config)
    else:
        _validate_prediction_inputs(config, dataset)
    nonempty, largest = _patch_statistics(dataset)
    if len(dataset) == 0 or nonempty == 0:
        raise ValueError(f"{split} split contains no patch with an eligible cell")
    if largest > config.data.max_cells_per_patch:
        raise ValueError(
            "An inference patch exceeds data.max_cells_per_patch: "
            f"observed={largest}, configured={config.data.max_cells_per_patch}"
        )
    return dataset


def _prepare_average_expression(
    config: HistCFMConfig,
    expected_count: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not config.model.average_expression_compatibility:
        if expected_count != 0:
            raise CheckpointError("Checkpoint reference-profile metadata is inconsistent")
        return None
    table = pd.read_csv(config.data.average_expression_path, index_col=0)
    if list(table.columns) != list(config.data.genes):
        raise ValueError("Average-expression genes must exactly match checkpoint genes")
    if int(table.shape[0]) != expected_count:
        raise CheckpointError(
            "Average-expression reference count does not match checkpoint metadata"
        )
    values = config.data.expression_scale * table.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Average-expression values must be finite")
    return torch.from_numpy(values).float().to(device)


def _resolve_device(config: HistCFMConfig) -> torch.device:
    if config.runtime.device == "cpu":
        return torch.device("cpu")
    if config.runtime.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device is cuda but CUDA is unavailable")
        if config.runtime.gpu_index >= torch.cuda.device_count():
            raise ValueError("runtime.gpu_index is outside the available CUDA devices")
        torch.cuda.set_device(config.runtime.gpu_index)
        return torch.device(f"cuda:{config.runtime.gpu_index}")
    if torch.cuda.is_available():
        if config.runtime.gpu_index >= torch.cuda.device_count():
            raise ValueError("runtime.gpu_index is outside the available CUDA devices")
        torch.cuda.set_device(config.runtime.gpu_index)
        return torch.device(f"cuda:{config.runtime.gpu_index}")
    return torch.device("cpu")


def _prepare_output(
    output_dir: Path,
    config: HistCFMConfig,
    checkpoint_path: Path,
) -> Path:
    resolved_output = output_dir.resolve()
    run_root = checkpoint_path.resolve().parent.parent
    if resolved_output == run_root or run_root in resolved_output.parents:
        raise ValueError("Inference output_dir must be outside the checkpoint training run")
    input_paths = [
        config.data.histology_path,
        config.data.nucleus_mask_path,
        config.data.matched_nuclei_path,
        config.data.expression_path,
        config.data.cell_type_path,
        config.data.average_expression_path,
    ]
    for raw in input_paths:
        if raw is None:
            continue
        input_dir = Path(raw).resolve().parent
        if resolved_output == input_dir or input_dir in resolved_output.parents:
            raise ValueError("Inference output_dir must be outside every input-data directory")
    if config.uni.enabled:
        for uni_path in (config.uni.index_path, config.uni.features_path):
            uni_dir = Path(uni_path).resolve().parent
            if resolved_output == uni_dir or uni_dir in resolved_output.parents:
                raise ValueError(
                    "Inference output_dir must be outside each UNI input directory"
                )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not config.runtime.overwrite_output:
        raise FileExistsError(
            f"Refusing non-empty output_dir without runtime.overwrite_output=true: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _sanitized_config(
    config: HistCFMConfig,
    normalization_artifact: str,
) -> Dict[str, Any]:
    result = asdict(config)
    for name in (
        "histology_path",
        "nucleus_mask_path",
        "matched_nuclei_path",
        "expression_path",
        "cell_type_path",
        "average_expression_path",
    ):
        value = result["data"][name]
        result["data"][name] = None if value is None else "<external-input>"
    result["data"]["normalization_path"] = normalization_artifact
    for name in ("index_path", "features_path"):
        if result["uni"][name] is not None:
            result["uni"][name] = "<external-feature-store>"
    return result


def _deduplicate(
    cell_ids: Sequence[int],
    areas: np.ndarray,
) -> List[int]:
    table = pd.DataFrame(
        {
            "cell_id": list(cell_ids),
            "area": areas.reshape(-1),
            "position": np.arange(len(cell_ids)),
        }
    )
    selected = table.sort_values("area", ascending=False).drop_duplicates(
        "cell_id", keep="first"
    )
    return sorted(selected["position"].astype(int).tolist())


def infer(
    config: ConfigInput,
    checkpoint_path: PathLike,
    output_dir: PathLike,
    split: str = "validation",
) -> Mapping[str, str]:
    """Run one CFM sample from one explicit formal HistCFM checkpoint."""

    if split not in SUPPORTED_SPLITS:
        raise ValueError("split must be 'validation' or 'prediction'")
    runtime = _load_inference_config(config)
    _validate_runtime_paths(runtime, split)
    checkpoint_file = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    trained = _checkpoint_config(checkpoint)
    _compare_checkpoint_config(runtime, trained, checkpoint)
    normalization = _normalization_path(checkpoint_file, checkpoint)
    dataset = _build_dataset_and_preflight(runtime, split, normalization)

    actual_seed = runtime.training.seed
    if actual_seed is None:
        actual_seed = checkpoint["seed"]
    if actual_seed is not None:
        set_seed(actual_seed, deterministic=runtime.runtime.deterministic)
    device = _resolve_device(runtime)
    reference_count = checkpoint["model_metadata"]["reference_profile_count"]
    average_expression = _prepare_average_expression(runtime, reference_count, device)

    trained.uni.index_path = runtime.uni.index_path
    trained.uni.features_path = runtime.uni.features_path
    trained.runtime.device = runtime.runtime.device
    trained.runtime.gpu_index = runtime.runtime.gpu_index
    model = build_model(trained, device, reference_count)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise CheckpointError(
            "Strict model-state loading failed; no legacy key fallback is supported"
        ) from error
    model.to(device)
    model.eval()

    generator = None
    worker_init = None
    if actual_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(actual_seed)
        worker_init = seed_worker
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=runtime.training.batch_size,
        shuffle=False,
        num_workers=runtime.training.workers,
        drop_last=False,
        worker_init_fn=worker_init,
        generator=generator,
    )
    if len(dataloader) == 0:
        raise ValueError("Inference DataLoader contains zero batches")

    output = _prepare_output(Path(output_dir), runtime, checkpoint_file)
    normalized_relative = checkpoint["normalization_artifact"]
    public_config = _sanitized_config(runtime, normalized_relative)
    write_config(public_config, output / "resolved_inference_config.yaml")

    all_ids: List[int] = []
    all_areas: List[np.ndarray] = []
    all_patch_keys: List[str] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    predicted_types: List[int] = []
    target_types: List[int] = []
    with torch.inference_mode():
        for (
            batch_nuclei,
            _,
            batch_histology,
            batch_expression,
            batch_n_cells,
            batch_cell_type,
            patch_ids,
            patch_keys,
        ) in tqdm(dataloader, desc="HistCFM inference"):
            batch_nuclei = batch_nuclei.to(device)
            batch_histology = batch_histology.to(device)
            batch_n_cells = batch_n_cells.to(device)
            patch_ids = patch_ids.to(device)
            if split == "validation":
                batch_expression_input = batch_expression.to(device)
                batch_cell_type_input = batch_cell_type.to(device)
            else:
                batch_expression_input = None
                batch_cell_type_input = None
            outputs = model(
                batch_histology,
                batch_nuclei,
                batch_n_cells,
                average_expression,
                batch_cell_type_input,
                batch_expression_input,
                patch_ids=patch_ids,
                patch_keys=patch_keys,
            )
            out_cell_type = outputs[0]
            out_expression = outputs[3]
            batch_cell_type_flat = outputs[2]
            batch_expression_flat = outputs[11]
            batch_area = outputs[13]
            patch_ids_flat = outputs[14]
            if out_expression.shape[0] == 0:
                continue
            out_expression = torch.relu(out_expression)
            predictions.append(
                (out_expression.detach().cpu().numpy() / runtime.data.expression_scale)
            )
            if split == "validation":
                targets.append(
                    batch_expression_flat.detach().cpu().numpy()
                    / runtime.data.expression_scale
                )
            area_array = batch_area.detach().cpu().numpy().reshape(-1)
            id_array = patch_ids_flat.detach().cpu().numpy().reshape(-1)
            all_areas.append(area_array)
            all_ids.extend(int(value) for value in id_array.tolist())
            for batch_index in range(int(batch_n_cells.shape[0])):
                count = int(batch_n_cells[batch_index])
                all_patch_keys.extend([str(patch_keys[batch_index])] * count)
            if runtime.model.use_cell_types:
                predicted_types.extend(
                    int(value)
                    for value in torch.argmax(out_cell_type, dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                if split == "validation":
                    target_types.extend(
                        int(value)
                        for value in batch_cell_type_flat.detach().cpu().numpy().tolist()
                    )

    if not predictions or not all_ids:
        raise RuntimeError("No eligible cells were produced by inference")
    prediction_values = np.vstack(predictions)
    area_values = np.concatenate(all_areas)
    if len(all_ids) != prediction_values.shape[0] or len(all_ids) != area_values.shape[0]:
        raise RuntimeError("Inference cell IDs, predictions, and areas are misaligned")
    selected = _deduplicate(all_ids, area_values)
    cell_ids = [all_ids[index] for index in selected]
    patch_keys_selected = [all_patch_keys[index] for index in selected]
    prediction_values = prediction_values[selected, :]
    area_values = area_values[selected]

    prediction_table = pd.DataFrame(prediction_values, columns=checkpoint["genes"])
    prediction_table.insert(0, "cell_id", cell_ids)
    prediction_table.to_csv(output / "predictions.csv", index=False)

    written: Dict[str, str] = {
        "predictions": str(output / "predictions.csv"),
        "cells": str(output / "cells.csv"),
        "metadata": str(output / "metadata.json"),
        "resolved_config": str(output / "resolved_inference_config.yaml"),
    }
    if split == "validation":
        target_values = np.vstack(targets)[selected, :]
        target_table = pd.DataFrame(target_values, columns=checkpoint["genes"])
        target_table.insert(0, "cell_id", cell_ids)
        target_table.to_csv(output / "targets.csv", index=False)
        written["targets"] = str(output / "targets.csv")

    if runtime.model.use_cell_types:
        classes = _ordered_cell_types(checkpoint["cell_type_mapping"])
        type_table = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "predicted_label": [classes[predicted_types[index]] for index in selected],
            }
        )
        if split == "validation":
            type_table.insert(
                1,
                "ground_truth_label",
                [classes[target_types[index]] for index in selected],
            )
        type_table.to_csv(output / "cell_types.csv", index=False)
        written["cell_types"] = str(output / "cell_types.csv")

    public_slide_id = "slide-" + hashlib.sha256(
        dataset.slide_id.encode("utf-8")
    ).hexdigest()[:16]
    patch_parts = [key.split("|") for key in patch_keys_selected]
    if any(len(parts) != 6 for parts in patch_parts):
        raise RuntimeError("Canonical patch-key parsing failed")
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "slide_id": [public_slide_id] * len(cell_ids),
            "patch_row_start": [int(parts[1]) for parts in patch_parts],
            "patch_column_start": [int(parts[2]) for parts in patch_parts],
            "patch_height": [int(parts[3]) for parts in patch_parts],
            "patch_width": [int(parts[4]) for parts in patch_parts],
            "pyramid_level": [int(parts[5]) for parts in patch_parts],
            "nucleus_area_pixels": area_values,
        }
    ).to_csv(output / "cells.csv", index=False)

    metadata = {
        "checkpoint_file": checkpoint_file.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_schema_version": checkpoint["schema_version"],
        "checkpoint_histcfm_version": checkpoint["histcfm_version"],
        "seed": actual_seed,
        "split": split,
        "genes": checkpoint["genes"],
        "cell_type_mapping": checkpoint["cell_type_mapping"],
        "normalization_artifact": normalized_relative,
        "model": {
            "embedding_dim": trained.model.embedding_dim,
            "flow_hidden_dim": trained.flow.hidden_dim,
            "flow_layers": trained.flow.num_layers,
            "inference_steps": trained.flow.inference_steps,
            "prior": trained.flow.prior,
            "inference_noise_std": trained.flow.inference_noise_std,
            "uni_enabled": trained.uni.enabled,
            "uni_mode": trained.uni.mode,
            "uni_feature_dim": trained.uni.feature_dim,
            "fusion_method": trained.uni.fusion_method,
        },
        "input": {
            "slide_id": public_slide_id,
            "patch_count": len(dataset),
        },
        "output_expression_scale": "log1p(count)",
        "sampling": "one stochastic CFM sample",
        "runtime": {
            "histcfm_version": __version__,
            "torch_version": str(torch.__version__),
            "cuda_version": getattr(torch.version, "cuda", None),
        },
    }
    _write_metadata_json(metadata, output / "metadata.json")
    return written


def _write_metadata_json(metadata: Mapping[str, Any], path: PathLike) -> None:
    """Write standards-compliant inference metadata and reject NaN/Infinity."""

    serialized = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def validate_inference_data(
    config: ConfigInput,
    checkpoint_path: PathLike,
    split: str = "validation",
) -> Mapping[str, Any]:
    """Read-only preflight for checkpoint-bound validation or prediction data."""

    if split not in SUPPORTED_SPLITS:
        raise ValueError("split must be 'validation' or 'prediction'")
    runtime = _load_inference_config(config)
    _validate_runtime_paths(runtime, split)
    checkpoint_file = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    trained = _checkpoint_config(checkpoint)
    _compare_checkpoint_config(runtime, trained, checkpoint)
    normalization = _normalization_path(checkpoint_file, checkpoint)
    dataset = _build_dataset_and_preflight(runtime, split, normalization)
    return {
        "split": split,
        "checkpoint_schema_version": checkpoint["schema_version"],
        "checkpoint_histcfm_version": checkpoint["histcfm_version"],
        "normalization_path": normalization,
        "patch_count": len(dataset),
    }
