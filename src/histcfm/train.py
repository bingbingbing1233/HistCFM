"""Formal HistCFM training entry.

Derived from the audited cell-level training entry in SydneyBioX/GHIST and
modified substantially for HistCFM on 2026-08-12. Changes provide formal names, strict
YAML configuration, explicit caller-owned output paths, preflight validation,
and metadata-rich checkpoints while retaining the audited training math and
loss order. Distributed under GNU GPL version 3 only; see ``LICENSE``.
"""

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .config import HistCFMConfig, PathLike, load_config, write_config
from .data import (
    HistCFMDataset,
    inspect_patch_inventory,
    validate_histcfm_ready_inputs,
)
from .data.dataset import validate_stain_augmentation_dependency
from .losses.sonrm import SONRMLoss
from .models import HistCFM


ConfigInput = Union[HistCFMConfig, PathLike]
NORMALIZATION_ARTIFACT = "artifacts/histology_normalization.npy"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed the sources covered by the audited entry and formal DataLoader."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the worker's PyTorch seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_device(config: HistCFMConfig) -> torch.device:
    requested = config.runtime.device
    index = config.runtime.gpu_index
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device is cuda but CUDA is unavailable")
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"runtime.gpu_index={index} but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if torch.cuda.is_available():
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"runtime.gpu_index={index} but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _prepare_output(output_dir: Path, overwrite: bool) -> Mapping[str, Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing non-empty output_dir without runtime.overwrite_output=true: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": output_dir,
        "artifacts": output_dir / "artifacts",
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
    }
    for name in ("artifacts", "checkpoints", "logs"):
        paths[name].mkdir(exist_ok=True)
    return paths


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("histcfm.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _patch_cell_statistics(dataset: HistCFMDataset) -> Tuple[int, int]:
    valid_ids = set(dataset.all_intersect)
    count = 0
    largest = 0
    for row_start, column_start in dataset.coords_starts:
        patch = dataset.nuclei[
            row_start : row_start + dataset.hsize,
            column_start : column_start + dataset.wsize,
        ]
        patch_ids = set(np.unique(patch).tolist())
        patch_ids.discard(0)
        eligible = patch_ids & valid_ids
        if eligible:
            count += 1
            largest = max(largest, len(eligible))
    return count, largest


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
            raise ValueError("Numeric cell-type labels must index data.cell_types")
        return
    labels = set(table["ct"].dropna().astype(str))
    unknown = sorted(labels - set(config.data.cell_types))
    if unknown:
        raise ValueError(f"Cell-type table contains unknown labels: {unknown[:10]}")


def _uni_store_paths(config: HistCFMConfig) -> Tuple[Optional[Path], Optional[Path]]:
    if not config.uni.enabled:
        return None, None
    return Path(config.uni.index_path), Path(config.uni.features_path)


def validate_training_data(config: ConfigInput) -> Mapping[str, Any]:
    """Read-only preflight for train and validation inputs without normalization."""

    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(resolved, HistCFMConfig):
        raise TypeError("config must be a HistCFMConfig or YAML path")
    if resolved.data.stain_augmentation:
        validate_stain_augmentation_dependency()
    if resolved.data.normalization_path is not None:
        raise ValueError(
            "Training validation requires data.normalization_path=null; "
            "training creates normalization later inside its output directory"
        )
    index_path, features_path = _uni_store_paths(resolved)
    summaries = {}
    for mode in ("train", "validation"):
        inventory = inspect_patch_inventory(
            histology_path=resolved.data.histology_path,
            nucleus_mask_path=resolved.data.nucleus_mask_path,
            matched_nuclei_path=resolved.data.matched_nuclei_path,
            expression_path=resolved.data.expression_path,
            cell_type_path=resolved.data.cell_type_path,
            validation_division=resolved.data.validation_split,
            patch_height=resolved.data.patch_height,
            patch_width=resolved.data.patch_width,
            overlap=resolved.data.overlap,
            min_nucleus_area=resolved.data.min_nucleus_area,
            mode=mode,
            require_cell_types=resolved.model.use_cell_types,
        )
        if inventory.largest_cell_count > resolved.data.max_cells_per_patch:
            raise ValueError(
                f"A {mode} patch exceeds data.max_cells_per_patch: "
                f"observed={inventory.largest_cell_count}, "
                f"configured={resolved.data.max_cells_per_patch}"
            )
        summaries[mode] = validate_histcfm_ready_inputs(
            histology_path=resolved.data.histology_path,
            nucleus_mask_path=resolved.data.nucleus_mask_path,
            matched_nuclei_path=resolved.data.matched_nuclei_path,
            expression_path=resolved.data.expression_path,
            cell_type_path=resolved.data.cell_type_path,
            average_expression_path=resolved.data.average_expression_path,
            gene_names=resolved.data.genes,
            validation_division=resolved.data.validation_split,
            patch_keys=inventory.patch_keys,
            uni_index_path=index_path,
            uni_feature_path=features_path,
            expected_uni_dim=resolved.uni.feature_dim,
            require_cell_types=resolved.model.use_cell_types,
            require_average_expression=resolved.model.average_expression_compatibility,
            require_uni=resolved.uni.enabled,
        )
    _validate_cell_type_values(resolved)
    return summaries


def _build_datasets_and_preflight(
    config: HistCFMConfig,
    artifacts_dir: Path,
) -> Tuple[HistCFMDataset, HistCFMDataset]:
    if config.data.normalization_path is not None:
        raise ValueError(
            "Training must compute its own normalization artifact; "
            "data.normalization_path must be null"
        )
    common = {
        "nucleus_mask_path": config.data.nucleus_mask_path,
        "histology_path": config.data.histology_path,
        "matched_nuclei_path": config.data.matched_nuclei_path,
        "expression_path": config.data.expression_path,
        "cell_type_path": config.data.cell_type_path,
        "gene_names": config.data.genes,
        "cell_types": config.data.cell_types,
        "patch_height": config.data.patch_height,
        "patch_width": config.data.patch_width,
        "overlap": config.data.overlap,
        "max_cells_per_patch": config.data.max_cells_per_patch,
        "min_nucleus_area": config.data.min_nucleus_area,
        "expression_scale": config.data.expression_scale,
        "validation_division": config.data.validation_split,
        "use_cell_types": config.model.use_cell_types,
    }
    train_dataset = HistCFMDataset(
        **common,
        mode="train",
        stain_augmentation=config.data.stain_augmentation,
        normalization_path=None,
        normalization_output_dir=artifacts_dir,
    )
    expected_normalization = artifacts_dir / "histology_normalization.npy"
    if Path(train_dataset.normalization_path).resolve() != expected_normalization.resolve():
        raise RuntimeError("Training normalization was not written to the formal artifact path")
    validation_dataset = HistCFMDataset(
        **common,
        mode="val",
        stain_augmentation=False,
        normalization_path=expected_normalization,
        normalization_output_dir=None,
    )
    train_count, train_largest = _patch_cell_statistics(train_dataset)
    validation_count, validation_largest = _patch_cell_statistics(validation_dataset)
    if train_count == 0:
        raise ValueError("Training split contains no patch with an eligible cell")
    if validation_count == 0:
        raise ValueError("Validation split contains no patch with an eligible cell")
    largest = max(train_largest, validation_largest)
    if largest > config.data.max_cells_per_patch:
        raise ValueError(
            "A formal patch exceeds data.max_cells_per_patch: "
            f"observed={largest}, configured={config.data.max_cells_per_patch}"
        )

    uni_index_path, uni_feature_path = _uni_store_paths(config)
    preflight_common = {
        "histology_path": config.data.histology_path,
        "nucleus_mask_path": config.data.nucleus_mask_path,
        "matched_nuclei_path": config.data.matched_nuclei_path,
        "expression_path": config.data.expression_path,
        "cell_type_path": config.data.cell_type_path,
        "average_expression_path": config.data.average_expression_path,
        "gene_names": config.data.genes,
        "validation_division": config.data.validation_split,
        "uni_index_path": uni_index_path,
        "uni_feature_path": uni_feature_path,
        "expected_uni_dim": config.uni.feature_dim,
        "require_cell_types": config.model.use_cell_types,
        "require_average_expression": config.model.average_expression_compatibility,
        "require_uni": config.uni.enabled,
    }
    validate_histcfm_ready_inputs(
        **preflight_common,
        patch_keys=train_dataset.patch_keys,
    )
    validate_histcfm_ready_inputs(
        **preflight_common,
        patch_keys=validation_dataset.patch_keys,
    )
    _validate_cell_type_values(config)
    return train_dataset, validation_dataset


def _prepare_average_expression(
    config: HistCFMConfig,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    if not config.model.average_expression_compatibility:
        return None, None
    table = pd.read_csv(config.data.average_expression_path, index_col=0)
    if list(table.columns) != list(config.data.genes):
        raise ValueError("Average-expression genes must exactly match configured genes")
    values = config.data.expression_scale * table.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Average-expression values must be numeric and finite")
    return torch.from_numpy(values).float().to(device), int(table.shape[0])


def build_model(
    config: HistCFMConfig,
    device: torch.device,
    n_ref: Optional[int],
) -> HistCFM:
    """Instantiate the formal model with an explicit old-to-new field mapping."""

    return HistCFM(
        len(config.data.cell_types) if config.model.use_cell_types else 0,
        len(config.data.genes),
        config.model.embedding_dim,
        device,
        n_ref,
        config.model.average_expression_compatibility,
        config.model.use_cell_types,
        config.model.use_neighborhood,
        use_flow_expr=True,
        flow_hidden_dim=config.flow.hidden_dim,
        flow_layers=config.flow.num_layers,
        flow_steps=config.flow.inference_steps,
        flow_k_neighbors=config.flow.k_neighbors,
        flow_use_zinb=config.flow.prior == "zinb",
        flow_noise_train_sigma=config.flow.train_noise_std,
        flow_noise_infer_sigma=config.flow.inference_noise_std,
        uni_enable=config.uni.enabled,
        uni_mode=config.uni.mode,
        uni_index_path=config.uni.index_path,
        uni_features_path=config.uni.features_path,
        uni_dim=config.uni.feature_dim,
        fusion_type=config.uni.fusion_method,
        fusion_hidden=config.uni.fusion_hidden_dim,
        fusion_dropout=config.uni.fusion_dropout,
    )


def build_optimizer(config: HistCFMConfig, model: HistCFM):
    """Build the source AdamW parameter groups in their audited order."""

    base_lr = config.training.learning_rate
    parameter_groups = [{"params": model.parameters(), "lr": base_lr, "lr_scale": 1.0}]
    for group in model.uni_provider.get_optimizer_params():
        scale = float(group.get("lr_scale", 1.0))
        parameter_groups.append(
            {"params": group["params"], "lr": base_lr * scale, "lr_scale": scale}
        )
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
        eps=config.training.epsilon,
    )


def _learning_rate(config: HistCFMConfig, epoch: int) -> float:
    schedule = config.training.learning_rate_schedule
    total_epochs = config.training.epochs
    if schedule == "two_stage":
        switch_point = config.training.switch_ratio * total_epochs
        if epoch < int(switch_point):
            return config.training.initial_learning_rate
        remaining = max(1, total_epochs - int(switch_point))
        progress = (epoch - int(switch_point)) / float(remaining)
        return config.training.initial_learning_rate + (
            config.training.final_learning_rate - config.training.initial_learning_rate
        ) * progress
    if schedule == "cosine":
        fraction = epoch / float(total_epochs)
        return config.training.final_learning_rate + 0.5 * (
            config.training.initial_learning_rate - config.training.final_learning_rate
        ) * (1 + np.cos(np.pi * fraction))
    return config.training.learning_rate * (1 - epoch / total_epochs)


def _finite_or_zero(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.isfinite(value).item():
        return torch.tensor(0.0).to(device)
    return value


def _loss_modules(config: HistCFMConfig) -> Mapping[str, nn.Module]:
    return {
        "map": nn.CrossEntropyLoss(reduction="mean"),
        "ct_hist": nn.CrossEntropyLoss(reduction="mean"),
        "expr_ct": nn.CrossEntropyLoss(reduction="mean"),
        "expr_ct_embed": nn.CosineEmbeddingLoss(reduction="mean"),
        "expr": nn.MSELoss(reduction="mean"),
        "expr_immune": nn.MSELoss(reduction="mean"),
        "expr_invasive": nn.MSELoss(reduction="mean"),
        "logits": nn.MSELoss(reduction="mean"),
        "comp_est": nn.KLDivLoss(reduction="batchmean"),
        "comp_gt": nn.KLDivLoss(reduction="batchmean"),
        "sonrm": SONRMLoss(
            k=config.sonrm.neighbors,
            hops=config.sonrm.hops,
            m12=config.sonrm.margin_12,
            m23=config.sonrm.margin_23,
        ),
    }


def compute_losses(
    *,
    config: HistCFMConfig,
    model: HistCFM,
    outputs: Tuple[Any, ...],
    batch_type_patch: torch.Tensor,
    batch_n_cells: torch.Tensor,
    device: torch.device,
    modules: Mapping[str, nn.Module],
) -> Mapping[str, torch.Tensor]:
    """Compute the audited losses without merging alias expression outputs."""

    if len(outputs) != 15:
        raise RuntimeError(f"HistCFM forward must return 15 items; got {len(outputs)}")
    (
        out_cell_type,
        out_map,
        batch_ct_pc,
        out_expr,
        out_expr_immune,
        out_expr_invasive,
        out_expr_hint,
        out_cell_type_expr,
        fv_cell_type_expr,
        out_cell_type_gt_expr,
        fv_cell_type_gt_expr,
        batch_expr_pc,
        comp_estimated,
        _,
        _,
    ) = outputs
    if batch_ct_pc is None or batch_ct_pc.shape[0] == 0:
        return {}

    batch_expr_pc_aug = batch_expr_pc + config.flow.target_noise_std * torch.randn_like(
        batch_expr_pc
    )
    expression = modules["expr"](out_expr, batch_expr_pc_aug)
    segmentation = modules["map"](out_map, batch_type_patch)
    if config.model.use_cell_types:
        histology_cell_type = modules["ct_hist"](out_cell_type, batch_ct_pc)
        expression_cell_type = modules["expr_ct"](out_cell_type_expr, batch_ct_pc)
        expression_embedding = config.loss.expression_embedding * modules[
            "expr_ct_embed"
        ](
            fv_cell_type_expr,
            fv_cell_type_gt_expr,
            target=torch.ones(batch_ct_pc.size(0)).to(device),
        )
        expression_logits = modules["logits"](
            out_cell_type_expr, out_cell_type_gt_expr
        )
    else:
        histology_cell_type = torch.tensor(0.0).to(device)
        expression_cell_type = torch.tensor(0.0).to(device)
        expression_embedding = torch.tensor(0.0).to(device)
        expression_logits = torch.tensor(0.0).to(device)

    if config.model.use_neighborhood:
        expression_immune = modules["expr_immune"](
            out_expr_immune, batch_expr_pc_aug
        )
        expression_invasive = modules["expr_invasive"](
            out_expr_invasive, batch_expr_pc_aug
        )
        denominator = torch.sum(batch_n_cells)
        if denominator.item() > 0:
            composition_estimated_sum = torch.zeros(len(config.data.cell_types)).to(device)
            for batch_index in range(int(batch_n_cells.shape[0])):
                n_cells = int(batch_n_cells[batch_index])
                if n_cells > 0:
                    composition_estimated_sum += n_cells * comp_estimated[batch_index, :]
            composition_estimated_sum = composition_estimated_sum / denominator
            composition_ground_truth = torch.nn.functional.one_hot(
                batch_ct_pc, num_classes=len(config.data.cell_types)
            ).float()
            composition_ground_truth = torch.mean(composition_ground_truth, 0)
            composition_histology_raw = torch.nn.functional.softmax(
                out_cell_type, dim=1
            )
            composition_histology_raw = torch.argmax(composition_histology_raw, 1)
            composition_histology = torch.nn.functional.one_hot(
                composition_histology_raw, num_classes=len(config.data.cell_types)
            ).float()
            composition_histology = torch.mean(composition_histology, 0)
            kl_epsilon = 10e-12
            composition_estimated_kl = F.softmax(
                composition_estimated_sum + kl_epsilon, dim=0
            )
            composition_histology_kl = F.softmax(
                composition_histology + kl_epsilon, dim=0
            )
            composition_ground_truth_kl = F.softmax(
                composition_ground_truth + kl_epsilon, dim=0
            )
            estimated_composition = modules["comp_est"](
                torch.log(composition_estimated_kl), composition_ground_truth_kl
            )
            histology_composition = modules["comp_gt"](
                torch.log(composition_histology_kl), composition_ground_truth_kl
            )
        else:
            estimated_composition = torch.tensor(0.0).to(device)
            histology_composition = torch.tensor(0.0).to(device)
    else:
        estimated_composition = torch.tensor(0.0).to(device)
        histology_composition = torch.tensor(0.0).to(device)
        expression_immune = torch.tensor(0.0).to(device)
        expression_invasive = torch.tensor(0.0).to(device)

    uni_hint = torch.tensor(0.0).to(device)
    if config.uni.enabled and out_expr_hint is not None:
        uni_hint = config.loss.uni_hint * modules["expr"](out_expr, out_expr_hint)

    guarded = {
        "segmentation": _finite_or_zero(segmentation, device),
        "histology_cell_type": _finite_or_zero(histology_cell_type, device),
        "expression_cell_type": _finite_or_zero(expression_cell_type, device),
        "endpoint_expression": _finite_or_zero(expression, device),
        "neighborhood_expression_immune": _finite_or_zero(expression_immune, device),
        "neighborhood_expression_invasive": _finite_or_zero(expression_invasive, device),
        "expression_embedding": _finite_or_zero(expression_embedding, device),
        "expression_logits": _finite_or_zero(expression_logits, device),
        "estimated_composition": _finite_or_zero(estimated_composition, device),
        "histology_composition": _finite_or_zero(histology_composition, device),
        "uni_hint": uni_hint,
    }
    total = (
        config.loss.segmentation * guarded["segmentation"]
        + config.loss.histology_cell_type * guarded["histology_cell_type"]
        + config.loss.expression_cell_type * guarded["expression_cell_type"]
        + config.loss.endpoint_expression * guarded["endpoint_expression"]
        + config.loss.neighborhood_expression_immune
        * guarded["neighborhood_expression_immune"]
        + config.loss.neighborhood_expression_invasive
        * guarded["neighborhood_expression_invasive"]
        + guarded["expression_embedding"]
        + config.loss.expression_logits * guarded["expression_logits"]
        + config.loss.estimated_composition * guarded["estimated_composition"]
        + config.loss.histology_composition * guarded["histology_composition"]
        + guarded["uni_hint"]
    )
    sonrm = torch.tensor(0.0).to(device)
    if config.sonrm.enabled:
        embeddings = getattr(model, "last_embeddings", None)
        coordinates = getattr(model, "last_coords", None)
        if embeddings is not None and coordinates is not None:
            sonrm = modules["sonrm"](embeddings, coordinates)
            total = total + config.loss.sonrm * sonrm
    guarded["sonrm"] = sonrm
    guarded["total"] = total
    return guarded


def _append_loss_row(path: Path, epoch: int, means: Mapping[str, float], batches: int) -> None:
    columns = [
        "total",
        "segmentation",
        "histology_cell_type",
        "expression_cell_type",
        "endpoint_expression",
        "neighborhood_expression_immune",
        "neighborhood_expression_invasive",
        "expression_embedding",
        "expression_logits",
        "estimated_composition",
        "histology_composition",
        "uni_hint",
        "sonrm",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["epoch"] + [f"loss_{name}" for name in columns] + ["num_batches"])
        writer.writerow([epoch] + [means[name] for name in columns] + [batches])


def train(config: ConfigInput, output_dir: PathLike) -> Mapping[str, Any]:
    """Run formal HistCFM training when explicitly called by the CLI or API."""

    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(resolved, HistCFMConfig):
        raise TypeError("config must be a HistCFMConfig or YAML path")
    resolved.validate_training_inputs()
    if resolved.data.stain_augmentation:
        validate_stain_augmentation_dependency()
    paths = _prepare_output(Path(output_dir), resolved.runtime.overwrite_output)
    logger = _configure_logger(paths["logs"] / "train.log")
    logger.info("Preparing and validating HistCFM-ready data")
    train_dataset, validation_dataset = _build_datasets_and_preflight(
        resolved, paths["artifacts"]
    )
    del validation_dataset
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    if resolved.training.drop_last and len(train_dataset) < resolved.training.batch_size:
        raise ValueError(
            "drop_last=true would produce zero training batches: "
            f"dataset={len(train_dataset)}, batch_size={resolved.training.batch_size}"
        )

    seed = resolved.training.seed
    if seed is not None:
        set_seed(seed, deterministic=resolved.runtime.deterministic)
    device = _resolve_device(resolved)
    average_expression, n_ref = _prepare_average_expression(resolved, device)
    model = build_model(resolved, device, n_ref)
    model.to(device)

    generator = None
    worker_init = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init = seed_worker
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=resolved.training.batch_size,
        shuffle=resolved.training.shuffle,
        num_workers=resolved.training.workers,
        drop_last=resolved.training.drop_last,
        worker_init_fn=worker_init,
        generator=generator,
    )
    if len(dataloader) == 0:
        raise ValueError("Configured DataLoader produces zero training batches")

    optimizer = build_optimizer(resolved, model)
    modules = _loss_modules(resolved)
    resolved_dict = resolved.to_dict()
    resolved_dict["data"]["normalization_path"] = NORMALIZATION_ARTIFACT
    write_config(resolved_dict, paths["root"] / "resolved_config.yaml")
    logger.info("Using device: %s", device)
    logger.info("Training patches: %d", len(train_dataset))
    logger.info("Begin training (endpoint flow matching + UNI hint + SONRM)")

    loss_log = paths["logs"] / "train_losses.csv"
    last_checkpoint = None
    for epoch in range(resolved.training.epochs):
        model.train()
        stage_pretrain = epoch < resolved.flow.warmup_epochs
        model.use_flow_expr = True
        current_lr = _learning_rate(resolved, epoch)
        for group in optimizer.param_groups:
            scale = float(group.get("lr_scale", 1.0))
            group["lr"] = max(
                current_lr * scale, resolved.training.minimum_learning_rate
            )
        logger.info(
            "Epoch %d/%d; learning rate %.12g",
            epoch + 1,
            resolved.training.epochs,
            optimizer.param_groups[0]["lr"],
        )
        sums: Dict[str, float] = {}
        effective_batches = 0
        flow_grad_norm = 0.0
        last_expression = None
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{resolved.training.epochs}")
        for (
            batch_nuclei,
            batch_type_patch,
            batch_histology,
            batch_expression,
            batch_n_cells,
            batch_cell_type,
            patch_ids,
            patch_keys,
        ) in progress:
            del patch_ids
            optimizer.zero_grad()
            batch_nuclei = batch_nuclei.to(device)
            batch_type_patch = batch_type_patch.to(device)
            batch_histology = batch_histology.to(device)
            batch_expression = batch_expression.to(device)
            batch_n_cells = batch_n_cells.to(device)
            batch_cell_type = batch_cell_type.to(device)
            outputs = model(
                batch_histology,
                batch_nuclei,
                batch_n_cells,
                average_expression,
                batch_cell_type,
                batch_expression,
                patch_ids=None,
                patch_keys=patch_keys,
                stage_pretrain=stage_pretrain,
            )
            batch_losses = compute_losses(
                config=resolved,
                model=model,
                outputs=outputs,
                batch_type_patch=batch_type_patch,
                batch_n_cells=batch_n_cells,
                device=device,
                modules=modules,
            )
            if not batch_losses:
                continue
            total_loss = batch_losses["total"]
            total_loss.backward()
            if resolved.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=resolved.training.gradient_clip
                )
            for name, parameter in model.named_parameters():
                if "flow" in name.lower() and parameter.grad is not None:
                    flow_grad_norm += parameter.grad.detach().data.norm(2).item()
            optimizer.step()
            for name, value in batch_losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.item())
            effective_batches += 1
            last_expression = outputs[3]
            progress.set_postfix(total_loss=float(total_loss.item()))

        if effective_batches == 0:
            raise RuntimeError("No effective training batch contained an eligible cell")
        means = {name: value / effective_batches for name, value in sums.items()}
        _append_loss_row(loss_log, epoch + 1, means, effective_batches)
        logger.info("Epoch %d mean total loss: %s", epoch + 1, means["total"])
        logger.info("Epoch %d mean SONRM loss: %s", epoch + 1, means["sonrm"])
        logger.info("Epoch %d mean UNI hint loss: %s", epoch + 1, means["uni_hint"])
        if last_expression is not None:
            with torch.no_grad():
                predicted_std = last_expression.std(dim=0)
                logger.info(
                    "Predicted std min/max this epoch: %s/%s",
                    float(predicted_std.min().item()),
                    float(predicted_std.max().item()),
                )
                logger.info("Flow grad norm this epoch: %s", flow_grad_norm)

        if epoch % resolved.training.checkpoint_frequency == 0:
            last_checkpoint = save_checkpoint(
                paths["checkpoints"] / f"epoch_{epoch + 1}.pth",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                include_optimizer=resolved.training.save_optimizer,
                resolved_config=resolved_dict,
                genes=resolved.data.genes,
                cell_types=(
                    resolved.data.cell_types if resolved.model.use_cell_types else []
                ),
                normalization_artifact=NORMALIZATION_ARTIFACT,
                seed=seed,
            )
            logger.info("Saved checkpoint: %s", last_checkpoint)

    logger.info("Training finished")
    return {
        "output_dir": str(paths["root"]),
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "resolved_config": str(paths["root"] / "resolved_config.yaml"),
        "normalization_artifact": str(paths["root"] / NORMALIZATION_ARTIFACT),
        "loss_log": str(loss_log),
    }
