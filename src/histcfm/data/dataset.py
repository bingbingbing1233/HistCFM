"""Formal cell-level HistCFM dataset.

Derived from the cell-level data loader in SydneyBioX/GHIST:
https://github.com/SydneyBioX/GHIST

Modified for HistCFM on 2026-08-12 with formal naming, repository-independent
paths, one canonical UNI patch-key builder, delayed optional stain
augmentation, input guards, and explicit normalization output control. The
audited patching, filtering, expression transform, tuple order, and tensor
dtypes are retained. Distributed under GNU GPL version 3 only; see LICENSE.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import natsort
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from .image_io import load_image
from .splitting import build_split_patch_coordinates, split_row_intervals
from .validation import build_patch_key


PathLike = Union[str, Path]
NORMALIZATION_FILENAME = "histology_normalization.npy"


torchvision.disable_beta_transforms_warning()


def _require_file(path: Optional[PathLike], label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"Missing {label}: <not provided>")
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def compute_histology_normalization(
    histology,
    patch_coordinates: Sequence[Tuple[int, int]],
    patch_height: int,
    patch_width: int,
):
    """Compute the exact audited mean-of-patch-means/stds statistic.

    The result is a float64 array of shape ``(2, 3)``. It is deliberately not
    replaced by a global-pixel statistic because that would change behavior.
    """

    hist_means = np.zeros(3)
    hist_stds = np.zeros(3)
    for row_start, column_start in tqdm(patch_coordinates):
        patch = histology[
            row_start : row_start + patch_height,
            column_start : column_start + patch_width,
        ]
        hist_means += np.mean(patch, (0, 1))
        hist_stds += np.std(patch, (0, 1))
    count = max(1, len(patch_coordinates))
    return np.vstack((hist_means / count, hist_stds / count))


def _assert_output_outside_input_dirs(
    output_dir: Path,
    input_paths: Sequence[Optional[PathLike]],
) -> None:
    output_resolved = output_dir.resolve()
    for raw_path in input_paths:
        if raw_path is None:
            continue
        input_dir = Path(raw_path).resolve().parent
        if output_resolved == input_dir or input_dir in output_resolved.parents:
            raise ValueError(
                "normalization output_dir must not be an input-data directory "
                "or one of its descendants"
            )


def _load_normalization(path: PathLike):
    normalization = np.load(path, allow_pickle=False)
    if normalization.shape != (2, 3):
        raise ValueError(
            f"Histology normalization must have shape (2, 3); got {normalization.shape}"
        )
    if not np.isfinite(normalization).all():
        raise ValueError("Histology normalization contains NaN or infinite values")
    return normalization


def load_or_create_normalization(
    *,
    mode: str,
    histology,
    patch_coordinates: Sequence[Tuple[int, int]],
    patch_height: int,
    patch_width: int,
    normalization_path: Optional[PathLike],
    output_dir: Optional[PathLike],
    input_paths: Sequence[Optional[PathLike]],
):
    """Load training statistics, or compute them only in explicit train output.

    Validation and prediction never estimate or write statistics. Training
    writes ``histology_normalization.npy`` as float64 with shape ``(2, 3)``
    only inside the caller-provided output directory.
    """

    if normalization_path is not None:
        path = _require_file(normalization_path, "histology normalization")
        return _load_normalization(path), path
    if mode != "train":
        raise FileNotFoundError(
            "Validation and prediction require normalization_path from training"
        )
    if output_dir is None:
        raise ValueError("Training requires an explicit normalization output_dir")
    output = Path(output_dir)
    _assert_output_outside_input_dirs(output, input_paths)
    output.mkdir(parents=True, exist_ok=True)
    path = output / NORMALIZATION_FILENAME
    if path.exists():
        return _load_normalization(path), path
    normalization = compute_histology_normalization(
        histology,
        patch_coordinates,
        patch_height,
        patch_width,
    )
    np.save(path, normalization)
    return normalization, path


def _prepare_input_data(
    *,
    nucleus_mask_path: PathLike,
    histology_path: PathLike,
    matched_nuclei_path: PathLike,
    mode: str,
    min_nucleus_area: float,
    expression_scale: float,
    patch_height: int,
    patch_width: int,
    overlap: int,
    gene_names: Sequence[str],
    validation_division: Sequence[float],
    expression_path: Optional[PathLike],
    cell_type_path: Optional[PathLike],
    cell_types: Optional[Sequence[str]],
    normalization_path: Optional[PathLike],
    normalization_output_dir: Optional[PathLike],
):
    if expression_path is not None:
        expression = pd.read_csv(expression_path, index_col=0)
        if not expression.columns.is_unique:
            raise ValueError("Expression gene names must be unique")
        missing_genes = [gene for gene in gene_names if gene not in expression.columns]
        if missing_genes:
            raise ValueError(
                "Expression is missing configured genes; silent dimension changes are "
                f"not allowed: {missing_genes[:10]}"
            )
        expression = expression.reindex(columns=gene_names)
    else:
        expression = None

    if cell_type_path is not None:
        cell_type = pd.read_csv(cell_type_path, index_col="c_id")
        is_all_numbers = pd.to_numeric(cell_type["ct"], errors="coerce").notna().all()
        if not is_all_numbers:
            type_mapping = dict(zip(cell_types, list(range(len(cell_types)))))
            cell_type["ct"] = cell_type["ct"].map(type_mapping).fillna(-1).astype(float)
        cell_type["ct"] = cell_type["ct"] + 1
    else:
        cell_type = None

    nuclei = load_image(nucleus_mask_path)
    histology = load_image(histology_path)
    whole_height = histology.shape[0]
    whole_width = histology.shape[1]
    row_intervals = split_row_intervals(whole_height, validation_division, mode)
    segmentation_ids = np.unique(
        np.concatenate(
            [nuclei[start:stop, :].reshape(-1) for start, stop in row_intervals]
        )
    )
    segmentation_ids = segmentation_ids[segmentation_ids != 0]

    matched_nuclei = pd.read_csv(matched_nuclei_path, index_col=0)
    matched_nuclei = matched_nuclei[
        matched_nuclei["size_pix_histology"] >= min_nucleus_area
    ]
    size_filtered_ids = matched_nuclei.index.tolist()
    all_intersect = list(set(segmentation_ids) & set(size_filtered_ids))

    if expression is not None:
        all_intersect = list(set(all_intersect) & set(expression.index.tolist()))
        expression = expression[expression.index.isin(all_intersect)]
        expression = expression.fillna(0)
        expression = expression.clip(lower=0)
        expression = expression_scale * np.log1p(expression)
        expression = expression.replace([np.inf, -np.inf], 0).fillna(0)
    if cell_type is not None:
        all_intersect = list(set(all_intersect) & set(cell_type.index))
        cell_type = cell_type.loc[all_intersect, :]

    all_intersect = natsort.natsorted(all_intersect)
    all_coordinates = build_split_patch_coordinates(
        image_height=whole_height,
        image_width=whole_width,
        validation_division=validation_division,
        patch_height=patch_height,
        patch_width=patch_width,
        overlap=overlap,
        mode=mode,
    )

    valid_coordinates = []
    for row_start, column_start in tqdm(all_coordinates):
        nuclei_patch = nuclei[
            row_start : row_start + patch_height,
            column_start : column_start + patch_width,
        ]
        patch_ids = np.unique(nuclei_patch)
        patch_ids = patch_ids[patch_ids != 0]
        valid_ids = list(set(patch_ids) & set(all_intersect))
        invalid_ids = list(set(patch_ids) - set(valid_ids))
        replacement = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_patch)
        for cell_id, value in replacement.items():
            nuclei_valid[nuclei_patch == cell_id] = value
        if np.sum(nuclei_valid) > 0:
            valid_coordinates.append((row_start, column_start))
    if len(valid_coordinates) == 0:
        raise ValueError(f"{mode} split contains no patch with an eligible cell")

    normalization, resolved_normalization_path = load_or_create_normalization(
        mode=mode,
        histology=histology,
        patch_coordinates=all_coordinates,
        patch_height=patch_height,
        patch_width=patch_width,
        normalization_path=normalization_path,
        output_dir=normalization_output_dir,
        input_paths=(
            nucleus_mask_path,
            histology_path,
            matched_nuclei_path,
            expression_path,
            cell_type_path,
        ),
    )
    slide_id = Path(histology_path).stem
    return (
        valid_coordinates,
        histology,
        nuclei,
        all_intersect,
        cell_type,
        expression,
        normalization,
        resolved_normalization_path,
        slide_id,
    )


def validate_stain_augmentation_dependency():
    """Require stainlib only when optional stain augmentation is enabled."""

    try:
        from stainlib.augmentation.augmenter import HedLighterColorAugmenter
    except ImportError as error:
        raise RuntimeError(
            "stain_augmentation=true requires the optional stainlib dependency; "
            "install HistCFM with the 'stain' extra (for example, "
            "python -m pip install -e '.[stain]')"
        ) from error
    return HedLighterColorAugmenter


def _create_stain_augmenter():
    HedLighterColorAugmenter = validate_stain_augmentation_dependency()
    try:
        return HedLighterColorAugmenter()
    except Exception as error:
        raise RuntimeError(
            "stainlib augmentation initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error


def _optional_stain_augmenter(enabled: bool):
    """Avoid importing stainlib when augmentation is disabled."""

    return _create_stain_augmenter() if bool(enabled) else None


class HistCFMDataset(data.Dataset):
    """One-slide, patch-indexed PyTorch dataset for cell-level HistCFM."""

    def __init__(
        self,
        *,
        nucleus_mask_path: PathLike,
        histology_path: PathLike,
        matched_nuclei_path: PathLike,
        expression_path: Optional[PathLike],
        cell_type_path: Optional[PathLike],
        gene_names: Sequence[str],
        cell_types: Optional[Sequence[str]],
        patch_height: int,
        patch_width: int,
        overlap: int,
        max_cells_per_patch: int,
        min_nucleus_area: float,
        expression_scale: float,
        validation_division: Sequence[float],
        mode: str = "train",
        use_cell_types: bool = True,
        stain_augmentation: bool = False,
        normalization_path: Optional[PathLike] = None,
        normalization_output_dir: Optional[PathLike] = None,
    ):
        if mode not in {"train", "val", "predict"}:
            raise ValueError("mode must be one of: train, val, predict")
        if int(patch_height) <= 0 or int(patch_width) <= 0:
            raise ValueError("patch_height and patch_width must be positive")
        if int(overlap) < 0 or int(overlap) >= min(int(patch_height), int(patch_width)):
            raise ValueError("overlap must be non-negative and smaller than each patch dimension")
        if int(max_cells_per_patch) <= 0:
            raise ValueError("max_cells_per_patch must be positive")
        if len(gene_names) == 0 or len(gene_names) != len(set(gene_names)):
            raise ValueError("gene_names must be non-empty and unique")
        _require_file(nucleus_mask_path, "nucleus mask")
        _require_file(histology_path, "histology image")
        _require_file(matched_nuclei_path, "matched-nuclei table")
        if mode != "predict":
            _require_file(expression_path, "expression table")
        else:
            expression_path = None
        if use_cell_types and mode != "predict":
            _require_file(cell_type_path, "cell-type table")
            if cell_types is None:
                raise ValueError("cell_types must be provided when use_cell_types is true")
            self.cell_types = list(cell_types)
            self.use_cell_types = True
        else:
            cell_type_path = None
            self.cell_types = None
            self.use_cell_types = False

        self.mode = mode
        self.gene_names = list(gene_names)
        self.max_cells_per_patch = int(max_cells_per_patch)
        self.hsize = int(patch_height)
        self.wsize = int(patch_width)
        self.stain_augmentation = bool(stain_augmentation)
        effective_overlap = 0 if mode == "train" else int(overlap)
        (
            coordinates,
            self.hist,
            self.nuclei,
            self.all_intersect,
            self.df_ct,
            self.df_expr,
            normalization,
            self.normalization_path,
            self.slide_id,
        ) = _prepare_input_data(
            nucleus_mask_path=nucleus_mask_path,
            histology_path=histology_path,
            matched_nuclei_path=matched_nuclei_path,
            mode=mode,
            min_nucleus_area=min_nucleus_area,
            expression_scale=expression_scale,
            patch_height=self.hsize,
            patch_width=self.wsize,
            overlap=effective_overlap,
            gene_names=self.gene_names,
            validation_division=validation_division,
            expression_path=expression_path,
            cell_type_path=cell_type_path,
            cell_types=self.cell_types,
            normalization_path=normalization_path,
            normalization_output_dir=normalization_output_dir,
        )
        self.norms_hist = normalization.copy()
        self.coords_starts = coordinates
        self.n_patches = len(self.coords_starts)
        self.tfs = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomApply([v2.RandomRotation((90, 90))], p=0.25),
                v2.RandomApply([v2.RandomRotation((180, 180))], p=0.25),
                v2.RandomApply([v2.RandomRotation((270, 270))], p=0.25),
                v2.ToDtype(torch.float32),
            ]
        )
        self.tfs_predict = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
        self.hed_lighter_aug = _optional_stain_augmenter(self.stain_augmentation)

    @property
    def patch_keys(self):
        return [
            build_patch_key(
                self.slide_id,
                row_start,
                column_start,
                self.hsize,
                self.wsize,
            )
            for row_start, column_start in self.coords_starts
        ]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, index):
        row_start, column_start = self.coords_starts[index]
        nuclei_patch = self.nuclei[
            row_start : row_start + self.hsize,
            column_start : column_start + self.wsize,
        ]
        histology_patch = self.hist[
            row_start : row_start + self.hsize,
            column_start : column_start + self.wsize,
        ]
        if (
            self.mode == "train"
            and self.stain_augmentation
            and self.hed_lighter_aug is not None
        ):
            self.hed_lighter_aug.randomize()
            histology_patch = self.hed_lighter_aug.transform(histology_patch)

        segmentation_ids = np.unique(nuclei_patch)
        segmentation_ids = segmentation_ids[segmentation_ids != 0]
        valid_ids = list(set(segmentation_ids) & set(self.all_intersect))
        invalid_ids = list(set(segmentation_ids) - set(valid_ids))
        replacement = dict(zip(invalid_ids, [0] * len(invalid_ids)))
        nuclei_valid = np.copy(nuclei_patch)
        for cell_id, value in replacement.items():
            nuclei_valid[nuclei_patch == cell_id] = value

        if self.use_cell_types and self.mode != "predict":
            values = self.df_ct.loc[valid_ids, "ct"].to_numpy()
            values = np.nan_to_num(values, nan=0).astype(int)
            type_lookup = dict(zip(valid_ids, values.tolist()))
            types_patch = np.copy(nuclei_valid)
            for cell_id, value in type_lookup.items():
                types_patch[nuclei_valid == cell_id] = value
        else:
            types_patch = np.where(nuclei_valid > 0, 1, 0)

        means = np.expand_dims(self.norms_hist[0, :], (0, 1))
        stds = np.expand_dims(self.norms_hist[1, :], (0, 1))
        stds = np.clip(stds, 1e-6, None)
        histology_patch = histology_patch - means
        histology_patch = histology_patch / stds

        patch_ids = np.unique(nuclei_valid)
        patch_ids = patch_ids[patch_ids != 0]
        n_cells = len(patch_ids)
        expression_pad = np.zeros(
            (self.max_cells_per_patch, len(self.gene_names))
        )
        if self.mode != "predict":
            expression = self.df_expr.loc[patch_ids, :].to_numpy()
            expression_pad[:n_cells, :] = expression.copy()
        ground_truth_types_pad = np.zeros(self.max_cells_per_patch)
        if self.use_cell_types and self.mode != "predict":
            ground_truth_values = self.df_ct.loc[patch_ids, "ct"].to_numpy()
            ground_truth_values = np.nan_to_num(
                ground_truth_values,
                nan=0,
            ).astype(int)
            ground_truth_types_pad[:n_cells] = ground_truth_values - 1
        ground_truth_types_tensor = torch.from_numpy(ground_truth_types_pad).long()
        patch_ids_pad = np.zeros(self.max_cells_per_patch)
        patch_ids_pad[:n_cells] = patch_ids.copy()
        patch_ids_tensor = torch.from_numpy(patch_ids_pad).long()
        n_cells_tensor = torch.from_numpy(np.array([n_cells])).long()

        model_input = np.concatenate(
            (
                np.expand_dims(nuclei_valid, -1),
                np.expand_dims(types_patch, -1),
                histology_patch,
            ),
            -1,
        )
        if self.mode == "train":
            model_input = self.tfs(model_input)
        else:
            model_input = self.tfs_predict(model_input)
        nuclei_tensor = model_input[0, :, :].type(torch.LongTensor)
        types_patch_tensor = model_input[1, :, :].type(torch.LongTensor)
        histology_tensor = model_input[2:, :, :]
        expression_tensor = torch.from_numpy(expression_pad).float()
        patch_key = build_patch_key(
            self.slide_id,
            row_start,
            column_start,
            self.hsize,
            self.wsize,
        )
        return (
            nuclei_tensor,
            types_patch_tensor,
            histology_tensor,
            expression_tensor,
            n_cells_tensor,
            ground_truth_types_tensor,
            patch_ids_tensor,
            patch_key,
        )
