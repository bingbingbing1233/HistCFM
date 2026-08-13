"""Strict formal configuration for HistCFM.

This module was written during the 2026-08-12 release reorganization. It does
not read research repositories, data, or environment variables and does not
instantiate the model. YAML parsing is performed only when ``load_config`` is
called. Distributed under GNU GPL version 3 only; see ``LICENSE``.
"""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


PathLike = Union[str, Path]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: Sequence[str], name: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _optional_path(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be null or a non-empty path string")
    return value


def _strings(value: Any, name: str, allow_empty: bool = False) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of strings")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique values")
    return list(value)


@dataclass
class DataConfig:
    histology_path: Optional[str] = None
    nucleus_mask_path: Optional[str] = None
    matched_nuclei_path: Optional[str] = None
    expression_path: Optional[str] = None
    cell_type_path: Optional[str] = None
    average_expression_path: Optional[str] = None
    genes: List[str] = field(default_factory=list)
    cell_types: List[str] = field(default_factory=list)
    validation_split: List[float] = field(default_factory=lambda: [0.0, 0.2])
    patch_height: int = 256
    patch_width: int = 256
    overlap: int = 30
    max_cells_per_patch: int = 200
    min_nucleus_area: float = 10.0
    expression_scale: float = 5.0
    normalization_path: Optional[str] = None
    stain_augmentation: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "DataConfig":
        value = _mapping(raw, "data")
        _strict_keys(value, cls.__dataclass_fields__, "data")
        result = cls(**dict(value))
        for name in (
            "histology_path",
            "nucleus_mask_path",
            "matched_nuclei_path",
            "expression_path",
            "cell_type_path",
            "average_expression_path",
            "normalization_path",
        ):
            setattr(result, name, _optional_path(getattr(result, name), f"data.{name}"))
        result.genes = _strings(result.genes, "data.genes")
        result.cell_types = _strings(result.cell_types, "data.cell_types", allow_empty=True)
        if not isinstance(result.validation_split, list) or len(result.validation_split) != 2:
            raise TypeError("data.validation_split must be a two-item list")
        result.validation_split = [
            _float(item, f"data.validation_split[{index}]", 0.0)
            for index, item in enumerate(result.validation_split)
        ]
        if any(item > 1.0 for item in result.validation_split):
            raise ValueError("data.validation_split values must be <= 1")
        result.patch_height = _int(result.patch_height, "data.patch_height", 1)
        result.patch_width = _int(result.patch_width, "data.patch_width", 1)
        result.overlap = _int(result.overlap, "data.overlap", 0)
        if result.overlap >= min(result.patch_height, result.patch_width):
            raise ValueError("data.overlap must be smaller than each patch dimension")
        result.max_cells_per_patch = _int(
            result.max_cells_per_patch, "data.max_cells_per_patch", 1
        )
        result.min_nucleus_area = _float(
            result.min_nucleus_area, "data.min_nucleus_area", 0.0
        )
        result.expression_scale = _float(result.expression_scale, "data.expression_scale", 0.0)
        if result.expression_scale == 0:
            raise ValueError("data.expression_scale must be positive")
        result.stain_augmentation = _bool(
            result.stain_augmentation, "data.stain_augmentation"
        )
        return result


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    use_cell_types: bool = True
    use_neighborhood: bool = True
    average_expression_compatibility: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "ModelConfig":
        value = _mapping(raw, "model")
        _strict_keys(value, cls.__dataclass_fields__, "model")
        result = cls(**dict(value))
        result.embedding_dim = _int(result.embedding_dim, "model.embedding_dim", 1)
        result.use_cell_types = _bool(result.use_cell_types, "model.use_cell_types")
        result.use_neighborhood = _bool(result.use_neighborhood, "model.use_neighborhood")
        result.average_expression_compatibility = _bool(
            result.average_expression_compatibility,
            "model.average_expression_compatibility",
        )
        if result.use_neighborhood and not result.use_cell_types:
            raise ValueError("model.use_neighborhood requires model.use_cell_types")
        return result


@dataclass
class FlowConfig:
    hidden_dim: int = 512
    num_layers: int = 8
    k_neighbors: int = 16
    inference_steps: int = 5
    prior: str = "gaussian"
    train_noise_std: float = 0.02
    inference_noise_std: float = 0.05
    target_noise_std: float = 0.001
    warmup_epochs: int = 0

    @classmethod
    def from_mapping(cls, raw: Any) -> "FlowConfig":
        value = _mapping(raw, "flow")
        _strict_keys(value, cls.__dataclass_fields__, "flow")
        result = cls(**dict(value))
        result.hidden_dim = _int(result.hidden_dim, "flow.hidden_dim", 1)
        result.num_layers = _int(result.num_layers, "flow.num_layers", 1)
        result.k_neighbors = _int(result.k_neighbors, "flow.k_neighbors", 1)
        result.inference_steps = _int(result.inference_steps, "flow.inference_steps", 1)
        if result.prior not in {"gaussian", "zinb"}:
            raise ValueError("flow.prior must be 'gaussian' or 'zinb'")
        result.train_noise_std = _float(result.train_noise_std, "flow.train_noise_std", 0.0)
        result.inference_noise_std = _float(
            result.inference_noise_std, "flow.inference_noise_std", 0.0
        )
        result.target_noise_std = _float(
            result.target_noise_std, "flow.target_noise_std", 0.0
        )
        result.warmup_epochs = _int(result.warmup_epochs, "flow.warmup_epochs", 0)
        return result


@dataclass
class UniConfig:
    enabled: bool = True
    mode: str = "precomputed"
    index_path: Optional[str] = None
    features_path: Optional[str] = None
    feature_dim: int = 1024
    fusion_method: str = "film"
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "UniConfig":
        value = _mapping(raw, "uni")
        _strict_keys(value, cls.__dataclass_fields__, "uni")
        result = cls(**dict(value))
        result.enabled = _bool(result.enabled, "uni.enabled")
        if result.mode != "precomputed":
            raise ValueError("uni.mode must be 'precomputed' in this release")
        result.index_path = _optional_path(result.index_path, "uni.index_path")
        result.features_path = _optional_path(result.features_path, "uni.features_path")
        result.feature_dim = _int(result.feature_dim, "uni.feature_dim", 1)
        if result.fusion_method not in {"film", "gate"}:
            raise ValueError("uni.fusion_method must be 'film' or 'gate'")
        result.fusion_hidden_dim = _int(
            result.fusion_hidden_dim, "uni.fusion_hidden_dim", 1
        )
        result.fusion_dropout = _float(result.fusion_dropout, "uni.fusion_dropout", 0.0)
        if result.fusion_dropout > 1.0:
            raise ValueError("uni.fusion_dropout must be <= 1")
        return result


@dataclass
class SonrmConfig:
    enabled: bool = True
    neighbors: int = 16
    hops: int = 3
    margin_12: float = 0.1
    margin_23: float = 0.1

    @classmethod
    def from_mapping(cls, raw: Any) -> "SonrmConfig":
        value = _mapping(raw, "sonrm")
        _strict_keys(value, cls.__dataclass_fields__, "sonrm")
        result = cls(**dict(value))
        result.enabled = _bool(result.enabled, "sonrm.enabled")
        result.neighbors = _int(result.neighbors, "sonrm.neighbors", 1)
        result.hops = _int(result.hops, "sonrm.hops", 1)
        result.margin_12 = _float(result.margin_12, "sonrm.margin_12", 0.0)
        result.margin_23 = _float(result.margin_23, "sonrm.margin_23", 0.0)
        return result


@dataclass
class LossConfig:
    segmentation: float = 1.0
    histology_cell_type: float = 1.0
    expression_cell_type: float = 1.0
    endpoint_expression: float = 10.0
    neighborhood_expression_immune: float = 10.0
    neighborhood_expression_invasive: float = 10.0
    expression_embedding: float = 100.0
    expression_logits: float = 1.0
    estimated_composition: float = 1.0
    histology_composition: float = 1.0
    uni_hint: float = 0.1
    sonrm: float = 0.05

    @classmethod
    def from_mapping(cls, raw: Any) -> "LossConfig":
        value = _mapping(raw, "loss")
        _strict_keys(value, cls.__dataclass_fields__, "loss")
        result = cls(**dict(value))
        for name in cls.__dataclass_fields__:
            setattr(result, name, _float(getattr(result, name), f"loss.{name}", 0.0))
        return result


@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 8
    workers: int = 1
    shuffle: bool = True
    drop_last: bool = True
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0001
    epsilon: float = 1e-8
    learning_rate_schedule: str = "linear"
    initial_learning_rate: float = 0.001
    final_learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-5
    switch_ratio: float = 0.5
    gradient_clip: float = 0.5
    checkpoint_frequency: int = 1
    save_optimizer: bool = True
    seed: Optional[int] = 42

    @classmethod
    def from_mapping(cls, raw: Any) -> "TrainingConfig":
        value = _mapping(raw, "training")
        _strict_keys(value, cls.__dataclass_fields__, "training")
        result = cls(**dict(value))
        result.epochs = _int(result.epochs, "training.epochs", 1)
        result.batch_size = _int(result.batch_size, "training.batch_size", 1)
        result.workers = _int(result.workers, "training.workers", 0)
        result.shuffle = _bool(result.shuffle, "training.shuffle")
        result.drop_last = _bool(result.drop_last, "training.drop_last")
        if result.optimizer != "adamw":
            raise ValueError("training.optimizer must be 'adamw'")
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "weight_decay",
            "epsilon",
            "initial_learning_rate",
            "final_learning_rate",
            "minimum_learning_rate",
            "switch_ratio",
            "gradient_clip",
        ):
            setattr(result, name, _float(getattr(result, name), f"training.{name}", 0.0))
        if not 0 <= result.beta1 < 1 or not 0 <= result.beta2 < 1:
            raise ValueError("training beta values must be in [0, 1)")
        if not 0 <= result.switch_ratio <= 1:
            raise ValueError("training.switch_ratio must be in [0, 1]")
        if result.learning_rate_schedule not in {"linear", "cosine", "two_stage"}:
            raise ValueError(
                "training.learning_rate_schedule must be linear, cosine, or two_stage"
            )
        result.checkpoint_frequency = _int(
            result.checkpoint_frequency, "training.checkpoint_frequency", 1
        )
        result.save_optimizer = _bool(result.save_optimizer, "training.save_optimizer")
        if result.seed is not None:
            result.seed = _int(result.seed, "training.seed", 0)
        return result


@dataclass
class RuntimeConfig:
    device: str = "auto"
    gpu_index: int = 0
    deterministic: bool = True
    overwrite_output: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "RuntimeConfig":
        value = _mapping(raw, "runtime")
        _strict_keys(value, cls.__dataclass_fields__, "runtime")
        result = cls(**dict(value))
        if result.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be auto, cpu, or cuda")
        result.gpu_index = _int(result.gpu_index, "runtime.gpu_index", 0)
        result.deterministic = _bool(result.deterministic, "runtime.deterministic")
        result.overwrite_output = _bool(result.overwrite_output, "runtime.overwrite_output")
        return result


@dataclass
class HistCFMConfig:
    data: DataConfig
    model: ModelConfig
    flow: FlowConfig
    uni: UniConfig
    sonrm: SonrmConfig
    loss: LossConfig
    training: TrainingConfig
    runtime: RuntimeConfig

    @classmethod
    def from_mapping(cls, raw: Any) -> "HistCFMConfig":
        value = _mapping(raw, "configuration")
        required = tuple(cls.__dataclass_fields__)
        _strict_keys(value, required, "top-level configuration")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"Missing top-level configuration group(s): {', '.join(missing)}")
        result = cls(
            data=DataConfig.from_mapping(value["data"]),
            model=ModelConfig.from_mapping(value["model"]),
            flow=FlowConfig.from_mapping(value["flow"]),
            uni=UniConfig.from_mapping(value["uni"]),
            sonrm=SonrmConfig.from_mapping(value["sonrm"]),
            loss=LossConfig.from_mapping(value["loss"]),
            training=TrainingConfig.from_mapping(value["training"]),
            runtime=RuntimeConfig.from_mapping(value["runtime"]),
        )
        result.validate_training_inputs()
        return result

    def validate_training_inputs(self) -> None:
        required_paths = {
            "data.histology_path": self.data.histology_path,
            "data.nucleus_mask_path": self.data.nucleus_mask_path,
            "data.matched_nuclei_path": self.data.matched_nuclei_path,
            "data.expression_path": self.data.expression_path,
        }
        if self.model.use_cell_types:
            required_paths["data.cell_type_path"] = self.data.cell_type_path
            if not self.data.cell_types:
                raise ValueError("data.cell_types is required when model.use_cell_types is true")
        if self.model.average_expression_compatibility:
            required_paths["data.average_expression_path"] = self.data.average_expression_path
        if self.uni.enabled:
            required_paths["uni.index_path"] = self.uni.index_path
            required_paths["uni.features_path"] = self.uni.features_path
        missing = [name for name, value in required_paths.items() if value is None]
        if missing:
            raise ValueError("Missing required training path(s): " + ", ".join(missing))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: PathLike) -> HistCFMConfig:
    """Read and strictly validate one formal YAML training configuration."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read HistCFM YAML configuration") from error
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return HistCFMConfig.from_mapping(raw)


def write_config(data: Mapping[str, Any], path: PathLike) -> None:
    """Write an already resolved mapping as safe YAML."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to write HistCFM YAML configuration") from error
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)
