"""Read the formal precomputed UNI feature store.

The public format is ``uni_index.json`` plus ``uni_features.npy``. Index
parsing and numeric validation are shared with the input preflight. This
module contains no online encoder, weights, downloads, or feature extraction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..data.validation import load_uni_feature_store


PathLike = Union[str, Path]


class UniFeatureProvider:
    def __init__(
        self,
        enable: bool,
        mode: str,
        index_path: Optional[PathLike],
        features_path: Optional[PathLike],
        uni_dim: int,
        device: Optional[torch.device] = None,
    ):
        self.enable = bool(enable)
        self.mode = str(mode or "precomputed")
        if self.mode != "precomputed":
            raise ValueError(
                "This HistCFM release supports mode='precomputed' only; "
                f"got {self.mode!r}"
            )
        self.index_path = None if index_path is None else Path(index_path)
        self.features_path = None if features_path is None else Path(features_path)
        self.uni_dim = int(uni_dim)
        self.device = device or torch.device("cpu")
        self.memmap_cache = None
        self.index_map: Dict[str, int] = {}
        if self.enable:
            self._load_precomputed_store()

    def _load_precomputed_store(self) -> None:
        if self.index_path is None or self.features_path is None:
            raise ValueError(
                "Precomputed UNI features require explicit index_path and features_path"
            )
        self.index_map, self.memmap_cache = load_uni_feature_store(
            self.index_path,
            self.features_path,
            expected_dim=self.uni_dim,
        )

    def get_optimizer_params(self):
        return []

    def fetch_patch_features(self, patch_keys: List[str]) -> torch.Tensor:
        if self.memmap_cache is None:
            raise RuntimeError("Precomputed UNI feature store is not loaded")
        missing = [key for key in patch_keys if key not in self.index_map]
        if missing:
            preview = ", ".join(repr(key) for key in missing[:5])
            raise KeyError(
                f"UNI features are missing {len(missing)} runtime patch keys: {preview}"
            )
        vectors = np.stack(
            [
                np.asarray(self.memmap_cache[self.index_map[key]], dtype=np.float32)
                for key in patch_keys
            ],
            axis=0,
        )
        output = torch.from_numpy(vectors).to(self.device)
        return F.normalize(output, p=2, dim=1)

    def get_patch_features(
        self,
        patch_keys: List[str],
        images_for_uni: Optional[List[np.ndarray]] = None,
    ) -> torch.Tensor:
        if not self.enable:
            return torch.zeros(
                (len(patch_keys), self.uni_dim),
                dtype=torch.float32,
                device=self.device,
            )
        return self.fetch_patch_features(patch_keys)

    def compute_match_rate(self, patch_keys: List[str]) -> Tuple[int, int, float]:
        total = len(patch_keys)
        if total == 0:
            return 0, 0, 0.0
        matched = sum(1 for key in patch_keys if key in self.index_map)
        return matched, total, float(matched) / float(total)

    def get_paths_info(self) -> Dict[str, str]:
        return {
            "mode": self.mode,
            "index_path": "" if self.index_path is None else str(self.index_path),
            "features_path": (
                "" if self.features_path is None else str(self.features_path)
            ),
        }

    def feature_scalar_stats(self, patch_keys: List[str]) -> Tuple[float, float]:
        if self.memmap_cache is None or not patch_keys:
            return 0.0, 0.0
        missing = [key for key in patch_keys if key not in self.index_map]
        if missing:
            raise KeyError(f"UNI feature key is missing: {missing[0]!r}")
        vectors = np.stack(
            [
                np.asarray(self.memmap_cache[self.index_map[key]], dtype=np.float32)
                for key in patch_keys
            ],
            axis=0,
        )
        return float(np.mean(vectors)), float(np.std(vectors))
