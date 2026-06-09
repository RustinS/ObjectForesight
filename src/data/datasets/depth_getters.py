from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .dataset_epic import SceneSequenceDataset


class _DepthArrayGetter:
    def __init__(self, depth_array: np.ndarray) -> None:
        self.depth_array = depth_array

    def __call__(self, idx: int) -> np.ndarray:
        return self.depth_array[idx]


class _DepthFileGetter:
    def __init__(self, npz_path: str, dataset_ref: "SceneSequenceDataset") -> None:
        self.npz_path = npz_path
        self.dataset_ref = dataset_ref

    def __call__(self, idx: int) -> Optional[np.ndarray]:
        return self.dataset_ref._load_depth_frame_lazy(self.npz_path, idx)


class _DepthMmapGetter:
    """Memory-mapped depth getter for efficient lazy loading without holding full array in RAM."""

    def __init__(self, npz_path: str, verbose: bool = False) -> None:
        self.npz_path = npz_path
        self.verbose = verbose
        self._mmap_depth: Optional[np.ndarray] = None
        self._init_mmap()

    def _init_mmap(self) -> None:
        try:
            # Open NPZ and find depth array key
            with np.load(self.npz_path, allow_pickle=False) as z:
                for k in ("depth", "depths", "D"):
                    if k in z:
                        dep = z[k]
                        # Store shape info for later mmap
                        self._depth_key = k
                        self._depth_shape = dep.shape
                        self._depth_dtype = dep.dtype
                        break
        except zipfile.BadZipFile:
            if self.verbose:
                print(f"[yellow]bad zip file[/yellow] {self.npz_path}")

    def __call__(self, idx: int) -> Optional[np.ndarray]:
        if self._mmap_depth is None:
            try:
                # Lazy init mmap on first access
                with np.load(self.npz_path, mmap_mode="r", allow_pickle=False) as z:
                    if hasattr(self, "_depth_key") and self._depth_key in z:
                        self._mmap_depth = z[self._depth_key]
            except zipfile.BadZipFile:
                return None

        if self._mmap_depth is not None:
            dep = self._mmap_depth[idx]
            if dep.ndim == 3 and dep.shape[0] == 1:
                return dep[0].astype(np.float32)
            return dep.astype(np.float32)
        return None
