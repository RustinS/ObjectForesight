from __future__ import annotations

import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from ..utils.logger import rprint as print

_C2W_KEYS = ("T_c_w", "c2w", "poses_c2w", "poses")
_W2C_KEYS = ("T_w_c", "w2c", "extrinsics")
_DEPTH_KEYS = ("depth", "depths", "D")


def _find_key(d: Dict[str, np.ndarray], names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        if n in d:
            return n
    return None


def _to_4x4(T: np.ndarray) -> np.ndarray:
    if T.ndim == 2:
        T = T[None]
    if T.shape[-2:] == (3, 4):
        B = T.shape[0]
        T4 = np.tile(np.eye(4, dtype=np.float32), (B, 1, 1))
        T4[:, :3, :4] = T.astype(np.float32)
        return T4
    return T.astype(np.float32)


def _norm_intrinsics(d: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if "K" in d and d["K"].ndim >= 2:
        K = d["K"]
        return (K[0] if K.ndim == 3 else K).astype(np.float32)
    if all(k in d for k in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = (float(d[k]) for k in ("fx", "fy", "cx", "cy"))
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if "intrinsics" in d and isinstance(d["intrinsics"], np.ndarray):
        K = d["intrinsics"]
        return (K[0] if K.ndim == 3 else K).astype(np.float32)
    return None


def _canon_extrinsics(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Optional[str]]:
    """Normalize to T_c_w (camera->world) with shape [T,4,4]."""
    key_c2w = _find_key(d, _C2W_KEYS)
    key_w2c = _find_key(d, _W2C_KEYS)

    if key_c2w is not None:
        return _to_4x4(d[key_c2w]), key_c2w

    if key_w2c is not None:
        T = _to_4x4(d[key_w2c])
        R, t = T[:, :3, :3], T[:, :3, 3]
        R_inv = np.transpose(R, (0, 2, 1))
        t_inv = -np.einsum("bij,bj->bi", R_inv, t)
        T_c_w = np.tile(np.eye(4, dtype=np.float32), (T.shape[0], 1, 1))
        T_c_w[:, :3, :3] = R_inv
        T_c_w[:, :3, 3] = t_inv
        return T_c_w, key_w2c

    raise ValueError("No extrinsics found in SPaTrack npz")


def load_spatrack(npz_path: str, verbose: bool = False, load_depth: bool = True) -> Dict[str, object]:
    """Load SPaTrack npz and normalize outputs.

    Returns: K (3x3), T_c_w (T,4,4), has_depth, depth (optional), used_ex, keys, H_src, W_src.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        keys_list = sorted(list(getattr(z, "files", [])))
        K = _norm_intrinsics(z)
        T_c_w, used_ex = _canon_extrinsics(z)

        has_depth, depth = False, None
        if load_depth:
            for k in _DEPTH_KEYS:
                if k in z:
                    has_depth, depth = True, z[k].astype(np.float32)
                    break
        else:
            has_depth = any(k in z for k in _DEPTH_KEYS)

        H_src = int(z["H"]) if "H" in z else 0
        W_src = int(z["W"]) if "W" in z else 0

    if verbose:
        keys = ",".join(keys_list)
        vid = npz_path.split("/")[-3] if "/spatrack/" in npz_path else "?"
        print(f"[bold cyan]SPaTrack[/bold cyan] {vid} • frames={T_c_w.shape[0]} • keys={{{keys}}} • extrinsics={used_ex or 'c2w'} • depth={'✓' if has_depth else '×'}")

    return {"K": K, "T_c_w": T_c_w, "has_depth": has_depth, "depth": depth, "used_ex": used_ex, "keys": keys_list, "H_src": H_src, "W_src": W_src}


def get_spatrack_length(npz_path: str) -> int:
    """Fast path to read only the time dimension from a SPaTrack npz."""
    candidates = tuple(f"{k}.npy" for k in _C2W_KEYS + _W2C_KEYS)

    with zipfile.ZipFile(npz_path, "r") as zf:
        names = set(zf.namelist())
        target_name = next((nm for nm in candidates if nm in names), None)
        if target_name is None:
            raise ValueError("No extrinsics array found in SPaTrack npz")

        with zf.open(target_name, "r") as fp:
            ver = read_magic(fp)
            if ver == (2, 0):
                shape, _, _ = read_array_header_2_0(fp)
            else:
                shape, _, _ = read_array_header_1_0(fp)

    return int(shape[0]) if isinstance(shape, tuple) and len(shape) >= 1 else 0
