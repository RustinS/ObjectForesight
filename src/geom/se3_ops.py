from __future__ import annotations

import math
from typing import Iterable, Literal, Tuple

import numpy as np
import torch

# ============================================================================
# PyTorch SE(3) Operations (Primary Interface)
# ============================================================================


def _det3x3_dtype_safe(R: torch.Tensor) -> torch.Tensor:
    """Return det(R) with a safe upcast when bf16/half on CUDA."""
    if R.dtype in (torch.bfloat16, torch.float16):
        return torch.det(R.to(torch.float32)).to(R.dtype)
    return torch.det(R)


def _svd_dtype_safe(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (U, S, Vh) with a safe upcast when bf16/half on CUDA."""
    if A.dtype in (torch.bfloat16, torch.float16):
        U, S, Vh = torch.linalg.svd(A.to(torch.float32))
        return U.to(A.dtype), S.to(A.dtype), Vh.to(A.dtype)
    return torch.linalg.svd(A)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 matrix (Zhou et al.)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    eps = 1e-6 if x.dtype not in (torch.bfloat16, torch.float16) else 1e-4
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
    u2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    u2_norm = torch.linalg.norm(u2, dim=-1, keepdim=True)
    need_fallback = u2_norm.squeeze(-1) < eps
    if need_fallback.any():
        basis = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype, device=x.device)
        alt = torch.where((b1.abs().argmin(dim=-1, keepdim=True) == 0).expand_as(b1), torch.tensor([0.0, 1.0, 0.0], dtype=x.dtype, device=x.device), basis)
        u2_fb = alt - (b1 * alt).sum(-1, keepdim=True) * b1
        u2 = torch.where(need_fallback.unsqueeze(-1), u2_fb, u2)
    b2 = torch.nn.functional.normalize(u2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (...,3,3) to 6D by taking first two columns."""
    return R[..., :3, :2].reshape(*R.shape[:-2], 6)


def compose_se3(t1: torch.Tensor, R1: torch.Tensor, t2: torch.Tensor, R2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two SE3 transforms T = T1 * T2."""
    return t1 + torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1), torch.matmul(R1, R2)


def invert_se3(t: torch.Tensor, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert SE3 transform."""
    R_inv = R.transpose(-1, -2)
    return -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1), R_inv


def se3_to_matrix(t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Pack to 4x4 matrix."""
    B = t.shape[0]
    T = torch.eye(4, device=t.device, dtype=t.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def matrix_to_se3(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract translation and rotation from 4x4 matrix."""
    return T[..., :3, 3], T[..., :3, :3]


def geodesic_distance_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance in degrees between rotations."""
    R = torch.matmul(R1.transpose(-1, -2), R2)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    eps = 1e-6 if R1.dtype not in (torch.bfloat16, torch.float16) else 1e-3
    cos = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    return torch.arccos(cos) * (180.0 / math.pi)


def _ensure_homogeneous(T: torch.Tensor) -> torch.Tensor:
    """Fail fast on malformed transforms: shape (H,4,4), last row [0,0,0,1]."""
    if T.dim() != 3 or T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (H,4,4) tensor, got {tuple(T.shape)}")
    last = T[:, 3, :]
    if not torch.allclose(last, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=T.dtype, device=T.device).unsqueeze(0), atol=1e-5):
        raise ValueError("SE(3) matrix last row must be [0,0,0,1]")
    return T


def compose_T(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compose batched 4x4 transforms. Broadcasts singleton batch."""
    A, B = _ensure_homogeneous(A), _ensure_homogeneous(B)
    batch = max(A.shape[0], B.shape[0])
    if A.shape[0] == 1 and batch > 1:
        A = A.repeat(batch, 1, 1)
    if B.shape[0] == 1 and batch > 1:
        B = B.repeat(batch, 1, 1)
    if A.shape[0] != batch or B.shape[0] != batch:
        raise ValueError(f"compose_T batch mismatch: A={A.shape[0]} B={B.shape[0]}")
    t1, R1 = A[:, :3, 3], A[:, :3, :3]
    t2, R2 = B[:, :3, 3], B[:, :3, :3]
    t, R = compose_se3(t1, R1, t2, R2)
    T = torch.eye(4, dtype=A.dtype, device=A.device).unsqueeze(0).repeat(batch, 1, 1)
    T[:, :3, :3], T[:, :3, 3] = R, t
    return T


def invert_T(T: torch.Tensor) -> torch.Tensor:
    """Invert batched 4x4 SE(3)."""
    T = _ensure_homogeneous(T)
    t, R = T[:, :3, 3], T[:, :3, :3]
    t_inv, R_inv = invert_se3(t, R)
    Tout = torch.eye(4, dtype=T.dtype, device=T.device).unsqueeze(0).repeat(T.shape[0], 1, 1)
    Tout[:, :3, :3] = R_inv
    Tout[:, :3, 3] = t_inv
    return Tout


# ============================================================================
# NumPy SE(3) Operations (Data Loading Interface)
# ============================================================================


def is_rotation_matrix(R: np.ndarray, atol: float = 1e-4) -> bool:
    """Check if R is a valid rotation matrix."""
    if R.shape[-2:] != (3, 3):
        return False
    RtR = R.T @ R
    ortho_err = float(np.linalg.norm(RtR - np.eye(3, dtype=R.dtype), ord="fro"))
    det = float(np.linalg.det(R))
    return abs(det - 1.0) <= 1e-3 and ortho_err <= atol


def matrix_info(T: np.ndarray) -> dict:
    """Return diagnostics for a 4x4 SE(3) matrix."""
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    R, t = T[:3, :3], T[:3, 3]
    RtR = R.T @ R
    return {
        "detR": float(np.linalg.det(R)),
        "ortho_err": float(np.linalg.norm(RtR - np.eye(3, dtype=R.dtype), ord="fro")),
        "t_norm": float(np.linalg.norm(t)),
        "last_row_ok": bool(np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)),
    }


def _to_4x4(M: np.ndarray) -> np.ndarray:
    if M.shape == (4, 4):
        return M.astype(np.float32)
    if M.shape == (3, 4):
        M4 = np.eye(4, dtype=np.float32)
        M4[:3, :4] = M.astype(np.float32)
        return M4
    raise ValueError(f"Pose has invalid shape {M.shape}")


def read_pose_4x4(path: str, strict: bool = True) -> np.ndarray:
    """Read a pose from a text file and return a validated 4x4 matrix."""
    M = _to_4x4(np.loadtxt(path).astype(np.float32))
    if not np.allclose(M[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        if strict:
            raise ValueError(f"Invalid last row in {path}: {M[3]}")
        M[3] = [0.0, 0.0, 0.0, 1.0]
    R = M[:3, :3]
    if not is_rotation_matrix(R):
        if strict:
            info = matrix_info(M)
            raise ValueError(f"Invalid rotation in {path}: det={info['detR']:.6f} ortho_err={info['ortho_err']:.2e}")
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        M[:3, :3] = R_ortho.astype(np.float32)
    return M.astype(np.float32)


def invert_T_np(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) transform."""
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    R, t = T[:3, :3], T[:3, 3]
    Tout = np.eye(4, dtype=T.dtype)
    Tout[:3, :3] = R.T
    Tout[:3, 3] = -R.T @ t
    return Tout


def compose_T_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transforms: result = A @ B."""
    if A.shape[-2:] != (4, 4) or B.shape[-2:] != (4, 4):
        raise ValueError("compose_T_np expects (4,4) matrices")
    return (A @ B).astype(np.float32)


def reexpress_object_pose(T_cam_k_obj: np.ndarray, E_k: np.ndarray, E_i0: np.ndarray, convention: Literal["c2w", "w2c"]) -> np.ndarray:
    """Express object pose from frame k in camera coordinates of frame i0."""
    if convention == "c2w":
        return compose_T_np(compose_T_np(invert_T_np(E_i0), E_k), T_cam_k_obj)
    elif convention == "w2c":
        return compose_T_np(compose_T_np(E_i0, invert_T_np(E_k)), T_cam_k_obj)
    raise ValueError("convention must be 'c2w' or 'w2c'")


def select_extrinsics_convention(T_cam_i0_obj_i0_gt: np.ndarray, E_i0_c2w: np.ndarray, E_i0_w2c: np.ndarray, T_cam_i0_obj_i0_cam: np.ndarray) -> Tuple[str, float, float]:
    """Auto-select extrinsics convention by self-consistency at k==i0."""

    def mat_err(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.linalg.norm(A[:3, :3] - B[:3, :3], ord="fro") + np.linalg.norm(A[:3, 3] - B[:3, 3]))

    T_cand_c2w = reexpress_object_pose(T_cam_i0_obj_i0_cam, E_i0_c2w, E_i0_c2w, "c2w")
    T_cand_w2c = reexpress_object_pose(T_cam_i0_obj_i0_cam, E_i0_w2c, E_i0_w2c, "w2c")
    err_c2w = mat_err(T_cand_c2w, T_cam_i0_obj_i0_gt)
    err_w2c = mat_err(T_cand_w2c, T_cam_i0_obj_i0_gt)
    return ("c2w" if err_c2w <= err_w2c else "w2c"), err_c2w, err_w2c


def batched_reexpress(
    T_cam_k_obj_list: Iterable[np.ndarray], E_c2w_list: Iterable[np.ndarray], E_w2c_list: Iterable[np.ndarray], i0_idx: int, convention: Literal["c2w", "w2c"]
) -> np.ndarray:
    """Vectorized helper to compute T_cam_i0_obj_k for all k."""
    E_list = list(E_c2w_list) if convention == "c2w" else list(E_w2c_list)
    E_i0 = E_list[i0_idx]
    results = [reexpress_object_pose(Tk, Ek, E_i0, convention) for Tk, Ek in zip(T_cam_k_obj_list, E_list, strict=False)]
    return np.stack(results, axis=0).astype(np.float32)
