from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import EncoderBase
from .ptv3_model import PointTransformerV3, offset2batch


class ContextFiLM(torch.nn.Module):
    def __init__(self, feature_dim: int, context_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(context_dim, feature_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, features: torch.Tensor, context_vec: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.mlp(context_vec)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return features * (1 + gamma) + beta


class PTV3Encoder(EncoderBase):
    """PointTransformerV3 encoder with FiLM context conditioning."""

    def __init__(
        self,
        embed_dim: int = 768,
        context_dim: int = 32,
        grid_size: float = 0.02,
        in_channels: int = 3,
        pooling: str = "mean",
        skip_film: bool = False,
        return_dict: bool = True,
        backbone_kwargs: dict | None = None,
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.grid_size = float(grid_size)
        self.in_channels = int(in_channels)
        self.pooling = str(pooling)
        self.skip_film = bool(skip_film)
        self.return_dict = return_dict
        self._use_fp16 = bool(use_fp16)

        bb_kwargs = dict(backbone_kwargs or {})
        bb_kwargs["in_channels"] = self.in_channels
        self.backbone = PointTransformerV3(**bb_kwargs)
        self.film = ContextFiLM(feature_dim=embed_dim, context_dim=context_dim)

        valid_pooling = {"mean", "max", "mean_max", "mean_std", "attn", "attn_ctx", "attn_obj"}
        if self.pooling not in valid_pooling:
            raise ValueError(f"Unknown pooling={self.pooling!r}. Valid: {sorted(valid_pooling)}")

        # Optional pooling heads
        self._pool_fuse: torch.nn.Linear | None = None
        if self.pooling in {"mean_max", "mean_std"}:
            self._pool_fuse = torch.nn.Linear(embed_dim * 2, embed_dim, bias=False)

        self._pool_score: torch.nn.Module | None = None
        if self.pooling == "attn":
            self._pool_score = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(embed_dim, 1, bias=False),
            )

        self._ctx_to_query: torch.nn.Linear | None = None
        if self.pooling in {"attn_ctx", "attn_obj"}:
            self._ctx_to_query = torch.nn.Linear(context_dim, embed_dim, bias=False)

        self._dist_bias: torch.nn.Module | None = None
        self._log_temp: torch.nn.Parameter | None = None
        if self.pooling == "attn_obj":
            hid = max(1, embed_dim // 4)
            self._dist_bias = torch.nn.Sequential(
                torch.nn.Linear(1, hid),
                torch.nn.SiLU(),
                torch.nn.Linear(hid, 1),
            )
            self._log_temp = torch.nn.Parameter(torch.tensor(0.0))

        # Eagerly initialize projection (fixes optimizer/DDP sync issues)
        # PTv3 output dim depends on cls_mode:
        #   cls_mode=False (default): returns dec_channels[0] (decoder output)
        #   cls_mode=True: returns enc_channels[-1] (encoder output)
        cls_mode = bb_kwargs.get("cls_mode", False)
        if cls_mode:
            ptv3_out_dim = bb_kwargs.get("enc_channels", (32, 64, 128, 256, 512))[-1]
        else:
            ptv3_out_dim = bb_kwargs.get("dec_channels", (64, 64, 128, 256))[0]
        self._proj = torch.nn.Linear(ptv3_out_dim, embed_dim, bias=False) if ptv3_out_dim != embed_dim else None

    def _maybe_project(self, x: torch.Tensor) -> torch.Tensor:
        if self._proj is None:
            return x
        assert x.shape[-1] == self._proj.in_features, f"PTv3 output dim mismatch: got {x.shape[-1]}, expected {self._proj.in_features}"
        return self._proj(x)

    def _pool_global(
        self,
        feat_all: torch.Tensor,
        batch_all: torch.Tensor,
        context_vec: torch.Tensor,
        B: int,
        *,
        obj_coords_all: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = feat_all.device
        dtype = feat_all.dtype
        C = feat_all.shape[-1]

        pooled = torch.zeros((B, C), device=device, dtype=dtype)

        scores_all = None
        if self.pooling == "attn":
            assert self._pool_score is not None
            scores_all = self._pool_score(feat_all).squeeze(-1)  # (N,)
        elif self.pooling == "attn_ctx":
            assert self._ctx_to_query is not None
            q = self._ctx_to_query(context_vec)  # (B, C)
            scores_all = (feat_all * q[batch_all]).sum(dim=-1) / math.sqrt(C)  # (N,)
        elif self.pooling == "attn_obj":
            if obj_coords_all is None:
                raise ValueError("pooling='attn_obj' requires obj_coords_all (derived from T_cam_anchor_obj).")
            assert self._ctx_to_query is not None
            assert self._dist_bias is not None
            assert self._log_temp is not None

            q = self._ctx_to_query(context_vec)  # (B, C)
            scores_content = (feat_all * q[batch_all]).sum(dim=-1) / math.sqrt(C)  # (N,)
            obj_dist = torch.linalg.norm(obj_coords_all, dim=-1)  # (N,)
            # Higher temp = softer distance weighting; division ensures correct semantics
            dist_in = (-obj_dist / torch.exp(self._log_temp).clamp(min=1e-3)).unsqueeze(-1)  # (N,1)
            scores_dist = self._dist_bias(dist_in).squeeze(-1)  # (N,)
            scores_all = scores_content + scores_dist

        for b in range(B):
            m = batch_all == b
            if not m.any():
                continue
            fb = feat_all[m]

            if self.pooling == "mean":
                pooled[b] = fb.mean(dim=0)
            elif self.pooling == "max":
                pooled[b] = fb.max(dim=0).values
            elif self.pooling == "mean_max":
                assert self._pool_fuse is not None
                fused = torch.cat([fb.mean(dim=0), fb.max(dim=0).values], dim=0).unsqueeze(0)
                pooled[b] = self._pool_fuse(fused).squeeze(0)
            elif self.pooling == "mean_std":
                assert self._pool_fuse is not None
                if fb.shape[0] == 1:
                    std = torch.zeros_like(fb[0])
                else:
                    std = fb.std(dim=0, unbiased=False)
                fused = torch.cat([fb.mean(dim=0), std], dim=0).unsqueeze(0)
                pooled[b] = self._pool_fuse(fused).squeeze(0)
            elif self.pooling in {"attn", "attn_ctx", "attn_obj"}:
                assert scores_all is not None
                sb = scores_all[m]
                wb = torch.softmax(sb.to(dtype=torch.float32), dim=0).to(dtype=dtype)
                pooled[b] = (fb * wb.unsqueeze(-1)).sum(dim=0)
            else:
                raise AssertionError(f"Unreachable pooling={self.pooling!r}")

        return pooled

    def forward(
        self,
        scene_pcd: torch.Tensor,
        context_vec: torch.Tensor,
        *,
        T_cam_anchor_obj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        B, N, _ = scene_pcd.shape
        device = scene_pcd.device

        # Optional FP16 for encoder (faster on tensor-core GPUs)
        # Enable via encoder_fp16=True in config or pass use_fp16=True
        use_fp16 = use_fp16 or getattr(self, "_use_fp16", False)
        if use_fp16 and torch.cuda.is_available():
            return self._forward_fp16(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj, mask=mask)
        return self._forward_impl(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj, mask=mask)

    def _forward_fp16(
        self,
        scene_pcd: torch.Tensor,
        context_vec: torch.Tensor,
        *,
        T_cam_anchor_obj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """FP16 forward pass using autocast."""
        with torch.amp.autocast("cuda", dtype=torch.float16):
            result = self._forward_impl(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj, mask=mask)
        # Cast output back to fp32
        if isinstance(result, dict):
            return {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
        return result.float()

    def _forward_impl(
        self,
        scene_pcd: torch.Tensor,
        context_vec: torch.Tensor,
        *,
        T_cam_anchor_obj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Core forward implementation."""
        B, N, _ = scene_pcd.shape
        device = scene_pcd.device

        if mask is not None:
            valid = mask.to(dtype=torch.bool, device=device).view(-1)
            coords = scene_pcd.view(-1, 3)[valid]
            batch = torch.arange(B, device=device).repeat_interleave(N)[valid]
            offset = torch.cumsum(valid.view(B, N).sum(dim=1), dim=0)
        else:
            coords = scene_pcd.reshape(B * N, 3)
            offset = torch.arange(1, B + 1, device=device, dtype=torch.long) * N
            batch = offset2batch(offset)

        if self.in_channels == 3:
            feat = coords
        elif self.in_channels == 6:
            # Object-centric transform: 6D features = [camera_coords, object_coords]
            if T_cam_anchor_obj is not None:
                T_obj_cam = torch.inverse(T_cam_anchor_obj)  # (B, 4, 4)
                pcd_homo = F.pad(scene_pcd, (0, 1), value=1.0)  # (B, N, 4)
                pcd_obj = (T_obj_cam @ pcd_homo.transpose(-1, -2)).transpose(-1, -2)[..., :3]  # (B, N, 3)
                if mask is not None:
                    pcd_obj_flat = pcd_obj.view(-1, 3)[valid]
                else:
                    pcd_obj_flat = pcd_obj.reshape(B * N, 3)
                feat = torch.cat([coords, pcd_obj_flat], dim=-1)  # (N_valid, 6)
            else:
                feat = torch.cat([coords, torch.zeros_like(coords)], dim=-1)  # fallback: zeros
        else:
            raise ValueError(f"PTV3Encoder only supports in_channels in {{3, 6}} right now, got in_channels={self.in_channels}.")

        point_dict = {"coord": coords, "feat": feat, "batch": batch, "grid_size": self.grid_size, "offset": offset, "context": context_vec}

        # Deterministic eval: disable order shuffling
        if not self.training:
            prev_shuffle = getattr(self.backbone, "shuffle_orders", False)
            self.backbone.shuffle_orders = False
            out_point = self.backbone(point_dict)
            self.backbone.shuffle_orders = prev_shuffle
        else:
            out_point = self.backbone(point_dict)

        feat_all = self._maybe_project(out_point["feat"])
        # Clone batch indices to avoid inference-mode tensor in autograd (PTv3 creates them in inference mode)
        batch_all = out_point["batch"].clone()

        obj_coords_all = None
        if self.pooling == "attn_obj":
            if T_cam_anchor_obj is None:
                raise ValueError("pooling='attn_obj' requires T_cam_anchor_obj.")
            coord_all = out_point["coord"]
            if coord_all.shape[0] != feat_all.shape[0]:
                raise RuntimeError(f"PTv3 output mismatch: coord N={coord_all.shape[0]} != feat N={feat_all.shape[0]}")
            # Use fp32 for inverse/transform for stability across dtypes.
            T_obj_cam = torch.inverse(T_cam_anchor_obj.to(dtype=torch.float32))  # (B,4,4)
            coord_homo = F.pad(coord_all.to(dtype=torch.float32), (0, 1), value=1.0)  # (N,4)
            obj_homo = torch.bmm(T_obj_cam[batch_all], coord_homo.unsqueeze(-1)).squeeze(-1)  # (N,4)
            obj_coords_all = obj_homo[:, :3].to(dtype=feat_all.dtype)

        pooled = self._pool_global(feat_all, batch_all, context_vec, B, obj_coords_all=obj_coords_all)
        global_feat = pooled if self.skip_film else self.film(pooled, context_vec)

        if not self.return_dict:
            return global_feat

        per_point = None
        if mask is None and feat_all.shape[0] == B * N:
            per_point = feat_all.reshape(B, N, -1)

        return {"per_point": per_point, "global": global_feat, "stages": None, "aux": None}
