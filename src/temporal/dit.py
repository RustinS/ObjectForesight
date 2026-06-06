from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(torch.nn.Module):
    """
    Sinusoidal positional embedding for timesteps.

    This implements the standard sinusoidal embedding used in transformers,
    adapted for diffusion timesteps. Creates unique embeddings for each timestep
    that the model can learn to associate with different noise levels.

    Args:
        dim: Embedding dimension (must be even for proper sinusoidal encoding)
    """

    def __init__(self, dim: int, T: int) -> None:
        super().__init__()
        self.dim = dim
        self.T = int(T)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.

        Args:
            t: Timestep tensor, shape (batch_size,)

        Returns:
            Sinusoidal embeddings, shape (batch_size, dim)
        """
        half_dim = self.dim // 2
        device = t.device

        # Create frequency scaling factor
        # log(10000) / (half_dim - 1) creates decreasing frequencies
        log_freq = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)

        # Create frequency scales: exp(-i * log_freq) for i in [0, half_dim-1]
        freq_scales = torch.exp(torch.arange(half_dim, device=device) * -log_freq)

        # Normalize timesteps to [0, 1] using training T, then scale by frequencies
        t_norm = t.float() / float(max(1, self.T - 1))
        # Scale timesteps by frequencies: t * freq_scales
        # Shape: (batch_size, half_dim)
        scaled_timesteps = t_norm.unsqueeze(1) * freq_scales.unsqueeze(0)

        # Create sinusoidal embeddings: [sin(scaled), cos(scaled)]
        # Shape: (batch_size, dim)
        embeddings = torch.cat([torch.sin(scaled_timesteps), torch.cos(scaled_timesteps)], dim=1)

        # Handle odd dimensions by padding with zeros
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))

        return embeddings


class SignedSinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        log_freq = math.log(10000.0) / (half - 1)
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -log_freq)
        self.register_buffer("freq", freq)

    def forward(self, offsets: torch.Tensor, dtype: torch.dtype = None) -> torch.Tensor:
        scaled = offsets.float().unsqueeze(-1) * self.freq
        emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        if dtype is not None:
            emb = emb.to(dtype)
        return emb


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning (per-layer MLP, like original DiT).

    AdaLN params ordering: [scale1, shift1, gate1, scale2, shift2, gate2]
      - scale: multiplicative modulation of LayerNorm output (1 + scale)
      - shift: additive modulation of LayerNorm output
      - gate: output scaling, zero-initialized for identity at init
    """

    def __init__(self, latent_dim: int, n_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0, cond_dim: int | None = None):
        super().__init__()
        # LayerNorm WITHOUT affine (AdaLN provides scale/shift)
        self.norm1 = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)  # Dropout on attention output

        mlp_hidden = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, latent_dim),
            nn.Dropout(dropout),
        )

        # Per-layer AdaLN MLP (like original DiT)
        # Takes conditioning vector, outputs 6 modulation params * latent_dim
        _cond_dim = cond_dim if cond_dim is not None else latent_dim
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(_cond_dim, 6 * latent_dim),
        )
        # Zero-init for identity at initialization
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, C) conditioning vector (combined timestep + scene)
        # Generate this block's AdaLN params
        adaln_params = self.adaln_mlp(cond)  # (B, 6*C)
        adaln_params = adaln_params.view(cond.shape[0], 6, -1)  # (B, 6, C)
        scale1, shift1, gate1, scale2, shift2, gate2 = adaln_params.unbind(dim=1)

        # Apply tanh to gates for stability (bounds to [-1, 1])
        gate1 = torch.tanh(gate1)
        gate2 = torch.tanh(gate2)

        # Attention with AdaLN modulation
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        h = self.attn_drop(h)
        x = x + gate1.unsqueeze(1) * h

        # MLP with AdaLN modulation
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


class DiTPose(torch.nn.Module):
    """
    Diffusion Transformer for pose prediction.

    This implements a simplified DiT architecture that processes flattened pose sequences
    (H*9 dimensions) through a transformer encoder to predict noise for diffusion.

    Architecture:
    - Input projection: flattened pose -> latent space
    - Timestep embedding: sinusoidal encoding of diffusion timesteps
    - Condition projection: conditioning information -> latent space
    - Transformer encoder: processes token with timestep and condition information
    - Output projection: latent space -> noise prediction

    Args:
        latent_dim: Hidden dimension of transformer (default: 768)
        n_layers: Number of transformer encoder layers (default: 12)
        n_heads: Number of attention heads (default: 12)
        dropout: Dropout rate (default: 0.1)
        out_dim: Output dimension (H*9 for pose sequences). Must be provided by config.
        T: Total diffusion timesteps (default: 1000)
        ddim_steps: DDIM sampling steps (default: 50)
    """

    def __init__(
        self,
        latent_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        out_dim: int | None = None,
        T: int = 1000,
        ddim_steps: int = 50,
        cond_dim: int | None = None,
        conditioning: str = "film",
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        if out_dim is None:
            # Do not silently assume a horizon; require config to wire H via out_dim=H*9.
            raise ValueError("DiTPose requires 'out_dim' to be set from config (expected out_dim = H * 9). No hard-coded default horizon.")
        self.out_dim = int(out_dim)
        self.T = T
        self.ddim_steps = ddim_steps
        self._cond_dim = int(cond_dim) if cond_dim is not None else int(latent_dim)

        # Normalize and validate conditioning mode (backup for paths that bypass config_adapter)
        conditioning = str(conditioning).lower().replace("-", "_")
        if conditioning not in ("film", "adaln_zero"):
            raise ValueError(f"Invalid conditioning='{conditioning}'. Must be 'film' or 'adaln_zero'.")
        self.conditioning = conditioning

        # Timestep embedding for diffusion timesteps (normalized by training T)
        self.t_embed = SinusoidalTimestepEmbedding(latent_dim, T=self.T)

        # Projection layers for per-token processing (9D per step)
        self.in_proj = torch.nn.Linear(9, latent_dim)
        self.out_proj = torch.nn.Linear(latent_dim, 9)
        # Conditioning projection accepts external encoder dim
        self.cond_proj = torch.nn.Linear(self._cond_dim, latent_dim)

        # Light MLPs for timestep/condition
        self.t_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        if self.conditioning == "adaln_zero":
            # Combined conditioning MLP (timestep + scene -> single conditioning vector)
            self.cond_combine = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )

            # DiT blocks with per-layer AdaLN MLPs (like original DiT)
            self.blocks = nn.ModuleList([DiTBlock(latent_dim, n_heads, dropout, mlp_ratio, cond_dim=latent_dim) for _ in range(n_layers)])

            # Final normalization
            self.final_norm = nn.LayerNorm(latent_dim)
        else:
            # FiLM conditioning (original implementation)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=n_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.pre_ln = nn.LayerNorm(self.latent_dim)
            # Per-channel FiLM-style gates (scale and bias) for timestep and conditioning
            # Initialize biases to 1.0 (strong initial effect), scales to 0.0 (no multiplicative effect initially)
            self.a_t = nn.Parameter(torch.zeros(self.latent_dim))  # multiplicative (per-channel) for t
            self.a_c = nn.Parameter(torch.zeros(self.latent_dim))  # multiplicative (per-channel) for cond
            self.b_t = nn.Parameter(torch.ones(self.latent_dim))  # additive (per-channel) for t
            self.b_c = nn.Parameter(torch.ones(self.latent_dim))  # additive (per-channel) for cond

        # Output projection: small init (not zero) to allow gradient flow
        # Zero-init blocks gradients since d(out)/d(x) = W = 0
        # Use small normal init scaled by 1/sqrt(n_layers) for stability
        nn.init.normal_(self.out_proj.weight, std=0.02 / (n_layers ** 0.5))
        nn.init.zeros_(self.out_proj.bias)

        # Learned positional encodings over H (max length 512)
        self.pos = torch.nn.Parameter(torch.zeros(1, 512, latent_dim))
        torch.nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.token_type_embed = nn.Embedding(2, latent_dim)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        self.token_type_scale = nn.Parameter(torch.ones(1))

        self.rel_temp_embed = SignedSinusoidalEmbedding(latent_dim)
        self.rel_temp_scale = nn.Parameter(torch.full((1,), 0.02))

    def forward(self, noisy_y: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor, ctx_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the DiT model.

        Args:
            noisy_y: Noisy pose data, shape (batch_size, out_dim=H*9)
            timestep: Diffusion timesteps, shape (batch_size,)
            cond: Conditioning information, shape (batch_size, cond_dim)
            ctx_tokens: Optional past-context tokens (depth-norm normalized), shape (batch_size, P, 9)

        Returns:
            Predicted noise over the future tokens, shape (batch_size, out_dim)
        """
        B, D = noisy_y.shape
        if D % 9 != 0:
            raise ValueError(f"DiTPose.forward expected noisy_y dim to be a multiple of 9, got D={D}.")
        H_from_input = D // 9
        H_cfg = int(self.out_dim) // 9
        if H_from_input != H_cfg:
            raise ValueError(
                f"DiTPose horizon mismatch: cfg-based H={H_cfg} (out_dim={int(self.out_dim)}) "
                f"but input has H={H_from_input} (D={D}). Check cfg.model.H / cfg.data.H vs cfg.temporal.out_dim."
            )
        y = noisy_y.view(B, H_from_input, 9)

        # Per-token input projection (future tokens)
        y_tok = self.in_proj(y)  # (B, H, C)

        # If context tokens are provided, embed and prepend
        if ctx_tokens is not None:
            if ctx_tokens.dim() == 2:
                ctx_tokens = ctx_tokens.unsqueeze(0)
            if ctx_tokens.size(-1) != 9:
                raise ValueError(f"ctx_tokens last dim must be 9, got {ctx_tokens.size(-1)}")
            P = int(ctx_tokens.shape[1])
            ctx_tok = self.in_proj(ctx_tokens)  # (B, P, C)
            tokens = torch.cat([ctx_tok, y_tok], dim=1)  # (B, P+H, C)
        else:
            P = 0
            tokens = y_tok  # (B, H, C)

        L = P + H_from_input
        device = tokens.device

        tokens = tokens + self.pos[:, :L, :]

        # Token type: context=0, future=1
        type_ids = torch.cat([torch.zeros(P, dtype=torch.long, device=device), torch.ones(H_from_input, dtype=torch.long, device=device)])
        tokens = tokens + self.token_type_scale * self.token_type_embed(type_ids)

        # Anchor-relative offsets: context ends at 0, future starts at 0
        ctx_offsets = torch.arange(-P + 1, 1, device=device)
        fut_offsets = torch.arange(0, H_from_input, device=device)
        rel_offsets = torch.cat([ctx_offsets, fut_offsets])
        tokens = tokens + self.rel_temp_scale * self.rel_temp_embed(rel_offsets, dtype=tokens.dtype)

        # Gated time/cond with light MLPs
        t_emb = self.t_embed(timestep)  # (B, C)
        t_emb = self.t_mlp(t_emb)  # (B, C)

        c_tok = self.cond_proj(cond)  # (B, C)
        c_tok = self.cond_mlp(c_tok)  # (B, C)

        if self.conditioning == "adaln_zero":
            # AdaLN-Zero: combine conditioning → pass to each block's per-layer MLP
            cond_combined = self.cond_combine(torch.cat([t_emb, c_tok], dim=-1))  # (B, C)

            x = tokens
            for block in self.blocks:
                x = block(x, cond_combined)

            x = self.final_norm(x)  # (B, L, C)
        else:
            # FiLM-style modulation: tokens * (1 + a_t * t + a_c * c) + (b_t * t + b_c * c)
            t_emb_exp = t_emb.unsqueeze(1).expand(B, L, -1)
            c_tok_exp = c_tok.unsqueeze(1).expand(B, L, -1)
            combined = tokens * (1.0 + self.a_t * t_emb_exp + self.a_c * c_tok_exp) + (self.b_t * t_emb_exp + self.b_c * c_tok_exp)
            combined = self.pre_ln(combined)

            # Transformer over L tokens
            x = self.transformer(combined)  # (B, L, C)

        # Project back to 9D per token and keep only future positions
        x9 = self.out_proj(x).reshape(B, L, 9)
        eps = x9[:, -H_from_input:, :].reshape(B, H_from_input * 9)

        # Gradient audit
        if getattr(self, "_grad_audit", False):
            try:
                self._ga = {"in_proj": tokens, "cond_proj": c_tok}
            except Exception:
                pass

        return eps
