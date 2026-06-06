from __future__ import annotations

from typing import Optional

import torch


class AutoregTransformerPose(torch.nn.Module):
    """
    Autoregressive Transformer for pose sequence prediction.

    - Inputs/Outputs follow the project convention: per-step 9D se3 token (t[3] + rot6d[6]).
    - Training uses teacher forcing over the ground-truth sequence to predict the next token at each step.
    - Inference performs causal rollout using the model's previous predictions.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        n_layers: int = 8,
        n_heads: int = 12,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        use_context_token: bool = True,
        w_rot: float = 2.0,
        w_trans: float = 2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_context_token = bool(use_context_token)
        self._w_rot = float(w_rot)
        self._w_trans = float(w_trans)

        # Token embeddings
        self.pose_in = torch.nn.Sequential(
            torch.nn.Linear(9, latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, latent_dim),
        )
        self.context_proj = torch.nn.Linear(latent_dim, latent_dim)
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, latent_dim))

        # Learned positional encodings (length >= H+1)
        self.pos_embed = torch.nn.Parameter(torch.zeros(self.max_seq_len, latent_dim))
        torch.nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Causal Transformer (GPT-like via TransformerEncoder + causal mask)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final LayerNorm (GPT-style, before output projection)
        self.ln_final = torch.nn.LayerNorm(latent_dim)

        # Output head for 9D se3 per step
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, 9),
        )

    @property
    def supports_scheduled_sampling(self) -> bool:
        return True

    # ar-transformer: build an upper-triangular boolean mask to prevent future leakage
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        # PyTorch Transformer supports boolean attn_mask (True = masked)
        return mask

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.shape[1]
        if L > self.pos_embed.shape[0]:
            raise ValueError(f"Sequence length L={L} exceeds pos_embed length={self.pos_embed.shape[0]}. Increase temporal.max_seq_len.")
        pos = self.pos_embed[:L].unsqueeze(0)  # [1, L, D]
        return x + pos

    # ar-transformer: teacher-forced training forward
    def forward_train(self, y_gt: torch.Tensor, cond_embed: torch.Tensor, ctx_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        y_gt: [B, H, 9] ground-truth sequence in canonical frame.
        cond_embed: [B, E] scene/mesh/context embedding (already projected by the encoder path).
        ctx_tokens: optional past-context tokens [B, P, 9] to prepend before START.

        The first predicted step is conditioned on the last context pose when available; otherwise falls back to START (or start_token_y0).

        Returns: y_pred [B, H, 9] aligned to GT steps 1..H (next-step predictions).
        """
        B, H, _ = y_gt.shape
        device = y_gt.device

        # Optional context prefix as tokens
        P = 0
        ctx_emb = None
        if ctx_tokens is not None:
            if ctx_tokens.dim() == 2:
                ctx_tokens = ctx_tokens.unsqueeze(0)
            ctx_tokens = ctx_tokens.to(device)
            ctx_emb = self.pose_in(ctx_tokens)  # [B, P, D]
            P = int(ctx_emb.shape[1])

        # Last-context seeding (fallback to first GT when P==0)
        last_ctx = ctx_tokens[:, -1, :] if (ctx_tokens is not None and P > 0) else y_gt[:, 0, :]  # [B,9]

        # Optional tiny jitter to improve robustness to tracker noise (train-time only)
        if self.training:
            eps_t = torch.randn_like(last_ctx[:, :3]) * 5e-3
            eps_r = torch.randn_like(last_ctx[:, 3:]) * 2e-2
            last_ctx = torch.cat([last_ctx[:, :3] + eps_t, last_ctx[:, 3:] + eps_r], dim=-1)

        seed_tok = self.pose_in(last_ctx).unsqueeze(1)  # [B,1,D]
        pose_tokens = self.pose_in(y_gt)  # [B, H, D]
        tokens = torch.cat([ctx_emb, seed_tok, pose_tokens], dim=1) if ctx_emb is not None else torch.cat([seed_tok, pose_tokens], dim=1)

        # Add context (simple additive FiLM-like bias)
        ctx = self.context_proj(cond_embed).unsqueeze(1)  # [B,1,D]
        if self.use_context_token:
            tokens = tokens + ctx  # broadcast add across sequence

        tokens = self._add_positional_encoding(tokens)
        attn_mask = AutoregTransformerPose._causal_mask(tokens.shape[1], device=device, dtype=tokens.dtype)
        h = self.transformer(tokens, mask=attn_mask)

        # Predict next token for positions (P+1)..(P+H) using hidden states at P..(P+H-1)
        h_shift = h[:, P : P + H, :]  # [B, H, D]
        y_pred = self.out_proj(self.ln_final(h_shift))  # [B, H, 9]
        return y_pred

    # Single-step causal prediction used for rollout/scheduled sampling
    def forward_step(self, y_prev: torch.Tensor, cond_embed: torch.Tensor, ctx_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            y_prev: (B, 9) last predicted token (t, rot6d)
            cond_embed: (B, D) conditioning embedding returned by core_model.condition_from_batch
            ctx_tokens: (B, P, 9) optional context tokens (first P tokens, if any)
        Returns:
            y_next: (B, 9)
        """
        if y_prev.dim() == 2:
            y_prev = y_prev.unsqueeze(1)  # (B,1,9)
        # Anchor device to model/conditioning embed to avoid accidental CPU tensors
        device = cond_embed.device
        y_prev = y_prev.to(device)
        # Optional context prefix
        ctx_emb = None
        if ctx_tokens is not None:
            if ctx_tokens.dim() == 2:
                ctx_tokens = ctx_tokens.unsqueeze(0)
            ctx_tokens = ctx_tokens.to(device)
            ctx_emb = self.pose_in(ctx_tokens)  # (B,P,D)
        # Build sequence = concat(context, y_prev)
        seq = self.pose_in(y_prev)  # (B,1,D)
        if ctx_emb is not None:
            seq = torch.cat([ctx_emb, seq], dim=1)  # (B,P+1,D)
        # Add conditioning and positions
        if self.use_context_token:
            ctx = self.context_proj(cond_embed).unsqueeze(1)  # (B,1,D)
            seq = seq + ctx
        seq = self._add_positional_encoding(seq)
        attn_mask = AutoregTransformerPose._causal_mask(seq.shape[1], device=device, dtype=seq.dtype)
        h = self.transformer(seq, mask=attn_mask)
        h_last = h[:, -1, :]
        y_next = self.out_proj(self.ln_final(h_last))  # (B,9)
        return y_next

    # ar-transformer: causal rollout for inference
    @torch.no_grad()
    def rollout_infer(self, H: int, cond_embed: torch.Tensor, start_token_y0: Optional[torch.Tensor] = None, ctx_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a sequence of length H.

        cond_embed: [B, E]
        start_token_y0 (optional): [B, 1, 9] explicit T0 pose token; if None, use learned START.
        ctx_tokens: optional past-context tokens [B, P, 9] to prepend before START.

        The first predicted step is conditioned on the last context pose when available; otherwise falls back to START (or start_token_y0).
        Returns: [B, H, 9]
        """
        device = cond_embed.device
        B = cond_embed.shape[0]
        preds: list[torch.Tensor] = []

        # Optional context prefix as tokens
        P = 0
        if ctx_tokens is not None:
            if ctx_tokens.dim() == 2:
                ctx_tokens = ctx_tokens.unsqueeze(0)
            ctx_tokens = ctx_tokens.to(device)
            ctx_emb = self.pose_in(ctx_tokens)  # [B,P,D]
            P = int(ctx_emb.shape[1])
        else:
            ctx_emb = None

        # Initialize with last context pose if available; else T0 if provided; else learned START
        if P > 0:
            init_tok = self.pose_in(ctx_tokens[:, -1:, :])  # [B,1,D]
        elif start_token_y0 is not None:
            init_tok = self.pose_in(start_token_y0.to(device))  # [B,1,D]
        else:
            init_tok = self.start_token.expand(B, 1, -1)  # [B,1,D]

        tokens = torch.cat([ctx_emb, init_tok], dim=1) if ctx_emb is not None else init_tok  # [B,P+1,D] or [B,1,D]

        ctx = self.context_proj(cond_embed).unsqueeze(1)  # [B,1,D]

        for _ in range(int(H)):
            # Prepare sequence with positional encoding and causal mask
            seq = tokens + (ctx if self.use_context_token else 0)
            seq = self._add_positional_encoding(seq)
            attn_mask = AutoregTransformerPose._causal_mask(seq.shape[1], device=device, dtype=seq.dtype)
            h = self.transformer(seq, mask=attn_mask)
            h_last = h[:, -1, :]
            y_next = self.out_proj(self.ln_final(h_last)).unsqueeze(1)  # [B,1,9]
            preds.append(y_next)
            # Append predicted token for next iteration
            tokens = torch.cat([tokens, self.pose_in(y_next)], dim=1)

        return torch.cat(preds, dim=1)  # [B,H,9]

    def compute_loss_ar(
        self,
        tokens_gt: torch.Tensor,
        cond_embed: torch.Tensor,
        ctx_tokens: Optional[torch.Tensor] = None,
        scheduled_sampling_p: float = 0.0,
    ) -> dict:
        """
        Teacher-forced AR loss with optional scheduled sampling.
        tokens_gt: (B, H, 9) ground-truth future tokens
        ctx_tokens: (B, P, 9) observed context tokens (must be non-empty)
        """
        if not isinstance(tokens_gt, torch.Tensor):
            tokens_gt = torch.as_tensor(tokens_gt)
        device = tokens_gt.device
        B, H, _ = tokens_gt.shape

        if ctx_tokens is None or ctx_tokens.shape[1] == 0:
            raise RuntimeError("AR training requires non-empty context tokens for seeding.")

        # Ensure ctx_tokens on same device/dtype as tokens_gt before any slicing
        ctx_tokens = ctx_tokens.to(device=device, dtype=tokens_gt.dtype, non_blocking=True)

        ss_p = float(scheduled_sampling_p)
        if ss_p <= 0.0:
            y_pred = self.forward_train(y_gt=tokens_gt, cond_embed=cond_embed, ctx_tokens=ctx_tokens)  # (B,H,9)
        else:
            # Scheduled sampling rollout that matches inference tokenization (growing prefix).
            ctx_emb = self.pose_in(ctx_tokens)  # (B,P,D)
            last_ctx = ctx_tokens[:, -1, :]  # (B,9)
            if self.training:
                eps_t = torch.randn_like(last_ctx[:, :3]) * 5e-3
                eps_r = torch.randn_like(last_ctx[:, 3:]) * 2e-2
                last_ctx = torch.cat([last_ctx[:, :3] + eps_t, last_ctx[:, 3:] + eps_r], dim=-1)
            seed_tok = self.pose_in(last_ctx).unsqueeze(1)  # (B,1,D)

            tokens = torch.cat([ctx_emb, seed_tok], dim=1)  # (B,P+1,D)
            ctx = self.context_proj(cond_embed).unsqueeze(1)  # (B,1,D)

            preds: list[torch.Tensor] = []
            for t in range(int(H)):
                seq = tokens + (ctx if self.use_context_token else 0)
                seq = self._add_positional_encoding(seq)
                attn_mask = AutoregTransformerPose._causal_mask(seq.shape[1], device=device, dtype=seq.dtype)
                h = self.transformer(seq, mask=attn_mask)
                y_next = self.out_proj(self.ln_final(h[:, -1, :]))  # (B,9)
                preds.append(y_next)

                with torch.no_grad():
                    m = (torch.rand(B, 1, device=device) < ss_p).float()
                    y_in = m * y_next + (1.0 - m) * tokens_gt[:, t, :]
                tokens = torch.cat([tokens, self.pose_in(y_in).unsqueeze(1)], dim=1)

            y_pred = torch.stack(preds, dim=1)  # (B,H,9)

        # Token-space loss; geometry-aware SE(3) loss is added in PoserV1 after unnormalization.
        # Compute per-step loss for horizon weighting support
        loss_per_step = ((y_pred - tokens_gt) ** 2).mean(dim=-1)  # [B, H]
        loss_main = loss_per_step.mean()
        return {
            "loss": loss_main,  # Non-detached for backprop (can be recomputed with horizon weighting in PoserV1)
            "loss_main": loss_main.detach(),
            "loss_aux": torch.tensor(0.0, device=device, dtype=loss_main.dtype),
            "pred_9d": y_pred,
            "loss_per_step": loss_per_step,  # [B, H] for horizon weighting in PoserV1
        }
