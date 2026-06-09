from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf, open_dict


def apply_config_adapter(cfg: DictConfig) -> DictConfig:
    """Idempotent adapter that normalizes temporal keys and selects backend.

    Selection order for backend kind:
    1) cfg.model.temporal_kind if present
    2) cfg.model.temporal.kind (legacy)
    3) cfg.temporal.kind (legacy)
    Defaults to "dit".
    """
    # Ensure sections exist
    for section, parent in [
        ("model", cfg),
        ("temporal", cfg),
        ("temporal", cfg.model if hasattr(cfg, "model") else None),
        ("temporal_ar", cfg.model if hasattr(cfg, "model") else None),
        ("temporal_dit", cfg.model if hasattr(cfg, "model") else None),
    ]:
        if parent is None:
            continue
        if not hasattr(parent, section) or getattr(parent, section) is None:
            with open_dict(parent):
                setattr(parent, section, OmegaConf.create({}))

    # Resolve selected backend kind
    selected_kind = getattr(cfg.model, "temporal_kind", None) or getattr(cfg.model.temporal, "kind", None) or getattr(cfg.temporal, "kind", None) or "dit"
    selected_kind = str(selected_kind).lower()

    def _pick(key: str, default: Any) -> Any:
        if selected_kind == "ar_transformer" and hasattr(cfg.model.temporal_ar, key):
            return getattr(cfg.model.temporal_ar, key)
        if selected_kind == "dit" and hasattr(cfg.model.temporal_dit, key):
            return getattr(cfg.model.temporal_dit, key)
        if hasattr(cfg.model.temporal, key):
            return getattr(cfg.model.temporal, key)
        if hasattr(cfg.temporal, key):
            return getattr(cfg.temporal, key)
        return default

    latent_dim = int(_pick("latent_dim", 768))
    n_layers = int(_pick("n_layers", 12 if selected_kind == "dit" else 8))
    n_heads = int(_pick("n_heads", 12))
    dropout = float(_pick("dropout", 0.1 if selected_kind == "dit" else 0.0))

    # DiT extras
    _default_H = getattr(getattr(cfg, "data", {}), "H", None) or getattr(getattr(cfg, "model", {}), "H", None)
    picked_out_dim = _pick("out_dim", None)
    if selected_kind == "dit":
        if picked_out_dim is None:
            if _default_H is None:
                raise ValueError("Temporal DiT config requires either temporal.out_dim or data.H/model.H")
            out_dim = int(_default_H) * 9
        else:
            out_dim = int(picked_out_dim)
    else:
        out_dim = None

    T = int(_pick("T", 1000))
    ddim_steps = int(_pick("ddim_steps", 50))

    # DiT conditioning mode: "film" (original) or "adaln_zero"
    conditioning = str(_pick("conditioning", "film")).lower().replace("-", "_")
    if conditioning not in ("film", "adaln_zero"):
        raise ValueError(f"Invalid conditioning mode '{conditioning}'. Must be 'film' or 'adaln_zero'.")
    mlp_ratio = float(_pick("mlp_ratio", 4.0))

    # AR-only extras
    use_context_token = bool(_pick("use_context_token", True))
    max_seq_len = int(_pick("max_seq_len", 512))
    if selected_kind == "ar_transformer":
        data_H = getattr(getattr(cfg, "data", None), "H", None)
        data_context_len = getattr(getattr(cfg, "data", None), "context_len", None)
        if data_H is not None and data_context_len is not None:
            # Dataset context tokens are typically `context_len + 1` (includes anchor frame). We also include the init token.
            required_len = int(data_H) + int(data_context_len) + 2
            max_seq_len = max(max_seq_len, required_len)

    # Write back compact set to top-level
    with open_dict(cfg.temporal):
        cfg.temporal.kind = selected_kind
        cfg.temporal.latent_dim = latent_dim
        cfg.temporal.n_layers = n_layers
        cfg.temporal.n_heads = n_heads
        cfg.temporal.dropout = dropout
        if selected_kind == "dit":
            cfg.temporal.out_dim = out_dim
            cfg.temporal.T = T
            cfg.temporal.ddim_steps = ddim_steps
            cfg.temporal.conditioning = conditioning
            cfg.temporal.mlp_ratio = mlp_ratio
        else:
            cfg.temporal.use_context_token = use_context_token
            cfg.temporal.max_seq_len = max_seq_len

    with open_dict(cfg.model):
        cfg.model.temporal_kind = selected_kind

    # Mirror into model.temporal
    with open_dict(cfg.model.temporal):
        for k in [
            "kind",
            "latent_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "out_dim",
            "T",
            "ddim_steps",
            "conditioning",
            "mlp_ratio",  # DiT conditioning mode params
            "use_context_token",
            "max_seq_len",
        ]:
            if hasattr(cfg.temporal, k):
                cfg.model.temporal[k] = getattr(cfg.temporal, k)

    # Wire skip_film to encoder based on conditioning mode
    # When adaln_zero, encoder should output clean features (no FiLM) since DiT handles all conditioning
    if hasattr(cfg.model, "encoder"):
        with open_dict(cfg.model.encoder):
            cfg.model.encoder.skip_film = conditioning == "adaln_zero"

    return cfg


def compact_view(cfg: DictConfig) -> Dict[str, Any]:
    """Return a small dict summarizing key runtime knobs for concise startup print."""
    return {
        "device": getattr(getattr(cfg, "train", {}), "device", "cpu"),
        "H": getattr(getattr(cfg, "data", {}), "H", None),
        "temporal.kind": getattr(getattr(cfg, "temporal", {}), "kind", None),
        "temporal.latent_dim": getattr(getattr(cfg, "temporal", {}), "latent_dim", None),
        "temporal.n_layers": getattr(getattr(cfg, "temporal", {}), "n_layers", None),
        "temporal.n_heads": getattr(getattr(cfg, "temporal", {}), "n_heads", None),
    }
