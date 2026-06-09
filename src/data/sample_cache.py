"""Sample-level cache utilities using LZ4-compressed torch tensors.

This module provides utilities for caching processed dataset samples to disk
using LZ4 compression for fast read/write performance. Point clouds are stored
as bfloat16 to reduce storage while maintaining sufficient precision.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import lz4.frame
import torch

CACHE_VERSION = 3  # Bumped: now includes post_train params in cache key


def make_sample_cache_key(cfg) -> str:
    """Generate SHA1 hash from relevant config parameters.

    The cache key ensures that cached samples are invalidated when
    configuration changes that affect sample computation.

    Args:
        cfg: Hydra config object with data section

    Returns:
        12-character hex string cache key
    """
    key_parts = [
        str(cfg.data.H),
        str(cfg.data.context_len),
        str(cfg.data.frame_skips),
        str(cfg.data.n_points),
        str(dict(cfg.data.get("downsample", {}))),
        str(cfg.data.get("anchor_mode", "last_context")),
        str(cfg.data.get("output_format", "abs_in_anchor_cam")),
        # Post-train filtering params (affects which windows are included)
        str(bool(cfg.data.get("post_train", False))),
        str(float(cfg.data.get("post_train_min_t", 0.0))),
        str(float(cfg.data.get("post_train_min_rot", 0.0))),
        str(CACHE_VERSION),
    ]
    # Note: load_hand_poses is NOT included in cache key because hand poses can be
    # loaded at runtime from the tar files without needing to be cached.
    return hashlib.sha1("|".join(key_parts).encode()).hexdigest()[:12]


def get_sample_path(cache_dir: Path, video_id: str, object_id: str, k0: int) -> Path:
    """Generate cache file path for a sample.

    Args:
        cache_dir: Base cache directory
        video_id: Video identifier
        object_id: Object identifier
        k0: Start frame index

    Returns:
        Path to the cached sample file
    """
    # Sanitize IDs to avoid path issues
    safe_video_id = video_id.replace("/", "_").replace("\\", "_")
    safe_object_id = object_id.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_video_id}_{safe_object_id}_{k0}.pt.lz4"


def save_sample(path: Path, sample: Dict[str, Any]) -> None:
    """Save sample dict as LZ4-compressed torch file.

    Point clouds (scene_pcd) are automatically converted to bfloat16
    for storage efficiency.

    Args:
        path: Output file path
        sample: Sample dictionary to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy for storage, converting point cloud to bfloat16
    cache_sample = {}
    for k, v in sample.items():
        if k == "scene_pcd" and v is not None:
            if isinstance(v, torch.Tensor):
                cache_sample[k] = v.to(torch.bfloat16)
            else:
                cache_sample[k] = torch.as_tensor(v, dtype=torch.bfloat16)
        elif isinstance(v, torch.Tensor):
            cache_sample[k] = v
        else:
            cache_sample[k] = v

    # Serialize to bytes
    buffer = io.BytesIO()
    torch.save(cache_sample, buffer)
    buffer.seek(0)

    # Compress with LZ4
    compressed = lz4.frame.compress(buffer.getvalue())

    # Write atomically using temp file with unique suffix to avoid race conditions
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(compressed)
        tmp_path.rename(path)
    except OSError:
        # Race condition: another process may have written the file already
        # This is benign - just clean up and continue
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


REQUIRED_SAMPLE_KEYS = {
    "video_id",
    "object_id",
    "scene_pcd",
    "init_pose",
    "target_future",
    "T_cam_anchor_obj",
    # Required for model conditioning
    "context_init_9d",
    "context_bbox_norm",
}


def load_sample(path: Path, validate: bool = True) -> Optional[Dict[str, Any]]:
    """Load sample from LZ4-compressed torch file.

    Point clouds are automatically converted back to float32 for model use.

    Args:
        path: Path to cached sample file
        validate: If True, verify sample has required keys

    Returns:
        Sample dictionary, or None if file doesn't exist or is invalid
    """
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            compressed = f.read()

        decompressed = lz4.frame.decompress(compressed)
        sample = torch.load(io.BytesIO(decompressed), weights_only=False)

        # Validate sample has required keys
        if validate:
            missing = REQUIRED_SAMPLE_KEYS - set(sample.keys())
            if missing:
                return None  # Treat as corrupted, will trigger recompute

        # Convert bfloat16 point cloud back to float32 numpy array
        # (dataset returns numpy arrays, so we need to match that)
        if sample.get("scene_pcd") is not None:
            pcd = sample["scene_pcd"]
            if isinstance(pcd, torch.Tensor):
                sample["scene_pcd"] = pcd.float().numpy()

        # Convert other tensor fields back to numpy if needed
        for key in ["target_future", "T_cam_anchor_obj", "T_c_w", "T_c_o", "context_init_9d", "context_bbox_norm"]:
            if key in sample and isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].numpy()

        return sample

    except Exception:
        # Corrupted cache file - return None to trigger recompute
        return None


def sample_exists(path: Path) -> bool:
    """Check if cached sample exists.

    Args:
        path: Path to cached sample file

    Returns:
        True if file exists
    """
    return path.exists()
