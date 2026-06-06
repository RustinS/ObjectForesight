from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset

from ..data.cache import (
    load_hot3d_shared_scenes_split,
    load_hot3d_split,
    load_split,
    save_hot3d_shared_scenes_split,
    save_hot3d_split,
    save_split,
)
from ..data.datasets import HOT3DClipsDataset, SceneSequenceDataset, SceneSequenceDatasetSynth
from ..dist.distrib import barrier as _barrier


def _build_base_kwargs(cfg: DictConfig, force_refresh: bool) -> dict:
    """Build common kwargs for SceneSequenceDataset construction."""
    return dict(
        dataset_root=cfg.data.dataset_root,
        H=cfg.data.H,
        window_stride=int(cfg.data.get("window_stride", 1)),
        frame_skips=int(cfg.data.get("frame_skips", 0)),
        min_frames=int(cfg.data.get("min_frames", cfg.data.H)),
        use_depth=bool(cfg.data.get("use_depth", True)),
        n_points=int(cfg.data.n_points),
        load_rgb0=bool(cfg.data.get("load_rgb0", True)),
        cache_index=bool(cfg.data.get("cache_index", True)),
        downsample_cfg=getattr(cfg.data, "downsample", None),
        context_len=int(cfg.data.context_len),
        filter_iou_drop=bool(getattr(cfg.data, "filter_iou_drop", False)),
        iou_drop_threshold=float(cfg.data.get("iou_drop_threshold", 0.1)),
        force_refresh_cache=force_refresh,
        force_rebuild_windows_cache=bool(getattr(cfg.data, "force_rebuild_windows_cache", False)),
        verbose=bool(getattr(cfg.data, "verbose", False)),
        # Sample cache settings
        use_sample_cache=bool(getattr(cfg.data, "use_sample_cache", False)),
        sample_cache_dir=getattr(cfg.data, "sample_cache_dir", None),
        precompute_cache=bool(getattr(cfg.data, "precompute_cache", False)),
        overwrite_sample_cache=bool(getattr(cfg.data, "overwrite_sample_cache", False)),
        cfg=cfg,
    )


def get_dataset(cfg: DictConfig) -> SceneSequenceDataset | SceneSequenceDatasetSynth | HOT3DClipsDataset:
    """Canonical dataset factory used by train/eval/infer/viz entrypoints."""
    ds_name = str(getattr(cfg.data, "dataset_name", "")).lower()
    is_post_train = bool(getattr(cfg.data, "post_train", False))
    if is_post_train and ds_name != "hot3d":
        ds_name = "post_train"

    # HOT3D dataset
    if ds_name == "hot3d":
        hot3d_cfg = getattr(cfg.data, "hot3d", None)
        if hot3d_cfg is None:
            raise ValueError("cfg.data.hot3d must be set for dataset_name=hot3d")
        # Post-train filtering params (stationary object removal)
        min_t = float(getattr(cfg.data, "post_train_min_t", 0.02))
        min_rot = float(getattr(cfg.data, "post_train_min_rot", 5.0))
        return HOT3DClipsDataset(
            clips_root=hot3d_cfg.clips_root,
            depth_cache_dir=hot3d_cfg.depth_cache_dir,
            H=cfg.data.H,
            context_len=int(cfg.data.context_len),
            n_points=int(cfg.data.n_points),
            window_stride=int(cfg.data.get("window_stride", 1)),
            frame_skips=int(cfg.data.get("frame_skips", 0)),
            split=str(hot3d_cfg.get("split", "train")),
            downsample_cfg=getattr(cfg.data, "downsample", None),
            object_library=str(hot3d_cfg.get("object_library", "")) or None,
            verbose=bool(getattr(cfg.data, "verbose", False)),
            # Sample cache settings
            use_sample_cache=bool(getattr(cfg.data, "use_sample_cache", False)),
            sample_cache_dir=getattr(cfg.data, "sample_cache_dir", None),
            precompute_cache=bool(getattr(cfg.data, "precompute_cache", False)),
            overwrite_sample_cache=bool(getattr(cfg.data, "overwrite_sample_cache", False)),
            cfg=cfg,
            # Post-train filtering
            post_train_mode=is_post_train,
            post_train_min_t=min_t,
            post_train_min_rot=min_rot,
            # Hand pose loading
            load_hand_poses=bool(getattr(cfg.data, "load_hand_poses", False)),
        )

    if ds_name in {"epic", "post_train", "synth"}:
        if not hasattr(cfg.data, "H"):
            raise ValueError("cfg.data.H must be set in the config")
        if not hasattr(cfg.data, "context_len"):
            raise ValueError("cfg.data.context_len must be set in the config")

        if ds_name == "synth":
            return SceneSequenceDatasetSynth(
                H=cfg.data.H,
                n_points=cfg.data.n_points,
                num_samples=cfg.data.get("num_samples", 8),
                image_size=cfg.data.get("image_size", [240, 320]),
                context_len=int(cfg.data.context_len),
            )

        root = cfg.data.get("dataset_root")
        from ..data.cache import remap_data_path
        root_check = remap_data_path(root) if isinstance(root, str) else root
        if not (isinstance(root_check, str) and os.path.isdir(root_check)):
            raise ValueError(f"data.dataset_name={ds_name} requires a valid data.dataset_root path")

        force_refresh = bool(getattr(cfg.data, "force_refresh_cache", False))
        base_kwargs = _build_base_kwargs(cfg, force_refresh)

        if ds_name == "epic":
            return SceneSequenceDataset(rot_smooth_alpha=float(getattr(cfg.data, "rot_smooth_alpha", 0.15)), **base_kwargs)

        # post_train
        iou_thresh = float(getattr(cfg.data, "post_train_IoU_thresh", 0.5))
        min_t = float(getattr(cfg.data, "post_train_min_t", 0.02))
        return SceneSequenceDataset(
            post_train_mode=True,
            post_train_iou_mean_thresh=iou_thresh,
            windows_cache_tag=f"post_train_iou{iou_thresh:.17g}_t{min_t:.17g}",
            rot_smooth_alpha=float(getattr(cfg.data, "rot_smooth_alpha", 0.0)),
            post_train_min_t=min_t,
            **base_kwargs,
        )

    # Legacy path: infer from dataset_root
    if cfg.data.get("dataset_root") and os.path.isdir(cfg.data.dataset_root):
        force_refresh = bool(cfg.data.get("force_refresh_cache", False))
        base = dict(
            dataset_root=cfg.data.dataset_root,
            H=cfg.data.H,
            window_stride=int(cfg.data.get("window_stride", 1)),
            frame_skips=int(cfg.data.get("frame_skips", 0)),
            min_frames=int(cfg.data.get("min_frames", cfg.data.H)),
            use_depth=bool(cfg.data.get("use_depth", True)),
            n_points=int(cfg.data.n_points),
            load_rgb0=bool(cfg.data.get("load_rgb0", True)),
            cache_index=bool(cfg.data.get("cache_index", True)),
            force_refresh_cache=force_refresh,
            force_rebuild_windows_cache=bool(getattr(cfg.data, "force_rebuild_windows_cache", False)),
            downsample_cfg=getattr(cfg.data, "downsample", None),
            verbose=bool(getattr(cfg.data, "verbose", False)),
        )
        if bool(getattr(cfg.data, "post_train", False)):
            iou_thresh = float(getattr(cfg.data, "post_train_IoU_thresh", 0.5))
            min_t = float(getattr(cfg.data, "post_train_min_t", 0.02))
            return SceneSequenceDataset(
                post_train_mode=True,
                post_train_iou_mean_thresh=iou_thresh,
                windows_cache_tag=f"post_train_iou{iou_thresh:.17g}_t{min_t:.17g}",
                rot_smooth_alpha=float(getattr(cfg.data, "rot_smooth_alpha", 0.0)),
                post_train_min_t=min_t,
                **base,
            )
        return SceneSequenceDataset(rot_smooth_alpha=float(getattr(cfg.data, "rot_smooth_alpha", 0.15)), **base)

    # Fallback: synthetic dataset
    return SceneSequenceDatasetSynth(
        H=cfg.data.H,
        n_points=cfg.data.n_points,
        num_samples=cfg.data.get("num_samples", 8),
        image_size=cfg.data.get("image_size", [240, 320]),
        context_len=int(cfg.data.context_len) if hasattr(cfg.data, "context_len") else 1,
    )


def _build_per_video_split(dataset: SceneSequenceDataset, seed: int, train_ratio: float) -> Tuple[list[str], list[str]]:
    """Build train/val split per video."""
    per_video: dict[str, list[str]] = {}
    for rec in dataset.records:
        if int(rec.get("n_frames", 0)) >= int(dataset.min_frames):
            rel = os.path.relpath(str(rec["object_dir"]), start=str(dataset.dataset_root))
            vid = str(rec.get("video_id", ""))
            per_video.setdefault(vid, []).append(rel)

    rng = np.random.default_rng(seed)
    train_ratio = min(1.0, max(0.0, train_ratio))
    train_rel, val_rel = [], []
    for vid in sorted(per_video.keys()):
        uniq = sorted(set(per_video[vid]))
        rng.shuffle(uniq)
        n_train = max(1, min(int(round(train_ratio * len(uniq))), len(uniq)))
        train_rel.extend(uniq[:n_train])
        val_rel.extend(uniq[n_train:])
    return train_rel, val_rel


def _build_hot3d_per_clip_split(
    windows: list[dict],
    seed: int,
    train_ratio: float,
) -> Tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Build train/val split per clip for HOT3D.

    Analogous to _build_per_video_split for EPIC:
    - Groups by clip_id (like video_id)
    - Within each clip, object_uids are shuffled and split

    IMPORTANT: Split is computed from (clip_id, object_uid) pairs that
    actually appear in windows, NOT from clip_metadata.object_uids.
    This ensures we only split "eligible" pairs that have windows.

    The split is stable regardless of post_train settings because:
    - post_train filtering happens BEFORE this function is called
    - We split whatever windows exist, so post_train just reduces the set
    """
    # Extract unique (clip_id, object_uid) pairs that actually have windows
    per_clip: dict[str, set[str]] = {}
    for w in windows:
        clip_id = w["clip_id"]
        obj_uid = w["object_uid"]
        per_clip.setdefault(clip_id, set()).add(obj_uid)

    rng = np.random.default_rng(seed)
    train_ratio = min(1.0, max(0.0, train_ratio))
    train_pairs: set[tuple[str, str]] = set()
    val_pairs: set[tuple[str, str]] = set()

    for clip_id in sorted(per_clip.keys()):
        object_uids = sorted(per_clip[clip_id])
        if not object_uids:
            continue

        rng.shuffle(object_uids)
        n_train = max(1, min(int(round(train_ratio * len(object_uids))), len(object_uids)))

        for uid in object_uids[:n_train]:
            train_pairs.add((clip_id, uid))
        for uid in object_uids[n_train:]:
            val_pairs.add((clip_id, uid))

    return train_pairs, val_pairs


def _build_hot3d_shared_scenes_split(
    windows: list[dict],
    seed: int,
    train_ratio: float,
) -> Tuple[
    set[tuple[str, str]],  # train_pairs: (clip_id, object_uid)
    set[tuple[str, str]],  # val_pairs: (clip_id, object_uid)
    set[tuple[str, str, int]],  # train_window_keys: (clip_id, object_uid, t0)
    set[tuple[str, str, int]],  # val_window_keys: (clip_id, object_uid, t0)
]:
    """Build train/val split ensuring all objects appear in both sets.

    Strategy:
    For each (clip_id, object_uid) pair, split its windows by train_ratio.
    This ensures every object's movement sequence is temporally split,
    with train_ratio of windows going to train and the rest to val.

    Objects with <2 windows are skipped (cannot split into both sets).

    Returns:
        train_pairs: Set of (clip_id, object_uid) pairs for training
        val_pairs: Set of (clip_id, object_uid) pairs for validation
        train_window_keys: Set of (clip_id, object_uid, t0) window keys for training
        val_window_keys: Set of (clip_id, object_uid, t0) window keys for validation
    """
    from ..utils.logger import rprint

    # Group windows by (clip_id, object_uid) -> list of (window_idx, t0)
    per_object: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for idx, w in enumerate(windows):
        key = (w["clip_id"], w["object_uid"])
        per_object.setdefault(key, []).append((idx, w["t0"]))

    rng = np.random.default_rng(seed)
    train_ratio = min(1.0, max(0.0, train_ratio))

    train_pairs: set[tuple[str, str]] = set()
    val_pairs: set[tuple[str, str]] = set()
    train_window_keys: set[tuple[str, str, int]] = set()
    val_window_keys: set[tuple[str, str, int]] = set()

    skipped_objects = 0

    for clip_id, obj_uid in sorted(per_object.keys()):
        window_list = per_object[(clip_id, obj_uid)].copy()

        if len(window_list) < 2:
            # Cannot split 1 window into both train and val - skip object
            skipped_objects += 1
            continue

        # Shuffle windows and split
        rng.shuffle(window_list)
        n_wins = len(window_list)
        # Ensure at least 1 window in each set: n_train_wins in [1, n_wins-1]
        n_train_wins = max(1, min(n_wins - 1, int(round(train_ratio * n_wins))))

        # Add to both train and val pairs (same object appears in both)
        train_pairs.add((clip_id, obj_uid))
        val_pairs.add((clip_id, obj_uid))

        for _, t0 in window_list[:n_train_wins]:
            train_window_keys.add((clip_id, obj_uid, t0))
        for _, t0 in window_list[n_train_wins:]:
            val_window_keys.add((clip_id, obj_uid, t0))

    if skipped_objects > 0:
        rprint(f"[yellow]shared_scenes_split[/yellow] skipped {skipped_objects} objects with <2 windows (cannot split)")

    return train_pairs, val_pairs, train_window_keys, val_window_keys


def build_train_val_datasets(
    dataset: SceneSequenceDataset | SceneSequenceDatasetSynth | HOT3DClipsDataset,
    cfg: DictConfig,
    seed: int,
    rank: int,
) -> Tuple[Dataset, Dataset]:
    """Build train/val datasets (with persistent split when possible)."""
    # --- Handle HOT3DClipsDataset ---
    if isinstance(dataset, HOT3DClipsDataset):
        # Skip internal splitting for official test set (no GT available anyway)
        if dataset.split != "train":
            return dataset, dataset

        redo_split = (not bool(getattr(cfg.data, "cache_index", True))) or bool(getattr(cfg.data, "force_refresh_cache", False))

        # Use base seed (rank-invariant) for deterministic split across all ranks
        base_seed = int(cfg.train.seed)
        train_ratio = float(getattr(cfg.data, "train_ratio_per_video", 0.70))

        # Check if shared_scenes_split mode is enabled
        shared_scenes_split = bool(getattr(cfg.data, "shared_scenes_split", False))

        # IMPORTANT: Barrier before checking split to avoid GPFS metadata caching issues.
        # Without this, some ranks may see stale filesystem state and take different code paths,
        # causing a deadlock (some ranks wait at the inner barrier, others skip ahead).
        _barrier()

        if shared_scenes_split:
            # ---- Shared-scenes split: all clips appear in both train and val ----
            # Get windows cache key to ensure split matches current windows (post_train params, etc.)
            windows_key = dataset._get_cache_key() if hasattr(dataset, "_get_cache_key") else ""

            split_tuple = load_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, windows_key)

            if redo_split or split_tuple is None:
                if rank == 0:
                    train_pairs, val_pairs, train_win_keys, val_win_keys = _build_hot3d_shared_scenes_split(dataset.windows, base_seed, train_ratio)
                    save_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, train_pairs, val_pairs, train_win_keys, val_win_keys, windows_key)
                _barrier()
                split_tuple = load_hot3d_shared_scenes_split(dataset.clips_root, dataset.split, windows_key)

            train_pairs, val_pairs, train_win_keys, val_win_keys = split_tuple

            if rank == 0:
                from ..utils.logger import rprint

                # All objects appear in both train and val (with different windows)
                rprint(f"[dim]shared-scenes split[/dim] objects={len(train_pairs):,} (all shared) train_windows={len(train_win_keys):,} val_windows={len(val_win_keys):,}")

            # Build indices with window-level granularity using (clip_id, object_uid, t0) keys
            train_indices = []
            val_indices = []
            for i, w in enumerate(dataset.windows):
                key = (w["clip_id"], w["object_uid"], w["t0"])
                if key in train_win_keys:
                    train_indices.append(i)
                elif key in val_win_keys:
                    val_indices.append(i)
                # else: window not covered (skipped objects with <2 windows)

            if not train_indices:
                raise ValueError(f"No train windows. Check {dataset.clips_root}/split/")
            if not val_indices:
                raise ValueError(f"No val windows. Check {dataset.clips_root}/split/")

            return Subset(dataset, train_indices), Subset(dataset, val_indices)

        else:
            # ---- Original per-clip split: objects within each clip are split ----
            split_tuple = load_hot3d_split(dataset.clips_root, dataset.split)

            if redo_split or split_tuple is None:
                if rank == 0:
                    train_pairs, val_pairs = _build_hot3d_per_clip_split(dataset.windows, base_seed, train_ratio)
                    save_hot3d_split(dataset.clips_root, dataset.split, train_pairs, val_pairs)
                _barrier()
                split_tuple = load_hot3d_split(dataset.clips_root, dataset.split)

            train_pairs, val_pairs = split_tuple

            if rank == 0:
                from ..utils.logger import rprint

                rprint(f"[dim]split[/dim] train_pairs={len(train_pairs):,} val_pairs={len(val_pairs):,}")

            # Coverage check: every window's (clip_id, object_uid) must be in train OR val
            all_pairs = train_pairs | val_pairs
            missing = []
            for w in dataset.windows:
                pair = (w["clip_id"], w["object_uid"])
                if pair not in all_pairs:
                    missing.append(pair)
            if missing:
                raise ValueError(
                    f"Split coverage error: {len(missing)} (clip_id, object_uid) pairs in "
                    f"windows not in split. This may happen if clips/objects were added after "
                    f"split creation. Delete {dataset.clips_root}/split/hot3d_{dataset.split}_*.txt "
                    f"to regenerate."
                )

            # Build indices
            train_indices = [i for i, w in enumerate(dataset.windows) if (w["clip_id"], w["object_uid"]) in train_pairs]
            val_indices = [i for i, w in enumerate(dataset.windows) if (w["clip_id"], w["object_uid"]) in val_pairs]

            if not train_indices:
                raise ValueError(f"No train windows. Check {dataset.clips_root}/split/")
            if not val_indices:
                raise ValueError(f"No val windows. Check {dataset.clips_root}/split/")

            if rank == 0:
                rprint(f"[dim]split[/dim] train_windows={len(train_indices):,} val_windows={len(val_indices):,}")

            return Subset(dataset, train_indices), Subset(dataset, val_indices)

    # --- Handle synthetic/other non-SceneSequenceDataset ---
    if not isinstance(dataset, SceneSequenceDataset):
        return dataset, dataset

    # --- Handle SceneSequenceDataset (EPIC) ---
    redo_split = (not bool(getattr(cfg.data, "cache_index", True))) or bool(getattr(cfg.data, "force_refresh_cache", False))

    # IMPORTANT: Barrier before checking split to avoid GPFS metadata caching issues.
    _barrier()

    split_tuple = load_split(dataset.dataset_root)

    if redo_split or split_tuple is None:
        if rank == 0:
            train_ratio = float(getattr(cfg.data, "train_ratio_per_video", 0.70))
            train_rel, val_rel = _build_per_video_split(dataset, seed, train_ratio)
            save_split(dataset.dataset_root, train_rel, val_rel)
        _barrier()
        split_tuple = load_split(dataset.dataset_root)

    if split_tuple is None:
        train_ratio = float(getattr(cfg.data, "train_ratio_per_video", 0.70))
        train_rel, val_rel = _build_per_video_split(dataset, seed, train_ratio)
    else:
        train_rel, val_rel = split_tuple

    train_set, val_set = set(train_rel), set(val_rel)

    def _rel(p: str) -> str:
        return os.path.relpath(str(p), start=str(dataset.dataset_root))

    train_indices = [i for i, w in enumerate(dataset.windows) if _rel(w.get("object_dir", "")) in train_set]
    val_indices = [i for i, w in enumerate(dataset.windows) if _rel(w.get("object_dir", "")) in val_set]

    if not train_indices and not val_indices:
        train_indices = list(range(len(dataset)))
    if not val_indices:
        val_indices = list(range(len(dataset)))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
