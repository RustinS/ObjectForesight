#!/usr/bin/env python3
"""Precompute sample cache for fast training.

This script precomputes and caches processed dataset samples using LZ4 compression.
Point clouds are stored as bfloat16 to reduce storage while maintaining precision.
Supports multi-GPU parallel processing via torchrun.

Usage (single process):
    uv run python scripts/precompute_sample_cache.py

Usage (8 GPUs with torchrun - RECOMMENDED):
    uv run torchrun --standalone --nproc_per_node=8 scripts/precompute_sample_cache.py

Usage (incremental build - skip existing):
    uv run torchrun --standalone --nproc_per_node=8 scripts/precompute_sample_cache.py --skip_existing

Usage (manual sharding, legacy):
    for i in {0..7}; do
        uv run python scripts/precompute_sample_cache.py --num_shards 8 --shard_idx $i &
    done
    wait
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from src.data.datasets import HOT3DClipsDataset, SceneSequenceDataset
from src.data.sample_cache import get_sample_path, load_sample, make_sample_cache_key, sample_exists
from torch.utils.data import Dataset
from tqdm import tqdm


_GCS_CLIENT = None
_GCS_BUCKET = None
_GCS_PREFIX = ""


def _gcs_init(bucket_uri: str) -> None:
    """Initialize a single GCS client + bucket handle for the process."""
    global _GCS_CLIENT, _GCS_BUCKET, _GCS_PREFIX
    if _GCS_CLIENT is not None:
        return
    from google.cloud import storage  # lazy import
    if not bucket_uri.startswith("gs://"):
        raise ValueError(f"--gcs_bucket must start with gs://; got {bucket_uri}")
    parts = bucket_uri[len("gs://"):].split("/", 1)
    bucket_name = parts[0]
    _GCS_PREFIX = (parts[1].rstrip("/") + "/") if len(parts) > 1 and parts[1] else ""
    # Pass a dummy project so user-creds ADC (no default project) still works.
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "no-project")
    _GCS_CLIENT = storage.Client(project=project)
    _GCS_BUCKET = _GCS_CLIENT.bucket(bucket_name)


def _gcs_fetch_video(bucket_uri: str, vid: str, dest_root: Path, inner_workers: int = 16) -> tuple[bool, str]:
    """Download a single video dir from GCS into dest_root/{vid}/ via the
    Python SDK with parallel blob downloads. Idempotent — skips if
    spatracker.npz is already present."""
    dst = dest_root / vid
    if (dst / "spatracker.npz").exists():
        return True, "already present"
    _gcs_init(bucket_uri)
    dst.mkdir(parents=True, exist_ok=True)
    prefix = f"{_GCS_PREFIX}{vid}/"
    try:
        blobs = [b for b in _GCS_BUCKET.list_blobs(prefix=prefix) if not b.name.endswith("/")]
    except Exception as e:
        return False, f"list_blobs: {e}"[:200]
    if not blobs:
        return False, f"no blobs under {prefix}"

    def _download_one(blob) -> tuple[bool, str]:
        rel = blob.name[len(prefix):]
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob.download_to_filename(str(out))
            return True, ""
        except Exception as e:
            return False, f"{rel}: {e}"

    n_ok = 0
    with ThreadPoolExecutor(max_workers=inner_workers) as inner:
        for ok, err in inner.map(_download_one, blobs):
            if ok:
                n_ok += 1
            else:
                return False, f"download {err}"[:200]
    return True, f"ok ({n_ok} files)"


def _cleanup_video(dest_root: Path, vid: str) -> None:
    dst = dest_root / vid
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)


def process_with_gcs_streaming(
    dataset: Dataset,
    my_indices: list[int],
    cache_dir: Path,
    bucket: str,
    dest_root: Path,
    skip_existing: bool,
    keep_local: bool,
    prefetch: int,
    rank: int,
    is_main: bool,
) -> tuple[int, int, int]:
    """Group window indices by video, JIT-fetch each video from GCS, process,
    optionally delete. Returns (processed, skipped, errors)."""
    by_vid: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for idx in my_indices:
        vid, obj, k0 = get_window_info(dataset, idx)
        by_vid[vid].append((idx, obj, k0))

    todo: list[str] = []
    pre_skipped = 0
    for vid, items in by_vid.items():
        if skip_existing and all(
            sample_exists(get_sample_path(cache_dir, vid, obj, k0))
            for _, obj, k0 in items
        ):
            pre_skipped += len(items)
            continue
        todo.append(vid)

    if is_main:
        print(f"GCS streaming • bucket={bucket}")
        print(f"  videos: total={len(by_vid):,} todo={len(todo):,} fully_cached_videos_skipped={len(by_vid) - len(todo):,}")
        print(f"  windows pre-skipped (fully cached): {pre_skipped:,}")
        print(f"  prefetch parallelism: {prefetch}")
        print(f"  keep_local: {keep_local}")
        print()

    executor = ThreadPoolExecutor(max_workers=max(1, prefetch))
    futures: dict[str, "Future[tuple[bool, str]]"] = {}

    def submit(vid_to_fetch: str) -> None:
        if vid_to_fetch not in futures:
            futures[vid_to_fetch] = executor.submit(_gcs_fetch_video, bucket, vid_to_fetch, dest_root)

    for vid in todo[:prefetch]:
        submit(vid)

    processed = 0
    skipped = pre_skipped
    errors = 0
    fetch_errors = 0

    pbar = tqdm(todo, desc=f"Rank {rank} videos", disable=not is_main)
    for i, vid in enumerate(pbar):
        nxt = i + prefetch
        if nxt < len(todo):
            submit(todo[nxt])

        ok, msg = futures.pop(vid).result()
        if not ok:
            n_skipped_for_vid = len(by_vid[vid])
            fetch_errors += 1
            errors += n_skipped_for_vid
            if is_main and fetch_errors <= 10:
                print(f"[fetch-error] {vid}: {msg}")
            continue

        for idx, obj, k0 in by_vid[vid]:
            if skip_existing:
                cache_path = get_sample_path(cache_dir, vid, obj, k0)
                if sample_exists(cache_path):
                    skipped += 1
                    continue
            try:
                _ = dataset[idx]
                processed += 1
            except Exception as e:
                errors += 1
                if is_main and errors <= 10:
                    print(f"[sample-error] {vid}/{obj}/{k0}: {e}")

        if not keep_local:
            _cleanup_video(dest_root, vid)

        if is_main:
            pbar.set_postfix(processed=processed, skipped=skipped, errors=errors, fetch_err=fetch_errors)

    executor.shutdown(wait=False)
    return processed, skipped, errors


def setup_distributed():
    """Initialize distributed processing if running under torchrun.

    Returns:
        Tuple of (rank, world_size).
        If not distributed, returns (0, 1).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank >= 0:
        # Running under torchrun - initialize distributed with gloo (CPU-only)
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size
    else:
        return 0, 1


def cleanup_distributed():
    """Clean up distributed processing."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def load_config(config_path: str) -> DictConfig:
    """Load Hydra config (composes `defaults:` if config_path lives under conf/)."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    p = Path(config_path)
    # If the file is inside a directory named "conf" we can use Hydra compose
    # to honor the `defaults:` list (so configs inherit debug.yaml).
    parents = list(p.parents)
    conf_root = next((parent for parent in parents if parent.name == "conf"), None)
    if conf_root is not None:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=str(conf_root.resolve())):
            cfg = compose(config_name=p.stem)
        return cfg
    return OmegaConf.load(config_path)


def build_dataset(cfg: DictConfig, precompute_cache: bool = True) -> Dataset:
    """Build dataset configured for cache precomputation."""
    ds_name = str(getattr(cfg.data, "dataset_name", "")).lower()

    # Get downsample_cfg
    downsample_cfg = getattr(cfg.data, "downsample", None)
    if isinstance(downsample_cfg, DictConfig):
        downsample_cfg = OmegaConf.to_container(downsample_cfg, resolve=True)

    # HOT3D dataset
    if ds_name == "hot3d":
        hot3d_cfg = getattr(cfg.data, "hot3d", None)
        if hot3d_cfg is None:
            raise ValueError("cfg.data.hot3d must be set for dataset_name=hot3d")
        # Post-train filtering params (stationary object removal)
        is_post_train = bool(getattr(cfg.data, "post_train", False))
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
            downsample_cfg=downsample_cfg,
            object_library=str(hot3d_cfg.get("object_library", "")) or None,
            verbose=False,
            # Sample cache settings
            use_sample_cache=False,  # Don't read from cache
            precompute_cache=precompute_cache,  # Write to cache
            overwrite_sample_cache=bool(getattr(cfg.data, "overwrite_sample_cache", False)),
            cfg=cfg,
            # Post-train filtering
            post_train_mode=is_post_train,
            post_train_min_t=min_t,
            post_train_min_rot=min_rot,
        )

    # EPIC/post_train dataset
    if ds_name not in {"epic", "post_train", ""}:
        raise ValueError(f"Sample cache only supports epic/post_train/hot3d datasets, got: {ds_name}")

    # Build base kwargs matching data_utils._build_base_kwargs
    kwargs = dict(
        dataset_root=cfg.data.dataset_root,
        H=cfg.data.H,
        window_stride=int(cfg.data.get("window_stride", 1)),
        frame_skips=int(cfg.data.get("frame_skips", 0)),
        min_frames=int(cfg.data.get("min_frames", cfg.data.H)),
        use_depth=bool(cfg.data.get("use_depth", True)),
        n_points=int(cfg.data.n_points),
        load_rgb0=bool(cfg.data.get("load_rgb0", True)),
        cache_index=bool(cfg.data.get("cache_index", True)),
        downsample_cfg=downsample_cfg,
        context_len=int(cfg.data.context_len),
        filter_iou_drop=bool(getattr(cfg.data, "filter_iou_drop", False)),
        iou_drop_threshold=float(cfg.data.get("iou_drop_threshold", 0.1)),
        force_refresh_cache=False,
        force_rebuild_windows_cache=False,
        verbose=False,
        rot_smooth_alpha=float(getattr(cfg.data, "rot_smooth_alpha", 0.15)),
        # Sample cache settings
        use_sample_cache=False,  # Don't read from cache
        precompute_cache=precompute_cache,  # Write to cache
        overwrite_sample_cache=bool(getattr(cfg.data, "overwrite_sample_cache", False)),
        cfg=cfg,
    )

    return SceneSequenceDataset(**kwargs)


def get_window_info(dataset: Dataset, idx: int) -> tuple:
    """Extract video_id, object_id, k0 from window, handling both dataset types.

    Returns:
        Tuple of (video_id, object_id, k0)
    """
    win = dataset.windows[idx]
    k0_raw = win["frame_ids"][0]
    k0 = int(k0_raw) if isinstance(k0_raw, str) and k0_raw.isdigit() else (k0_raw if isinstance(k0_raw, int) else 0)

    # HOT3D uses clip_id/object_uid, EPIC uses video_id/object_id
    if "clip_id" in win:
        return win["clip_id"], win["object_uid"], k0
    else:
        return win["video_id"], win["object_id"], k0


def get_cache_root(cfg: DictConfig) -> Path:
    """Get cache root directory based on dataset type. Applies DATA_PATH_REMAP
    so the script's skip-existing checks hit the same physical location the
    dataset writes to."""
    from src.data.cache import remap_data_path

    ds_name = str(getattr(cfg.data, "dataset_name", "")).lower()
    if ds_name == "hot3d":
        hot3d_cfg = getattr(cfg.data, "hot3d", None)
        if hot3d_cfg is not None:
            return Path(remap_data_path(str(hot3d_cfg.clips_root)))
    return Path(remap_data_path(str(cfg.data.dataset_root)))


def main():
    parser = argparse.ArgumentParser(description="Precompute sample cache for fast training")
    parser.add_argument("--config", type=str, default="conf/debug.yaml", help="Path to config file")
    parser.add_argument("--num_shards", type=int, default=None, help="Override world size (manual sharding)")
    parser.add_argument("--shard_idx", type=int, default=None, help="Override rank (manual sharding)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples that already have cache files")
    parser.add_argument("--dry_run", action="store_true", help="Count samples without processing")
    parser.add_argument("--verify", action="store_true", help="Verify cached samples can be loaded and have required keys")
    parser.add_argument("--gcs_bucket", type=str, default=None, help="Stream raw video data from this GCS bucket (e.g., gs://your-bucket/manip_data) instead of expecting it on local disk. Per video: rsync→process→delete.")
    parser.add_argument("--gcs_prefetch", type=int, default=4, help="How many videos to prefetch in parallel via ThreadPool (default 4)")
    parser.add_argument("--gcs_keep_local", action="store_true", help="Keep raw video dirs on disk after processing (default: delete to save space)")
    args = parser.parse_args()

    # Setup distributed
    dist_rank, dist_world = setup_distributed()

    # Two layers of sharding (composable):
    # - Outer (--num_shards/--shard_idx): take a static slice of the full
    #   dataset. Used for smoke tests or when running multiple independent
    #   beaker jobs each handling a subset.
    # - Inner (torchrun): within that slice, distribute work across the
    #   torchrun-spawned workers via setup_distributed().
    # When neither outer nor distributed is set, processes the full dataset
    # in a single rank.
    if args.num_shards is not None or args.shard_idx is not None:
        outer_world = args.num_shards if args.num_shards is not None else 1
        outer_rank = args.shard_idx if args.shard_idx is not None else 0
    else:
        outer_world, outer_rank = 1, 0

    # Each worker's effective identity in the global flattened layout
    rank = outer_rank * dist_world + dist_rank
    world_size = outer_world * dist_world

    is_main = rank == 0

    if is_main:
        print("Precompute Sample Cache")
        print(f"  Config: {args.config}")
        print(f"  Rank: {rank}/{world_size}")
        print(f"  Skip existing: {args.skip_existing}")
        print()

    # Load config
    cfg = load_config(args.config)

    # Build dataset
    dataset = build_dataset(cfg, precompute_cache=not args.dry_run)
    total_samples = len(dataset)

    # Get cache directory
    cache_key = make_sample_cache_key(cfg)
    cache_root = get_cache_root(cfg)
    cache_dir = cache_root / ".sample_cache" / cache_key

    if is_main:
        ds_name = str(getattr(cfg.data, "dataset_name", "")).lower()
        print(f"Dataset: {ds_name} • {total_samples:,} samples")
        print(f"Cache dir: {cache_dir}")
        print()

    # Shard indices across workers. When streaming from GCS we shard by video
    # rather than by stride so each worker pays the per-video fetch cost only
    # once, then processes all that video's windows back-to-back.
    if args.gcs_bucket:
        from collections import defaultdict
        by_vid_all: dict[str, list[int]] = defaultdict(list)
        for i in range(total_samples):
            vid, _obj, _k0 = get_window_info(dataset, i)
            by_vid_all[vid].append(i)
        sorted_vids = sorted(by_vid_all.keys())
        my_vids = sorted_vids[rank::world_size]
        my_indices: list[int] = []
        for v in my_vids:
            my_indices.extend(by_vid_all[v])
    else:
        all_indices = list(range(total_samples))
        my_indices = all_indices[rank::world_size]

    if is_main:
        print(f"Processing {len(my_indices):,} samples on rank {rank}")
        print()

    if args.dry_run:
        # Just count
        existing = 0
        missing = 0
        for idx in tqdm(my_indices, desc=f"Counting (rank {rank})", disable=not is_main):
            video_id, object_id, k0 = get_window_info(dataset, idx)
            cache_path = get_sample_path(cache_dir, video_id, object_id, k0)
            if sample_exists(cache_path):
                existing += 1
            else:
                missing += 1

        if is_main:
            print(f"Existing: {existing:,}, Missing: {missing:,}")

        cleanup_distributed()
        return

    if args.verify:
        # Verify cached samples can be loaded and have required keys
        valid = 0
        invalid = 0
        missing = 0
        invalid_paths = []

        pbar = tqdm(my_indices, desc=f"Verifying (rank {rank})", disable=not is_main)
        for idx in pbar:
            video_id, object_id, k0 = get_window_info(dataset, idx)
            cache_path = get_sample_path(cache_dir, video_id, object_id, k0)

            if not sample_exists(cache_path):
                missing += 1
                continue

            sample = load_sample(cache_path, validate=True)
            if sample is None:
                invalid += 1
                if len(invalid_paths) < 10:
                    invalid_paths.append(str(cache_path))
            else:
                # Additional validation: check shapes
                pcd = sample.get("scene_pcd")
                target = sample.get("target_future")
                if pcd is None or len(pcd.shape) != 2 or pcd.shape[1] != 3:
                    invalid += 1
                    if len(invalid_paths) < 10:
                        invalid_paths.append(f"{cache_path} (bad pcd shape)")
                elif target is None or len(target.shape) != 2:
                    invalid += 1
                    if len(invalid_paths) < 10:
                        invalid_paths.append(f"{cache_path} (bad target shape)")
                else:
                    valid += 1

            if is_main and (valid + invalid + missing) % 100 == 0:
                pbar.set_postfix(valid=valid, invalid=invalid, missing=missing)

        if is_main:
            print()
            print("Verification Results:")
            print(f"  Valid: {valid:,}")
            print(f"  Invalid: {invalid:,}")
            print(f"  Missing: {missing:,}")
            if invalid_paths:
                print()
                print("Sample invalid files:")
                for p in invalid_paths:
                    print(f"  - {p}")

        cleanup_distributed()
        return

    # Process samples
    if args.gcs_bucket:
        ds_root_attr = getattr(dataset, "dataset_root", None) or getattr(dataset, "clips_root", None)
        if ds_root_attr is None:
            raise ValueError("Could not determine dataset root for GCS streaming")
        processed, skipped, errors = process_with_gcs_streaming(
            dataset=dataset,
            my_indices=my_indices,
            cache_dir=cache_dir,
            bucket=args.gcs_bucket,
            dest_root=Path(ds_root_attr),
            skip_existing=args.skip_existing,
            keep_local=args.gcs_keep_local,
            prefetch=args.gcs_prefetch,
            rank=rank,
            is_main=is_main,
        )
    else:
        processed = 0
        skipped = 0
        errors = 0

        pbar = tqdm(my_indices, desc=f"Rank {rank}", disable=not is_main)
        for idx in pbar:
            try:
                # Check if should skip
                if args.skip_existing:
                    video_id, object_id, k0 = get_window_info(dataset, idx)
                    cache_path = get_sample_path(cache_dir, video_id, object_id, k0)
                    if sample_exists(cache_path):
                        skipped += 1
                        continue

                # Fetch sample - this triggers cache write via precompute_cache=True
                _ = dataset[idx]
                processed += 1

                # Update progress
                if is_main and processed % 100 == 0:
                    pbar.set_postfix(processed=processed, skipped=skipped, errors=errors)

            except Exception as e:
                errors += 1
                if is_main and errors <= 10:
                    print(f"Error on sample {idx}: {e}")

    if is_main:
        print()
        print("Done!")
        print(f"  Processed: {processed:,}")
        print(f"  Skipped: {skipped:,}")
        print(f"  Errors: {errors:,}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
