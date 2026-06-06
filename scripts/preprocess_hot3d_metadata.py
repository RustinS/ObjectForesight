#!/usr/bin/env python3
"""Pre-extract HOT3D clip metadata to pickle files for fast loading.

This script extracts JSON metadata (objects.json, cameras.json) from HOT3D clip tar
files and saves them as pickle files. The HOT3DClipsDataset will load these pickle
files directly instead of opening tar files, providing ~20-50x speedup on cache misses.

Usage:
    python scripts/preprocess_hot3d_metadata.py \
        --clips_dir /path/to/hot3d-clips/train_aria \
        --output_dir /path/to/hot3d-clips/depth_cache

    # With sharding for parallel processing:
    python scripts/preprocess_hot3d_metadata.py \
        --clips_dir /path/to/hot3d-clips/train_aria \
        --output_dir /path/to/hot3d-clips/depth_cache \
        --num_shards 8 --shard_idx 0
"""

import argparse
import json
import os
import pickle
import tarfile
from glob import glob

from tqdm import tqdm


def extract_clip_metadata(clip_path: str, output_dir: str) -> bool:
    """Extract all JSON metadata from a clip tar file and save as pickle.

    Args:
        clip_path: Path to the clip tar file.
        output_dir: Directory to save the pickle file.

    Returns:
        True if extraction was successful, False otherwise.
    """
    clip_id = os.path.basename(clip_path).replace(".tar", "")
    pkl_path = os.path.join(output_dir, f"{clip_id}_meta.pkl")

    # Skip if already exists
    if os.path.exists(pkl_path):
        return True

    clip_data = {"frames": {}, "cameras": None}

    try:
        with tarfile.open(clip_path, "r") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(".objects.json"):
                    frame_id = name.split(".")[0]
                    content = json.load(tar.extractfile(member))
                    clip_data["frames"].setdefault(frame_id, {})["objects"] = content
                elif name.endswith(".cameras.json"):
                    frame_id = name.split(".")[0]
                    content = json.load(tar.extractfile(member))
                    clip_data["frames"].setdefault(frame_id, {})["cameras"] = content
                    if clip_data["cameras"] is None:
                        clip_data["cameras"] = content

        # Save as pickle with highest protocol for fastest loading
        with open(pkl_path, "wb") as f:
            pickle.dump(clip_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    except Exception as e:
        print(f"Error processing {clip_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-extract HOT3D clip metadata to pickle files.")
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing clip tar files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for pickle files")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing")
    parser.add_argument("--shard_idx", type=int, default=0, help="Current shard index (0-indexed)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip clips that already have pickle files")
    args = parser.parse_args()

    # Find all clip tar files
    clip_paths = sorted(glob(os.path.join(args.clips_dir, "*.tar")))
    if not clip_paths:
        print(f"No clip tar files found in {args.clips_dir}")
        return

    print(f"Found {len(clip_paths)} clips total")

    # Shard selection
    num_shards = max(1, args.num_shards)
    shard_idx = args.shard_idx % num_shards
    sharded_paths = [p for i, p in enumerate(clip_paths) if i % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} clips in this shard")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip existing if requested
    if args.skip_existing:
        existing = set(os.listdir(args.output_dir))
        sharded_paths = [p for p in sharded_paths if f"{os.path.basename(p).replace('.tar', '')}_meta.pkl" not in existing]
        print(f"After skipping existing: {len(sharded_paths)} clips to process")

    if not sharded_paths:
        print("Nothing to process")
        return

    # Process clips
    success_count = 0
    for clip_path in tqdm(sharded_paths, desc="Extracting metadata"):
        if extract_clip_metadata(clip_path, args.output_dir):
            success_count += 1

    print(f"\nDone. Successfully processed {success_count}/{len(sharded_paths)} clips.")
    print(f"Pickle files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
