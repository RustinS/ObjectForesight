#!/usr/bin/env python3
"""Run SpaTrackerV2 on HOT3D-Clips to extract depth maps with fisheye undistortion.

This script extracts RGB frames from HOT3D clip tar files, undistorts them from
fisheye to pinhole camera model, runs SpaTrackerV2 to estimate depth, and saves
the depth maps along with pinhole intrinsics as NPZ files.

Supports multi-GPU processing via torchrun for parallel preprocessing.

Usage (single GPU):
    python scripts/preprocess_hot3d_spatracker.py \
        --clips_root /path/to/hot3d-clips/train_aria \
        --output_dir /path/to/hot3d-clips/depth_cache_pinhole

Usage (8 GPUs with torchrun):
    torchrun --standalone --nproc_per_node=8 scripts/preprocess_hot3d_spatracker.py \
        --clips_root /path/to/hot3d-clips/train_aria \
        --output_dir /path/to/hot3d-clips/depth_cache_pinhole \
        --skip_existing

Usage (manual sharding, legacy):
    python scripts/preprocess_hot3d_spatracker.py \
        --clips_root /path/to/hot3d-clips/train_aria \
        --output_dir /path/to/hot3d-clips/depth_cache_pinhole \
        --num_shards 8 --shard_idx 0
"""

import argparse
import glob
import json
import os
import random
import sys
import tarfile
import traceback

import cv2
import numpy as np
import torch
import torch.distributed as dist

# hand_tracking_toolkit for fisheye undistortion
from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image

# Add SpaTrackerV2 to path (use parent 3dmanip directory)
sys.path.insert(0, "./SpaTrackerV2")


def setup_distributed():
    """Initialize distributed processing if running under torchrun.

    Returns:
        Tuple of (rank, world_size, local_rank, device).
        If not distributed, returns (0, 1, 0, cuda:0 or cpu).
    """
    # Check if running under torchrun (sets LOCAL_RANK env var)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank >= 0:
        # Running under torchrun - initialize distributed
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device
    else:
        # Single process mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device


def cleanup_distributed():
    """Clean up distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Run SpaTrackerV2 on HOT3D-Clips to extract depth with fisheye undistortion.")
    p.add_argument("--clips_root", type=str, required=True, help="Path to HOT3D clips directory (e.g., train_aria)")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for depth NPZ files (e.g., depth_cache_pinhole)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards for distributed processing")
    p.add_argument("--shard_idx", type=int, default=0, help="Current shard index (0-indexed)")
    p.add_argument("--skip_existing", action="store_true", help="Skip clips that already have depth files")
    p.add_argument("--max_frames", type=int, default=0, help="Max frames per clip (0 = all frames)")
    p.add_argument("--focal_scale", type=float, default=1.0, help="Focal length scale for pinhole conversion (controls FOV)")
    return p.parse_args()


def convert_to_pinhole(fisheye_cam: camera.CameraModel, focal_scale: float = 1.0) -> camera.CameraModel:
    """Convert fisheye camera to pinhole equivalent.

    Args:
        fisheye_cam: Input fisheye camera model from HOT3D.
        focal_scale: Scale factor for focal length (controls field of view in pinhole).

    Returns:
        Pinhole camera model with same resolution and principal point.
    """
    return camera.PinholePlaneCameraModel(
        width=fisheye_cam.width,
        height=fisheye_cam.height,
        f=[fisheye_cam.f[0] * focal_scale, fisheye_cam.f[1] * focal_scale],
        c=fisheye_cam.c,
        distort_coeffs=[],
        T_world_from_eye=fisheye_cam.T_world_from_eye,
    )


def extract_rgb_frames_from_tar(clip_path: str, max_frames: int = 0, focal_scale: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract and undistort RGB frames from HOT3D clip tar file.

    Args:
        clip_path: Path to the clip tar file.
        max_frames: Maximum number of frames to extract (0 = all).
        focal_scale: Scale factor for pinhole focal length.

    Returns:
        Tuple of (frames, K_pinhole) where:
        - frames: List of undistorted RGB images as numpy arrays
        - K_pinhole: 3x3 pinhole intrinsics matrix
    """
    frames = []
    frame_names = []
    fisheye_cam = None
    pinhole_cam = None

    with tarfile.open(clip_path, "r") as tar:
        # First pass: collect frame names and load camera calibration
        for member in tar.getmembers():
            # Aria RGB stream is 214-1
            if member.name.endswith(".image_214-1.jpg"):
                frame_id = member.name.split(".")[0]
                frame_names.append((frame_id, member))
            # Load camera calibration from first cameras.json found
            elif member.name.endswith(".cameras.json") and fisheye_cam is None:
                cameras_json = json.load(tar.extractfile(member))
                cam_data = cameras_json.get("214-1", {})
                if cam_data:
                    fisheye_cam = camera.from_json(cam_data)
                    pinhole_cam = convert_to_pinhole(fisheye_cam, focal_scale=focal_scale)

    # Sort by frame ID
    frame_names.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    if max_frames > 0:
        frame_names = frame_names[:max_frames]

    # Second pass: extract and undistort frames
    with tarfile.open(clip_path, "r") as tar:
        for _frame_id, member in frame_names:
            img_data = tar.extractfile(member).read()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Undistort fisheye -> pinhole
                if fisheye_cam is not None and pinhole_cam is not None:
                    img = warp_image(src_camera=fisheye_cam, dst_camera=pinhole_cam, src_image=img)
                frames.append(img)

    # Build pinhole K matrix (3x3)
    if pinhole_cam is not None:
        K_pinhole = np.array(
            [[pinhole_cam.f[0], 0, pinhole_cam.c[0]], [0, pinhole_cam.f[1], pinhole_cam.c[1]], [0, 0, 1]],
            dtype=np.float32,
        )
    else:
        # Fallback: assume centered pinhole with reasonable focal length
        h, w = frames[0].shape[:2] if frames else (1408, 1408)
        K_pinhole = np.array(
            [[w * 0.7, 0, w / 2], [0, w * 0.7, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )

    return frames, K_pinhole


@torch.no_grad()
def process_clip(clip_path: str, output_dir: str, models, device: torch.device, max_frames: int = 0, focal_scale: float = 1.0):
    """Process a single HOT3D clip to extract depth maps with fisheye undistortion.

    Args:
        clip_path: Path to the clip tar file.
        output_dir: Directory to save depth NPZ files.
        models: Tuple of (vggt4track_model, spatracker_model).
        device: Torch device to use for inference.
        max_frames: Maximum frames to process (0 = all).
        focal_scale: Scale factor for pinhole focal length.
    """
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    vggt4track_model, model = models
    clip_id = os.path.basename(clip_path).replace(".tar", "")

    # Extract and undistort RGB frames, get pinhole intrinsics
    frames, K_pinhole = extract_rgb_frames_from_tar(clip_path, max_frames, focal_scale=focal_scale)
    if len(frames) == 0:
        print(f"[yellow]No frames[/yellow] in {clip_id}")
        return

    # Get original undistorted image dimensions
    orig_h, orig_w = frames[0].shape[:2]

    print(f"Processing {clip_id}: {len(frames)} frames ({orig_w}x{orig_h})")

    # Stack frames into video tensor
    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video_tensor = preprocess_image(video_tensor, keep_ratio=True)[None]  # [1, T, C, H, W]

    # Run VGGT4Track for initial predictions
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = vggt4track_model(video_tensor.to(device) / 255.0)

    depth_tensor = preds["points_map"][..., 2].squeeze().detach().cpu().numpy()
    extrs = preds["poses_pred"].squeeze().detach().cpu().numpy()
    intrs = preds["intrs"].squeeze().detach().cpu().numpy()
    unc_metric = preds["unc_metric"].squeeze().detach().cpu().numpy() > 0.5
    video_tensor = video_tensor.squeeze()  # [T, C, H, W]

    # Set up grid points for tracking
    grid_pts = get_points_on_a_grid(10, video_tensor.shape[2:], device="cpu")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    # Run full SpaTrackerV2 model for refined depth
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        c2w_traj, intrs_out, point_map, conf_depth, track3d_pred, _, vis_pred, *_ = model.forward(
            video_tensor,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=1,
            full_point=False,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=unc_metric,
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.2,
        )

    # Extract refined depth with confidence masking
    depth_save = point_map[:, 2, ...].clone()
    depth_save[conf_depth < 0.5] = 0

    # Save depth NPZ with pinhole intrinsics and original dimensions
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{clip_id}_depth.npz")
    np.savez_compressed(
        out_path,
        depths=depth_save.cpu().numpy().astype(np.float32),
        K_pinhole=K_pinhole,  # Pinhole intrinsics (3x3)
        orig_h=np.array(orig_h, dtype=np.int32),  # Original undistorted height
        orig_w=np.array(orig_w, dtype=np.int32),  # Original undistorted width
    )
    print(f"Saved: {out_path} (depths={depth_save.shape}, K_pinhole={K_pinhole.shape})")

    torch.cuda.empty_cache()


def main():
    args = parse_args()

    # Setup distributed processing (auto-detects torchrun)
    rank, world_size, local_rank, device = setup_distributed()
    is_main = is_main_process(rank)

    # Set seed with rank offset for reproducibility
    set_seed(args.seed + rank)

    if is_main:
        print(f"Distributed: world_size={world_size}, using device={device}")

    # Find all clip tar files
    clip_paths = sorted(glob.glob(os.path.join(args.clips_root, "*.tar")))
    if not clip_paths:
        if is_main:
            print(f"No clip tar files found in {args.clips_root}")
        cleanup_distributed()
        return

    if is_main:
        print(f"Found {len(clip_paths)} clips total")

    # Shard selection: use distributed rank/world_size if available, else use manual args
    if world_size > 1:
        # Distributed mode: auto-shard by rank
        num_shards = world_size
        shard_idx = rank
    else:
        # Single process mode: use manual sharding args
        num_shards = max(1, args.num_shards)
        shard_idx = args.shard_idx % num_shards

    sharded_paths = [p for i, p in enumerate(clip_paths) if i % num_shards == shard_idx]
    print(f"[Rank {rank}] Shard {shard_idx}/{num_shards}: {len(sharded_paths)} clips")

    # Skip existing if requested
    if args.skip_existing:
        if is_main:
            os.makedirs(args.output_dir, exist_ok=True)
        # Sync to ensure directory exists before all ranks check
        if world_size > 1:
            dist.barrier()
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        existing = set(os.listdir(args.output_dir))
        sharded_paths = [p for p in sharded_paths if f"{os.path.basename(p).replace('.tar', '')}_depth.npz" not in existing]
        print(f"[Rank {rank}] After skipping existing: {len(sharded_paths)} clips to process")

    if not sharded_paths:
        print(f"[Rank {rank}] Nothing to process")
        cleanup_distributed()
        return

    # Load SpaTrackerV2 models on this device
    print(f"[Rank {rank}] Loading SpaTrackerV2 models on {device}...")
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(device)
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.spatrack.track_num = 100
    model.eval().to(device)
    models = (vggt4track_model, model)
    print(f"[Rank {rank}] Models loaded")

    # Process clips
    for i, clip_path in enumerate(sharded_paths, 1):
        try:
            print(f"\n[Rank {rank}] [{i}/{len(sharded_paths)}] {os.path.basename(clip_path)}")
            process_clip(clip_path, args.output_dir, models, device, args.max_frames, focal_scale=args.focal_scale)
        except Exception as e:
            print(f"[Rank {rank}] Error processing {clip_path}: {e}\n{traceback.format_exc()}")

    print(f"\n[Rank {rank}] Done.")

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
