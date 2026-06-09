from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from ..dist.distrib import is_rank0
from ..utils.logger import rprint as print
from ..utils.progress import tqdm_auto as tqdm
from .cache import read_index, spot_check_index, write_index

# ============================================================================
# Path construction helpers (formerly paths.py)
# ============================================================================

ACTION_MP4 = "action.mp4"
SPATRACK_NPZ = "spatracker.npz"
OBJECTS_DIRNAME = "objects"
TRELLIS_MESH_REL = os.path.join("trellis", "model.glb")
FPOSE_DIR_REL = os.path.join("foundationpose10", "ob_in_cam")


def list_video_dirs(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        return []
    return sorted(os.path.join(dataset_root, n) for n in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, n)))


def spatrack_npz_path(video_dir: str) -> str:
    return os.path.join(video_dir, SPATRACK_NPZ)


def action_mp4_path(video_dir: str) -> str:
    return os.path.join(video_dir, ACTION_MP4)


def objects_root(video_dir: str) -> str:
    return os.path.join(video_dir, OBJECTS_DIRNAME)


def list_object_dirs(video_dir: str) -> List[str]:
    obj_root = objects_root(video_dir)
    if not os.path.isdir(obj_root):
        return []
    return sorted(os.path.join(obj_root, n) for n in os.listdir(obj_root) if os.path.isdir(os.path.join(obj_root, n)))


def parse_object_id_name(obj_dir: str) -> Tuple[str, str]:
    """Parse object directory name into (object_id, object_name)."""
    base = os.path.basename(obj_dir)
    for sep in ["+", "_", "-"]:
        if sep in base:
            left, right = base.split(sep, 1)
            obj_id = left.strip().lstrip("0") or left.strip()
            return obj_id, right.strip()
    # Digits prefix is id
    digits = ""
    for ch in base:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        return digits.lstrip("0") or digits, base[len(digits) :].strip("-_+") or base
    return base, base


def mesh_glb_path(obj_dir: str) -> str:
    return os.path.join(obj_dir, TRELLIS_MESH_REL)


def fpose_dir(obj_dir: str) -> str:
    return os.path.join(obj_dir, FPOSE_DIR_REL)


def _pose_txt_key(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(stem)
    except ValueError:
        return 10**9


def list_pose_txts(obj_dir: str) -> List[str]:
    dirp = fpose_dir(obj_dir)
    if not os.path.isdir(dirp):
        return []
    txts = [os.path.join(dirp, f) for f in os.listdir(dirp) if f.lower().endswith(".txt")]
    return sorted(txts, key=_pose_txt_key)


# ============================================================================
# Dataset scanning
# ============================================================================


def scan_dataset(
    dataset_root: str,
    H: int,
    cache_index: bool = True,
    min_frames: int | None = None,
    force_refresh_cache: bool = False,
) -> List[Dict[str, Any]]:
    """Scan filesystem for usable trajectories. Optionally read/write JSON index cache."""
    if cache_index and not force_refresh_cache:
        cached = read_index(dataset_root)
        if cached is not None and spot_check_index(cached):
            videos = len({rec["video_id"] for rec in cached})
            usable = sum(1 for rec in cached if rec.get("n_frames", 0) >= (min_frames or H))
            print(f"[bold]Scan Summary[/bold] • root={dataset_root} • videos={videos} • objects={len(cached)} • usable_trajectories={usable}")
            return cached

    video_dirs = list_video_dirs(dataset_root)
    records: List[Dict[str, Any]] = []
    missing_pose, missing_spa, missing_mesh, total_frames = 0, 0, 0, 0

    for vdir in tqdm(video_dirs, desc="[cyan]scan[/cyan] videos", unit="vid", disable=not is_rank0()):
        vid = os.path.basename(vdir)
        spa = spatrack_npz_path(vdir)
        if not os.path.exists(spa):
            missing_spa += 1
            continue

        rgb = action_mp4_path(vdir)
        for odir in list_object_dirs(vdir):
            oid, oname = parse_object_id_name(odir)
            mesh = mesh_glb_path(odir)
            pose_txts = list_pose_txts(odir)
            n = len(pose_txts)
            total_frames += n

            if n == 0:
                missing_pose += 1
                continue
            if not os.path.exists(mesh):
                missing_mesh += 1

            records.append(
                {
                    "video_id": vid,
                    "video_dir": vdir,
                    "spatrack_npz": spa,
                    "rgb_path": rgb,
                    "object_id": oid,
                    "object_name": oname,
                    "object_dir": odir,
                    "mesh_path": mesh,
                    "pose_txts": pose_txts,
                    "n_frames": n,
                }
            )

    usable = sum(1 for r in records if r["n_frames"] >= (min_frames or H))
    print(
        f"[bold]Scan Summary[/bold] • root={dataset_root}\n"
        f"videos={len(video_dirs)} • objects={len(records)} • total_frames={total_frames:,} • usable_trajectories={usable}"
        f"\nmissing: {missing_pose} objects (no poses), {missing_spa} spatrack, {missing_mesh} mesh"
    )

    if cache_index:
        # Don't clobber a non-empty cached index with an empty rescan (happens
        # when raw clips are absent on a portable-snapshot host but the cached
        # index is intentionally trusted via SAMPLE_CACHE_TRUST_INDEX).
        if records or read_index(dataset_root) is None:
            write_index(dataset_root, records)
    return records
