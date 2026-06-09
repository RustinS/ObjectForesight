from __future__ import annotations

import tarfile
from typing import Optional

import cv2
import numpy as np


def read_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Read a specific frame from an mp4 video. Returns BGR uint8 (H,W,3) or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_video_frame_size(video_path: str) -> Optional[tuple[int, int]]:
    """Return (H, W) for the given video without decoding a frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (h, w) if w > 0 and h > 0 else None


def read_frame_from_hot3d_tar(tar_path: str, frame_id: str, stream_id: str = "214-1") -> Optional[np.ndarray]:
    """
    Read a specific RGB frame from a HOT3D clip tar file.

    Args:
        tar_path: Path to the HOT3D clip tar file
        frame_id: Frame identifier string (e.g., "0000000123456789" nanosecond timestamp)
        stream_id: Camera stream ID (default "214-1" for Aria RGB)

    Returns:
        BGR uint8 (H,W,3) image or None if not found
    """
    try:
        # HOT3D frame IDs on disk are typically zero-padded (e.g., "000047").
        # Accept "47" / "000047" interchangeably for convenience.
        if isinstance(frame_id, str) and frame_id.isdigit() and len(frame_id) <= 6:
            frame_id = frame_id.zfill(6)

        # Construct expected filename pattern: <frame_id>.image_<stream_id>.jpg
        target_suffix = f".image_{stream_id}.jpg"
        target_prefix = f"{frame_id}."

        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                name = member.name
                # Match pattern: <frame_id>.image_<stream_id>.jpg
                if name.startswith(target_prefix) and name.endswith(target_suffix):
                    img_data = tar.extractfile(member).read()
                    img_arr = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    return img
        return None
    except Exception:
        return None


def read_any_frame_from_hot3d_tar(tar_path: str, stream_id: str = "214-1") -> Optional[np.ndarray]:
    """
    Read any RGB frame from a HOT3D clip tar file (first found).

    Args:
        tar_path: Path to the HOT3D clip tar file
        stream_id: Camera stream ID (default "214-1" for Aria RGB)

    Returns:
        BGR uint8 (H,W,3) image or None if not found
    """
    try:
        target_suffix = f".image_{stream_id}.jpg"

        with tarfile.open(tar_path, "r") as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                if member.name.endswith(target_suffix):
                    img_data = tar.extractfile(member).read()
                    img_arr = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    return img
        return None
    except Exception:
        return None


def get_hot3d_frame_size(tar_path: str, stream_id: str = "214-1") -> Optional[tuple[int, int]]:
    """
    Get (H, W) for a HOT3D clip tar file by reading the first frame.

    Args:
        tar_path: Path to the HOT3D clip tar file
        stream_id: Camera stream ID (default "214-1" for Aria RGB)

    Returns:
        (H, W) tuple or None if not found
    """
    img = read_any_frame_from_hot3d_tar(tar_path, stream_id)
    if img is not None:
        return (img.shape[0], img.shape[1])
    return None
