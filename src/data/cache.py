from __future__ import annotations

import hashlib
import json
import os
import pickle


def remap_data_path(path: str) -> str:
    """Translate snapshot-source paths to local mounts via DATA_PATH_REMAP env var.

    Format: DATA_PATH_REMAP="from1=to1;from2=to2". Lets a portable snapshot keep
    its original cache hashes (which encode source absolute paths) while file IO
    is redirected to wherever the data was rehydrated locally. No-op if env var
    is unset or no prefix matches.
    """
    spec = os.environ.get("DATA_PATH_REMAP", "")
    if not spec or not isinstance(path, str) or not path:
        return path
    for rule in spec.split(";"):
        if "=" not in rule:
            continue
        src, dst = rule.split("=", 1)
        src = src.rstrip("/")
        dst = dst.rstrip("/")
        if not src:
            continue
        if path == src:
            return dst
        if path.startswith(src + "/"):
            return dst + path[len(src):]
    return path


def _read_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_index(dataset_root: str, index: list[dict]) -> None:
    with open(os.path.join(dataset_root, ".index.json"), "w") as f:
        json.dump(index, f)


def read_index(dataset_root: str) -> list[dict] | None:
    return _read_json(os.path.join(dataset_root, ".index.json"))


def spot_check_index(index: list[dict], n_samples: int = 5) -> bool:
    if not index:
        return False
    # Trust the cached index without verifying raw-data paths. Used when running
    # from a portable snapshot whose index records reference files only present
    # on the source HPC. Sample cache + windows pkl short-circuit __getitem__.
    if os.environ.get("SAMPLE_CACHE_TRUST_INDEX", "0") == "1":
        return True
    step = max(1, len(index) // max(1, n_samples))
    for rec in [index[i] for i in range(0, len(index), step)][:n_samples]:
        for key in ("spatrack_npz", "rgb_path", "mesh_path"):
            if not os.path.exists(rec.get(key, "")):
                return False
        pose_txts = rec.get("pose_txts", [])
        _npz = os.path.join(rec.get("object_dir", ""), "foundationpose10", "poses.npz")
        if not os.path.exists(_npz) and (not pose_txts or not os.path.exists(pose_txts[0])):
            return False
    return True


def read_conv_map(dataset_root: str) -> dict[str, str] | None:
    path = os.path.join(dataset_root, "split", "extrinsics_conv.json")
    data = _read_json(path)
    if isinstance(data, dict):
        return {str(k): ("c2w" if str(v) == "c2w" else "w2c") for k, v in data.items()}
    return None


def write_conv_map(dataset_root: str, mapping: dict[str, str]) -> None:
    path = os.path.join(dataset_root, "split", "extrinsics_conv.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def read_spatrack_len_map(dataset_root: str) -> dict:
    obj = _read_json(os.path.join(dataset_root, ".spatrack_len.json"))
    return obj if isinstance(obj, dict) else {}


def write_spatrack_len_map(dataset_root: str, mapping: dict) -> None:
    with open(os.path.join(dataset_root, ".spatrack_len.json"), "w") as f:
        json.dump(mapping, f)


def load_split(dataset_root: str) -> tuple[list[str], list[str]] | None:
    split_dir = os.path.join(dataset_root, "split")
    tfile, vfile = os.path.join(split_dir, "train.txt"), os.path.join(split_dir, "val.txt")
    if not (os.path.exists(tfile) and os.path.exists(vfile)):
        return None
    with open(tfile) as f:
        train_rel = [ln.strip() for ln in f if ln.strip()]
    with open(vfile) as f:
        val_rel = [ln.strip() for ln in f if ln.strip()]
    return train_rel, val_rel


def save_split(dataset_root: str, train_rel: list[str], val_rel: list[str]) -> None:
    split_dir = os.path.join(dataset_root, "split")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_rel) + "\n")
    with open(os.path.join(split_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_rel) + "\n")


def load_hot3d_split(clips_root: str, official_split: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]] | None:
    """Load HOT3D train/val split as sets of (clip_id, object_uid) tuples.

    Args:
        clips_root: Path to HOT3D clips directory
        official_split: The official HOT3D split ("train" or "test")

    Returns:
        Tuple of (train_pairs, val_pairs) sets, or None if split files don't exist.
        Each pair is (clip_id, object_uid).
    """
    split_dir = os.path.join(clips_root, "split")
    # Namespace by official split to avoid cross-split contamination
    tfile = os.path.join(split_dir, f"hot3d_{official_split}_train.txt")
    vfile = os.path.join(split_dir, f"hot3d_{official_split}_val.txt")
    if not (os.path.exists(tfile) and os.path.exists(vfile)):
        return None

    def parse_file(path: str) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln and "," in ln:
                    clip_id, obj_uid = ln.split(",", 1)
                    pairs.add((clip_id.strip(), obj_uid.strip()))
        return pairs

    train_pairs, val_pairs = parse_file(tfile), parse_file(vfile)

    # Validate no leakage
    overlap = train_pairs & val_pairs
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} pairs in both train and val")

    return train_pairs, val_pairs


def save_hot3d_split(
    clips_root: str,
    official_split: str,
    train_pairs: set[tuple[str, str]],
    val_pairs: set[tuple[str, str]],
) -> None:
    """Save HOT3D train/val split as (clip_id, object_uid) pairs.

    Args:
        clips_root: Path to HOT3D clips directory
        official_split: The official HOT3D split ("train" or "test")
        train_pairs: Set of (clip_id, object_uid) tuples for training
        val_pairs: Set of (clip_id, object_uid) tuples for validation
    """
    split_dir = os.path.join(clips_root, "split")
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, f"hot3d_{official_split}_train.txt"), "w") as f:
        for clip_id, obj_uid in sorted(train_pairs):
            f.write(f"{clip_id},{obj_uid}\n")

    with open(os.path.join(split_dir, f"hot3d_{official_split}_val.txt"), "w") as f:
        for clip_id, obj_uid in sorted(val_pairs):
            f.write(f"{clip_id},{obj_uid}\n")


def load_hot3d_shared_scenes_split(
    clips_root: str,
    official_split: str,
    windows_key: str = "",
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str, int]], set[tuple[str, str, int]]] | None:
    """Load HOT3D shared-scenes split with window-level granularity.

    Args:
        clips_root: Path to HOT3D clips directory
        official_split: The official HOT3D split ("train" or "test")
        windows_key: Hash of windows cache parameters (ensures split matches current windows)

    Returns:
        Tuple of (train_pairs, val_pairs, train_window_keys, val_window_keys) or None if files don't exist.
        - train_pairs/val_pairs: Set of (clip_id, object_uid) tuples
        - train_window_keys/val_window_keys: Set of (clip_id, object_uid, t0) tuples for window-level split
    """
    split_dir = os.path.join(clips_root, "split")
    # Include windows_key in filenames to invalidate when windows change (post_train params, etc.)
    key_suffix = f"_{windows_key}" if windows_key else ""
    tfile = os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_train.txt")
    vfile = os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_val.txt")
    t_win_file = os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_train_windows.json")
    v_win_file = os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_val_windows.json")

    if not all(os.path.exists(f) for f in [tfile, vfile, t_win_file, v_win_file]):
        return None

    def parse_pairs_file(path: str) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln and "," in ln:
                    clip_id, obj_uid = ln.split(",", 1)
                    pairs.add((clip_id.strip(), obj_uid.strip()))
        return pairs

    def deserialize_window_keys(path: str) -> set[tuple[str, str, int]]:
        """Load window keys from nested dict JSON: {clip_id: {object_uid: [t0_list]}}"""
        with open(path) as f:
            data = json.load(f)
        result: set[tuple[str, str, int]] = set()
        for clip_id, obj_dict in data.items():
            for obj_uid, t0_list in obj_dict.items():
                for t0 in t0_list:
                    result.add((clip_id, obj_uid, int(t0)))
        return result

    train_pairs = parse_pairs_file(tfile)
    val_pairs = parse_pairs_file(vfile)
    train_window_keys = deserialize_window_keys(t_win_file)
    val_window_keys = deserialize_window_keys(v_win_file)

    # Validate no window-level overlap (same t0 should not appear in both train and val for same object)
    overlap = train_window_keys & val_window_keys
    if overlap:
        raise ValueError(f"Window-level leakage: {len(overlap)} windows in both train and val")

    return train_pairs, val_pairs, train_window_keys, val_window_keys


def save_hot3d_shared_scenes_split(
    clips_root: str,
    official_split: str,
    train_pairs: set[tuple[str, str]],
    val_pairs: set[tuple[str, str]],
    train_window_keys: set[tuple[str, str, int]],
    val_window_keys: set[tuple[str, str, int]],
    windows_key: str = "",
) -> None:
    """Save HOT3D shared-scenes split with window-level granularity.

    Args:
        clips_root: Path to HOT3D clips directory
        official_split: The official HOT3D split ("train" or "test")
        train_pairs: Set of (clip_id, object_uid) tuples for training
        val_pairs: Set of (clip_id, object_uid) tuples for validation
        train_window_keys: Set of (clip_id, object_uid, t0) tuples for train windows
        val_window_keys: Set of (clip_id, object_uid, t0) tuples for val windows
        windows_key: Hash of windows cache parameters (ensures split matches current windows)
    """
    split_dir = os.path.join(clips_root, "split")
    os.makedirs(split_dir, exist_ok=True)

    # Include windows_key in filenames to invalidate when windows change (post_train params, etc.)
    key_suffix = f"_{windows_key}" if windows_key else ""

    # Save pairs (same format as original)
    with open(os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_train.txt"), "w") as f:
        for clip_id, obj_uid in sorted(train_pairs):
            f.write(f"{clip_id},{obj_uid}\n")

    with open(os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_val.txt"), "w") as f:
        for clip_id, obj_uid in sorted(val_pairs):
            f.write(f"{clip_id},{obj_uid}\n")

    def serialize_window_keys(keys: set[tuple[str, str, int]]) -> dict[str, dict[str, list[int]]]:
        """Convert window keys to nested dict: {clip_id: {object_uid: [t0_list]}}"""
        result: dict[str, dict[str, list[int]]] = {}
        for clip_id, obj_uid, t0 in keys:
            result.setdefault(clip_id, {}).setdefault(obj_uid, []).append(t0)
        # Sort t0 lists for determinism
        for clip_id in result:
            for obj_uid in result[clip_id]:
                result[clip_id][obj_uid].sort()
        return result

    # Save window keys as nested JSON
    with open(os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_train_windows.json"), "w") as f:
        json.dump(serialize_window_keys(train_window_keys), f, sort_keys=True)

    with open(os.path.join(split_dir, f"hot3d_{official_split}_shared{key_suffix}_val_windows.json"), "w") as f:
        json.dump(serialize_window_keys(val_window_keys), f, sort_keys=True)


def compute_records_fingerprint(records: list[dict]) -> str:
    h = hashlib.sha1()
    vid_to_spa: dict[str, str] = {}
    for r in records:
        vid_to_spa.setdefault(str(r.get("video_id", "")), str(r.get("spatrack_npz", "")))
    for vid in sorted(vid_to_spa):
        try:
            st = os.stat(vid_to_spa[vid])
            h.update(f"V{vid}{st.st_mtime_ns}{st.st_size}".encode())
        except FileNotFoundError:
            # Portable-snapshot mode: raw spatrack files not on this host. Use the
            # path string so the fingerprint is still deterministic. The resulting
            # hash won't match what the source HPC produced, so EPIC also relies
            # on the windows-cache fallback in _maybe_load_windows_cache.
            h.update(f"V{vid}{vid_to_spa[vid]}".encode())
    h.update(f"R{len(records)}".encode())
    total_frames = 0
    for r in records[:: max(1, len(records) // 1024)]:
        pose_txts = r.get("pose_txts") or []
        _fids = r.get("frame_ids") or []
        total_frames += int(r.get("n_frames", 0))
        h.update(
            f"O{r.get('object_id', '')}{os.path.basename(str(r.get('object_dir', '')))}"
            f"{r.get('n_frames', 0)}{os.path.basename(pose_txts[0]) if pose_txts else (_fids[0] if _fids else '')}"
            f"{os.path.basename(pose_txts[-1]) if pose_txts else (_fids[-1] if _fids else '')}".encode()
        )
    h.update(str(total_frames).encode())
    return h.hexdigest()


def make_windows_cache_key(
    dataset_root: str,
    H: int,
    window_stride: int,
    context_len: int,
    respect_init_from_frame: bool,
    filter_iou_drop: bool,
    iou_drop_threshold: float,
    min_frames: int,
    records_fingerprint: str,
    anchor_mode: str = "window_start",
    anchor_frame_idx: int = 0,
    output_format: str = "abs_in_anchor",
    anchor_policy: str = "horizon_start",
    mode_tag: str = "",
    frame_skips: int = 0,
) -> str:
    params = {
        "root": os.path.abspath(dataset_root),
        "H": H,
        "S": window_stride,
        "P": context_len,
        "respect_init": respect_init_from_frame,
        "filter_iou_drop": filter_iou_drop,
        "iou_thr": iou_drop_threshold,
        "min_frames": min_frames,
        "rec_fp": records_fingerprint,
        "anchor_mode": anchor_mode,
        "anchor_frame_idx": anchor_frame_idx,
        "output_format": output_format,
        "anchor_policy": anchor_policy,
        "mode_tag": mode_tag or "",
    }
    if frame_skips > 0:
        params["frame_skips"] = frame_skips
    return hashlib.sha1(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_windows_cache(dataset_root: str, key: str) -> list[dict] | None:
    path = os.path.join(dataset_root, f".windows.{key}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, list) else None


def write_windows_cache(dataset_root: str, key: str, windows: list[dict]) -> None:
    with open(os.path.join(dataset_root, f".windows.{key}.pkl"), "wb") as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
