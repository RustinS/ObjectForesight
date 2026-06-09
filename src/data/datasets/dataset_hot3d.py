from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as _ScipyRotation
from torch.utils.data import Dataset

from ...dist.distrib import is_rank0
from ...utils.logger import rprint as print
from ...utils.progress import tqdm_auto as tqdm
from ..pointcloud import pad_points_by_interpolation, subsample_points, voxel_downsample
from ..sample_cache import get_sample_path, load_sample, make_sample_cache_key, sample_exists, save_sample
from .helpers import _poses_to_9d
from .lru_cache import _LRU


class HOT3DClipsDataset(Dataset):
    """HOT3D-Clips dataset for 6-DoF object pose prediction (Aria RGB only).

    Uses HOT3D ground-truth poses (motion capture) + SpatialTracker v2 depth maps.
    Expects pre-computed depth NPZ files from SpaTrackerV2 preprocessing.
    """

    STREAM_ID = "214-1"  # Aria RGB stream
    WINDOWS_CACHE_VERSION = 3  # Bumped for post_train filtering support
    output_format = "abs_in_anchor"

    def __init__(
        self,
        clips_root: str,
        depth_cache_dir: str,
        H: int,
        context_len: int = 1,
        n_points: int = 20000,
        window_stride: int = 1,
        frame_skips: int = 0,
        split: str = "train",
        downsample_cfg: Optional[dict] = None,
        object_library: Optional[str] = None,
        verbose: bool = False,
        use_sample_cache: bool = False,
        sample_cache_dir: Optional[str] = None,
        precompute_cache: bool = False,
        overwrite_sample_cache: bool = False,
        cfg: Optional[DictConfig] = None,
        # Post-train filtering (stationary object removal)
        post_train_mode: bool = False,
        post_train_min_t: float = 0.02,
        post_train_min_rot: float = 5.0,
        # Hand pose loading (MANO parameters)
        load_hand_poses: bool = False,
    ) -> None:
        # Preserve the original (config-supplied) paths for cache-key hashing —
        # which is computed from os.path.abspath(clips_root) etc. — while letting
        # IO go to wherever the data actually lives via DATA_PATH_REMAP.
        from ..cache import remap_data_path
        self._clips_root_orig = clips_root
        self._depth_cache_dir_orig = depth_cache_dir
        self.clips_root = remap_data_path(clips_root)
        self.depth_cache_dir = remap_data_path(depth_cache_dir)
        if isinstance(object_library, str):
            object_library = remap_data_path(object_library)
        self.H = H
        self.context_len = context_len
        self.n_points = n_points
        self.window_stride = window_stride
        self.frame_skips = max(0, frame_skips)
        self.split = split
        self.object_library = object_library
        self.verbose = verbose
        self.use_sample_cache = use_sample_cache
        self.precompute_cache = precompute_cache
        self.overwrite_sample_cache = overwrite_sample_cache
        self._cfg = cfg
        self._sample_cache_dir: Optional[str] = None
        # Post-train filtering params
        self.post_train_mode = post_train_mode
        self.post_train_min_t = post_train_min_t
        self.post_train_min_rot = post_train_min_rot
        # Hand pose loading
        self.load_hand_poses = load_hand_poses

        self.downsample_cfg = OmegaConf.to_container(downsample_cfg, resolve=True) if isinstance(downsample_cfg, DictConfig) else downsample_cfg

        hot3d_cfg = getattr(getattr(cfg, "data", None), "hot3d", None)
        cap_clip = int(getattr(hot3d_cfg, "clip_cache_cap", 0) or os.environ.get("DATASET_CAP_CLIP", 2000))
        cap_depth = int(getattr(hot3d_cfg, "depth_cache_cap", 0) or os.environ.get("DATASET_CAP_DEPTH", 500))
        self._clip_cache = _LRU(cap=cap_clip)
        self._depth_array_cache = _LRU(cap=cap_depth)
        self._pixel_grids: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._warned_missing_depth_meta = False

        # Optional mesh mapping (object_uid -> BOP id) for setting mesh_path like EPIC.
        self._uid_to_bop: Dict[str, int] = {}
        if isinstance(self.object_library, str) and self.object_library:
            info_path = os.path.join(self.object_library, "models_info.json")
            if os.path.isfile(info_path):
                try:
                    with open(info_path, "r") as f:
                        models_info = json.load(f)
                    for bop_id_str, info in (models_info or {}).items():
                        if not isinstance(info, dict) or "original_id" not in info:
                            continue
                        try:
                            self._uid_to_bop[str(info["original_id"])] = int(bop_id_str)
                        except Exception:
                            continue
                except Exception:
                    # Non-fatal: mesh_path will remain unset.
                    self._uid_to_bop = {}

        # Scan clips and build windows (with caching)
        self.clip_paths = self._scan_clips()
        self.clip_metadata, self.windows = self._load_or_build_windows_cached()

        print(
            f"[bold]HOT3D[/bold] • clips={len(self.clip_paths)} • objects={sum(len(m['object_uids']) for m in self.clip_metadata.values())} "
            f"• windows={len(self.windows):,} • H={self.H} • P={self.context_len} • stride={self.window_stride} • fs={self.frame_skips} • split={self.split}"
        )
        if self.post_train_mode:
            print(f"[bold yellow]post_train[/bold yellow] HOT3D enabled • min_t={self.post_train_min_t:.3f}m • min_rot={self.post_train_min_rot:.1f}deg")
        if self.load_hand_poses:
            print("[bold cyan]load_hand_poses[/bold cyan] HOT3D enabled • loading MANO parameters from hands.json")

        if self.use_sample_cache or self.precompute_cache:
            if cfg is not None:
                cache_key = make_sample_cache_key(cfg)
            else:
                key_parts = [str(self.H), str(self.context_len), str(self.frame_skips), str(self.n_points), str(self.downsample_cfg or {})]
                cache_key = hashlib.sha1("|".join(key_parts).encode()).hexdigest()[:12]
            self._sample_cache_dir = sample_cache_dir or os.path.join(self.clips_root, ".sample_cache", cache_key)
            if is_rank0():
                print(f"[dim]sample cache[/dim] {self._sample_cache_dir}")

    def _scan_clips(self) -> List[str]:
        clips_dir = os.path.join(self.clips_root, f"{self.split}_aria")
        if not os.path.isdir(clips_dir):
            # Portable-snapshot mode: raw clip tars aren't on this host. The
            # cached windows pkl + sample cache short-circuit __getitem__.
            if os.environ.get("SAMPLE_CACHE_TRUST_INDEX", "0") == "1":
                return []
            raise ValueError(f"HOT3D clips directory not found: {clips_dir}")
        clip_paths = sorted(glob.glob(os.path.join(clips_dir, "*.tar")))
        if not clip_paths:
            if os.environ.get("SAMPLE_CACHE_TRUST_INDEX", "0") == "1":
                return []
            raise ValueError(f"No clip tar files found in {clips_dir}")
        return clip_paths

    def _get_cache_key(self) -> str:
        params = {
            "clips_root": os.path.abspath(self._clips_root_orig),
            "depth_cache_dir": os.path.abspath(self._depth_cache_dir_orig),
            "split": self.split,
            "H": self.H,
            "context_len": self.context_len,
            "window_stride": self.window_stride,
            "frame_skips": self.frame_skips,
            "n_clips": len(self.clip_paths),
            "windows_cache_version": int(self.WINDOWS_CACHE_VERSION),
            # Include post_train params (only affect cache key when filtering is active)
            "post_train_mode": self.post_train_mode,
            "post_train_min_t": self.post_train_min_t if self.post_train_mode else 0.0,
            "post_train_min_rot": self.post_train_min_rot if self.post_train_mode else 0.0,
        }
        return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

    def _upgrade_cached_windows(self, metadata: Dict[str, Dict], windows: List[Dict[str, Any]]) -> bool:
        """Backwards-compat: older caches did not store clip-local `t0` indices."""
        updated = False
        frame_idx_by_clip: Dict[str, Dict[str, int]] = {}
        for w in windows:
            if not isinstance(w, dict) or "t0" in w:
                continue
            clip_id = w.get("clip_id")
            if not isinstance(clip_id, str) or clip_id not in metadata:
                continue

            frame_ids = w.get("frame_ids")
            if isinstance(frame_ids, np.ndarray):
                if frame_ids.size == 0:
                    continue
                first_fid = str(frame_ids.reshape(-1)[0])
            elif isinstance(frame_ids, (list, tuple)):
                if len(frame_ids) == 0:
                    continue
                first_fid = str(frame_ids[0])
            else:
                continue

            idx_map = frame_idx_by_clip.get(clip_id)
            if idx_map is None:
                clip_fids = metadata.get(clip_id, {}).get("frame_ids")
                idx_map = {str(fid): i for i, fid in enumerate(clip_fids)} if isinstance(clip_fids, list) else {}
                frame_idx_by_clip[clip_id] = idx_map

            t0 = idx_map.get(first_fid)
            w["t0"] = int(t0) if t0 is not None else 0
            updated = True

        return updated

    def _load_or_build_windows_cached(self):
        cache_path = os.path.join(self.clips_root, f".hot3d_windows_{self.split}_{self._get_cache_key()}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and "metadata" in cached and "windows" in cached:
                metadata, windows = cached["metadata"], cached["windows"]
                if isinstance(metadata, dict) and isinstance(windows, list):
                    if self._upgrade_cached_windows(metadata, windows) and is_rank0():
                        tmp_path = f"{cache_path}.tmp"
                        with open(tmp_path, "wb") as f:
                            pickle.dump({"metadata": metadata, "windows": windows}, f)
                        os.replace(tmp_path, cache_path)
                        print(f"[dim]updated HOT3D cache[/dim] {cache_path}")
                print(f"[dim]loaded HOT3D cache[/dim] {cache_path}")
                return cached["metadata"], cached["windows"]

        # Portable-snapshot fallback: when n_clips isn't computable (raw clip
        # tars absent), the source-HPC hash can't be reproduced. Pick the pkl
        # whose (clip_id, object_uid, k0) tuples are most contained in the
        # paired sample cache.
        if os.environ.get("SAMPLE_CACHE_TRUST_INDEX", "0") == "1" and self._cfg is not None:
            cached_keys: set[tuple[str, str, int]] = set()
            sc_key = make_sample_cache_key(self._cfg)
            sc_dir = os.path.join(self.clips_root, ".sample_cache", sc_key)
            if os.path.isdir(sc_dir):
                for fname in os.listdir(sc_dir):
                    if not fname.endswith(".pt.lz4"):
                        continue
                    stem = fname[:-len(".pt.lz4")]
                    try:
                        cid, ouid, k0 = stem.rsplit("_", 2)
                        cached_keys.add((cid, ouid, int(k0)))
                    except ValueError:
                        continue
            expected_window_len = int(self.context_len) + int(self.H)
            best: Optional[tuple[int, int, str, dict, list]] = None
            for pkl_path in sorted(glob.glob(os.path.join(self.clips_root, f".hot3d_windows_{self.split}_*.pkl"))):
                try:
                    with open(pkl_path, "rb") as f:
                        cached = pickle.load(f)
                except Exception:
                    continue
                if not (isinstance(cached, dict) and "metadata" in cached and "windows" in cached):
                    continue
                wins = cached["windows"]
                if not isinstance(wins, list) or not wins:
                    continue
                w0 = wins[0]
                if len(w0.get("frame_ids", [])) != expected_window_len:
                    continue
                if int(w0.get("frame_skips", self.frame_skips)) != int(self.frame_skips):
                    continue
                if cached_keys:
                    hits = 0
                    for w in wins:
                        wfids = w.get("frame_ids", [])
                        if not wfids:
                            continue
                        k0_str = str(wfids[0])
                        k0 = int(k0_str) if k0_str.isdigit() else 0
                        if (str(w.get("clip_id", "")), str(w.get("object_uid", "")), k0) in cached_keys:
                            hits += 1
                else:
                    hits = len(wins)
                cur = (hits, -len(wins), pkl_path, cached["metadata"], wins)
                if best is None or cur > best:
                    best = cur
            if best is not None:
                hits, _neg_n, pkl_path, meta, wins = best
                key_in_name = os.path.basename(pkl_path)[len(f".hot3d_windows_{self.split}_"):-len(".pkl")]
                if is_rank0():
                    print(f"[dim]HOT3D windows cache[/dim] portable-snapshot fallback • key={key_in_name[:12]}… ({len(wins):,} windows, {hits:,} cache hits)")
                self._upgrade_cached_windows(meta, wins)
                return meta, wins

        metadata = self._load_all_clip_metadata()
        windows = self._build_windows(metadata)
        with open(cache_path, "wb") as f:
            pickle.dump({"metadata": metadata, "windows": windows}, f)
        if is_rank0():
            print(f"[dim]saved HOT3D cache[/dim] {cache_path}")
        return metadata, windows

    def _load_all_clip_metadata(self) -> Dict[str, Dict]:
        metadata = {}
        for clip_path in tqdm(self.clip_paths, desc="Scanning [cyan]HOT3D clips[/cyan]", disable=not is_rank0()):
            clip_id = os.path.basename(clip_path).replace(".tar", "")
            meta = self._scan_clip_metadata(clip_path)
            if meta["n_frames"] > 0 and meta["object_uids"]:
                meta["clip_path"] = clip_path
                metadata[clip_id] = meta
        return metadata

    def _scan_clip_metadata(self, clip_path: str) -> Dict:
        clip_id = os.path.basename(clip_path).replace(".tar", "")

        # Fast path: use pre-extracted metadata pickle if present (from scripts/preprocess_hot3d_metadata.py)
        pkl_path = os.path.join(self.depth_cache_dir, f"{clip_id}_meta.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                clip_data = pickle.load(f)
            frames_dict = clip_data.get("frames", {}) if isinstance(clip_data, dict) else {}
            cameras = clip_data.get("cameras") if isinstance(clip_data, dict) else None

            frames = [fid for fid, fd in frames_dict.items() if isinstance(fid, str) and isinstance(fd, dict) and fd.get("objects") is not None]
            sorted_frames = sorted(frames, key=lambda x: int(x) if x.isdigit() else x)
            frame_order = {fid: i for i, fid in enumerate(sorted_frames)}

            object_uids: set[str] = set()
            object_frames: Dict[str, set[str]] = {}
            for fid in sorted_frames:
                fd = frames_dict.get(fid, {})
                objs = fd.get("objects") if isinstance(fd, dict) else None
                if not isinstance(objs, dict):
                    continue
                for obj_list in objs.values():
                    if isinstance(obj_list, list):
                        for o in obj_list:
                            if isinstance(o, dict) and "object_uid" in o:
                                uid = str(o["object_uid"])
                                object_uids.add(uid)
                                object_frames.setdefault(uid, set()).add(fid)
                    elif isinstance(obj_list, dict) and "object_uid" in obj_list:
                        uid = str(obj_list["object_uid"])
                        object_uids.add(uid)
                        object_frames.setdefault(uid, set()).add(fid)

            object_frames_sorted = {uid: sorted(list(fids), key=lambda f: frame_order.get(f, 10**9)) for uid, fids in object_frames.items()}
            return {
                "n_frames": len(sorted_frames),
                "frame_ids": sorted_frames,
                "object_uids": sorted(list(object_uids)),
                "object_frames": object_frames_sorted,
                "cameras": cameras,
            }

        frames, object_uids, cameras = set(), set(), None
        object_frames: Dict[str, set[str]] = {}
        with tarfile.open(clip_path, "r") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(".objects.json"):
                    frame_id = name.split(".")[0]
                    frames.add(frame_id)
                    for obj_list in json.load(tar.extractfile(member)).values():
                        if isinstance(obj_list, list):
                            for o in obj_list:
                                if isinstance(o, dict) and "object_uid" in o:
                                    uid = str(o["object_uid"])
                                    object_uids.add(uid)
                                    object_frames.setdefault(uid, set()).add(frame_id)
                        elif isinstance(obj_list, dict) and "object_uid" in obj_list:
                            uid = str(obj_list["object_uid"])
                            object_uids.add(uid)
                            object_frames.setdefault(uid, set()).add(frame_id)
                elif name.endswith(".cameras.json") and cameras is None:
                    cameras = json.load(tar.extractfile(member))
        sorted_frames = sorted(frames, key=lambda x: int(x) if x.isdigit() else x)
        frame_order = {fid: i for i, fid in enumerate(sorted_frames)}
        object_frames_sorted = {uid: sorted(list(fids), key=lambda f: frame_order.get(f, 10**9)) for uid, fids in object_frames.items()}
        return {
            "n_frames": len(sorted_frames),
            "frame_ids": sorted_frames,
            "object_uids": sorted(list(object_uids)),
            "object_frames": object_frames_sorted,
            "cameras": cameras,
        }

    def _build_windows(self, clip_metadata: Dict[str, Dict]) -> List[Dict[str, Any]]:
        wins = []
        win_len = self.context_len + self.H
        step = self.frame_skips + 1
        span = (win_len - 1) * step + 1

        # Track filtering stats for logging
        total_candidates, dropped_motion = 0, 0

        for clip_id, meta in tqdm(clip_metadata.items(), desc="Building [cyan]windows[/cyan]", disable=not is_rank0()):
            frame_ids = meta["frame_ids"]
            object_frames = meta.get("object_frames") if isinstance(meta, dict) else None
            # Depth cache supports both legacy NPZ and memory-mapped NPY variants; accept either.
            has_depth_npz = os.path.exists(os.path.join(self.depth_cache_dir, f"{clip_id}_depth.npz"))
            has_depth_npy = os.path.exists(os.path.join(self.depth_cache_dir, f"{clip_id}_depths.npy"))
            if len(frame_ids) < span or not (has_depth_npz or has_depth_npy):
                continue

            # Load clip data once per clip if post_train filtering is enabled
            clip_data = None
            cameras = None
            if self.post_train_mode:
                clip_data = self._load_clip_data(clip_id)
                cameras = clip_data.get("cameras")

            for obj_uid in meta["object_uids"]:
                present_set = set(object_frames.get(obj_uid, [])) if isinstance(object_frames, dict) else None
                if not present_set:
                    continue
                for t0 in range(0, len(frame_ids) - span + 1, self.window_stride):
                    win_fids = [frame_ids[t0 + i * step] for i in range(win_len)]
                    if any(fid not in present_set for fid in win_fids):
                        continue

                    total_candidates += 1

                    # Post-train motion filtering: require minimum translation OR rotation
                    if self.post_train_mode:
                        anchor_local = min(self.context_len, len(win_fids) - 1)
                        trans_m, rot_deg = self._compute_window_motion(clip_data, cameras, win_fids, obj_uid, anchor_local)
                        # Keep window if EITHER translation OR rotation exceeds threshold
                        if trans_m < self.post_train_min_t and rot_deg < self.post_train_min_rot:
                            dropped_motion += 1
                            continue

                    wins.append({"clip_id": clip_id, "object_uid": obj_uid, "frame_ids": win_fids, "clip_path": meta["clip_path"], "t0": t0})

        # Log filtering stats
        if self.post_train_mode and is_rank0() and total_candidates > 0:
            kept = total_candidates - dropped_motion
            print(
                f"[dim]post_train[/dim] HOT3D motion filter • min_t={self.post_train_min_t:.3f}m • min_rot={self.post_train_min_rot:.1f}deg "
                f"• candidates={total_candidates:,} kept={kept:,} dropped={dropped_motion:,}"
            )

        return wins

    def _load_clip_data(self, clip_id: str) -> Dict:
        cached = self._clip_cache.get(clip_id)
        if cached is not None:
            # If load_hand_poses enabled but cache doesn't have hands, populate from tar
            if self.load_hand_poses and "hands" not in cached:
                cached["hands"] = self._load_hands_from_tar(clip_id)
            return cached

        pkl_path = os.path.join(self.depth_cache_dir, f"{clip_id}_meta.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                clip_data = pickle.load(f)
            # If load_hand_poses enabled but pickle doesn't have hands, load from tar
            if self.load_hand_poses and "hands" not in clip_data:
                clip_data["hands"] = self._load_hands_from_tar(clip_id)
            self._clip_cache.put(clip_id, clip_data)
            return clip_data

        clip_data = {"frames": {}, "cameras": None, "hands": {}}
        with tarfile.open(self.clip_metadata[clip_id]["clip_path"], "r") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(".objects.json"):
                    clip_data["frames"].setdefault(name.split(".")[0], {})["objects"] = json.load(tar.extractfile(member))
                elif name.endswith(".cameras.json"):
                    content = json.load(tar.extractfile(member))
                    clip_data["frames"].setdefault(name.split(".")[0], {})["cameras"] = content
                    if clip_data["cameras"] is None:
                        clip_data["cameras"] = content
                elif self.load_hand_poses and name.endswith(".hands.json"):
                    fid = name.split(".")[0]
                    clip_data["hands"][fid] = json.load(tar.extractfile(member))
        self._clip_cache.put(clip_id, clip_data)
        return clip_data

    def _load_hands_from_tar(self, clip_id: str) -> Dict[str, Any]:
        """Load hands.json files from tar for a clip."""
        hands: Dict[str, Any] = {}
        clip_path = self.clip_metadata[clip_id]["clip_path"]
        with tarfile.open(clip_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".hands.json"):
                    fid = member.name.split(".")[0]
                    hands[fid] = json.load(tar.extractfile(member))
        return hands

    def _get_hand_features(self, hands_data: Dict, frame_id: str) -> np.ndarray:
        """Extract 42D hand features (21D left + 21D right) for a frame.

        Each hand has:
          - thetas: 15D PCA-compressed finger pose
          - wrist_xform: 6D (axis-angle rotation + translation)
        """
        frame_hands = hands_data.get(frame_id, {})
        features = []
        for side in ["left", "right"]:
            hand = frame_hands.get(side, {})
            mano = hand.get("mano_pose", {}) if isinstance(hand, dict) else {}
            thetas = mano.get("thetas", [0.0] * 15)
            wrist = mano.get("wrist_xform", [0.0] * 6)
            features.extend(thetas)
            features.extend(wrist)
        return np.array(features, dtype=np.float32)  # 42D total

    def _get_depth_array(self, clip_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int]]]:
        cached = self._depth_array_cache.get(clip_id)
        if cached is not None:
            return cached

        npy_path = os.path.join(self.depth_cache_dir, f"{clip_id}_depths.npy")
        if os.path.exists(npy_path):
            depths = np.load(npy_path, mmap_mode="r")
            meta = np.load(os.path.join(self.depth_cache_dir, f"{clip_id}_meta.npy"), allow_pickle=True).item()
            K_pinhole = np.asarray(meta["K_pinhole"], dtype=np.float32) if meta.get("K_pinhole") is not None else None
            orig_hw = (meta["orig_h"], meta["orig_w"]) if meta.get("orig_h") is not None else None
        else:
            with np.load(os.path.join(self.depth_cache_dir, f"{clip_id}_depth.npz"), allow_pickle=False) as z:
                depths = z["depths"].astype(np.float32)
                K_pinhole = z["K_pinhole"].astype(np.float32) if "K_pinhole" in z else None
                orig_hw = (int(z["orig_h"]), int(z["orig_w"])) if "orig_h" in z else None

        if self.verbose and (K_pinhole is None or orig_hw is None) and not self._warned_missing_depth_meta:
            self._warned_missing_depth_meta = True
            print(
                f"[yellow]HOT3D[/yellow] depth metadata missing (K_pinhole/orig_hw); "
                f"point clouds will use a pinhole approximation of fisheye intrinsics scaled to depth resolution. "
                f"(example clip: {clip_id})"
            )

        result = (depths, K_pinhole, orig_hw)
        self._depth_array_cache.put(clip_id, result)
        return result

    def _load_depth_frame(self, clip_id: str, frame_idx: int) -> np.ndarray:
        depths, _, _ = self._get_depth_array(clip_id)
        idx = frame_idx if frame_idx < len(depths) else 0
        return np.ascontiguousarray(depths[idx], dtype=np.float32) if len(depths) > 0 else np.zeros((256, 256), dtype=np.float32)

    @staticmethod
    def _quat_to_se3(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = _ScipyRotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        T[:3, 3] = [tx, ty, tz]
        return T

    @staticmethod
    def _invert_se3_batch(T: np.ndarray) -> np.ndarray:
        """Invert SE(3) as (N,4,4) assuming a valid rotation matrix."""
        if T.ndim != 3 or T.shape[-2:] != (4, 4):
            raise ValueError(f"Expected (N,4,4), got {T.shape}")
        R = T[:, :3, :3].astype(np.float32)
        t = T[:, :3, 3].astype(np.float32)
        R_inv = np.transpose(R, (0, 2, 1))
        t_inv = -np.einsum("bij,bj->bi", R_inv, t)
        Tout = np.tile(np.eye(4, dtype=np.float32), (T.shape[0], 1, 1))
        Tout[:, :3, :3] = R_inv
        Tout[:, :3, 3] = t_inv
        return Tout

    @staticmethod
    def _invert_se3_single(T: np.ndarray) -> np.ndarray:
        """Invert a single 4x4 SE(3) matrix."""
        R, t = T[:3, :3], T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        Tout = np.eye(4, dtype=np.float32)
        Tout[:3, :3] = R_inv
        Tout[:3, 3] = t_inv
        return Tout

    def _get_camera_extrinsics(self, cameras: Dict, frame_cameras: Optional[Dict] = None) -> np.ndarray:
        t_wc = (frame_cameras or cameras).get(self.STREAM_ID, {}).get("T_world_from_camera", {})
        t, q = t_wc.get("translation_xyz", [0, 0, 0]), t_wc.get("quaternion_wxyz", [1, 0, 0, 0])
        return self._quat_to_se3(q[0], q[1], q[2], q[3], t[0], t[1], t[2])

    def _get_intrinsics(self, cameras: Dict) -> np.ndarray:
        calib = cameras.get(self.STREAM_ID, {}).get("calibration", cameras.get(self.STREAM_ID, {}))
        params = calib.get("projection_params", [])
        h, w = calib.get("image_height", 1408), calib.get("image_width", 1408)
        if len(params) >= 3:
            fx = fy = params[2]
            cx, cy = params[0], params[1]
        else:
            fx = fy = max(h, w) * 0.5
            cx, cy = w / 2.0, h / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def _find_object(self, frame_data: Dict, obj_uid: str) -> Optional[Dict]:
        for obj_list in frame_data.get("objects", {}).values():
            if isinstance(obj_list, list):
                for obj in obj_list:
                    if isinstance(obj, dict) and obj.get("object_uid") == obj_uid:
                        return obj
            elif isinstance(obj_list, dict) and obj_list.get("object_uid") == obj_uid:
                return obj_list
        return None

    def _get_object_pose_world(self, frame_data: Dict, obj_uid: str) -> Optional[np.ndarray]:
        obj = self._find_object(frame_data, obj_uid)
        if obj is None:
            return None
        t_wo = obj.get("T_world_from_object", {})
        t, q = t_wo.get("translation_xyz", [0, 0, 0]), t_wo.get("quaternion_wxyz", [1, 0, 0, 0])
        return self._quat_to_se3(q[0], q[1], q[2], q[3], t[0], t[1], t[2])

    def _get_object_bbox(self, frame_data: Dict, obj_uid: str) -> Optional[List[float]]:
        obj = self._find_object(frame_data, obj_uid)
        if obj is None:
            return None
        return obj.get("boxes_amodal", {}).get(self.STREAM_ID) or obj.get("boxes_modal", {}).get(self.STREAM_ID)

    @staticmethod
    def _decode_rle(rle: list[int], height: int, width: int) -> np.ndarray:
        """Decode HOT3D uncompressed RLE into a binary mask.

        HOT3D-Clips stores masks as a flat list of (start_index, run_length) pairs
        in row-major (C) order, where indices are into a flattened (H*W,) array.
        """
        h = int(height)
        w = int(width)
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), dtype=np.bool_)
        if not isinstance(rle, list) or len(rle) == 0:
            return np.zeros((h, w), dtype=np.bool_)

        flat = np.zeros(h * w, dtype=np.uint8)
        for start, length in zip(rle[0::2], rle[1::2], strict=False):
            s = int(start)
            n = int(length)
            if n <= 0:
                continue
            if s < 0:
                continue
            if s >= flat.size:
                break
            end = min(s + n, flat.size)
            flat[s:end] = 1

        return flat.reshape((h, w)).astype(np.bool_)

    def _get_object_mask(self, frame_data: Dict, obj_uid: str) -> Optional[np.ndarray]:
        """Return the object mask for the configured stream (modal preferred)."""
        obj = self._find_object(frame_data, obj_uid)
        if obj is None:
            return None
        mask_info = obj.get("masks_modal", {}).get(self.STREAM_ID) or obj.get("masks_amodal", {}).get(self.STREAM_ID)
        if not isinstance(mask_info, dict):
            return None
        rle = mask_info.get("rle")
        if not isinstance(rle, list):
            return None
        h = int(mask_info.get("height", 0) or 0)
        w = int(mask_info.get("width", 0) or 0)
        return self._decode_rle(rle, h, w)

    def _compute_window_motion(
        self,
        clip_data: Dict,
        cameras: Dict,
        frame_ids: List[str],
        obj_uid: str,
        anchor_local: int,
    ) -> Tuple[float, float]:
        """Compute total translation and max rotation for a window in anchor camera frame.

        Args:
            clip_data: Full clip data dict with 'frames' containing per-frame data
            cameras: Camera calibration dict
            frame_ids: List of frame IDs in this window
            obj_uid: Object UID to track
            anchor_local: Local index of anchor frame within window

        Returns:
            (trans_total_m, rot_max_deg): Total translation in meters and max rotation in degrees.
            Returns (0.0, 0.0) if any pose data is missing.
        """
        # Get anchor camera pose (world<-camera)
        anchor_fid = frame_ids[anchor_local]
        anchor_frame_data = clip_data.get("frames", {}).get(anchor_fid, {})
        anchor_frame_cameras = anchor_frame_data.get("cameras", cameras)
        T_c_w_anchor = self._get_camera_extrinsics(cameras, anchor_frame_cameras)

        # Invert to get cam(anchor)<-world
        T_cam_anchor_w = self._invert_se3_single(T_c_w_anchor)

        # Collect object poses in anchor camera frame
        T_cam_anchor_obj_list = []
        for fid in frame_ids:
            frame_data = clip_data.get("frames", {}).get(fid, {})
            T_w_o = self._get_object_pose_world(frame_data, obj_uid)
            if T_w_o is None:
                return 0.0, 0.0  # Skip window if any pose is missing
            T_cam_anchor_obj = T_cam_anchor_w @ T_w_o
            T_cam_anchor_obj_list.append(T_cam_anchor_obj)

        if len(T_cam_anchor_obj_list) < 2:
            return 0.0, 0.0

        T_seq = np.stack(T_cam_anchor_obj_list, axis=0).astype(np.float32)  # (N, 4, 4)

        # Compute total translation (sum of frame-to-frame displacements)
        trans = T_seq[:, :3, 3]  # (N, 3)
        diffs = np.diff(trans, axis=0)  # (N-1, 3)
        trans_total = float(np.linalg.norm(diffs, axis=1).sum())

        # Compute max rotation difference (geodesic distance in degrees)
        R_seq = T_seq[:, :3, :3]  # (N, 3, 3)
        rot_max_deg = 0.0
        for i in range(len(R_seq) - 1):
            R1, R2 = R_seq[i], R_seq[i + 1]
            dR = R1.T @ R2
            trace = np.clip(dR[0, 0] + dR[1, 1] + dR[2, 2], -1.0, 3.0)
            angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            rot_max_deg = max(rot_max_deg, float(np.degrees(angle_rad)))

        return trans_total, rot_max_deg

    def _get_pixel_grid(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        if (H, W) not in self._pixel_grids:
            self._pixel_grids[(H, W)] = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        return self._pixel_grids[(H, W)]

    def _build_pointcloud(self, depth: np.ndarray, K: np.ndarray, T_c_w: Optional[np.ndarray] = None) -> np.ndarray:
        H, W = depth.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u, v = self._get_pixel_grid(H, W)
        z = depth
        x, y = (u - cx) * z / fx, (v - cy) * z / fy
        z_flat = z.ravel()
        valid = (z_flat > 0.01) & (z_flat < 100.0)
        points = np.column_stack([x.ravel(), y.ravel(), z_flat])[valid]

        if T_c_w is not None:
            points = (points @ T_c_w[:3, :3].T) + T_c_w[:3, 3]

        if isinstance(self.downsample_cfg, dict) and self.downsample_cfg.get("enabled"):
            method = self.downsample_cfg.get("method", "voxel")
            if method == "voxel":
                points = voxel_downsample(points, float(self.downsample_cfg.get("voxel_size", 0.01)))
            elif method == "random":
                points = subsample_points(points, int(self.downsample_cfg.get("target_n", self.n_points)))
            tgt = int(self.downsample_cfg.get("target_n", 0))
            if tgt > 0 and points.shape[0] > tgt:
                points = subsample_points(points, tgt)
            elif tgt > 0 and points.shape[0] < tgt:
                points = pad_points_by_interpolation(points, tgt)
        else:
            # Deterministic sampling/padding (matches pointcloud.py helpers).
            if points.shape[0] > self.n_points:
                points = subsample_points(points, self.n_points)
            elif points.shape[0] < self.n_points:
                points = pad_points_by_interpolation(points, self.n_points)

        return points.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        clip_id = win["clip_id"]
        obj_uid = win["object_uid"]
        frame_ids = win["frame_ids"]
        k0_str = frame_ids[0]
        k0 = int(k0_str) if k0_str.isdigit() else 0

        # Backward/robustness: older pickled dataset instances (or mismatched imports across workers)
        # may miss `overwrite_sample_cache`. Default to False.
        if self.use_sample_cache and (not getattr(self, "overwrite_sample_cache", False)) and self._sample_cache_dir is not None:
            cache_path = get_sample_path(Path(self._sample_cache_dir), clip_id, obj_uid, k0)
            cached = load_sample(cache_path)
            if cached is not None:
                # Cache compatibility: backfill mesh_path if missing (doesn't require recomputing heavy fields).
                if (not cached.get("mesh_path")) and isinstance(self.object_library, str) and self.object_library and self._uid_to_bop:
                    bop_id = self._uid_to_bop.get(str(obj_uid))
                    if bop_id is not None:
                        mp = os.path.join(self.object_library, f"obj_{int(bop_id):06d}.glb")
                        if os.path.isfile(mp):
                            cached["mesh_path"] = mp
                # Cache compatibility: backfill object mask if placeholder/dummy.
                try:
                    obj = cached.get("object")
                    m = obj.get("mask") if isinstance(obj, dict) else None
                    if isinstance(m, np.ndarray) and m.shape == (1, 1):
                        clip_data = self._load_clip_data(clip_id)
                        a = cached.get("anchor_frame_idx", k0)
                        a_int = int(a) if str(a).isdigit() else k0
                        fid = f"{a_int:06d}"
                        frame_data = clip_data.get("frames", {}).get(fid, {})
                        m2 = self._get_object_mask(frame_data, obj_uid)
                        if isinstance(m2, np.ndarray) and m2.ndim == 2 and m2.size > 0:
                            obj["mask"] = m2.astype(np.bool_)
                            obj["mask_frame_idx"] = a_int
                except Exception:
                    pass
                # Load hand poses at runtime if enabled (not stored in cache)
                if self.load_hand_poses:
                    clip_data = self._load_clip_data(clip_id)
                    hands_data = clip_data.get("hands", {})
                    context_frame_ids = cached.get("context_frame_ids", np.array([], dtype=np.int32))
                    hand_poses = []
                    for fid_int in context_frame_ids:
                        fid = f"{int(fid_int):06d}"  # Convert to 6-digit zero-padded string
                        hand_feats = self._get_hand_features(hands_data, fid)
                        hand_poses.append(hand_feats)
                    if hand_poses:
                        cached["context_hand_poses"] = np.stack(hand_poses, axis=0)
                    else:
                        P_ctx2 = len(context_frame_ids) if len(context_frame_ids) > 0 else (self.context_len + 1)
                        cached["context_hand_poses"] = np.zeros((P_ctx2, 42), dtype=np.float32)
                return cached

        # Load clip data (depth loaded lazily per-frame to save memory)
        clip_data = self._load_clip_data(clip_id)
        cameras = clip_data["cameras"]

        # Get depth array + pinhole intrinsics (if available from new NPZ format)
        _, K_pinhole_cached, orig_hw_cached = self._get_depth_array(clip_id)

        # Prefer pinhole K from NPZ (correct for undistorted depth), fallback to cameras.json
        if K_pinhole_cached is not None:
            K = K_pinhole_cached.copy()
        else:
            K = self._get_intrinsics(cameras)

        # Build pose arrays for all frames in window
        P_ctx = self.context_len
        anchor_local = min(P_ctx, len(frame_ids) - 1)
        step = self.frame_skips + 1

        T_c_w_list: list[np.ndarray] = []  # world<-camera (camera->world)
        T_w_o_list: list[np.ndarray] = []  # world<-object (object->world)

        for fid in frame_ids:
            frame_data = clip_data["frames"].get(fid, {})
            frame_cameras = frame_data.get("cameras", cameras)

            T_c_w = self._get_camera_extrinsics(cameras, frame_cameras)
            T_w_o = self._get_object_pose_world(frame_data, obj_uid)

            if T_w_o is None:
                raise ValueError(f"HOT3D window contains a frame without object pose: clip_id={clip_id} obj_uid={obj_uid} fid={fid}")

            T_c_w_list.append(T_c_w)
            T_w_o_list.append(T_w_o)

        T_c_w_win = np.stack(T_c_w_list, axis=0).astype(np.float32)
        T_w_o_win = np.stack(T_w_o_list, axis=0).astype(np.float32)

        # Compute camera<-world and camera<-object for each frame
        T_w_c_win = self._invert_se3_batch(T_c_w_win)  # camera<-world (world->camera)
        T_c_o_win = np.matmul(T_w_c_win, T_w_o_win).astype(np.float32)  # camera<-object

        # Canonicalize to anchor camera frame: cam(anchor)<-object(k) = cam(anchor)<-world * world<-object(k)
        T_cam_anchor_w = T_w_c_win[anchor_local].astype(np.float32)  # cam(anchor)<-world
        T_cam_anchor_obj = np.matmul(T_cam_anchor_w[None, ...], T_w_o_win).astype(np.float32)

        # Extract target future (H frames after context)
        T_sel = T_cam_anchor_obj[P_ctx : P_ctx + self.H]
        target_future = _poses_to_9d(T_sel)

        # Initial pose (first frame in world)
        T_w_o0 = T_w_o_win[0]
        t0_vec = T_w_o0[:3, 3].astype(np.float32)
        R0 = T_w_o0[:3, :3]
        r6_0 = np.concatenate([R0[:, 0], R0[:, 1]], axis=0).astype(np.float32)

        # Camera extrinsics at k0
        T_c_w0 = T_c_w_win[0]  # world<-camera
        T_wc0 = T_w_c_win[0]  # camera<-world

        # Build point cloud from depth (lazy per-frame loading to save memory)
        # Use clip-local index from window dict, not timestamp string
        # k0_depth_idx = win["t0"]  # clip-local index for first window frame (kept for reference)
        anchor_depth_idx = win["t0"] + anchor_local * step  # clip-local index for anchor frame
        depth_frame = self._load_depth_frame(clip_id, anchor_depth_idx)  # Use anchor frame depth

        # Scale intrinsics to match depth resolution (SpaTrackerV2 outputs at different resolution)
        cam_info = cameras.get(self.STREAM_ID, {})
        calib = cam_info.get("calibration", cam_info)
        rgb_h = int(calib.get("image_height", 1408))
        rgb_w = int(calib.get("image_width", 1408))
        depth_orig_h, depth_orig_w = orig_hw_cached if orig_hw_cached is not None else (rgb_h, rgb_w)
        depth_h, depth_w = depth_frame.shape
        K_scaled = K.copy()
        if depth_h != depth_orig_h or depth_w != depth_orig_w:
            scale_x = depth_w / max(depth_orig_w, 1)
            scale_y = depth_h / max(depth_orig_h, 1)
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy

        # Output scene_pcd in anchor camera frame (not world) to match target_future coords
        scene_pcd = self._build_pointcloud(depth_frame, K_scaled, T_c_w=None)

        # Context data
        P_ctx2 = P_ctx + 1
        context_T_cam_anchor_obj = T_cam_anchor_obj[:P_ctx2]
        context_init_9d = _poses_to_9d(context_T_cam_anchor_obj)

        # Bounding boxes (reuse orig_h, orig_w from intrinsics scaling above)
        bbox_norm = [0.0, 0.0, 1.0, 1.0]
        context_bbox_list = []
        h_img = rgb_h
        w_img = rgb_w

        for i, fid in enumerate(frame_ids[:P_ctx2]):
            frame_data = clip_data["frames"].get(fid, {})
            bb = self._get_object_bbox(frame_data, obj_uid)
            if bb is not None and len(bb) == 4 and w_img > 0 and h_img > 0:
                x1, y1, x2, y2 = bb
                context_bbox_list.append([x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img])
                if i == 0:
                    bbox_norm = [x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img]
            else:
                context_bbox_list.append(list(bbox_norm))

        context_bbox_norm = np.array(context_bbox_list, dtype=np.float32)

        # Load hand poses for context frames (P_ctx2 = P_ctx + 1 frames, same as context_init_9d)
        if self.load_hand_poses:
            hand_poses = []
            hands_data = clip_data.get("hands", {})
            for fid in frame_ids[:P_ctx2]:  # Same slice as bbox/init9d
                hand_feats = self._get_hand_features(hands_data, fid)
                hand_poses.append(hand_feats)
            context_hand_poses = np.stack(hand_poses, axis=0)  # (P_ctx2, 42)
        else:
            context_hand_poses = np.zeros((P_ctx2, 42), dtype=np.float32)

        # Load object mask for the anchor frame (modal preferred; falls back to dummy if missing)
        anchor_fid = frame_ids[anchor_local]
        anchor_frame_data = clip_data["frames"].get(anchor_fid, {})
        mask_used = self._get_object_mask(anchor_frame_data, obj_uid)
        mask_frame_idx = int(anchor_fid) if str(anchor_fid).isdigit() else anchor_local
        if not (isinstance(mask_used, np.ndarray) and mask_used.ndim == 2 and mask_used.size > 0):
            mask_used = np.zeros((1, 1), dtype=np.bool_)
            mask_frame_idx = k0

        sample = {
            "video_id": clip_id,
            "object_id": obj_uid,
            "K": K,
            "T_wc0": T_wc0.astype(np.float32),
            "T_c_w0": T_c_w0.astype(np.float32),
            "scene_pcd": scene_pcd,
            "init_pose": {"t0": t0_vec, "rot6d0": r6_0},
            "target_future": target_future,
            "mesh_path": "",
            "object": {
                "bbox": np.array([bbox_norm[0] * w_img, bbox_norm[1] * h_img, bbox_norm[2] * w_img, bbox_norm[3] * h_img], dtype=np.float32),
                "bbox_norm": np.array(bbox_norm, dtype=np.float32),
                "mask": mask_used.astype(np.bool_) if isinstance(mask_used, np.ndarray) else None,
                "mask_frame_idx": int(mask_frame_idx),
            },
            "frame_ids": np.array([int(f) if f.isdigit() else i for i, f in enumerate(frame_ids)], dtype=np.int32),
            "rgb_path": win["clip_path"],
            "spatrack_npz": os.path.join(self.depth_cache_dir, f"{clip_id}_depth.npz"),
            "object_dir": "",
            "T_c_o": T_c_o_win.astype(np.float32),
            "T_c_w": T_c_w_win.astype(np.float32),
            "T_cam_anchor_obj": T_cam_anchor_obj,
            "anchor_mode": "window_start",
            "anchor_frame_idx": int(frame_ids[anchor_local]) if frame_ids[anchor_local].isdigit() else anchor_local,
            "anchor_local_idx": anchor_local,
            "anchor_depth_idx": anchor_depth_idx,  # clip-local index for depth array (for sample_picker/viz)
            "extrinsics_convention": "c2w",
            "context_len": P_ctx2,
            "context_frame_ids": np.array([int(f) if f.isdigit() else i for i, f in enumerate(frame_ids[:P_ctx2])], dtype=np.int32),
            "context_T_cam_anchor_obj": context_T_cam_anchor_obj,
            "context_init_9d": context_init_9d,
            "context_bbox_norm": context_bbox_norm,
            "context_hand_poses": context_hand_poses,
        }

        # If we have a mesh library, set mesh_path (EPIC-parity; used by viz/render tools).
        if isinstance(self.object_library, str) and self.object_library and self._uid_to_bop:
            bop_id = self._uid_to_bop.get(str(obj_uid))
            if bop_id is not None:
                mp = os.path.join(self.object_library, f"obj_{int(bop_id):06d}.glb")
                if os.path.isfile(mp):
                    sample["mesh_path"] = mp

        if self.verbose and not hasattr(self, "_printed_loads"):
            self._printed_loads = set()
        if self.verbose and f"{clip_id}:{obj_uid}" not in getattr(self, "_printed_loads", set()):
            self._printed_loads.add(f"{clip_id}:{obj_uid}")
            print(f"[dim]load[/dim] {clip_id} • obj={obj_uid} • k0={k0}")

        if self.precompute_cache and self._sample_cache_dir is not None:
            cache_path = get_sample_path(Path(self._sample_cache_dir), clip_id, obj_uid, k0)
            if getattr(self, "overwrite_sample_cache", False) or (not sample_exists(cache_path)):
                save_sample(cache_path, sample)

        return sample
