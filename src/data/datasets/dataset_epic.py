from __future__ import annotations

import csv
import glob
import hashlib
import os
import resource
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import TqdmExperimentalWarning

from ...dist.distrib import is_rank0
from ...geom.se3_ops import invert_T_np as invert_T
from ...geom.se3_ops import select_extrinsics_convention
from ...utils.geometry import infer_image_size_from_K, scale_intrinsics
from ...utils.logger import rprint as print
from ...utils.progress import tqdm_auto as tqdm
from ..cache import (
    compute_records_fingerprint,
    make_windows_cache_key,
    read_conv_map,
    read_spatrack_len_map,
    read_windows_cache,
    write_conv_map,
    write_spatrack_len_map,
    write_windows_cache,
)
from ..fpose_io import load_object_poses
from ..loader import scan_dataset
from ..pointcloud import maybe_build_scene_pointcloud
from ..sample_cache import get_sample_path, load_sample, make_sample_cache_key, sample_exists, save_sample
from ..spatrack_io import get_spatrack_length, load_spatrack
from ..video_io import get_video_frame_size
from .depth_getters import _DepthArrayGetter, _DepthFileGetter
from .helpers import _poses_to_9d, _stem_to_fid
from .lru_cache import _LRU

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class SceneSequenceDataset(Dataset):
    """Filesystem-backed dataset that scans 3DManip results and builds H-step windows.

    Each item corresponds to one object trajectory window of length H+1 frames, returning
    targets for steps 1..H relative to the first frame k0.
    """

    def __init__(
        self,
        dataset_root: str,
        H: int,
        window_stride: int = 1,
        frame_skips: int = 0,
        min_frames: Optional[int] = None,
        use_depth: bool = True,
        n_points: int = 20000,
        load_rgb0: bool = True,
        cache_index: bool = True,
        verbose: bool = False,
        anchor_mode: Literal["window_start", "global_frame", "absolute"] = "window_start",
        anchor_frame_idx: int = 0,
        output_format: Literal["abs_in_anchor", "delta_from_prev", "delta_from_anchor"] = "abs_in_anchor",
        return_rotations: bool = True,
        debug_checks: bool = False,
        downsample_cfg: Optional[dict] = None,
        respect_init_from_frame: bool = True,
        rot_smooth_alpha: float = 0.15,
        context_len: int = 1,
        filter_iou_drop: bool = False,
        iou_drop_threshold: float = 0.1,
        force_refresh_cache: bool = False,
        force_rebuild_windows_cache: bool = False,
        post_train_mode: bool = False,
        post_train_iou_mean_thresh: float = 0.0,
        post_train_min_t: float = 0.0,
        windows_cache_tag: Optional[str] = None,
        use_sample_cache: bool = False,
        sample_cache_dir: Optional[str] = None,
        precompute_cache: bool = False,
        overwrite_sample_cache: bool = False,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        from ..cache import remap_data_path
        self._dataset_root_orig = dataset_root
        self.dataset_root = remap_data_path(dataset_root)
        self.H = H
        self.window_stride = window_stride
        self.frame_skips = max(0, frame_skips)
        self.min_frames = min_frames if min_frames is not None else H
        self.use_depth = use_depth
        self.n_points = n_points
        self.load_rgb0 = load_rgb0
        self.cache_index = cache_index
        self.verbose = verbose
        self.anchor_mode = anchor_mode
        self.anchor_frame_idx = anchor_frame_idx
        self.output_format = output_format
        self.return_rotations = return_rotations
        self.debug_checks = debug_checks
        self.downsample_cfg = OmegaConf.to_container(downsample_cfg, resolve=True) if isinstance(downsample_cfg, DictConfig) else downsample_cfg
        self.respect_init_from_frame = respect_init_from_frame
        self.rot_smooth_alpha = rot_smooth_alpha
        self.context_len = context_len
        self.filter_iou_drop = filter_iou_drop
        self.iou_drop_threshold = iou_drop_threshold
        self.force_refresh_cache = force_refresh_cache
        self.force_rebuild_windows_cache = force_rebuild_windows_cache
        self.post_train_mode = post_train_mode
        self.post_train_iou_mean_thresh = post_train_iou_mean_thresh
        self.post_train_min_t = post_train_min_t
        self._windows_cache_tag = str(windows_cache_tag) if windows_cache_tag else ""
        self.use_sample_cache = use_sample_cache
        self.precompute_cache = precompute_cache
        self.overwrite_sample_cache = overwrite_sample_cache
        self._cfg = cfg
        self._sample_cache_dir: Optional[str] = None

        if self.use_sample_cache or self.precompute_cache:
            if cfg is not None:
                cache_key = make_sample_cache_key(cfg)
            else:
                key_parts = [str(H), str(context_len), str(frame_skips), str(n_points), str(downsample_cfg), str(anchor_mode), str(output_format), "1"]
                cache_key = hashlib.sha1("|".join(key_parts).encode()).hexdigest()[:12]
            base_dir = sample_cache_dir or self.dataset_root
            self._sample_cache_dir = str(Path(base_dir) / ".sample_cache" / cache_key)
            Path(self._sample_cache_dir).mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"[bold cyan]Sample cache[/bold cyan] dir={self._sample_cache_dir}")

        if self.post_train_mode:
            tag = self._windows_cache_tag
            print(
                f"[bold yellow]post_train[/bold yellow] enabled • iou_mean_thr={self.post_train_iou_mean_thresh:.3f} • min_t={self.post_train_min_t:.3f} m"
                + (f" • cache_tag='{tag}'" if tag else "")
            )
        self._init_from_map_cache: Dict[str, Dict[int, int]] = {}
        self._iou_map_cache: Dict[str, Dict[int, float]] = {}

        self.records: List[Dict[str, Any]] = scan_dataset(
            self.dataset_root,
            self.H,
            cache_index=self.cache_index,
            min_frames=self.min_frames,
            force_refresh_cache=self.force_refresh_cache,
        )

        cap_video = int(os.environ.get("DATASET_CAP_VIDEO", 10000))
        cap_obj = int(os.environ.get("DATASET_CAP_OBJ", 100000))
        cap_masks = int(os.environ.get("DATASET_CAP_MASKS", 1024))

        self._video_pack = _LRU(cap_video)
        self._video_len: Dict[str, int] = {}
        self._video_loaded: Dict[str, bool] = {}
        self._video_conv: Dict[str, str] = {}
        conv_map = read_conv_map(self.dataset_root)
        if isinstance(conv_map, dict):
            self._video_conv.update(conv_map)
        self._obj_cache = _LRU(cap_obj)
        self._obj_warned_mismatch: Dict[str, bool] = {}
        self._mask_array_cache = _LRU(cap_masks)
        self._mask_warned_missing: Dict[str, bool] = {}
        self._mem_log_every = int(os.environ.get("DATASET_MEM_LOG_EVERY", "0"))

        rec_fp = compute_records_fingerprint(self.records)
        self._windows_cache_key: Optional[str] = self._window_cache_key(rec_fp)
        if self.cache_index and not self.force_refresh_cache:
            cached_wins = self._maybe_load_windows_cache(rec_fp)
            if cached_wins:
                self.windows = cached_wins
                self._remap_window_paths(self.windows)
                print(
                    f"[bold]Dataset[/bold] • videos={len({r['video_id'] for r in self.records})} "
                    f"• objects={len(self.records)} • windows={len(self.windows):,} • H={self.H} "
                    f"• P={self.context_len} • stride={self.window_stride} "
                    f"• fs={self.frame_skips} • [dim](windows cache)[/dim]"
                )
                return

        self.windows = self._build_windows()
        self._remap_window_paths(self.windows)
        print(
            f"[bold]Dataset[/bold] • videos={len({r['video_id'] for r in self.records})} "
            f"• objects={len(self.records)} • windows={len(self.windows):,} • H={self.H} "
            f"• P={self.context_len} • stride={self.window_stride} • fs={self.frame_skips}"
        )

    def _window_cache_key(self, rec_fp: str, mode_tag: Optional[str] = None) -> str:
        return make_windows_cache_key(
            self._dataset_root_orig,
            self.H,
            self.window_stride,
            self.context_len,
            self.respect_init_from_frame,
            self.filter_iou_drop,
            self.iou_drop_threshold,
            self.min_frames,
            rec_fp,
            anchor_mode=self.anchor_mode,
            anchor_frame_idx=self.anchor_frame_idx,
            output_format=self.output_format,
            anchor_policy="horizon_start_v1",
            mode_tag=mode_tag if mode_tag is not None else self._windows_cache_tag,
            frame_skips=self.frame_skips,
        )

    def _legacy_cache_tags(self) -> List[str]:
        if not self.post_train_mode:
            return []
        tags = ["post_train"]
        if self.post_train_iou_mean_thresh > 0.0 or self.post_train_min_t > 0.0:
            tags.append(f"post_train_iou{self.post_train_iou_mean_thresh:.3f}_t{self.post_train_min_t:.3f}")
        return [t for t in tags if t != self._windows_cache_tag]

    @staticmethod
    def _remap_window_paths(windows: List[Dict[str, Any]]) -> None:
        """In-place: remap snapshot-source absolute paths in window dicts to
        local mounts via DATA_PATH_REMAP. No-op when env var unset."""
        from ..cache import remap_data_path
        if not os.environ.get("DATA_PATH_REMAP"):
            return
        path_keys = ("spatrack_npz", "object_dir", "mesh_path", "rgb_path")
        for w in windows:
            for k in path_keys:
                v = w.get(k)
                if isinstance(v, str) and v:
                    w[k] = remap_data_path(v)

    def _maybe_load_windows_cache(self, rec_fp: str) -> Optional[List[Dict[str, Any]]]:
        if self.force_rebuild_windows_cache:
            return None
        cached = read_windows_cache(self.dataset_root, self._windows_cache_key)
        if cached:
            return cached
        for tag in self._legacy_cache_tags():
            alt_key = self._window_cache_key(rec_fp, mode_tag=tag)
            alt = read_windows_cache(self.dataset_root, alt_key)
            if alt:
                self._windows_cache_key = alt_key
                print(f"[dim]windows cache[/dim] using legacy tag='{tag}' (found {len(alt):,} windows)")
                return alt
        # Portable-snapshot fallback: EPIC's windows-cache key includes a
        # records fingerprint that depends on raw-file mtimes, which can't be
        # recomputed without raw data. Pick the snapshot's .windows.*.pkl whose
        # (video_id, object_id, k0) tuples are most contained in the prebuilt
        # sample cache — that's the pkl the cache was actually built from.
        if os.environ.get("SAMPLE_CACHE_TRUST_INDEX", "0") == "1" and self._sample_cache_dir is not None and os.path.isdir(self._sample_cache_dir):
            import glob, pickle
            cached_keys: set[tuple[str, str, int]] = set()
            for fname in os.listdir(self._sample_cache_dir):
                if not fname.endswith(".pt.lz4"):
                    continue
                stem = fname[:-len(".pt.lz4")]
                try:
                    cid, ouid, k0 = stem.rsplit("_", 2)
                    cached_keys.add((cid, ouid, int(k0)))
                except ValueError:
                    continue
            expected_window_len = int(self.context_len) + int(self.H)
            best: Optional[tuple[int, int, str, list]] = None  # (hits, -nwins, path, wins)
            for pkl_path in sorted(glob.glob(os.path.join(self.dataset_root, ".windows.*.pkl"))):
                try:
                    with open(pkl_path, "rb") as f:
                        wins = pickle.load(f)
                except Exception:
                    continue
                if not isinstance(wins, list) or not wins:
                    continue
                w0 = wins[0]
                if len(w0.get("frame_ids", [])) != expected_window_len:
                    continue
                if int(w0.get("frame_skips", self.frame_skips)) != int(self.frame_skips):
                    continue
                hits = 0
                for w in wins:
                    fids = w.get("frame_ids", [])
                    if not fids:
                        continue
                    k0 = int(fids[0]) if isinstance(fids[0], int) else (int(fids[0]) if str(fids[0]).lstrip("-").isdigit() else 0)
                    if (str(w.get("video_id", "")), str(w.get("object_id", "")), k0) in cached_keys:
                        hits += 1
                cur = (hits, -len(wins), pkl_path, wins)
                if best is None or cur > best:
                    best = cur
            if best is not None:
                hits, _neg_n, pkl_path, wins = best
                key_from_path = os.path.basename(pkl_path)[len(".windows."):-len(".pkl")]
                self._windows_cache_key = key_from_path
                # Default: cache-only filter so __getitem__ never falls through to
                # load_spatrack on an absent dataset path. When precompute_cache=True
                # we *want* the full window list so the script can compute the
                # missing samples (typically with raw data streamed JIT from remote storage).
                if self.precompute_cache:
                    print(f"[dim]windows cache[/dim] portable-snapshot fallback • key={key_from_path[:12]}… ({len(wins):,} windows, {hits:,} cache hits) • precompute mode: filter disabled")
                    return wins
                filtered = [
                    w for w in wins
                    if (
                        w.get("frame_ids")
                        and (
                            str(w.get("video_id", "")),
                            str(w.get("object_id", "")),
                            int(w["frame_ids"][0]),
                        ) in cached_keys
                    )
                ]
                print(f"[dim]windows cache[/dim] portable-snapshot fallback • key={key_from_path[:12]}… ({len(wins):,} windows total, {hits:,} cache hits, retained {len(filtered):,})")
                return filtered
        return None

    def _ensure_video_pack(self, video_id: str, spatrack_npz: str) -> Dict[str, Any]:
        from ..cache import remap_data_path
        spatrack_npz = remap_data_path(spatrack_npz)
        pk = self._video_pack.get(video_id)
        if pk is None:
            pk = load_spatrack(spatrack_npz, verbose=False, load_depth=False)
            self._video_pack.put(video_id, pk)
            self._video_len[video_id] = pk["T_c_w"].shape[0]
            self._video_loaded[video_id] = True

            if self.verbose:
                keys = ",".join(pk.get("keys", []))
                extr_name = pk.get("used_ex") or "c2w"
                print(
                    f"[bold cyan]SpaTrack[/bold cyan] {video_id} • frames={pk['T_c_w'].shape[0]} "
                    f"• keys={{{keys}}} • extrinsics={extr_name} "
                    f"• depth={'✓' if pk.get('has_depth') else '×'}"
                )
            if self._mem_log_every and len(self._video_pack) % self._mem_log_every == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"[mem] videos={len(self._video_pack)} objects={len(self._obj_cache)} rss≈{rss:.0f} MB pid={os.getpid()}")
        return pk

    def _obj_key(self, object_dir: str) -> str:
        return object_dir

    def _ensure_object(self, object_dir: str) -> Dict[str, np.ndarray]:
        key = self._obj_key(object_dir)
        obj = self._obj_cache.get(key)
        if obj is None:
            obj = load_object_poses(os.path.join(object_dir, "foundationpose10", "ob_in_cam"), verbose=self.verbose)
            obj = {"frame_ids": obj.get("frame_ids", np.zeros((0,), dtype=np.int32)), "T_c_o": obj.get("T_c_o", np.zeros((0, 4, 4), dtype=np.float32))}
            self._obj_cache.put(key, obj)
        return obj

    def _sorted_object_poses(self, object_dir: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self._ensure_object(object_dir)
        fids = obj["frame_ids"]
        order = np.argsort(fids.astype(np.int64))
        if np.all(order == np.arange(order.shape[0])):
            return fids, obj["T_c_o"].astype(np.float32)
        fids_sorted = fids[order]
        T_sorted = obj["T_c_o"][order].astype(np.float32)
        obj["frame_ids"], obj["T_c_o"] = fids_sorted, T_sorted
        self._obj_cache[self._obj_key(object_dir)] = obj
        return fids_sorted, T_sorted

    def _load_mask_frame(self, object_dir: str, fid: int) -> Optional[np.ndarray]:
        key = (object_dir, fid)
        cached = self._mask_array_cache.get(key)
        if cached is not None:
            self._mask_array_cache.move_to_end(key)
            return cached

        p = os.path.join(object_dir, "masks.npz")
        if not os.path.exists(p):
            if not self._mask_warned_missing.get(object_dir, False):
                print(f"[yellow]missing masks[/yellow] {object_dir}/masks.npz")
                self._mask_warned_missing[object_dir] = True
            return None

        try:
            with np.load(p, allow_pickle=False) as z:
                # Try frame ID key first (original format: keys are '0', '1', '2', ...)
                arr = z.get(str(fid))
                if arr is None:
                    # Fallback: check for 'arr' key (sandbox converted format)
                    # Could be a 3D array (num_frames, H, W) indexed by frame ID
                    if "arr" in z:
                        full_arr = z["arr"]
                        if full_arr.ndim == 3 and fid < full_arr.shape[0]:
                            arr = full_arr[fid]
                        elif full_arr.ndim == 2:
                            # Single 2D mask, use for all frames
                            arr = full_arr
        except zipfile.BadZipFile:
            if self.verbose:
                print(f"[yellow]bad zip file[/yellow] {p}")
            return None

        if arr is None:
            return None
        out = arr.astype(bool)
        self._mask_array_cache.put(key, out)
        return out

    def _load_depth_frame_lazy(self, spatrack_npz: str, idx: int) -> Optional[np.ndarray]:
        try:
            with np.load(spatrack_npz, allow_pickle=False) as z:
                for k in ("depth", "depths", "D"):
                    if k in z:
                        dep = z[k]
                        if dep.ndim == 4 and dep.shape[1] == 1:
                            return dep[idx, 0].astype(np.float32)
                        return dep[idx].astype(np.float32)
        except zipfile.BadZipFile:
            if self.verbose:
                print(f"[yellow]bad zip file[/yellow] {spatrack_npz}")
            return None
        return None

    def _find_object_track_csv(self, object_dir: str) -> Optional[str]:
        """Find CSV with 'frame' and 'init_from_frame' columns in foundationpose10 dirs."""
        candidates: list[str] = []
        for d in [os.path.join(object_dir, "foundationpose10", "ob_in_cam"), os.path.join(object_dir, "foundationpose10")]:
            if os.path.isdir(d):
                candidates.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))
        for p in candidates:
            with open(p, "r") as f:
                header = next(csv.reader(f), None)
            if header and {"frame", "init_from_frame"} <= {h.strip().lower() for h in header}:
                return p
        return None

    def _ensure_init_map(self, object_dir: str) -> Dict[int, int]:
        cached = self._init_from_map_cache.get(object_dir)
        if cached is not None:
            return cached
        path = self._find_object_track_csv(object_dir)
        mapping: Dict[int, int] = {}
        if path is None:
            self._init_from_map_cache[object_dir] = mapping
            return mapping
        rows = []
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                if "frame" in row and "init_from_frame" in row:
                    rows.append((int(row["frame"]), int(row["init_from_frame"])))
        rows.sort(key=lambda t: t[0])
        mapping = {fr: init_fr for fr, init_fr in rows}
        self._init_from_map_cache[object_dir] = mapping
        return mapping

    def _ensure_iou_map(self, object_dir: str) -> Dict[int, float]:
        cached = self._iou_map_cache.get(object_dir)
        if cached is not None:
            return cached
        path = self._find_object_track_csv(object_dir)
        mapping: Dict[int, float] = {}
        if path is None:
            self._iou_map_cache[object_dir] = mapping
            return mapping
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                row_low = {k.strip().lower(): v for k, v in row.items()}
                if "frame" in row_low and "iou" in row_low:
                    mapping[int(row_low["frame"])] = max(0.0, min(1.0, float(row_low["iou"])))
        self._iou_map_cache[object_dir] = mapping
        return mapping

    @staticmethod
    def _mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
        """Convert (N,3,3) rotation matrices to (N,4) unit quaternions [w,x,y,z]."""
        N = R.shape[0]
        q = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            r = R[i]
            tr = r[0, 0] + r[1, 1] + r[2, 2]
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2.0
                qw, qx, qy, qz = 0.25 * S, (r[2, 1] - r[1, 2]) / S, (r[0, 2] - r[2, 0]) / S, (r[1, 0] - r[0, 1]) / S
            elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                S = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                qw, qx, qy, qz = (r[2, 1] - r[1, 2]) / S, 0.25 * S, (r[0, 1] + r[1, 0]) / S, (r[0, 2] + r[2, 0]) / S
            elif r[1, 1] > r[2, 2]:
                S = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
                qw, qx, qy, qz = (r[0, 2] - r[2, 0]) / S, (r[0, 1] + r[1, 0]) / S, 0.25 * S, (r[1, 2] + r[2, 1]) / S
            else:
                S = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
                qw, qx, qy, qz = (r[1, 0] - r[0, 1]) / S, (r[0, 2] + r[2, 0]) / S, (r[1, 2] + r[2, 1]) / S, 0.25 * S
            q[i] = np.array([qw, qx, qy, qz], dtype=np.float32)
            q[i] /= np.linalg.norm(q[i]) + 1e-8
        return q

    @staticmethod
    def _quat_wxyz_to_mat3(q: np.ndarray) -> np.ndarray:
        """Convert (N,4) quaternions [w,x,y,z] to (N,3,3) rotation matrices."""
        N = q.shape[0]
        R = np.zeros((N, 3, 3), dtype=np.float32)
        qq = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        w, x, y, z = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    @staticmethod
    def _quat_make_continuous(q: np.ndarray) -> np.ndarray:
        """Flip q[i] if dot(q[i], q[i-1]) < 0 for sign continuity."""
        qq = q.copy()
        for i in range(1, qq.shape[0]):
            if np.dot(qq[i - 1], qq[i]) < 0.0:
                qq[i] = -qq[i]
        return qq

    def _smooth_rotations_ema(self, T_seq: np.ndarray, alpha: float) -> np.ndarray:
        """Apply EMA smoothing to rotations in a (N,4,4) pose sequence."""
        if T_seq.ndim != 3 or T_seq.shape[1:] != (4, 4) or not (0.0 < alpha <= 1.0):
            return T_seq
        N = T_seq.shape[0]
        R_in = T_seq[:, :3, :3].astype(np.float32)
        q = self._mat3_to_quat_wxyz(R_in)
        q = self._quat_make_continuous(q)
        q_s = np.empty_like(q)
        q_s[0] = q[0]
        for i in range(1, N):
            q_s[i] = (1.0 - alpha) * q_s[i - 1] + alpha * q[i]
            q_s[i] /= np.linalg.norm(q_s[i]) + 1e-8
        R_s = self._quat_wxyz_to_mat3(q_s)
        T_out = T_seq.copy()
        T_out[:, :3, :3] = R_s
        return T_out

    @staticmethod
    def _bbox_from_mask_xyxy(mask: np.ndarray) -> Optional[List[int]]:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @staticmethod
    def _mask_valid(mask: Optional[np.ndarray]) -> bool:
        return isinstance(mask, np.ndarray) and mask.ndim == 2 and mask.size > 0 and np.any(mask)

    @staticmethod
    def _mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
        if not (isinstance(mask_a, np.ndarray) and isinstance(mask_b, np.ndarray) and mask_a.ndim == 2 and mask_b.ndim == 2):
            return 1.0
        a, b = mask_a.astype(bool), mask_b.astype(bool)
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        return float(inter) / float(union) if union > 0 else 1.0

    def _build_windows(self) -> List[Dict[str, Any]]:
        wins: List[Dict[str, Any]] = []
        before_total, after_total, dropped_trans, dropped_iou_mean = 0, 0, 0, 0
        # Pre-fill SPaTrack lengths from cache
        sp_len_map = {} if self.force_refresh_cache else read_spatrack_len_map(self.dataset_root)
        if sp_len_map:
            seen_vids = {}
            for rec in self.records:
                vid = rec["video_id"]
                if vid in seen_vids:
                    continue
                npz_path = rec["spatrack_npz"]
                st = os.stat(npz_path)
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                size = st.st_size
                seen_vids[vid] = (npz_path, mtime_ns, size)
                entry = sp_len_map.get(vid)
                if isinstance(entry, dict) and entry.get("npz") == npz_path and entry.get("mtime_ns") == mtime_ns and entry.get("size") == size:
                    self._video_len[vid] = entry.get("len", 0)

        for rec in tqdm(self.records, desc="Reading [cyan]SpaTrack headers[/cyan]", unit="vid", disable=not is_rank0()):
            vid = rec["video_id"]
            if vid not in self._video_len:
                self._video_len[vid] = get_spatrack_length(rec["spatrack_npz"])

        for rec in tqdm(self.records, desc="Building [cyan]windows[/cyan]", unit="obj", disable=not is_rank0()):
            vid = rec["video_id"]
            Tlen = self._video_len[vid]
            pose_txts = rec["pose_txts"]
            init_map = self._ensure_init_map(rec["object_dir"])
            fids_all = [_stem_to_fid(p) for p in pose_txts]
            fids = [k for k in fids_all if 0 <= k < Tlen]
            if len(fids) < len(fids_all) and not self._obj_warned_mismatch.get(rec["object_dir"], False):
                print(f"[yellow]mismatch[/yellow] {vid}/{rec['object_id']}_{rec['object_name']} • spatrack={Tlen} • obj={len(fids_all)} • using intersection={len(fids)}")
                self._obj_warned_mismatch[rec["object_dir"]] = True

            win_len = self.context_len + self.H
            step = self.frame_skips + 1
            span = (win_len - 1) * step + 1
            if len(fids) < span:
                continue
            stride = max(1, self.window_stride)
            before_total += 1 + (len(fids) - span) // stride

            for t0 in range(0, len(fids) - span + 1, self.window_stride):
                frame_ids = [fids[t0 + i * step] for i in range(win_len)]
                # Init-from-frame grouping
                if self.respect_init_from_frame and init_map:
                    group_vals = [init_map.get(f) for f in frame_ids]
                    if any(g is None for g in group_vals) or len(set(group_vals)) != 1:
                        continue
                # IoU drop filter
                if self.filter_iou_drop:
                    obj_dir = rec["object_dir"]
                    fid0 = frame_ids[0]
                    iou_map = self._ensure_iou_map(obj_dir)
                    baseline = iou_map.get(fid0)
                    if baseline is not None:
                        vals = [iou_map[f] for f in frame_ids if f in iou_map]
                        if vals and (baseline - min(vals)) > self.iou_drop_threshold:
                            continue
                    else:
                        m0 = self._load_mask_frame(obj_dir, fid0)
                        if self._mask_valid(m0):
                            min_iou = 1.0
                            for f in frame_ids[1:]:
                                mj = self._load_mask_frame(obj_dir, f)
                                if self._mask_valid(mj):
                                    min_iou = min(min_iou, self._mask_iou(m0, mj))
                                    if (1.0 - min_iou) > self.iou_drop_threshold:
                                        break
                            if (1.0 - min_iou) > self.iou_drop_threshold:
                                continue
                # Post-train IoU mean filter
                if self.post_train_mode and self.post_train_iou_mean_thresh > 0.0:
                    iou_map = self._ensure_iou_map(rec["object_dir"])
                    vals = [iou_map[f] for f in frame_ids if f in iou_map]
                    if not vals or np.mean(vals) < self.post_train_iou_mean_thresh:
                        dropped_iou_mean += 1
                        continue
                # Post-train translation filter
                if self.post_train_mode and self.post_train_min_t > 0.0:
                    pk = self._ensure_video_pack(vid, rec["spatrack_npz"])
                    T_c_w = pk["T_c_w"]
                    fids_obj, T_c_o_all = self._sorted_object_poses(rec["object_dir"])
                    map_idx = {int(f): i for i, f in enumerate(fids_obj.tolist())}
                    t_list = []
                    for f in frame_ids:
                        idx_local = map_idx.get(f)
                        if idx_local is None:
                            t_list = []
                            break
                        Tw_o = T_c_w[f].astype(np.float32) @ T_c_o_all[idx_local].astype(np.float32)
                        t_list.append(Tw_o[:3, 3])
                    if len(t_list) >= 2:
                        diffs = np.diff(np.stack(t_list, axis=0), axis=0)
                        if np.linalg.norm(diffs, axis=1).sum() < self.post_train_min_t:
                            dropped_trans += 1
                            continue

                wins.append(
                    {
                        "video_id": vid,
                        "object_id": rec["object_id"],
                        "object_name": rec["object_name"],
                        "frame_ids": frame_ids,
                        "rgb_path": rec["rgb_path"],
                        "spatrack_npz": rec["spatrack_npz"],
                        "mesh_path": rec["mesh_path"],
                        "object_dir": rec["object_dir"],
                    }
                )
                after_total += 1

        if before_total > 0:
            print(
                f"[dim]csv_filter[/dim] windows: before={before_total:,} "
                f"after={after_total:,} dropped={before_total - after_total:,} "
                f"keep={after_total / before_total:.2f}x • P={self.context_len} • H={self.H}"
            )
        if self.post_train_mode and self.post_train_iou_mean_thresh > 0.0 and before_total > 0:
            print(f"[dim]post_train[/dim] IoU-mean filter • thr={self.post_train_iou_mean_thresh:.3f} • dropped_windows={dropped_iou_mean}")
        if self.post_train_mode and self.post_train_min_t > 0.0 and before_total > 0:
            print(f"[dim]post_train[/dim] translation filter • min_t={self.post_train_min_t:.4f} m • dropped_windows={dropped_trans}")
        if self.respect_init_from_frame and self.verbose:
            print("[dim]windows[/dim] built with init_from_frame grouping enforced")

        # Persist SPaTrack length map
        uniq_vids = {}
        for rec in self.records:
            vid = rec["video_id"]
            if vid not in uniq_vids:
                uniq_vids[vid] = rec["spatrack_npz"]
        out_map = {}
        for vid, npz_path in uniq_vids.items():
            st = os.stat(npz_path)
            out_map[vid] = {
                "npz": npz_path,
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                "size": st.st_size,
                "len": self._video_len.get(vid, 0),
            }
        write_spatrack_len_map(self.dataset_root, out_map)
        if self.cache_index and self._windows_cache_key:
            write_windows_cache(self.dataset_root, self._windows_cache_key, wins)
        return wins

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        vid = win["video_id"]
        obj_dir = win["object_dir"]
        frame_ids: List[int] = win["frame_ids"]
        k0 = frame_ids[0]

        # Backward/robustness: older pickled dataset instances (or mismatched imports across workers)
        # may miss `overwrite_sample_cache`. Default to False.
        if self.use_sample_cache and (not getattr(self, "overwrite_sample_cache", False)) and self._sample_cache_dir is not None:
            cache_path = get_sample_path(Path(self._sample_cache_dir), vid, win["object_id"], k0)
            cached = load_sample(cache_path)
            if cached is not None:
                return cached

        P_ctx = self.context_len
        horizon_anchor_local = min(P_ctx, len(frame_ids) - 1)

        pk = self._ensure_video_pack(vid, win["spatrack_npz"])
        K = pk["K"].astype(np.float32) if pk.get("K") is not None else None
        T_c_w = pk["T_c_w"].astype(np.float32)
        has_depth = pk["has_depth"] and self.use_depth
        depth = pk["depth"] if has_depth else None

        fids_obj, T_c_o_all = self._sorted_object_poses(obj_dir)
        if self.rot_smooth_alpha > 0.0:
            T_c_o_all = self._smooth_rotations_ema(T_c_o_all, alpha=self.rot_smooth_alpha)
        map_idx = {int(f): i for i, f in enumerate(fids_obj.tolist())}
        idxs = [map_idx[f] for f in frame_ids]
        T_c_o = T_c_o_all[idxs]

        def _invert4x4(T: np.ndarray) -> np.ndarray:
            Rcw, tcw = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
            Rwc = Rcw.T
            Tinv = np.eye(4, dtype=np.float32)
            Tinv[:3, :3] = Rwc
            Tinv[:3, 3] = -Rwc @ tcw
            return Tinv

        conv = self._video_conv.get(vid)
        if conv is None:
            a = frame_ids[0]
            a_local_obj = {int(f): i for i, f in enumerate(fids_obj.tolist())}.get(a)
            if a_local_obj is not None:
                T_i0_gt = T_c_o_all[a_local_obj]
                conv, _, _ = select_extrinsics_convention(T_i0_gt, T_c_w[a], _invert4x4(T_c_w[a]), T_i0_gt)
            else:
                conv = "c2w"
            self._video_conv[vid] = conv
            if is_rank0():
                write_conv_map(self.dataset_root, self._video_conv)

        # Anchor index
        if self.anchor_mode == "window_start":
            a = frame_ids[horizon_anchor_local]
            a_local = horizon_anchor_local
        elif self.anchor_mode == "global_frame":
            a = int(np.clip(self.anchor_frame_idx, 0, T_c_w.shape[0] - 1))
            matches = np.where(np.array(frame_ids, dtype=np.int32) == a)[0]
            a_local = int(matches[0]) if len(matches) > 0 else -1
        else:
            a = frame_ids[0]
            a_local = 0

        # World poses
        if conv == "c2w":
            T_w_o = np.matmul(T_c_w[frame_ids], T_c_o)
            T_cam_a_w = _invert4x4(T_c_w[a])
        else:
            T_w_c = np.stack([_invert4x4(T_c_w[k]) for k in frame_ids], axis=0)
            T_w_o = np.matmul(T_w_c, T_c_o)
            T_cam_a_w = T_c_w[a]

        if self.anchor_mode == "absolute":
            T_cam_anchor_obj = T_c_o.astype(np.float32)
        else:
            T_cam_anchor_obj = np.stack([T_cam_a_w @ T_w_o[j] for j in range(len(frame_ids))], axis=0).astype(np.float32)

        T_c_w0 = T_c_w[k0]
        Rcw, tcw = T_c_w0[:3, :3], T_c_w0[:3, 3]
        Rwc = Rcw.T
        T_w_c0 = np.eye(4, dtype=np.float32)
        T_w_c0[:3, :3] = Rwc
        T_w_c0[:3, 3] = -Rwc @ tcw

        L_total = T_cam_anchor_obj.shape[0]

        if self.output_format == "abs_in_anchor":
            T_all = T_cam_anchor_obj
        elif self.output_format == "delta_from_prev":
            dT_prev_list = [np.eye(4, dtype=np.float32)]
            for j in range(1, L_total):
                dT_prev_list.append(invert_T(T_cam_anchor_obj[j - 1]) @ T_cam_anchor_obj[j])
            T_all = np.stack(dT_prev_list, axis=0).astype(np.float32)
        elif self.output_format == "delta_from_anchor":
            base_T = T_cam_anchor_obj[a_local] if 0 <= a_local < L_total else T_cam_anchor_obj[0]
            T_all = np.stack([invert_T(base_T) @ T for T in T_cam_anchor_obj], axis=0).astype(np.float32)
        else:
            T_w_o0 = T_w_o[0]
            R0, t0 = T_w_o0[:3, :3], T_w_o0[:3, 3]
            T_o0_w = np.eye(4, dtype=np.float32)
            T_o0_w[:3, :3] = R0.T
            T_o0_w[:3, 3] = -R0.T @ t0
            T_all = np.stack([(T_o0_w @ T_w_o[j]).astype(np.float32) for j in range(L_total)], axis=0)

        T_sel = T_all[P_ctx : P_ctx + self.H]
        t = T_sel[:, :3, 3]
        R = T_sel[:, :3, :3]
        r6 = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)
        target_future = np.concatenate([t, r6], axis=1).astype(np.float32)

        T_w_o0 = T_w_o[0]
        t0_vec = T_w_o0[:3, 3].astype(np.float32)
        R0 = T_w_o0[:3, :3]
        r6_0 = np.concatenate([R0[:, 0], R0[:, 1]], axis=0).astype(np.float32)

        depth_getter = None
        if has_depth:
            depth_getter = _DepthArrayGetter(depth) if depth is not None else _DepthFileGetter(win["spatrack_npz"], self)
        scene_pcd0 = maybe_build_scene_pointcloud(vid, has_depth, depth_getter, k0, K, T_c_w, self.n_points, verbose=self.verbose, downsample_cfg=self.downsample_cfg)
        if not isinstance(scene_pcd0, np.ndarray):
            scene_pcd0 = np.zeros((self.n_points, 3), dtype=np.float32)

        image0_path, image0_idx = None, None
        if self.load_rgb0:
            image0_path, image0_idx = win["rgb_path"], k0

        if K is not None and image0_path is not None:
            H_src_infer, W_src_infer = infer_image_size_from_K(K)
            H_dst, W_dst = 0, 0
            wh = get_video_frame_size(image0_path)
            if isinstance(wh, tuple):
                H_dst, W_dst = wh[0], wh[1]
            elif has_depth and depth is not None:
                H_dst, W_dst = depth.shape[1], depth.shape[2]
            if H_dst > 0 and W_dst > 0 and (H_src_infer != H_dst or W_src_infer != W_dst):
                K = scale_intrinsics(K, (H_src_infer, W_src_infer), (H_dst, W_dst)).astype(np.float32)

        # Load mask and compute bbox
        mask_used, mask_frame_used = None, None
        for k in [k0] + [k for k in frame_ids if k != k0]:
            m = self._load_mask_frame(obj_dir, k)
            if self._mask_valid(m):
                mask_used, mask_frame_used = m.astype(bool), k
                break
        if mask_used is None:
            wh = get_video_frame_size(win["rgb_path"])
            if isinstance(wh, tuple) and wh[0] > 0 and wh[1] > 0:
                mask_used = np.zeros((wh[0], wh[1]), dtype=bool)

        bbox_xyxy, bbox_norm = [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]
        if self._mask_valid(mask_used):
            Hm, Wm = mask_used.shape[:2]
            bb = self._bbox_from_mask_xyxy(mask_used)
            if bb is not None:
                x1, y1, x2, y2 = bb
                bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
                if Wm > 0 and Hm > 0:
                    bbox_norm = [x1 / Wm, y1 / Hm, x2 / Wm, y2 / Hm]

        sample = {
            "video_id": vid,
            "object_id": str(win["object_id"]),
            "K": K.astype(np.float32) if K is not None else None,
            "T_wc0": T_w_c0.astype(np.float32),
            "T_c_w0": T_c_w0.astype(np.float32),
            "scene_pcd": scene_pcd0.astype(np.float32) if isinstance(scene_pcd0, np.ndarray) else None,
            "init_pose": {"t0": t0_vec, "rot6d0": r6_0},
            "target_future": target_future,
            "mesh_path": win["mesh_path"],
            "object": {
                "bbox": np.array(bbox_xyxy, dtype=np.float32),
                "bbox_norm": np.array(bbox_norm, dtype=np.float32),
                "mask": mask_used.astype(np.bool_) if isinstance(mask_used, np.ndarray) else None,
                "mask_frame_idx": mask_frame_used if mask_frame_used is not None else k0,
            },
            "frame_ids": np.array(frame_ids, dtype=np.int32),
            "rgb_path": win["rgb_path"],
            "spatrack_npz": win["spatrack_npz"],
            "object_dir": win["object_dir"],
            "T_c_o": T_c_o.astype(np.float32),
            "T_c_w": T_c_w[frame_ids].astype(np.float32),
            "T_cam_anchor_obj": T_cam_anchor_obj.astype(np.float32),
            "anchor_mode": self.anchor_mode,
            "anchor_frame_idx": a,
            "anchor_local_idx": a_local,
            "extrinsics_convention": self._video_conv.get(vid, "c2w"),
        }

        if self.load_rgb0 and image0_path is not None:
            sample["image0_path"] = image0_path
            sample["image0_idx"] = image0_idx

        P_ctx2 = P_ctx + 1
        context_T_cam_anchor_obj = T_cam_anchor_obj[:P_ctx2]
        context_init_9d = _poses_to_9d(context_T_cam_anchor_obj)
        _context_bbox = []
        for j in range(P_ctx2):
            loaded_mask = self._load_mask_frame(obj_dir, frame_ids[j])
            mj = loaded_mask if loaded_mask is not None else mask_used
            if self._mask_valid(mj):
                Hm, Wm = mj.shape[:2]
                bbj = self._bbox_from_mask_xyxy(mj)
                if bbj is not None and Wm > 0 and Hm > 0:
                    x1, y1, x2, y2 = bbj
                    _context_bbox.append([x1 / Wm, y1 / Hm, x2 / Wm, y2 / Hm])
                    continue
            _context_bbox.append(list(bbox_norm))
        context_bbox_norm = np.asarray(_context_bbox, dtype=np.float32)

        sample.update(
            {
                "context_len": P_ctx2,
                "context_frame_ids": np.array(frame_ids[:P_ctx2], dtype=np.int32),
                "context_T_cam_anchor_obj": context_T_cam_anchor_obj.astype(np.float32),
                "context_init_9d": context_init_9d,
                "context_bbox_norm": context_bbox_norm,
            }
        )

        if self.verbose:
            key_once = f"{vid}:{win['object_id']}"
            if not hasattr(self, "_printed_loads"):
                self._printed_loads = set()
            if key_once not in self._printed_loads:
                self._printed_loads.add(key_once)
                depth_mark = "✓" if has_depth else "×"
                mesh_mark = "✓" if os.path.exists(win["mesh_path"]) else "×"
                print(f"[dim]load[/dim] {vid} • obj={win['object_id']}_{win['object_name']} • k0={k0} • mesh={mesh_mark} • depth={depth_mark}")

        if self.debug_checks and (not hasattr(self, "_dbg_count") or self._dbg_count < 3):
            self._dbg_count = getattr(self, "_dbg_count", 0) + 1
            conv_s = self._video_conv.get(vid, "?")
            err_anchor = 0.0
            if 0 <= a_local < len(frame_ids):
                fp_idx = {int(f): i for i, f in enumerate(fids_obj.tolist())}.get(a)
                if fp_idx is not None:
                    T_fp = T_c_o_all[fp_idx]
                    T_cmp = T_cam_anchor_obj[a_local]
                    err_anchor = float(np.linalg.norm(T_fp[:3, 3] - T_cmp[:3, 3]))
            print(f"[dim]dbg[/dim] {vid} conv={conv_s} a={a} • anchor_t_err={err_anchor:.3e}")

        if self.precompute_cache and self._sample_cache_dir is not None:
            cache_path = get_sample_path(Path(self._sample_cache_dir), vid, win["object_id"], k0)
            if getattr(self, "overwrite_sample_cache", False) or (not sample_exists(cache_path)):
                save_sample(cache_path, sample)

        return sample
