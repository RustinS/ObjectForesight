"""Verify the new precomputed cache files are picked up by training without
falling through to raw-data loaders. Loads a few NEW windows (post-snapshot)
through the dataset and checks the sample-cache short-circuit fires."""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO = "."
sys.path.insert(0, REPO)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.datasets import SceneSequenceDataset
from src.data.sample_cache import get_sample_path, load_sample, make_sample_cache_key
from src.utils.config_adapter import apply_config_adapter


@hydra.main(config_path="../conf", config_name="debug", version_base=None)
def main(cfg: DictConfig) -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))
    cfg = apply_config_adapter(cfg)

    # Force use_sample_cache=True for this check
    OmegaConf.set_struct(cfg, False)
    cfg.data.use_sample_cache = True

    # Build dataset (window pkl will load full 2.07M list because precompute_cache=False here)
    from src.utils.data_utils import get_dataset
    ds = get_dataset(cfg)
    print(f"[total] {len(ds):,} windows")

    cache_key = make_sample_cache_key(cfg)
    cache_dir = Path(ds.dataset_root) / ".sample_cache" / cache_key
    print(f"[cache_dir] {cache_dir}")

    # Pick 5 NEW samples (written after snapshot — files mtime > 2026-05-06 03:00)
    import time
    snapshot_cutoff_unix = time.mktime(time.strptime("2026-05-06 03:18:53", "%Y-%m-%d %H:%M:%S"))
    new_files = []
    for i, p in enumerate(cache_dir.glob("*.pt.lz4")):
        if p.stat().st_mtime > snapshot_cutoff_unix:
            new_files.append(p)
        if len(new_files) >= 1000:
            break
    print(f"[new files] {len(new_files):,} candidate post-job samples (sampled subset)")

    # Pick samples whose (vid, obj, k0) tuple matches a window
    # cache filename: {vid}_{obj_id}_{k0}.pt.lz4
    by_key = {}
    for p in new_files:
        stem = p.stem.removesuffix(".pt")
        try:
            vid, obj, k0 = stem.rsplit("_", 2)
            by_key[(vid, obj, int(k0))] = p
        except ValueError:
            continue

    matched_indices = []
    for i, w in enumerate(ds.windows):
        key = (str(w.get("video_id", "")), str(w.get("object_id", "")), int(w["frame_ids"][0]))
        if key in by_key:
            matched_indices.append((i, key, by_key[key]))
        if len(matched_indices) >= 5:
            break

    print(f"[matched] {len(matched_indices)} new-cache windows mapped to dataset indices")
    print()

    for idx, key, p in matched_indices:
        # Direct cache load (control)
        direct = load_sample(p, validate=True)
        if direct is None:
            print(f"[FAIL] direct load returned None for {p.name}")
            continue
        # Now load via dataset.__getitem__ — should hit the cache short-circuit
        sample = ds[idx]
        # Compare: tensors should be equal
        import torch, numpy as np
        ok = True
        for k in ("scene_pcd", "target_future"):
            d, s = direct.get(k), sample.get(k)
            if d is None or s is None:
                ok = False; print(f"  [FAIL] {p.name}: missing {k}")
                continue
            if isinstance(d, np.ndarray): d = torch.from_numpy(d)
            if isinstance(s, np.ndarray): s = torch.from_numpy(s)
            if not torch.allclose(d.float(), s.float(), atol=1e-5):
                ok = False; print(f"  [FAIL] {p.name}: {k} differs (cache vs __getitem__)")
                continue
        status = "OK" if ok else "FAIL"
        print(f"[{status}] idx={idx} {key} {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
