"""Find which dataset windows are still missing from cache."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATA_PATH_REMAP", "")
os.environ.setdefault("SAMPLE_CACHE_TRUST_INDEX", "1")
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.sample_cache import make_sample_cache_key
from src.utils.config_adapter import apply_config_adapter
from src.utils.data_utils import get_dataset

@hydra.main(config_path="../conf", config_name="debug", version_base=None)
def main(cfg: DictConfig) -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))
    cfg = apply_config_adapter(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.data.use_sample_cache = True
    ds = get_dataset(cfg)
    cache_key = make_sample_cache_key(cfg)
    cache_dir = Path(ds.dataset_root) / ".sample_cache" / cache_key
    have = set()
    for f in cache_dir.iterdir():
        if f.suffix == ".lz4":
            stem = f.stem.removesuffix(".pt")
            try:
                vid, obj, k0 = stem.rsplit("_", 2)
                have.add((vid, obj, int(k0)))
            except ValueError:
                pass
    print(f"cache files: {len(have):,}")
    missing = []
    for w in ds.windows:
        key = (str(w.get("video_id", "")), str(w.get("object_id", "")), int(w["frame_ids"][0]))
        if key not in have:
            missing.append((key, w.get("spatrack_npz", "")))
    print(f"missing windows: {len(missing)}")
    for k, p in missing[:10]:
        print(f"  {k}  ({p})")

if __name__ == "__main__":
    main()
