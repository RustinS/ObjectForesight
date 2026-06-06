#!/usr/bin/env python3
"""Find samples that cause CUDA scatter/gather index errors.

Uses the exact same data loading as `src/train_main.py` to reproduce training conditions.

Examples:
    # Scan a contiguous range (fast-ish, then bisect to exact samples)
    uv run python scripts/find_bad_samples.py start=0 end=10000 batch_size=128 output=bad_samples.txt

    # Test an explicit set of train indices (one per line / comma-separated)
    uv run python scripts/find_bad_samples.py indices_file=./outputs/debug/bad_batch_indices.txt do_backward=true
"""

import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from src.utils.config_adapter import apply_config_adapter
from src.utils.data_utils import build_train_val_datasets, get_dataset
from src.utils.train_utils import debug_collate
from tqdm import tqdm


def _parse_indices_text(text: str) -> list[int]:
    tokens: list[str] = []
    for line in text.splitlines():
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        # Allow either one index per line, or comma/space-separated lists.
        for raw in line_s.replace(",", " ").split():
            s = raw.strip()
            if not s:
                continue
            tokens.append(s)
    return [int(t) for t in tokens]


def _load_indices(indices_csv: str | None, indices_file: str | None) -> list[int]:
    indices: list[int] = []
    if indices_file:
        p = Path(indices_file)
        if not p.exists():
            raise FileNotFoundError(f"indices_file does not exist: {p}")
        indices.extend(_parse_indices_text(p.read_text()))
    if indices_csv:
        indices.extend(_parse_indices_text(indices_csv))
    # Deduplicate while preserving order
    out: list[int] = []
    seen: set[int] = set()
    for i in indices:
        if i in seen:
            continue
        out.append(i)
        seen.add(i)
    return out


def test_sample_through_model(model, batch, device, sample_indices, *, do_backward: bool, repeats: int = 1):
    """Test a batch through the full model. Returns error message if fails."""
    try:
        # Move batch to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v

        # Synchronize before forward
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for _ in range(int(repeats)):
            if do_backward:
                model.train()
                model.zero_grad(set_to_none=True)
                out = model(batch_gpu)
                loss = out.get("loss") if isinstance(out, dict) else None
                if loss is None:
                    raise RuntimeError("do_backward=true requires model(batch) to return a dict with key 'loss'.")
                loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                model.eval()
                # Run forward pass (same as training)
                with torch.no_grad():
                    _ = model(batch_gpu)

            # Synchronize after forward/backward to surface async CUDA asserts as early as possible.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return None  # Success

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        return f"Samples {sample_indices}: {type(e).__name__}: {str(e)[:300]}\n{tb[-500:]}"


def test_single_sample(model, sample, device, sample_idx, *, do_backward: bool, repeats: int = 1):
    """Test a single sample through the model."""
    try:
        # Collate single sample into batch
        batch = debug_collate([sample])
        return test_sample_through_model(model, batch, device, [sample_idx], do_backward=do_backward, repeats=repeats)
    except Exception as e:
        return f"Sample {sample_idx}: collate failed: {type(e).__name__}: {str(e)[:200]}"


@hydra.main(config_path="../conf", config_name="debug", version_base=None)
def main(cfg: DictConfig) -> None:
    # Get scan parameters from config (set via command line overrides)
    start_idx = int(cfg.get("start", 0))
    end_idx = cfg.get("end", None)  # None means all
    batch_size = int(cfg.get("batch_size", 128))
    output_file = str(cfg.get("output", "bad_samples.txt"))
    indices_csv = cfg.get("indices_csv", None)
    indices_file = cfg.get("indices_file", None)
    do_backward = bool(cfg.get("do_backward", False))
    indices_batch_size = int(cfg.get("indices_batch_size", 1))
    repeats = int(cfg.get("repeats", 1))
    seed = int(cfg.get("seed", int(getattr(cfg.train, "seed", 42))))

    # Suppress warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"do_backward={do_backward}")
    print(f"repeats={repeats}")
    print(f"seed={seed}")

    # Match train_main.py-style seeding (important for reproducing shuffle_orders / dropout-dependent crashes).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set CUDA to synchronous mode for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Register eval resolver if not present
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    # Apply config adapter (same as train_main.py) - builds cfg.temporal from temporal_ar/temporal_dit
    cfg = apply_config_adapter(cfg)

    # Load dataset exactly as in train_main.py
    print("Loading dataset (same as train_main.py)...")
    dataset = get_dataset(cfg)
    print(f"Base dataset size: {len(dataset)}")

    # Build train/val split (same as train_main.py)
    seed = int(cfg.train.seed)
    rank = 0  # Single GPU
    train_dataset, val_dataset = build_train_val_datasets(dataset, cfg, seed, rank)
    print(f"Train dataset size: {len(train_dataset)}")

    # Build model exactly as in train_main.py
    print("Building model...")
    from hydra.utils import instantiate

    model = instantiate(cfg.model, _recursive_=False)
    model.to(device)
    model.eval()

    explicit_indices = _load_indices(indices_csv, indices_file) if (indices_csv or indices_file) else []

    # Try to load checkpoint if available (to match training state)
    ckpt_path = Path(cfg.train.out_dir) / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        print("Checkpoint loaded")

    # Set end index
    if end_idx is None:
        end_idx = len(train_dataset)
    else:
        end_idx = int(end_idx)

    bad_samples = []

    if explicit_indices:
        print(f"\nTesting explicit indices (n={len(explicit_indices)})...")
        print(f"indices_batch_size={indices_batch_size}")
    else:
        print(f"\nTesting samples {start_idx} to {end_idx}...")
        print(f"Batch size: {batch_size}")

    if explicit_indices:
        if indices_batch_size <= 1:
            for i in tqdm(explicit_indices, desc="Testing indices"):
                try:
                    sample = train_dataset[i]
                except Exception as e:
                    bad_samples.append(f"Sample {i}: failed to load: {type(e).__name__}: {str(e)[:200]}")
                    tqdm.write(f"LOAD ERROR: Sample {i}")
                    continue

                error = test_single_sample(model, sample, device, i, do_backward=do_backward, repeats=repeats)
                if error:
                    bad_samples.append(error)
                    tqdm.write(f"FOUND: Sample {i}")
        else:
            # Test indices in chunks, to reproduce batch-only failures (e.g., variable point counts per sample).
            for chunk_start in tqdm(range(0, len(explicit_indices), indices_batch_size), desc="Testing index batches"):
                chunk = explicit_indices[chunk_start : chunk_start + indices_batch_size]
                samples = []
                load_failed = False
                for i in chunk:
                    try:
                        samples.append(train_dataset[i])
                    except Exception as e:
                        load_failed = True
                        bad_samples.append(f"Sample {i}: failed to load: {type(e).__name__}: {str(e)[:200]}")
                        tqdm.write(f"LOAD ERROR: Sample {i}")
                if load_failed:
                    continue

                batch = debug_collate(samples)
                error = test_sample_through_model(model, batch, device, chunk, do_backward=do_backward, repeats=repeats)
                if error:
                    bad_samples.append(error)
                    tqdm.write(f"FOUND: Batch {chunk[0]}-{chunk[-1]} (size={len(chunk)})")
                    # Re-test individually to pinpoint
                    for i in chunk:
                        try:
                            sample = train_dataset[i]
                            single_error = test_single_sample(model, sample, device, i, do_backward=do_backward, repeats=repeats)
                            if single_error:
                                tqdm.write(f"  -> Sample {i} is bad")
                        except Exception:
                            tqdm.write(f"  -> Sample {i} failed to load")
    else:
        if batch_size == 1:
            # Test individual samples
            for i in tqdm(range(start_idx, end_idx), desc="Testing samples"):
                try:
                    sample = train_dataset[i]
                except Exception as e:
                    bad_samples.append(f"Sample {i}: failed to load: {type(e).__name__}: {str(e)[:200]}")
                    tqdm.write(f"LOAD ERROR: Sample {i}")
                    continue

                error = test_single_sample(model, sample, device, i, do_backward=do_backward, repeats=repeats)
                if error:
                    bad_samples.append(error)
                    tqdm.write(f"FOUND: Sample {i}")
        else:
            # Test in batches (faster but less precise error location)
            from torch.utils.data import DataLoader, Subset

            subset = Subset(train_dataset, list(range(start_idx, end_idx)))
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Single-threaded for debugging
                collate_fn=debug_collate,
            )
            batch_start = start_idx
            for batch in tqdm(loader, desc="Testing batches"):
                batch_end = min(batch_start + batch_size, end_idx)
                indices = list(range(batch_start, batch_end))
                error = test_sample_through_model(model, batch, device, indices, do_backward=do_backward, repeats=repeats)
                if error:
                    bad_samples.append(error)
                    tqdm.write(f"FOUND: Batch {indices[0]}-{indices[-1]}")
                    # Re-test individually to find exact sample
                    for i in indices:
                        try:
                            sample = train_dataset[i]
                            single_error = test_single_sample(model, sample, device, i, do_backward=do_backward, repeats=repeats)
                            if single_error:
                                tqdm.write(f"  -> Sample {i} is bad")
                        except Exception:
                            tqdm.write(f"  -> Sample {i} failed to load")
                batch_start = batch_end

    # Write results
    print(f"\nFound {len(bad_samples)} problematic samples/batches")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Problematic samples found: {len(bad_samples)}\n")
        f.write(f"# Range tested: {start_idx} to {end_idx}\n")
        f.write(f"# Dataset: {cfg.data.dataset_name}\n")
        f.write(f"# Dataset root: {cfg.data.dataset_root}\n\n")
        for s in bad_samples:
            f.write(s + "\n\n")

    print(f"Results written to {output_path}")

    # Print summary
    if bad_samples:
        print("\nProblematic samples (first 10):")
        for s in bad_samples[:10]:
            print(f"  {s[:200]}...")
        if len(bad_samples) > 10:
            print(f"  ... and {len(bad_samples) - 10} more")
    else:
        print("\nNo problematic samples found in tested range!")


if __name__ == "__main__":
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    main()
