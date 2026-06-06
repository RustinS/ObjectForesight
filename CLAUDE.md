# CLAUDE.md

Future 6-DoF pose prediction using PyTorch + Hydra. Predicts H future object poses from scene point clouds.

## Environment

**CRITICAL**: All Python code must be run on GPU allocations. Use `srun` before running any commands:
```bash
srun --gpus=1 --time=2:00:00 --pty bash
# Then run commands in the allocated shell
```

**Running Commands**: Use `uv run` (no activation needed):
```bash
uv run python -m src.train_main
```

**Setup**: On GPU node, run `./scripts/setup.sh`

## Common Commands

```bash
# Training (uses conf/debug.yaml by default)
uv run python -m src.train_main
uv run python -m src.train_main train.resume=latest  # Resume

# Multi-GPU
uv run torchrun --standalone --nproc_per_node=8 -m src.train_main
bash scripts/submit.sh --job-name experiment  # Slurm cluster

# Evaluation & Inference
uv run python -m src.eval_main eval.ckpt=./outputs/checkpoints/best.pt
uv run python -m src.infer_main infer.ckpt=./outputs/checkpoints/last.pt

# Debug/test
uv run python -m src.train_main train.tiny_overfit=true train.tiny_n=50
uv run python -m src.train_main data.dataset_name=synth data.num_samples=16

# Code formatting (line width 175)
uv run ruff format . && uv run ruff check --fix .
```

## Configuration

**Primary config: `conf/debug.yaml`** - used for most training runs.

Other configs in `conf/`:
- `default.yaml`: Base config with all defaults
- `p10_eval.yaml`, `p3_h32_eval.yaml`: Evaluation configs

### Key Config Sections

**Data**: `dataset_name` (epic/synth), `H` (horizon), `context_len`, `frame_skips`, `n_points`

**Model**: `temporal_kind` (ar_transformer/dit), `encoder`, `temporal_ar`, `temporal_dit`, `w_rot`, `w_trans`

**Training**: `batch_size`, `lr`, `epochs`, `amp`, `ema`, `lr_schedule`, `eval_every_steps`, `save_every`

**Eval**: `eval_mode` (loss/sampler), `steps` (DDIM), `compute_canon_metrics`

## Architecture

### Key Components

**Models** (`src/models/poser_v1/`):
- `model.py`: Core PoserV1 with AR/DiT backends
- `builder.py`: Model instantiation via Hydra
- `interfaces.py`: Batch contracts and types

**Encoders** (`src/encoders/`):
- `ptv3_adapter.py`: PointTransformer V3 adapter (primary)
- `ptv3_model.py`: Raw PTv3 implementation

**Temporal** (`src/temporal/`):
- `ar_transformer.py`: Autoregressive transformer
- `dit.py`: Diffusion transformer with DDIM
- `diffusion.py`: Beta schedules, q_sample, ddim_sample

**Data** (`src/data/`):
- `dataset.py`: SceneSequenceDataset for EPIC/3DManip
- `pointcloud.py`: Point cloud from depth
- `fpose_io.py`: FoundationPose loading

**Geometry** (`src/geom/`, `src/utils/se3*.py`):
- `canonicalize.py`: Single source of truth for pose canonicalization
- SE(3) transformations, 6D rotation, geodesic distance

## Pipeline Contract

### Batch Contract (Training/Inference)

Required keys per item:
- `scene_pcd`: float32 [N,3] → batched to [B,N,3]
- `init_pose`: dict with `t0` [3], `rot6d0` [6]
- `target_future`: float32 [H,9] or [B,H,9]

Context sequence (P frames):
- `context_len`: int P
- `context_frame_ids`: int32 [P]
- `context_T_cam_anchor_obj`: float32 [P,4,4]
- `context_init_9d`: float32 [P,9]
- `context_bbox_norm`: float32 [P,4]

Canonicalization/camera meta (window of length P+H):
- `frame_ids`: int32 [P+H]
- `T_c_w`: float32 [P+H,4,4] (camera←world if extrinsics_convention=="c2w")
- `T_c_o`: float32 [P+H,4,4] (camera←object, FoundationPose track)
- `T_cam_anchor_obj`: float32 [P+H,4,4] (object poses in anchor camera)
- `anchor_mode`: str, usually "window_start"
- `anchor_frame_idx`: int (global)
- `anchor_local_idx`: int (window-local index of dataset anchor)
- `extrinsics_convention`: "c2w" or "w2c"

### Token Format (9D)

All inputs/outputs: `[t_x, t_y, t_z, rot6d(6)]`
- Rotation is 6D; projection to SO(3) via `rot6d_to_matrix`
- Model inputs/outputs: `(B,H,9)`

### Temporal Contract and Indices

- `P`: context length (`context_len`)
- `H`: prediction horizon (`cfg.data.H`)
- Dataset anchor: `aLoc = sample['anchor_local_idx']`
- Visualization anchor (last context): `anchor_idx = aLoc + (P - 1)`
- First predicted frame (window-local): `start = anchor_idx + 1`
- Predicted range: `[start, stop)` where `stop = start + H`
- GT slice for metrics: `T_cam_anchor_obj[start:stop]` shape `[H,4,4]`

### Canonicalization

Single source of truth: `geom.canonicalize.canonicalize_preds_to_anchor(...)`
- Call with `do_denorm=True`
- Pass `t_mean=[0,0,0]`, `t_std=[t_scale, t_scale, t_scale]`
- For DiT/AR, translations are metric; default `t_scale=1.0`
- **Never re-scale translations outside this call** (prevents double scaling)

Supported pred modes: `"abs_in_anchor_cam"`, `"deltas_from_prev_cam"`, `"deltas_from_anchor_cam"`, `"relative_world_from_o0"`

### Train vs Infer vs Viz Usage

**Training** (`train_main.py`):
- Uses dataset anchor `aLoc` for canonicalization within metrics
- Slices GT with `aLoc + P : aLoc + P + H`

**Inference** (`infer_main.py`):
- Uses `aLoc` when canonicalizing for metrics
- `anchor_idx = aLoc + (P - 1)`, `start = anchor_idx + 1`, `stop = start + H`
- Saves `pred_meta*.json` with: frames, `t_scale`, `ctx_len`, `anchor_idx`, range, `H`, `pred_mode`

**Visualization** (`viz_main.py`):
- `sample_v['anchor_local_idx'] = anchor_idx`
- Canonicalize via `canonicalize_preds_to_anchor(..., do_denorm=True)` using `t_scale` from meta
- Slice GT identically: `T_cam_anchor_obj[start:stop]`

### HOT3D Depth

Use pinhole-undistorted depth cache (`depth_cache_pinhole/`). NPZ format:
- `depths`: [T, H, W] float32
- `K_pinhole`: [3, 3] float32 intrinsics
- `orig_h`, `orig_w`: Original dimensions

## Development

### Debugging
1. Use debug config with tiny overfit: `train.tiny_overfit=true train.tiny_n=50`
2. Enable anomaly detection: `train.anomaly_detection=true`
3. Check tensor shapes at failure points
4. For NaNs: check learning rate, gradient norms, data normalization

### Multi-GPU / Slurm

```bash
# Single-node multi-GPU
uv run torchrun --standalone --nproc_per_node=8 -m src.train_main

# Slurm cluster
./scripts/submit.sh --nodes 1 --gpus-per-node 8
./scripts/submit.sh --nodes 2 --gpus-per-node 8  # Multi-node
```

## Constraints

- **GPU access**: Use `srun` for GPU allocation before running any Python code
- **Bash timeouts**: Set 5+ min for Python commands (training/eval/inference)
- **No destructive ops**: Never broad `rm -rf` on outputs/checkpoints/data
- **Determinism**: Set random seeds in new scripts
- **Dependencies**: Update `pyproject.toml` then `uv sync`
- **Canonicalization**: Never re-scale translations outside `canonicalize_preds_to_anchor`
