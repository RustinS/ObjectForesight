## Purpose

This file defines project-specific AI instructions for the future pose prediction / 6D object pose forecasting codebase in this repository. It is intended only for this project’s training, inference, evaluation, and visualization workflows.

## Tool calls and environment

- VERY IMPORTANT!! When using the bash tool, set a timeout AT LEAST 5 minutes.
- This project uses `uv` for environment management. No activation needed - use `uv run` to execute commands.
- Assume all commands are run from the repo root unless explicitly stated otherwise.

**Common entrypoints (run from repo root):**

```bash
uv run python -m src.train_main     # training
uv run python -m src.eval_main      # evaluation
uv run python -m src.infer_main     # inference
uv run python -m src.viz_main       # visualization overlays
uv run python -m src.viz_gather     # gather trajectories for viz
```

For long-running jobs (training, large eval, bulk inference):

- Prefer commands that are easy to copy-paste into a job script or scheduler.
- Avoid suggesting commands that assume interactive GPUs on login nodes.

---

## Overall structure

- `/src` follows a Python package structure (models, data, geometry, utils, training/eval/inference/viz entrypoints).
- `/conf` contains Hydra configs (datasets, models, training/eval settings, debug configs, etc.).
- `/outputs` contains training/eval/inference artifacts (checkpoints, logs, metrics JSON, predictions, viz assets).

When suggesting changes:

- Respect existing boundaries:
  - Data loading and windowing in `src/data/`.
  - Model definitions in `src/models/`.
  - Geometry and SE(3) utilities in `src/geom/` and `src/utils/se3*.py`.
  - Training/evaluation/inference logic in `src/train_main.py`, `src/eval_main.py`, `src/infer_main.py`, and `src/viz_*.py`.

---

## Code conventions

- All Python code must be formatted with ruff with line width 175. Always format.
- Prefer PyTorch tensors over NumPy arrays in core code paths.
- Keep functions focused and composable:
  - Avoid huge functions that mix data I/O, model logic, and visualization.
  - Prefer small helpers wired together in the main entrypoints.
- Use relative imports within the library or package (`src/...`).
- Keep imports clean:
  - Remove unused imports.
  - Group standard library, third-party, and local imports logically.

When modifying existing functionality:

- Prefer local, minimal diffs that preserve behavior unless explicitly refactoring.
- Clearly annotate any behavioral changes in comments or docstrings, especially if they affect metrics, datasets, or canonicalization.

---

## Tooling and dependency conventions

- Use `uv run python -m src.<module>` style entrypoints instead of running module files directly.
- For dependencies:
  - Add any new Python dependencies to `pyproject.toml`, not just to the environment manually.
  - Run `uv sync` after modifying dependencies.
  - Keep training, evaluation, and inference reproducible by avoiding ad hoc local-only dependencies.

Logging and reproducibility:

- Prefer deterministic behavior where possible:
  - Set random seeds consistently (PyTorch, NumPy, Python, and so on) when suggesting new scripts.
- Ensure that key hyperparameters (learning rate, batch size, dataset name, temporal backend, and so on) are configurable via Hydra and not hard-coded into Python files.

---

## Hydra and config best practices

- When adding new features that require settings:
  - Wire them through Hydra configs under `conf/` (for example `conf/debug.yaml`, model, data, and train sub-configs).
  - Document any new config fields with short inline comments or docstrings.
- When updating configs:
  - Explain clearly which keys to change and in which file (for example `conf/debug.yaml: data.dataset_name`).
  - Avoid duplicating configs when a parameterized variant can express the change.

---

## Miscellaneous good practices

- Be explicit about file paths and functions:
  - Reference paths like `src/models/poser_v1/model.py::PoserV1.compute_loss` when suggesting edits.
- When debugging:
  - For shape mismatches, print tensor shapes at the point of failure rather than guessing.
  - For NaNs or divergence, check learning rate, gradient norms, and data normalization first.
- When working with cloud paths such as `s3://` or `gs://`:
  - Prefer omitting trailing slashes unless the existing code expects them.
- Avoid destructive shell operations:
  - Do not suggest broad `rm -rf` commands (especially on `/`, `~`, or `outputs/` without filters).
- When proposing new visualizations or debugging plots:
  - Keep plotting helpers in dedicated modules (for example `src/plot_render.py` or `src/utils/viz.py`) instead of scattering ad hoc plotting across the codebase.

The goal of these instructions is to keep the codebase consistent, safe, and easy to reason about while you iterate on training, evaluation, inference, and visualization for future pose prediction and 6D object pose forecasting.
