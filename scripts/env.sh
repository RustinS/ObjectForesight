#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[error] env.sh failed at line $LINENO with exit code $?" >&2' ERR
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[env] DEBUG=1 (trace on)"
  set -x
fi
echo "[env] Starting environment setup"

# Minimal environment bootstrap for Slurm jobs.
# - Restores module environment (if available)
# - Uses uv for Python environment management
# - Configures NCCL networking with safe auto-detection and clear override
# - Sets reproducible, sane runtime defaults

# ---------------
# Modules
# ---------------
if command -v module >/dev/null 2>&1; then
  echo "[env] 'module' command found; restoring default module set"
  module --quiet restore >/dev/null 2>&1 || true
else
  echo "[env] 'module' command not found; skipping module restore"
fi

# ---------------
# uv environment
# ---------------
# uv run automatically uses the project's .venv - no activation needed
# Just ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  # Try common install locations
  if [[ -f "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  elif [[ -f "$HOME/.cargo/bin/uv" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "[fatal] uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
fi
echo "[env] uv: $(command -v uv) ($(uv --version))"

# Avoid concurrent venv mutation when running on Slurm (multi-node / multi-rank).
# If you need to update deps, do it once ahead of time via `uv sync`.
UV_RUN=(uv run)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  UV_RUN=(uv run --no-sync --frozen)
  export UV_NO_PROGRESS=${UV_NO_PROGRESS:-1}
fi

# Verify project venv exists
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  echo "[fatal] .venv not found in ${PROJECT_ROOT}. Run ./scripts/setup.sh first." >&2
  exit 1
fi
echo "[env] Using venv at ${PROJECT_ROOT}/.venv"

# Show Python version via uv
echo "[env] python: $("${UV_RUN[@]}" python -c 'import sys; print(sys.executable)')"
"${UV_RUN[@]}" python -V || true

# ---------------
# NCCL / Runtime defaults
# ---------------
export NCCL_DEBUG=${NCCL_DEBUG:-warn}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

# Detect suitable network interface for NCCL
detect_nic_and_ib() {
  local nic=""
  local ib_disable=1

  if [[ -d /sys/class/infiniband ]]; then
    # Prefer Infiniband if present; common iface names: ib0, ib1
    # Note: \b word boundary not supported by all awk versions, use simpler pattern
    nic=$(ip -o link | awk -F': ' '/ib[0-9]+/{print $2; exit}') || true
    if [[ -n "$nic" ]]; then
      ib_disable=0
    fi
  fi

  if [[ -z "$nic" ]]; then
    # Fall back to first non-loopback interface
    nic=$(ip -o link | awk -F': ' '$2!="lo"{print $2; exit}') || true
  fi

  printf '%s %s' "${nic}" "${ib_disable}"
}

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  __detect_out="$(detect_nic_and_ib || true)"
  __nic="${__detect_out%% *}"
  __ib_disable="${__detect_out#* }"
  export NCCL_SOCKET_IFNAME="${__nic:-eth0}"
  export NCCL_IB_DISABLE="${__ib_disable:-1}"
else
  # Honor manual override
  if [[ -z "${NCCL_IB_DISABLE:-}" ]]; then
    export NCCL_IB_DISABLE=0
  fi
fi

echo "[env] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} (override by exporting before submit)"
echo "[env] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

# Reasonable thread/env defaults
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
echo "[env] OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# Precision defaults (training can still override via CLI/env)
export DEFAULT_PRECISION=${DEFAULT_PRECISION:-bf16}
export PRECISION=${PRECISION:-${DEFAULT_PRECISION}}
echo "[env] PRECISION=${PRECISION}"

export IBV_FORK_SAFE=1

# Disable JIT compilation for spconv/cumm (more stable in multi-node DDP)
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
echo "[env] CUMM_DISABLE_JIT=1, SPCONV_DISABLE_JIT=1"

# PyTorch CUDA allocator tuning (helps avoid fragmentation on long runs)
# PyTorch's OOM guidance recommends PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
# Keep the newer name (if supported) in sync as well.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"${PYTORCH_CUDA_ALLOC_CONF}"}

# Suppress Python warnings in logs by default (can override by exporting PYTHONWARNINGS)
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}

# Debug: CUDA launch blocking for precise error location (set CUDA_DEBUG=1 to enable)
if [[ "${CUDA_DEBUG:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export HYDRA_FULL_ERROR=1
  echo "[env] CUDA_DEBUG=1: CUDA_LAUNCH_BLOCKING=1, HYDRA_FULL_ERROR=1"
fi

# PTv3 index validation debug mode (set PTV3_DEBUG=1 to enable backward index checks)
if [[ "${PTV3_DEBUG:-0}" == "1" ]]; then
  export PTV3_ASSERT_BACKWARD=1
  export PTV3_ASSERT_INDICES=1
  export GRAD_NAN_DEBUG=1
  export CUDA_LAUNCH_BLOCKING=1
  export HYDRA_FULL_ERROR=1
  echo "[env] PTV3_DEBUG=1: Backward index validation + NaN grad detection + CUDA sync"
fi
