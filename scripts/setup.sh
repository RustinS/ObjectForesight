#!/usr/bin/env bash
# setup.sh - One-command environment setup for future-pose-pred
# Usage: ./scripts/setup.sh [--skip-gpu] [--cuda-arch 90]
#
# Requirements:
#   - CUDA 12.x installed and in PATH (nvcc available)
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Git installed
#   - Run on a GPU node for GPU package builds

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CUDA_ARCH="${CUDA_ARCH:-90}"  # Default: H100/H200 (sm_90)
MAX_JOBS="${MAX_JOBS:-8}"
SKIP_GPU="${SKIP_GPU:-false}"
EXT_DIR="${EXT_DIR:-.ext}"    # Directory for cloned extensions

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
error() { echo -e "${RED}[error]${NC} $1"; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gpu) SKIP_GPU=true; shift ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --ext-dir) EXT_DIR="$2"; shift 2 ;;
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-gpu       Skip GPU packages (for CPU-only development)"
            echo "  --cuda-arch N    CUDA architecture (default: 90 for H100/H200)"
            echo "  --ext-dir DIR    Directory for cloned extensions (default: .ext)"
            echo "  --max-jobs N     Max parallel build jobs (default: 8)"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *) error "Unknown option: $1. Use --help for usage." ;;
    esac
done

log "Setting up future-pose-pred environment..."
log "Project root: $PROJECT_ROOT"

# Detect CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
    CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
    log "Found CUDA $CUDA_VERSION at $CUDA_HOME"
else
    warn "nvcc not found. GPU packages will fail to build."
    if [ "$SKIP_GPU" = false ]; then
        error "Set --skip-gpu to continue without GPU packages, or install CUDA first."
    fi
fi

# Check uv
if ! command -v uv &> /dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "Using uv $(uv --version)"

# Clean up old build artifacts
log "Cleaning up old build artifacts..."
rm -rf .venv
rm -rf uv.lock
rm -rf src/*.egg-info
rm -rf src/**/*.egg-info
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Create virtual environment
log "Creating virtual environment..."
uv venv --python 3.11

# Install base dependencies and project
log "Installing dependencies and project..."
uv sync

if [ "$SKIP_GPU" = true ]; then
    log "Skipping GPU packages (--skip-gpu specified)"
    # Verify base installation
    source .venv/bin/activate
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    python -c "import src; print('src module OK')" || error "src module not importable"
    log ""
    log "Setup complete (CPU-only)!"
    log "  Activate:  source .venv/bin/activate"
    log "  Run code:  uv run python -m src.train_main"
    log ""
    exit 0
fi

# Set build environment
export CUDA_HOME
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
export MAX_JOBS="$MAX_JOBS"
export FORCE_CUDA=1
# Convert CUDA arch to PyTorch format (90 -> 9.0)
TORCH_ARCH="${CUDA_ARCH:0:1}.${CUDA_ARCH:1}"
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"

log "Build environment:"
log "  CUDA_HOME=$CUDA_HOME"
log "  CUDA_ARCH=$CUDA_ARCH (TORCH_CUDA_ARCH_LIST=$TORCH_ARCH)"
log "  MAX_JOBS=$MAX_JOBS"

# Install GPU packages from PyPI
log "Installing GPU packages (torch-scatter, flash-attn, pytorch3d)..."
log "This may take 20-30 minutes for building from source..."

# torch-scatter
log "Building torch-scatter..."
uv pip install torch-scatter --no-build-isolation || warn "torch-scatter failed, continuing..."

# flash-attn - force build from source (no prebuilt wheels)
log "Building flash-attn from source (CUDA arch: $CUDA_ARCH)..."
uv pip install flash-attn --no-build-isolation --no-binary flash-attn -v || warn "flash-attn failed, continuing..."

# pytorch3d - install from GitHub for better compatibility
log "Building pytorch3d..."
uv pip install "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation || warn "pytorch3d failed, continuing..."

# spconv/cumm - build from source with patched CUDA path detection
# cumm uses CUMM_CUDA_ARCH_LIST for JIT compilation at runtime
export CUMM_CUDA_ARCH_LIST="$TORCH_ARCH"

log "Setting up extensions directory: $EXT_DIR"
mkdir -p "$EXT_DIR"

# Disable GUI credential prompts for headless servers
unset GIT_ASKPASS
unset SSH_ASKPASS
export GIT_TERMINAL_PROMPT=0

# Clone cumm
if [ ! -d "$EXT_DIR/cumm" ]; then
    log "Cloning cumm..."
    git clone --depth 1 https://github.com/FindDefinition/cumm.git "$EXT_DIR/cumm"
fi

# Patch cumm to fix CUDA path detection and set default arch
log "Patching cumm for CUDA path detection and default arch ($TORCH_ARCH)..."
CUMM_COMMON="$EXT_DIR/cumm/cumm/common.py"
if grep -q 'lib = Path(nvcc_path).parent.parent / "lib"' "$CUMM_COMMON"; then
    sed -i 's|lib = Path(nvcc_path).parent.parent / "lib"|cuda_root = Path(nvcc_path).parent.parent\n                # Try lib64 first (standard CUDA toolkit), then lib (conda)\n                lib = cuda_root / "lib64" if (cuda_root / "lib64").exists() else cuda_root / "lib"|' "$CUMM_COMMON"
    sed -i 's|include = Path(nvcc_path).parent.parent / "targets/x86_64-linux/include"|include = cuda_root / "targets/x86_64-linux/include"\n                if not include.exists():\n                    include = cuda_root / "include"|' "$CUMM_COMMON"
    sed -i 's|linux_cuda_root = Path("/usr/local/cuda")|linux_cuda_root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))|' "$CUMM_COMMON"
    log "cumm patched successfully"
else
    log "cumm already patched or patch not needed"
fi
# Patch default CUDA arch so env var isn't required at runtime
if grep -q "os.getenv('CUMM_CUDA_ARCH_LIST', None)" "$CUMM_COMMON"; then
    sed -i "s|os.getenv('CUMM_CUDA_ARCH_LIST', None)|os.getenv('CUMM_CUDA_ARCH_LIST', '$TORCH_ARCH')|" "$CUMM_COMMON"
    log "cumm default arch set to $TORCH_ARCH"
fi

# Install pccm (cumm build dependency)
log "Installing pccm..."
uv pip install pccm || warn "pccm install failed"

# Install cumm from source
log "Installing cumm from source..."
uv pip install -e "$EXT_DIR/cumm" --no-build-isolation || warn "cumm build failed"

# Clone spconv
if [ ! -d "$EXT_DIR/spconv" ]; then
    log "Cloning spconv..."
    git clone --depth 1 https://github.com/traveller59/spconv.git "$EXT_DIR/spconv"
fi

# Install spconv from source (this will pull cumm from PyPI, we'll fix that next)
log "Installing spconv from source..."
uv pip install -e "$EXT_DIR/spconv" --no-build-isolation || warn "spconv build failed"

# Reinstall local cumm to override PyPI version that spconv pulled
log "Reinstalling local cumm (overriding PyPI version)..."
uv pip install -e "$EXT_DIR/cumm" --no-build-isolation || warn "cumm reinstall failed"

# Verification
log "Verifying installation..."
export CUDA_HOME
export CUMM_CUDA_ARCH_LIST="$TORCH_ARCH"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import src; print('src module OK')" || error "src module not importable"
uv run python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')" 2>/dev/null || warn "flash-attn not available"
uv run python -c "import pytorch3d; print('pytorch3d OK')" 2>/dev/null || warn "pytorch3d not available"
uv run python -c "import cumm; print(f'cumm {cumm.__version__} OK')" 2>/dev/null || warn "cumm not available"
uv run python -c "import spconv; print(f'spconv {spconv.__version__} OK')" 2>/dev/null || warn "spconv not available"

# Note: spconv/cumm use JIT compilation on first import.
# Default arch is patched into cumm source, so no env vars needed at runtime.
# Compiled kernels are cached in ~/.cumm for subsequent runs.

log ""
log "Setup complete!"
log ""
