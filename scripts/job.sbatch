#!/usr/bin/env bash
#SBATCH --comment=futurepos
set -euo pipefail
trap 'echo "[error] job.sbatch failed at line $LINENO with exit code $?" >&2' ERR
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[job] DEBUG=1 (trace on)"
  set -x
fi

# Resolve project paths from the Slurm submit directory (not the spool copy)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
HERE="${PROJECT_ROOT}/scripts"
echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "[job] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[job] HERE=${HERE}"

# Load environment (modules, uv, NCCL)
echo "[job] Sourcing ${HERE}/env.sh"
source "${HERE}/env.sh"
echo "[job] Environment loaded"

# Ensure we run from the project root so uv can find pyproject.toml and .venv
cd "${PROJECT_ROOT}"
echo "[job] PWD=$(pwd)"
ls -l "${HERE}" || true
echo "[job] uv: $(command -v uv)"

echo "========== SLURM JOB =========="
echo "Host:        $(hostname)"
echo "Node list:   ${SLURM_JOB_NODELIST:-unknown}"
echo "Tasks:       ${SLURM_NTASKS:-1} (per-node=${SLURM_TASKS_PER_NODE:-?})"
echo "Cores/task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "GPUs/task:   1"
echo "==============================="

"${HERE}/run_srun.sh"

