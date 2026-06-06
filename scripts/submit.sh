#!/usr/bin/env bash
set -euo pipefail

# Slurm submit helper for multi-node, multi-GPU training with srun (one task per GPU).
# Defaults are tailored for an 8x H200/node cluster.

# -----------------------------
# Defaults (can be overridden)
# -----------------------------
NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
# If not provided, derived from MEM_PER_GPU_GB * GPUS_PER_NODE.
MEM_PER_NODE=${MEM_PER_NODE:-}
TIME=${TIME:-24:00:00}
QOS=${QOS:-normal}
ACCOUNT=${ACCOUNT:-}
JOB_NAME=${JOB_NAME:-futurepos}
SRUN_GPU_FLAG=${SRUN_GPU_FLAG:-gpus-per-node}  # or: gres | gpus
ARRAY=${ARRAY:-}
LOG_DIR=${LOG_DIR:-./logs}
CKPT_DIR=${CKPT_DIR:-./ckpts}
PRECISION=${PRECISION:-bf16}
FSDP_ENABLE=${FSDP_ENABLE:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-200}
EVAL=${EVAL:-0}
DRY_RUN=${DRY_RUN:-0}
EXCLUDE=${EXCLUDE:-}
DEPENDENCY=${DEPENDENCY:-}

# -----------------------------
# Parse --key value pairs
# -----------------------------
to_var_name() {
  # Convert --kebab-case to UPPER_SNAKE
  local key="$1"
  key="${key#--}"
  key="${key//-/_}"
  printf '%s' "${key^^}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval)
      EVAL=1
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --*)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for '$1'" >&2
        exit 1
      fi
      VAR_NAME=$(to_var_name "$1")
      declare -g "$VAR_NAME"="$2"
      shift 2
      ;;
    *)
      echo "Unknown positional arg '$1' (ignored)" >&2
      shift 1
      ;;
  esac
done

# -----------------------------
# Compute ntasks
# -----------------------------
REQUESTED=$(( NODES * GPUS_PER_NODE ))
NTASKS=$REQUESTED

if [[ -z "${MEM_PER_NODE}" ]]; then
  MEM_PER_NODE="$(( MEM_PER_GPU_GB * GPUS_PER_NODE ))G"
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# Export runtime env for the job step; values are taken from current shell
export LOG_DIR CKPT_DIR PRECISION FSDP_ENABLE EXTRA_ARGS SRUN_GPU_FLAG GPUS_PER_NODE EVAL

# -----------------------------
# Build sbatch arguments
# -----------------------------
SBATCH_COMMON=(
  --job-name="$JOB_NAME"
  --qos="$QOS"
  --time="$TIME"
  --nodes="$NODES"
  --ntasks="$NTASKS"
  --ntasks-per-node="$GPUS_PER_NODE"
  --gpus-per-task=1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM_PER_NODE"
  --output="$LOG_DIR/%x_%A_%a.log"
  --error="$LOG_DIR/%x_%A_%a.log"
  --export=ALL,LOG_DIR,CKPT_DIR,PRECISION,FSDP_ENABLE,EXTRA_ARGS,SRUN_GPU_FLAG,GPUS_PER_NODE,EVAL
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_COMMON+=( --account="$ACCOUNT" )
fi

if [[ -n "${ARRAY}" ]]; then
  SBATCH_COMMON+=( --array="$ARRAY" )
fi

if [[ -n "${EXCLUDE}" ]]; then
  SBATCH_COMMON+=( --exclude="$EXCLUDE" )
fi

if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_COMMON+=( --dependency="$DEPENDENCY" )
fi

echo "Submitting job: name=${JOB_NAME} nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} ntasks=${NTASKS} time=${TIME} mem=${MEM_PER_NODE}"
echo "[submit] sbatch ${SBATCH_COMMON[*]} scripts/job.sbatch"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[submit] DRY_RUN=1 (not submitting)"
  exit 0
fi

sbatch "${SBATCH_COMMON[@]}" scripts/job.sbatch
