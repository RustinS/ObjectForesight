#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] run_srun.sh failed at line $LINENO with exit code $?" >&2' ERR
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[run] DEBUG=1 (trace on)"
  set -x
fi

# Build the srun line with one task per GPU across nodes.

GPU_FLAG=${SRUN_GPU_FLAG:-gpus-per-node}
PER_NODE=${GPUS_PER_NODE:-8}
echo "[run] GPU_FLAG=${GPU_FLAG} PER_NODE=${PER_NODE}"

UV_RUN=(uv run --no-sync --frozen)

SRUN_GPU_ARGS=()
case "$GPU_FLAG" in
  gpus-per-node)
    SRUN_GPU_ARGS+=( --gpus-per-node="${PER_NODE}" )
    ;;
  gres)
    SRUN_GPU_ARGS+=( --gres="gpu:${PER_NODE}" )
    ;;
  gpus)
    SRUN_GPU_ARGS+=( --gpus="${PER_NODE}" )
    ;;
  *)
    echo "[warn] Unknown SRUN_GPU_FLAG='${GPU_FLAG}', defaulting to --gpus-per-node" >&2
    SRUN_GPU_ARGS+=( --gpus-per-node="${PER_NODE}" )
    ;;
esac

# Determine total tasks robustly to avoid unbound var errors under 'set -u'
TOTAL_TASKS="${SLURM_NTASKS:-}"
if [[ -z "${TOTAL_TASKS}" ]]; then
  if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
    TOTAL_TASKS=$(( SLURM_JOB_NUM_NODES * PER_NODE ))
  else
    TOTAL_TASKS="${PER_NODE}"
  fi
fi

echo "[run] uv: $(command -v uv)"
echo "[run] python (via uv): $("${UV_RUN[@]}" python -c 'import sys; print(sys.executable)')"
"${UV_RUN[@]}" python -V || true
echo "[run] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-unset}"
echo "[run] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} PRECISION=${PRECISION:-unset}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run] EVAL=${EVAL:-0}"
echo "[run] srun extra gpu args: ${SRUN_GPU_ARGS[*]}"

echo "[run] srun ntasks=${TOTAL_TASKS} ntasks-per-node=${PER_NODE} gpu-flag=${GPU_FLAG}"

# Determine launcher flags
EVAL_FLAG=()
if [[ "${EVAL:-0}" == "1" ]]; then
  EVAL_FLAG=( --eval )
fi

srun \
  --ntasks="${TOTAL_TASKS}" \
  --ntasks-per-node="${PER_NODE}" \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  --cpu-bind=cores \
  "${SRUN_GPU_ARGS[@]}" \
  "${UV_RUN[@]}" python -u -m src.dist.launch_ddp \
    --precision "${PRECISION:-bf16}" \
    --fsdp.enable "${FSDP_ENABLE:-0}" \
    --log-dir "${LOG_DIR}" \
    --ckpt-dir "${CKPT_DIR}" \
    "${EVAL_FLAG[@]}" \
    -- ${EXTRA_ARGS:-}
