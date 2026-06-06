#!/usr/bin/env bash
# Submit a Slurm dependency chain of N x TIME jobs that auto-resumes from the
# latest checkpoint between jobs. Useful when a single job cannot exceed the
# cluster's 24h wall-clock cap but training needs longer.
#
# Usage:
#   ./scripts/submit_chain.sh --config-name debug --job-name my_run --num-jobs 2
#
# Required:
#   --config-name <name>   Hydra config name (e.g., debug).
#   --job-name <name>      Slurm job name prefix; jobs land as <name>_seg00, _seg01, ...
#
# Optional (forwarded to submit.sh, except where overridden per segment):
#   --num-jobs N           Chain length. Default: 2.
#   --nodes, --gpus-per-node, --time, --qos, --mem-per-gpu-gb, etc.
#
# All other --key value pairs are passed through to submit.sh untouched.
# Continuation segments (segment >= 1) automatically append `train.resume=latest`
# to EXTRA_ARGS so every job after the first picks up where the previous left off.

set -euo pipefail

NUM_JOBS=2
CONFIG_NAME=""
JOB_NAME_BASE="abl_chain"
PASSTHROUGH=()
USER_EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-jobs)     NUM_JOBS="$2"; shift 2 ;;
    --config-name)  CONFIG_NAME="$2"; shift 2 ;;
    --job-name)     JOB_NAME_BASE="$2"; shift 2 ;;
    --extra-args)   USER_EXTRA_ARGS="$2"; shift 2 ;;
    --*)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for '$1'" >&2
        exit 1
      fi
      PASSTHROUGH+=( "$1" "$2" )
      shift 2
      ;;
    *)
      echo "Unknown positional arg '$1' (ignored)" >&2
      shift 1
      ;;
  esac
done

if [[ -z "${CONFIG_NAME}" ]]; then
  echo "Error: --config-name is required" >&2
  exit 1
fi
if (( NUM_JOBS < 1 )); then
  echo "Error: --num-jobs must be >= 1" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PREV_JOB_ID=""
for (( seg=0; seg<NUM_JOBS; seg++ )); do
  SEG_NAME="$(printf '%s_seg%02d' "${JOB_NAME_BASE}" "${seg}")"

  # First segment: from-scratch (uses config's resume="" default).
  # Continuation segments: append train.resume=latest so the run picks up the
  # most recent checkpoint written by the previous segment.
  if (( seg == 0 )); then
    SEG_EXTRA_ARGS="--config-name ${CONFIG_NAME} ${USER_EXTRA_ARGS}"
  else
    SEG_EXTRA_ARGS="--config-name ${CONFIG_NAME} train.resume=latest ${USER_EXTRA_ARGS}"
  fi

  SUBMIT_ENV=()
  SUBMIT_ENV+=( "EXTRA_ARGS=${SEG_EXTRA_ARGS}" )
  SUBMIT_ENV+=( "JOB_NAME=${SEG_NAME}" )
  if [[ -n "${PREV_JOB_ID}" ]]; then
    SUBMIT_ENV+=( "DEPENDENCY=afterany:${PREV_JOB_ID}" )
  fi

  echo "[chain] Submitting segment ${seg}/${NUM_JOBS}: ${SEG_NAME}"
  echo "[chain]   EXTRA_ARGS=${SEG_EXTRA_ARGS}"
  if [[ -n "${PREV_JOB_ID}" ]]; then
    echo "[chain]   depends on afterany:${PREV_JOB_ID}"
  fi

  # Capture sbatch's "Submitted batch job <id>" line.
  SUBMIT_OUTPUT="$(env "${SUBMIT_ENV[@]}" ./scripts/submit.sh "${PASSTHROUGH[@]}" 2>&1)"
  echo "${SUBMIT_OUTPUT}"
  PREV_JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | grep -oE 'Submitted batch job [0-9]+' | tail -1 | awk '{print $4}')"
  if [[ -z "${PREV_JOB_ID}" ]]; then
    echo "[chain] ERROR: failed to capture job ID from submit.sh output; aborting chain." >&2
    exit 1
  fi
  echo "[chain]   submitted job ${PREV_JOB_ID}"
done

echo "[chain] Done — submitted ${NUM_JOBS} segment(s); last job=${PREV_JOB_ID}"
