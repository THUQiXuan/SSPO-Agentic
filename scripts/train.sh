#!/usr/bin/env bash
# Train one method.
#   bash scripts/train.sh --model qwen35_9b|qwen3_4b --method drgrpo|rft_grpo|pivotopd|rbd [--task code_mt]
#                         [--seed 0] [--name RUN] [--gpus 0,1,...] [--steps N --tasks-per-step N --group-size N]
# Output: runs/<RUN>/ ; re-running the same command resumes from the last finished step.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# optional runtime switches for training jobs (e.g. RBD_SLEEP_MODE=1 RBD_MEM_UTIL=0.72), read when the job starts
[[ -f configs/runtime_train.env ]] && { source configs/runtime_train.env; export RBD_SLEEP_MODE RBD_MEM_UTIL RBD_MAX_NUM_SEQS 2>/dev/null || true; }
MODEL=""; METHOD=""; TASK=code_mt; SEED=0; NAME=""; GPUS=""; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;; --method) METHOD="$2"; shift 2;; --task) TASK="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;; --name) NAME="$2"; shift 2;; --gpus) GPUS="$2"; shift 2;;
    --steps|--tasks-per-step|--group-size|--max-turns) EXTRA+=("$1" "$2"); shift 2;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;; *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ -n "${MODEL}" && -n "${METHOD}" ]] || { sed -n '2,6p' "$0"; exit 2; }
NAME="${NAME:-${MODEL}_${TASK}_${METHOD}_s${SEED}}"; RUN="runs/${NAME}"
preset "${MODEL}" "${GPUS}"
NTRAIN=$(awk -F, '{print NF}' <<< "${RBD_TRAIN_GPUS}")
mkdir -p "${RUN}/logs"; trap 'stop_servers' EXIT
for attempt in 1 2 3; do
  [[ -f "${RUN}/done" ]] && break
  start_servers "${MODEL}" "${RUN}/servers.json" "${RUN}/logs" "${RBD_SERVER_GPUS}" "${RBD_MEM_UTIL}" "${RBD_MAX_NUM_SEQS}"
  set +e
  CUDA_VISIBLE_DEVICES="${RBD_TRAIN_GPUS}" ${PYTHON} -m torch.distributed.run --standalone --nproc_per_node="${NTRAIN}" \
    -m rbd.train.main --run-dir "${RUN}" --model "${MODEL}" --task "${TASK}" --method "${METHOD}" \
    --servers "${RUN}/servers.json" --seed "${SEED}" --micro-tokens "${RBD_MICRO_TOKENS}" "${EXTRA[@]}" 2>&1 | tee -a "${RUN}/logs/train.log"
  rc=${PIPESTATUS[0]}; set -e
  [[ ${rc} -eq 0 && -f "${RUN}/done" ]] && break
  log "training exited with ${rc}; resuming"; stop_servers; sleep 10
done
[[ -f "${RUN}/done" ]] && log "done: ${RUN}" || { log "ERROR: not finished, see ${RUN}/logs/train.log"; exit 1; }
