#!/usr/bin/env bash
# Checker helpers with the frozen Qwen3-8B on every visible GPU: [SHARD=i NSHARDS=n] bash scripts/sel_helpers.sh FILE [FILE ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preset qwen3_8b
RUN=runs/helpers_${SHARD:-0}; mkdir -p ${RUN}/logs; trap 'stop_servers' EXIT
start_servers qwen3_8b ${RUN}/servers.json ${RUN}/logs "${RBD_SERVER_GPUS}" "${RBD_MEM_UTIL}" "${RBD_MAX_NUM_SEQS}"
${PYTHON} -m rbd.sspo.helpers --servers ${RUN}/servers.json --model-path "${RBD_MODEL_PATH}" --shard "${SHARD:-0}" --nshards "${NSHARDS:-1}" --problems "$@"
log "helpers done"
