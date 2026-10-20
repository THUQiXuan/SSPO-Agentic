#!/usr/bin/env bash
# Shared settings (sourced). Paths are relative to the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${PYTHON:-python}"
export PYTHON
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p runs logs
log() { echo "[$(date '+%F %T')] $*"; }

# model_field <model> <key>  (reads configs/models/<model>.yaml)
model_field() { ${PYTHON} -c "from rbd.config import load_yaml; print(load_yaml('configs/models/$1.yaml').get('$2', ''))"; }

# start_servers <model> <state> <logdir> <gpus> <mem_util> <max_num_seqs>
start_servers() {
  local model="$1" state="$2" logdir="$3" gpus="$4" util="$5" seqs="$6"
  mkdir -p "${logdir}"; rm -f "${state}"
  local mpath mml lmo
  mpath="$(${PYTHON} -c "from rbd.config import load_yaml, model_path; print(model_path(load_yaml('configs/models/${model}.yaml')))")"
  mml="$(model_field "${model}" max_model_len)"; lmo="$(model_field "${model}" language_model_only)"
  ${PYTHON} -m rbd.serving.servers launch --gpus "${gpus}" --model "${mpath}" --state "${state}" --log-dir "${logdir}" \
    --mem-util "${util}" --max-num-seqs "${seqs}" --max-model-len "${mml:-0}" \
    --language-model-only "$([[ "${lmo}" == True ]] && echo 1 || echo 0)" >> "${logdir}/launcher.log" 2>&1 &
  SERVERS_PID=$!
  log "vLLM launcher pid ${SERVERS_PID} (GPUs ${gpus}, util ${util}, max_model_len ${mml:-model default})"
  if ! ${PYTHON} -m rbd.serving.servers wait --state "${state}" --pid "${SERVERS_PID}" --timeout 3600 > /dev/null; then
    log "ERROR: vLLM servers did not start"; tail -n 60 "${logdir}/launcher.log" >&2; stop_servers; return 3
  fi
  log "vLLM servers ready"
}

stop_servers() {
  if [[ -n "${SERVERS_PID:-}" ]] && kill -0 "${SERVERS_PID}" 2>/dev/null; then
    kill -TERM "${SERVERS_PID}" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "${SERVERS_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${SERVERS_PID}" 2>/dev/null || true
  fi
  SERVERS_PID=""
}

# preset <model> [gpus] -> RBD_TIER, RBD_PLACEMENT, RBD_SERVER_GPUS, RBD_TRAIN_GPUS, RBD_MEM_UTIL, ...
preset() {
  eval "$(${PYTHON} -m rbd.config preset --model "$1" ${2:+--gpus "$2"})"
  export RBD_TIER RBD_PLACEMENT RBD_SERVER_GPUS RBD_TRAIN_GPUS RBD_MEM_UTIL RBD_MAX_NUM_SEQS RBD_MICRO_TOKENS RBD_GPU_MEM_GB
  log "tier ${RBD_TIER} (${RBD_GPU_MEM_GB} GB), placement ${RBD_PLACEMENT}: servers [${RBD_SERVER_GPUS}] util ${RBD_MEM_UTIL}, trainers [${RBD_TRAIN_GPUS}], micro-batch ${RBD_MICRO_TOKENS} tokens"
}
