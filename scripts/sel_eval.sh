#!/usr/bin/env bash
# Agent runs of one policy on benchmark splits (generation on GPUs; scoring needs the checker cases).
#   bash scripts/sel_eval.sh --name NAME [--adapter none|DIR] [--splits "dev test"] [--n 16] [--active 1]
# Splits: dev, test (data/sel/<split>.jsonl.gz) and cand (calibration on data/sel/candidates.jsonl.gz).
# With --active 1 the calibrated training pool data/sel/train.jsonl.gz is written after the cand split.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NAME=""; ADAPTER=none; SPLITS="dev test"; N=16; ACTIVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in --name) NAME="$2"; shift 2;; --adapter) ADAPTER="$2"; shift 2;; --splits) SPLITS="$2"; shift 2;;
    --n) N="$2"; shift 2;; --active) ACTIVE="$2"; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac
done
file_of() { case "$1" in cand) echo data/sel/candidates.jsonl.gz;; pool) echo data/sel2/pool.jsonl.gz;; *) echo data/sel/$1.jsonl.gz;; esac; }
preset qwen3_4b
RUN=runs/eval/${NAME}; mkdir -p ${RUN}/logs; SRV=${RUN}/servers_${SHARD:-0}.json; trap 'stop_servers' EXIT
start_servers qwen3_4b ${SRV} ${RUN}/logs/srv_${SHARD:-0} "${RBD_SERVER_GPUS}" "${RBD_MEM_UTIL}" "${RBD_MAX_NUM_SEQS}"
for s in ${SPLITS}; do
  ${PYTHON} -m rbd.sspo.evaluate gen --servers ${SRV} --shard "${SHARD:-0}" --nshards "${NSHARDS:-1}" --split-file "$(file_of $s)" --n ${N} --seed "${EVAL_SEED:-20261018}" \
    --adapter "${ADAPTER}" --chunk "${SEL_CHUNK:-96}" --out ${RUN}/${s} 2>&1 | tee -a ${RUN}/logs/gen_${s}_${SHARD:-0}.log
done
stop_servers
[[ "${NSHARDS:-1}" != 1 ]] && { log "shard ${SHARD} of ${NAME} generated"; exit 0; }
if [[ "${ACTIVE}" == 1 ]]; then
  ${PYTHON} -m rbd.sspo.evaluate active --split-file data/sel/candidates.jsonl.gz --out ${RUN}/cand \
    --train-out data/sel/train.jsonl.gz 2>&1 | tee -a ${RUN}/logs/active.log
fi
for s in ${SPLITS}; do
  [[ "$s" == cand ]] && continue
  ${PYTHON} -m rbd.sspo.evaluate score --split-file "$(file_of $s)" --out ${RUN}/${s} 2>&1 | tee -a ${RUN}/logs/score_${s}.log \
    || log "scoring of ${s} deferred (checker cases missing?)"
done
log "eval ${NAME} done"
