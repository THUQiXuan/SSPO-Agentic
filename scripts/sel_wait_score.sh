#!/usr/bin/env bash
# Wait until every generation shard of an evaluation has finished, then score its splits (CPU job; needs cases).
#   bash scripts/sel_wait_score.sh NAME NSHARDS "dev test"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NAME="$1"; NS="$2"; SPLITS="${3:-dev test}"
# the re-selected splits need their checker cases first (marker written after the pool case builds)
until [[ -f data/sel2/cases_ready ]]; do sleep 60; done
for s in ${SPLITS}; do
  until [[ $(ls runs/eval/${NAME}/${s}.gen_done_s*.json 2>/dev/null | wc -l) -ge ${NS} ]]; do sleep 60; done
done
for s in ${SPLITS}; do
  f=data/sel/${s}.jsonl.gz; [[ "$s" == cand ]] && f=data/sel/train.jsonl.gz
  ${PYTHON} -m rbd.sspo.evaluate score --split-file "$f" --out runs/eval/${NAME}/${s}
done
log "scored ${NAME}"
