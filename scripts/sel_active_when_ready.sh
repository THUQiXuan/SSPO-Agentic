#!/usr/bin/env bash
# Wait until every calibration shard has written its samples, then build the training pool (CPU job).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NS="${1:-8}"
until [[ $(ls runs/eval/base_calib/cand.samples/chunk_s*_0000.jsonl.gz 2>/dev/null | wc -l) -ge ${NS} ]]; do sleep 60; done
log "all ${NS} calibration shards present"
${PYTHON} -m rbd.sspo.evaluate active --split-file data/sel/candidates.jsonl.gz --out runs/eval/base_calib/cand \
  --train-out data/sel/train.jsonl.gz
