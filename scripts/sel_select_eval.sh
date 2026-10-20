#!/usr/bin/env bash
# Re-select dev / test with the training-pool rule once every calibration shard of the pool is in (CPU job):
# keep pool problems the frozen base agent solves 1-7 times in 8 runs (seed 777, independent of the evaluation seed),
# then split them by whole contests into data/sel/dev.jsonl.gz and data/sel/test.jsonl.gz.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NS="${1:-12}"
until [[ $(ls runs/eval/test_calib/pool.gen_done_s*.json 2>/dev/null | wc -l) -ge ${NS} ]]; do sleep 60; done
log "all ${NS} pool calibration shards present"
${PYTHON} -m rbd.sspo.evaluate active --split-file data/sel2/pool.jsonl.gz --out runs/eval/test_calib/pool \
  --train-out data/sel2/eval_active.jsonl.gz
${PYTHON} -m rbd.sspo.data split-eval --selected data/sel2/eval_active.jsonl.gz --out data/sel --dev-frac 0.33
