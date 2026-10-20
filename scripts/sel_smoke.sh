#!/usr/bin/env bash
# Few-step end-to-end check of the selection-aware trainer (no checker cases needed).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export RBD_SEL_ALLOW_NOCASES=1
METHOD="${1:-sel_sspo}"; NAME="smoke_${METHOD}${SMOKE_TAG:+_${SMOKE_TAG}}"; rm -rf runs/${NAME}
bash scripts/train.sh --model qwen3_4b --task code_sel_smoke --method ${METHOD} --name ${NAME} \
  --steps 2 --tasks-per-step 4 --group-size 4 --max-turns 4
${PYTHON} -c "
import json; rows=[json.loads(l) for l in open('runs/${NAME}/metrics.jsonl')]
print(json.dumps(rows, indent=1)[:5000]); assert len(rows)==2; print('SMOKE PASS')"
