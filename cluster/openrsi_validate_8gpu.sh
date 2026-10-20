#!/usr/bin/env bash
# Validation of the OpenRSI task lane on one node with 8 H200 (docs/OPENRSI_TASK.md, sections 2-3; operator guide:
# cluster/remote/README.md). Stages, each resumable (re-running the script skips finished stages):
#   A. Judge lane: score the released SSPO step-30 adapter (1-GPU training, seed 0) on dev + test with all 8 GPUs;
#   B. Work lane: for every seed in SEEDS (default 0) train SSPO on all 8 GPUs (30 steps), then score it on dev + test;
#   C. runs/openrsi_validate/report.{md,json} and artifacts.tar.gz (cluster/remote/summarize.py): metrics against the
#      released reference rows, per-stage wall time, environment, asset hashes. No raw completions are packed.
#   SSPO_ROOT=/path/to/SSPO-Agentic SSPO_ENV="source /path/to/venv/bin/activate" \
#     sbatch --nodes=1 --gpus-per-node=8 --cpus-per-task=64 --mem=400G cluster/slurm/job.sbatch \
#     "bash cluster/openrsi_validate_8gpu.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"
OUT=runs/openrsi_validate; mkdir -p ${OUT}
BASE="${BASE_ADAPTER:-reference/adapters/sel_sspo_s0}"          # from the release asset adapters_step30.tar.gz
stamp() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s)" >> ${OUT}/timing.tsv; }

ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "${ngpu}" == 8 ]] || { echo "expected 8 visible GPUs, found ${ngpu}" >&2; exit 2; }
for f in data/sel/train.jsonl.gz data/sel/dev.jsonl.gz data/sel/test.jsonl.gz "${BASE}/adapter_config.json"; do
  [[ -f "${f}" ]] || { echo "missing ${f}: run cluster/remote/fetch_assets.sh first" >&2; exit 2; }
done

evaluate() {  # NAME ADAPTER
  [[ -f runs/eval/$1/dev.rows.jsonl && -f runs/eval/$1/test.rows.jsonl ]] && return 0
  stamp "$1" eval_start
  RBD_MEM_UTIL=0.85 RBD_MAX_NUM_SEQS=256 bash scripts/sel_eval.sh --name "$1" --adapter "$2" --splits "dev test"
  [[ -f runs/eval/$1/dev.rows.jsonl && -f runs/eval/$1/test.rows.jsonl ]] || { echo "evaluation $1 incomplete" >&2; exit 1; }
  stamp "$1" eval_end
}

train() {  # SEED
  local run=sel_sspo_s$1_8g
  [[ -f runs/${run}/done ]] && return 0
  stamp ${run} train_start
  bash scripts/train.sh --model qwen3_4b --task code_sel --method sel_sspo --name ${run} --seed $1
  stamp ${run} train_end
}

evaluate val_released_s0_8g "${BASE}"
for s in ${SEEDS}; do
  train $s
  evaluate val_sspo_s${s}_8g runs/sel_sspo_s${s}_8g/ckpt/rl_0030/adapter
done
SEEDS="${SEEDS}" ${PYTHON} cluster/remote/summarize.py
