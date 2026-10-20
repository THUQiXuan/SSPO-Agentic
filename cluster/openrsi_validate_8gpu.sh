#!/usr/bin/env bash
# One-time validation of the OpenRSI task lane on one node with 8 GPUs (docs/OPENRSI_TASK.md, sections 2-3):
#   A. Judge lane: score the released SSPO step-30 adapter (seed 0) on the 128-problem test set with all 8 GPUs;
#   B. Work lane: train SSPO with seed 1 on all 8 GPUs (30 steps), then score it;
#   C. train and score SSPO with seed 2;
#   D. write runs/openrsi_validate/{test_seeds.md,timing.md}: select@8 of the three seeds and the wall time of
#      every stage (Work = training, Judge = test evaluation).
# Re-running the script skips finished stages (training and evaluation are resumable after preemption).
#   SSPO_ROOT=/path/to/SSPO-Agentic SSPO_ENV="source /path/to/venv/bin/activate" RBD_MODEL=qwen3_4b \
#     sbatch --gpus-per-node=8 --cpus-per-task=64 --mem=400G --constraint=h200 \
#     cluster/slurm/job.sbatch "bash cluster/openrsi_validate_8gpu.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
OUT=runs/openrsi_validate; mkdir -p ${OUT}
BASE="${BASE_ADAPTER:-runs/sel_sspo_s0/ckpt/rl_0030/adapter}"   # from the release asset adapters_step30.tar.gz
stamp() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s)" >> ${OUT}/timing.tsv; }

evaluate() {  # NAME ADAPTER
  [[ -f runs/eval/$1/test.rows.jsonl ]] && return 0
  stamp "$1" eval_start
  RBD_MEM_UTIL=0.85 RBD_MAX_NUM_SEQS=256 bash scripts/sel_eval.sh --name "$1" --adapter "$2" --splits test
  stamp "$1" eval_end
}

train() {  # SEED
  local run=sel_sspo_s$1_8g
  [[ -f runs/${run}/done ]] && return 0
  stamp ${run} train_start
  bash scripts/train.sh --model qwen3_4b --task code_sel --method sel_sspo --name ${run} --seed $1
  stamp ${run} train_end
}

evaluate val_sspo_s0_8g "${BASE}"
for s in 1 2; do
  train $s
  evaluate val_sspo_s${s}_8g runs/sel_sspo_s${s}_8g/ckpt/rl_0030/adapter
done

${PYTHON} -m rbd.sspo.report --split test --ref s0 --out ${OUT}/test_seeds.md \
  --runs s0=runs/eval/val_sspo_s0_8g s1=runs/eval/val_sspo_s1_8g s2=runs/eval/val_sspo_s2_8g
${PYTHON} - <<'EOF'
import json, statistics as st
from pathlib import Path
out = Path("runs/openrsi_validate")
last = {}
for line in (out / "timing.tsv").read_text().splitlines():
    name, ev, t = line.split("\t")
    last[(name, ev)] = int(t)          # the last attempt of a stage (earlier ones were preempted)
sel = {}
for s in (0, 1, 2):
    rows = [json.loads(l) for l in Path(f"runs/eval/val_sspo_s{s}_8g/test.rows.jsonl").read_text().splitlines() if l.strip()]
    sel[s] = 100 * st.mean(r["select@8"] for r in rows)
lines = ["| stage | wall time (min) |", "|---|---:|"]
for s in (0, 1, 2):
    n = f"val_sspo_s{s}_8g"
    if (n, "eval_end") in last:
        lines.append(f"| Judge: test evaluation, seed {s} | {(last[(n, 'eval_end')] - last[(n, 'eval_start')]) / 60:.1f} |")
for s in (1, 2):
    m = [json.loads(l) for l in Path(f"runs/sel_sspo_s{s}_8g/metrics.jsonl").read_text().splitlines() if l.strip()]
    steps = [r["t_step_s"] for r in m]
    lines.append(f"| Work: training seed {s}, sum of 30 steps (median step {st.median(steps) / 60:.1f} min, "
                 f"median rollout {st.median(r['t_rollout_s'] for r in m) / 60:.1f} min) | {sum(steps) / 60:.1f} |")
sd = st.stdev(sel.values())
lines += ["", "test select@8 by seed: " + ", ".join(f"s{s} {v:.2f}" for s, v in sel.items()),
          f"mean {st.mean(sel.values()):.2f}, standard deviation s = {sd:.2f} points"]
(out / "timing.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
