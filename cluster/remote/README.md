# Operator guide: 8 × H200 validation run (one seed)

This run checks that the OpenRSI task lane of this repository (docs/OPENRSI_TASK.md) works on one node with 8 GPUs
and measures its cost. It is a fixed protocol: nothing below needs to be tuned, and nothing should be changed to
work around a failure (in particular never shorten the 40,960-token context or add a generation cap).

## What runs

| stage | work | GPUs | estimate |
|---|---|---|---|
| A | score the released SSPO step-30 adapter (trained on 1 H200, seed 0) on dev 65 + test 128 problems × 16 agent runs | 8 | 30–40 min |
| B | train SSPO, seed 0, on all 8 GPUs: Qwen3-4B + LoRA, 30 steps × 16 problems × 16 multi-turn agent runs | 8 | 2.5–3.5 h |
| C | score the new step-30 adapter on dev + test | 8 | 30–40 min |
| D | `cluster/remote/summarize.py`: report and `artifacts.tar.gz` (adapter, per-problem rows, metrics; no completions) | 0 | 1 min |

Total about 4–5 h. Every stage resumes after preemption: rerun the same command and finished stages are skipped
(training resumes from its last completed step).

## Resources

One node; 8 × H200 (SXM or NVL, ≥ 140 GB each — 80 GB cards stalled on the uncapped 40,960-token rollouts in our
runs); 64 CPU cores (≥ 32 works, slower: agent commands and the checker run on the CPU); 400 GB RAM; about 60 GB
free disk; internet once for `fetch_assets.sh` (a login node is fine).

## Environment

Same stack as SelectRL-Bench: Python 3.12, `vllm==0.29.0` (with its `torch==2.13.0`), `transformers==5.12.1`,
`peft==0.19.1`, plus numpy, pyyaml, aiohttp, safetensors, huggingface_hub. An existing SelectRL vLLM 0.29 environment
should work unchanged; these scripts install nothing. Also needed: `git` on the compute node; `unshare -rn` is used
when available (agent commands without network), otherwise skipped.

## Steps

```bash
git clone https://github.com/THUQiXuan/SSPO-Agentic && cd SSPO-Agentic && git checkout <pinned commit from the issue>
source <your vLLM 0.29 environment>
bash cluster/remote/check_env.sh                 # read-only; prints RESULT: OK or what is missing
bash cluster/remote/fetch_assets.sh              # release v1.0 assets (sha256-checked) + Qwen/Qwen3-4B pinned revision
SSPO_ROOT=$PWD SSPO_ENV="source <your vLLM 0.29 environment>" \
  sbatch --nodes=1 --gpus-per-node=8 --cpus-per-task=64 --mem=400G [--account/--partition/--constraint for H200] \
  cluster/slurm/job.sbatch "bash cluster/openrsi_validate_8gpu.sh"
```

`cluster/slurm/job.sbatch` asks for 24 h and requeues itself on preemption; adapt or drop those lines to your
scheduler's rules. The first lines of the job log show the GPU tier and placement (8 colocated vLLM servers and 8
training ranks, vLLM sleep mode during the update).

## What to send back

* `runs/openrsi_validate/report.md` (paste into the issue) and the sha256 printed for `artifacts.tar.gz`;
* `artifacts.tar.gz` itself (a few hundred MB, mostly the LoRA adapter) by any route you allow, or its path on the server;
* on failure: the last 100 lines of `runs/sel_sspo_s0_8g/logs/train.log` or of the failing `runs/eval/*/logs/*`.

## Health signals while it runs

* `runs/sel_sspo_s0_8g/metrics.jsonl` gains one line per step; `t_step_s` should be about 300–420 s;
  `mismatch_abs_lp` stays below 0.02 (the run stops itself above 0.08).
* `runs/eval/<name>/logs/gen_*.log` reports finished chunks; a few runs per step take minutes because they use the
  full context, which is expected.
