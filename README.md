# SSPO-Agentic: selection-aware RL for multi-turn coding agents

Coding agents are deployed with a fixed test-time selector: sample *k* runs, keep the programs that pass an
execution checker, vote on their output signatures and submit one program. Standard RL trains for pass@1
(or pass@k), which is not what the deployed system is scored on. **SSPO** (selection-aware set policy
optimization) trains the policy on the deployed metric **select@k** directly: every run of a problem receives the
exact leave-one-out credit of its effect on the selector's success, computed by a dynamic program over all
*k*-subsets of the *n* sampled runs.

This repository contains the complete multi-turn agentic RL framework, the benchmark construction pipeline, three
training algorithms (SSPO and two published baselines), the evaluation code and the results of the first seed.

| | |
|---|---|
| Policy | `Qwen/Qwen3-4B` (thinking off), default context **40,960 tokens**, no generation cap; LoRA r = 64 |
| Agent | up to 8 turns; one bash command per turn in a git repository (edit `solution.py`, run public examples, submit) |
| Deployed system | k = 8 runs → frozen execution checker (Chk) → majority vote over output signatures → one submission |
| Benchmark | CodeContests (unique-answer problems), contest-disjoint splits; **test set: 128 problems / 114 contests** |
| Algorithms | `sspo` (ours), `drgrpo` (Dr. GRPO pass@1, arXiv 2503.20783), `pkpo` (PKPO pass@8, arXiv 2505.15201) |

## Result (seed 0, step 30, test set of 128 problems)

Each problem is evaluated with 16 agent runs (seeds depend only on problem and run index, so all policies are
paired); select@8 and cov@8 are exact expectations over all C(16, 8) subsets. Numbers in %.

| policy | pass@1 | cov@8 | **select@8** | selection gap (cov@8 − select@8) | Chk pass rate | mean turns |
|---|---:|---:|---:|---:|---:|---:|
| frozen Qwen3-4B | 42.48 | 64.84 | 58.20 | 6.65 | 38.8 | 7.04 |
| **SSPO (ours)** | 49.95 | **69.19** | **64.28** | **4.91** | **46.6** | 6.88 |
| Dr. GRPO | **50.10** | 66.02 | 59.34 | 6.69 | 44.6 | 5.83 |
| PKPO pass@8 | 45.61 | 65.51 | 59.02 | 6.49 | 41.3 | 7.15 |

Paired select@8 differences (SSPO minus other; 95% percentile bootstrap resampling whole contests; Holm over the
three comparisons):

| comparison | difference (points) | 95% interval | p | p (Holm) |
|---|---:|---|---:|---:|
| SSPO − frozen | +6.08 | [+0.65, +11.63] | 0.029 | 0.058 |
| SSPO − Dr. GRPO | +4.94 | [+0.26, +9.78] | 0.040 | 0.058 |
| SSPO − PKPO | +5.26 | [+0.92, +9.85] | 0.016 | 0.048 |

* SSPO's gain over Dr. GRPO splits into a higher cov@8 (+3.17 points: more problems with a correct program that
  passes the checker) and a smaller selection gap (+1.78 points: fewer wrong programs that pass the checker and
  win the vote).
* Dr. GRPO and SSPO reach the same pass@1 (50.10 vs 49.95) but differ by 4.9 points in select@8: optimizing
  pass@1 does not optimize the deployed system.
* At step 15 the ordering was the same but the differences were within noise
  ([results/test128_step15.md](results/test128_step15.md)); the step-30 checkpoint was fixed in advance as the
  primary read-out. One training seed so far; the intervals reflect problem sampling only.

Full tables: [results/test128_step30.md](results/test128_step30.md). Run history and design decisions:
[docs/RUN_LOG.md](docs/RUN_LOG.md).

## Method

For one problem, the *n* = 16 runs of a training step are scored by their representative program y(τ) (the last
version of `solution.py` that passes all public examples, else the last version): hidden-test correctness g,
checker verdict Chk and output signature sig. With C = {g = 1, Chk = 1}, k = 8 and α = 0.7,

    A_i = (k/n) · [ (1 − α) · (A_i^cov − b_i^cov) + α · A_i^(S,U) ]

* A^cov − b^cov: leave-one-out coverage advantage on C with its leave-one-out baseline (PKPO closed form on C);
* A^(S,U): uncentered leave-one-out credit of J⁻(K) = J_S(K) · 1[K has no W*], the selector's success on a
  k-subset K where wrong programs whose signature is already known to be correct (Σ⁺_x, fixed before the step)
  count as failures; exact dynamic program under cluster purity, enumeration of all subsets otherwise.

Every assistant token of run i receives A_i; the loss is the PPO-clipped surrogate with a constant (Dr. GRPO)
normalizer. Details: [docs/METHOD.md](docs/METHOD.md).

## Repository layout

| path | content |
|---|---|
| `rbd/agent`, `rbd/envs`, `rbd/serving`, `rbd/train` | multi-turn agent loop, sandboxed git-repository environment, per-GPU vLLM servers, LoRA trainer (token-exact multi-turn sequences, chunked log-probabilities, gradient checkpointing, vLLM sleep mode) |
| `rbd/sspo` | representative program and signals, advantages, step builder, plug-in interface, evaluation, data and report scripts |
| `selectrl/checker`, `selectrl/algolib`, `selectrl/sandbox`, `selectrl/data` | frozen execution checker, selector and SSPO dynamic program, program sandbox, CodeContests conversion (from [SelectRL-Bench](https://github.com/THUQiXuan/SelectRL-Bench)) |
| `algorithms/` | the algorithm plug-in: `sspo.py`, `drgrpo.py`, `pkpo.py` (identical to the arms used for the results) and `candidate.py` |
| `configs/` | model, task, method and GPU-tier presets |
| `scripts/` | training, evaluation, checker building, calibration |
| `cluster/slurm/job.sbatch` | generic resumable Slurm wrapper |
| `docs/OPENRSI_TASK.md` | draft of an OpenRSI-Index AutoResearch task built on this repository, with a requirements check |

## Reproduce

Environment (as used for the results): Python 3.12, `vllm==0.29.0`, `torch==2.13.0` (CUDA 13), `transformers==5.12.1`,
`peft==0.19.1`, plus `requirements.txt`. Unit tests (CPU, ~15 s): `python -m pytest -q tests`.

1. **Problems.** `python -m selectrl.data.codecontests --out data/src` downloads `deepmind/code_contests`, removes
   multi-answer problems and writes contest-disjoint `train_candidates / val / test` files. Then
   `python -m rbd.sspo.data prepare --src data/src --out data/sel --n-cand 768` selects the training candidates.
   The released split files (see [data/README.md](data/README.md)) contain the exact problems used.
2. **Frozen checker.** `bash scripts/sel_helpers.sh <problem files>` (frozen Qwen3-8B writes 4 input generators,
   4 brute-force solvers and 2 validators per problem; replies capped at 8,192 tokens), then
   `bash scripts/sel_build.sh 0 1 <problem files>` builds the check cases on CPU. Released cases are provided.
3. **Training pool.** `bash scripts/sel_eval.sh --name base_calib --splits cand --n 8 --active 1` keeps the
   candidates the frozen agent solves 1–7 times in 8 runs (231 problems).
4. **Train.** `bash scripts/train.sh --model qwen3_4b --task code_sel --method sel_sspo --name sel_sspo_s0 --seed 0`
   (also `sel_drgrpo`, `sel_pkpo`, or `sel_file` for `algorithms/candidate.py`). 30 steps × 16 problems × 16 runs;
   resumable after preemption. `configs/runtime_train.env` enables vLLM sleep mode (72% of GPU memory for rollouts).
5. **Evaluate.** `bash scripts/sel_eval.sh --name <name> --adapter runs/<run>/ckpt/rl_0030/adapter --splits test`
   then `python -m rbd.sspo.report --split test --runs frozen=... sspo=... --ref sspo`.

Hardware: one H200 per training run took about 10 minutes per step (rollout 4–8 min, scoring 1–1.5 min, update
1–2 min). In our runs 48 GB cards and A100-80G stalled when a few runs filled the KV cache with uncapped
40,960-token contexts; the presets therefore target ≥ 80 GB and H200 was used for every reported number.

## Assets

Released with the repository (GitHub release `v1.0`; see [data/README.md](data/README.md)): problem files of the
training pool (231), dev (65) and test (128) sets, frozen-checker cases, the LoRA adapters of the three algorithms
at step 30, and per-problem evaluation rows.

## License

Code: Apache-2.0. Third-party data and models keep their own terms ([THIRD_PARTY.md](THIRD_PARTY.md)).
