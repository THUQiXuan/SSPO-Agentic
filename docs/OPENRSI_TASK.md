# OpenRSI-Index AutoResearch task: draft and requirements check

This file prepares a submission to [OpenRSI-Index](https://github.com/OpenRSI-Foundation/OpenRSI-Index) through
its `proposal-agent` skill. It is **not** a submitted proposal. Rows marked *contributor decision* must be
confirmed by the contributor in the proposal-agent session; repository facts can be verified there against a
pinned commit of this repository.

## 1. Task in one paragraph

A research agent starts from this repository with SSPO as the human reference method. It may change only the
credit-assignment rule in `algorithms/candidate.py` (per-run advantages for the 16 agent runs of a training
problem). Each candidate is trained with the fixed protocol (Qwen3-4B, LoRA, 30 steps × 16 problems × 16 runs,
multi-turn coding agent with the full 40,960-token context) and the resulting LoRA adapter is scored by the fixed
evaluator: **exact select@8 of the deployed execution selector on the hidden 128-problem test set**. The question
is whether the agent can discover a credit-assignment rule that beats SSPO (64.28 select@8 for seed 0) under the
same budget, and why.

## 2. Draft of the proposal fields

| Field | Draft |
|---|---|
| Full name / email | Xuan Qi / *contributor provides the publication email* |
| Repository URL / commit | `https://github.com/THUQiXuan/SSPO-Agentic` at the commit pinned at submission time |
| Project-selection route | Coauthored route: the contributor wrote this repository (and the SelectRL-Bench components it vendors). *Contributor confirms.* |
| Scientific question | Under a fixed RL budget for a multi-turn coding agent, which per-run credit-assignment rule maximizes the deployed select@8 (checker filter + signature majority vote)? Falsifiable: a candidate rule either exceeds the SSPO adapter's test select@8 under the matched protocol or it does not. |
| Why research, not engineering | Distinct hypotheses lead to different rules: exact selection credit (SSPO), coverage-only credit (PKPO), pass@1 credit (Dr. GRPO), variance-reduced or problem-weighted variants, use of the known-correct signature set Σ⁺, run-length or turn-aware credit. Each trial shows whether a rule changes coverage (cov@8) or the selection gap, which tells the agent which mechanism helped. |
| Starting environment and artifacts | This repository; `Qwen/Qwen3-4B` (Hugging Face, pinned revision `1cfa9a7208912126459214e8b04321603b3df60c`); released problem files (training pool 231, dev 65, test 128), released frozen-checker cases, released SSPO / Dr. GRPO / PKPO step-30 adapters (GitHub release `v1.0`). Test problems, their hidden tests and their checker cases are Judge-only. |
| Agent's final deliverable | The LoRA adapter (`runs/sel_file_s0/ckpt/rl_0030/adapter`) produced in Work by `scripts/train.sh --method sel_file`, together with `algorithms/candidate.py` and the run manifest / metrics. |
| Files the agent may modify | `algorithms/candidate.py` only (pure-Python function `compute_advantages(group, ctx)`; interface in `rbd/sspo/plugin.py`). |
| Prohibited actions | Changing the model, LoRA shape, sampling, number of steps, problems per step, runs per problem, learning rate, optimizer, loss form, context or turn limits; training on dev or test problems or any data outside the training pool; reading or copying Judge-only test files or checker cases; modifying the checker, selector, evaluator or report code; submitting an adapter not produced by the fixed training command; editing logs or metrics. |
| Reference baseline | The released SSPO step-30 adapter (evaluation-ready artifact). Solution materializes it by copying the release asset; no training or evaluation is needed. Dr. GRPO and PKPO adapters are additional references. *Contributor chooses the reference baseline.* |
| Repository evidence paths | `rbd/sspo/advantages.py` (arms), `algorithms/*.py`, `rbd/sspo/builder.py` (training step), `configs/methods/sel_*.yaml` (protocol), `rbd/sspo/evaluate.py` (evaluator), `selectrl/algolib/selector.py` (exact select@k), `rbd/sspo/report.py` (paired bootstrap), `results/test128_step30.md` (reported numbers). |
| Baseline evaluation path | `bash scripts/sel_eval.sh --name ref --adapter <sspo adapter> --splits test` then `python -m rbd.sspo.evaluate score --split-file data/sel/test.jsonl.gz --out runs/eval/ref/test`. |
| Baseline metric reported by the repository | select@8 = 64.28% on the 128-problem test set (seed 0, step 30); frozen 58.20, Dr. GRPO 59.34, PKPO 59.02. Not yet reproduced by OpenRSI. |
| Fixed evaluation protocol | 128 test problems × 16 agent runs; run seeds depend only on (problem, run index), identical for every candidate; up to 8 turns per run inside the model's 40,960-token context, no generation cap; counted unit = one agent run; no resampling or replacement; infrastructure retries rerun the same seed. select@8 and cov@8 exact over all C(16, 8) subsets; candidate-only evaluation (the baseline is not re-run per submission). |
| Evaluation mode and input | Direct evaluation of the submitted LoRA adapter. *Alternative (contributor decision):* evaluation-time retraining from the submitted `algorithms/candidate.py`, which lets Judge enforce the training budget but adds one full training run (~3 h on 8 GPUs) to every Judge step. |
| Score | Mean select@8 over the 128 problems, in percent, higher is better. Adapter that fails to load (wrong rank, missing keys, non-finite weights) or a training run that did not finish: unscored. Agent-environment command failures inside an evaluation run are part of the score (the run simply fails), not candidate failures. |
| Feedback visible to the agent | Aggregate test select@8, cov@8, pass@1 and checker pass rate per submission. The agent can additionally evaluate on the dev set (65 problems, provided in Work) with the same evaluator. |
| Hidden from the agent | Test problem statements, hidden tests, test checker cases, per-problem outcomes and samples. |
| Reward-hacking measures | Judge evaluates only the adapter with its own copy of the evaluator and test files; Judge checks the run manifest (config hash of `configs/methods/sel_file.yaml`, 30 completed steps) and that only `algorithms/candidate.py` differs from the pinned repository. Residual limitation of the shared Base/Work/Judge snapshot: Work is agent-controlled, so adherence to the training protocol is checked from artifacts, not re-executed. |
| Noise and meaningful improvement | One seed: 95% contest-bootstrap intervals have half-width ≈ 5 points; SSPO vs Dr. GRPO = +4.9 [+0.3, +9.8]. Gains of ~5 points are distinguishable; after reproduction, repeat training seeds if observed seed variance could change a conclusion. |
| Web search / external services / extra data | Disabled / none / none (training data fixed to the 231-problem pool). |
| Compute per candidate | *Contributor estimate.* Measured on 1× H200: training ≈ 30 × 10 min ≈ 5 h, test evaluation ≈ 1–1.5 h. Single-node 8-GPU lane (estimate, not yet measured): training ≈ 2.5–3.5 h, Judge ≈ 0.5 h, cycle ≈ 3–4 h → about 12–16 loops in 48 h. 1-GPU lane: cycle ≈ 6.5 h → about 7 loops in 48 h (non-blocking Flag). CPU: ≥ 32 cores for program execution; RAM ≈ 200 GB. |

## 3. Requirements check (proposal-agent rubric, nine gates)

| Gate | Status | Evidence and open items |
|---|---|---|
| 1. Contributor expertise | Likely pass | Public record of the contributor: LLM post-training (DPO data selection, arXiv 2508.04149; shallow preference signals, arXiv 2505.17122), self-evolving agents (Alita, arXiv 2505.20286; survey, arXiv 2507.21046). Needs the contributor-provided email. |
| 2. Source repository | Pass with one risk | Open source, every asset public or released with the repository; single container (vLLM servers and the program sandbox are subprocesses; network isolation via `unshare -rn` is optional); single node, ≤ 8 GPUs. Risk: the onboarding asks for a *representative* project; this repository is new and has no external adoption yet. The coauthored route is allowed, but reviewers may weigh influence. |
| 3. Model-development scope | Pass | RL post-training objective for a coding agent; each trial changes the rule, trains, observes select@8 / cov@8 / selection gap and updates the hypothesis. |
| 4. Traceable baseline | Pass after release | SSPO adapter as an evaluation-ready artifact with its evaluator and reported metric; requires the GitHub release assets. |
| 5. Objective and metric | Pass | Fixed workload (128 × 16 runs), exact metric, counted unit, no resampling, unscored-invalid semantics defined above. |
| 6. Action space | Pass | Credit assignment admits many distinct hypotheses (see "Why research"). Contributor may widen it (e.g. per-turn advantages) — currently out of the interface. |
| 7. Evaluation integrity | Pass with stated residual | Test content Judge-only; aggregate feedback only; Work-produced adapter checked against manifest. CodeContests is public, so the only leakage path is copying test data into training, which is prohibited and needs network access or Judge files that Work does not have. |
| 8. Data and network | Pass | No web, no services, no extra data. |
| 9. Task-generation readiness | Pending contributor decisions | Email; evaluation mode (direct vs retraining); reference baseline (SSPO vs Dr. GRPO); compute lane (1 vs 8 GPUs, GPU type) and runtime confirmation. |

Compute check: within the normal reference for the 8-GPU lane (estimate); Flag for a 1-GPU lane.

## 4. Things to finish before submitting

1. Publish the release assets (problem files, checker cases, three adapters, evaluation rows) and pin the commit.
2. Validate the 8-GPU colocated training path once (all reported runs used one H200 per run).
3. Decide the open contributor items in section 3, gate 9.
4. Optional: a second training seed for SSPO and Dr. GRPO to estimate seed variance.
