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

## 2. Fixed setting

| Item | Value | Reason |
|---|---|---|
| Reference baseline | SSPO, released step-30 adapter (training seed 0) | the human reference method of the task |
| Evaluation mode | direct evaluation of the submitted LoRA adapter (candidate-only) | the evaluator scores the adapter itself; retraining in Judge would double every cycle |
| Training per candidate (Work) | 30 steps × 16 problems × 16 agent runs = 7,680 runs; problems from the 231-problem pool (≈ 2.1 epochs); training seed fixed to 0 for every candidate (same problem order and rollout seeds as the reference adapter); LoRA r = 64, lr 1e-5, 2 PPO mini-batches per step; up to 8 turns, 40,960-token context, no generation cap; the step-30 adapter is the deliverable | identical to the protocol of the reported results, so the released SSPO adapter is a matched baseline without retraining. At step 15 the methods were not yet distinguishable on the test set (SSPO − Dr. GRPO +2.35 [−2.26, +6.77]); at step 30 they were (+4.94 [+0.26, +9.78]), so a shorter run would remove the signal the task is about |
| Evaluation per candidate (Judge) | 128 test problems (114 contests) × 16 agent runs = 2,048 runs, run seeds fixed (eval seed 20261018); exact select@8 over all C(16, 8) subsets | unchanged test set: all re-selected held-out problems not used for the dev set |
| Optional evaluation in Work | dev set, 65 problems × 16 runs (≈ 10–15 min on 8 GPUs), counted in Work time | lets the agent compare candidates without spending Judge submissions |
| Execution lane | one node, 8 × H200 (141 GB), ≥ 64 CPU cores, ≥ 400 GB RAM; Work and Judge use the same 8 GPUs one after the other | H200 is the only card measured to run uncapped 40,960-token agent rollouts without stalling the KV cache; H100-80G is not validated for this lane |
| Estimated cycle | Work 2.5–3.5 h + Judge 0.3–0.5 h + startup 0.2 h ≈ 3.0–4.2 h → about 11–16 complete loops in 48 h | within the normal reference (≥ 10 loops in 48 h); above the 3-hour soft preference because every rollout step waits for its longest 40,960-token run |

Basis of the cycle estimate (to be replaced by the measurement of section 3): on one H200 a training step took 417–944 s
(rollout of 256 runs 304–825 s, checker scoring ≈ 90 s, update ≈ 80 s) and the test evaluation 1–1.5 h. With 8 GPUs
each vLLM server holds 32 runs per step, so the rollout is bounded by the longest single run (≈ 3–5 min when a run fills
the context), not by throughput; the update is data-parallel with the same global batch and loss normalizer, so the
8-GPU lane trains with the same protocol as the 1-GPU runs.

## 3. Seeds: what is repeated and what is not

* **Every candidate of the research loop: one training seed (0) and one evaluation, not repeated.** A second seed per
  candidate would roughly double the cycle (≈ 7–8 h, fewer than 7 loops in 48 h). The evaluation of a fixed adapter
  uses fixed run seeds, so repeating it measures nothing new.
* **SSPO baseline: two additional training seeds (1 and 2), once, before the agent run.** The only evidence for the
  level of SSPO is one seed, while the gains the task is about are a few points (SSPO beat Dr. GRPO by 4.9), the same
  order as plausible seed-to-seed variation of a 30-step LoRA RL run. The agent also keeps the best of about 12
  single-seed results, which inflates that result when the spread is not small. Without the spread no meaningful-gain
  threshold can be set. The same job validates the 8-GPU lane and measures Work and Judge time
  (`cluster/openrsi_validate_8gpu.sh`: re-score the released seed-0 adapter, train and score seeds 1 and 2;
  about 8 h on one 8 × H200 node).
* **Threshold after that job.** Let s be the standard deviation of test select@8 across SSPO seeds 0, 1, 2 and m their
  mean. A final candidate is a meaningful improvement when its seed-0 score exceeds m by at least G = max(2, 3s)
  points (one seed against a three-seed mean has standard deviation ≈ 1.15 s, and the final candidate is the best of
  about 12, so 3s ≈ 2.6 standard deviations). Only when the final candidate lies above m but below m + G is it
  retrained with seeds 1 and 2; the conclusion then uses its three-seed mean against m with threshold 2s (≈ 2.4
  standard deviations of a difference of two three-seed means). Otherwise no further seeds are run.
* Dr. GRPO and PKPO are not baselines of this task and need no additional seeds for it.

## 4. Draft of the proposal fields

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
| Prohibited actions | Changing the model, LoRA shape, sampling, training seed, number of steps, problems per step, runs per problem, learning rate, optimizer, loss form, context or turn limits; training on dev or test problems or any data outside the training pool; reading or copying Judge-only test files or checker cases; modifying the checker, selector, evaluator or report code; submitting an adapter not produced by the fixed training command; editing logs or metrics. |
| Reference baseline | The released SSPO step-30 adapter, training seed 0 (evaluation-ready artifact). Solution materializes it by copying the release asset; no training or evaluation is needed. |
| Repository evidence paths | `rbd/sspo/advantages.py` (arms), `algorithms/*.py`, `rbd/sspo/builder.py` (training step), `configs/methods/sel_*.yaml` (protocol), `rbd/sspo/evaluate.py` (evaluator), `selectrl/algolib/selector.py` (exact select@k), `rbd/sspo/report.py` (paired bootstrap), `results/test128_step30.md` (reported numbers). |
| Baseline evaluation path | `bash scripts/sel_eval.sh --name ref --adapter <sspo adapter> --splits test` then `python -m rbd.sspo.evaluate score --split-file data/sel/test.jsonl.gz --out runs/eval/ref/test`. |
| Baseline metric reported by the repository | select@8 = 64.28% on the 128-problem test set (seed 0, step 30); frozen 58.20, Dr. GRPO 59.34, PKPO 59.02. Not yet reproduced by OpenRSI. |
| Fixed evaluation protocol | 128 test problems × 16 agent runs; run seeds depend only on (problem, run index), identical for every candidate; up to 8 turns per run inside the model's 40,960-token context, no generation cap; counted unit = one agent run; no resampling or replacement; infrastructure retries rerun the same seed. select@8 and cov@8 exact over all C(16, 8) subsets; candidate-only evaluation (the baseline is not re-run per submission). The candidate adapter must come from the fixed training protocol of section 2 with training seed 0. |
| Evaluation mode and input | Direct evaluation of the submitted LoRA adapter. |
| Score | Mean select@8 over the 128 problems, in percent, higher is better. Adapter that fails to load (wrong rank, missing keys, non-finite weights) or a training run that did not finish: unscored. Agent-environment command failures inside an evaluation run are part of the score (the run simply fails), not candidate failures. |
| Feedback visible to the agent | Aggregate test select@8, cov@8, pass@1 and checker pass rate per submission. The agent can additionally evaluate on the dev set (65 problems, provided in Work) with the same evaluator. |
| Hidden from the agent | Test problem statements, hidden tests, test checker cases, per-problem outcomes and samples. |
| Reward-hacking measures | Judge evaluates only the adapter with its own copy of the evaluator and test files; Judge checks the run manifest (config hash of `configs/methods/sel_file.yaml`, 30 completed steps) and that only `algorithms/candidate.py` differs from the pinned repository. Residual limitation of the shared Base/Work/Judge snapshot: Work is agent-controlled, so adherence to the training protocol is checked from artifacts, not re-executed. |
| Noise and meaningful improvement | Proposal stage: with one seed, the three credit rules already separated by about 5 points on this test set (SSPO − Dr. GRPO = +4.9 [+0.3, +9.8]), so a change of the credit rule can move the score by an amount that one seed already resolves on this test set. After baseline reproduction: SSPO seeds 1 and 2 give the seed spread s; the meaningful-gain threshold and the rule for retraining the final candidate with more seeds are in section 3. |
| Web search / external services / extra data | Disabled / none / none (training data fixed to the 231-problem pool). |
| Compute per candidate | One node, 8 × H200, ≥ 64 CPU cores, ≥ 400 GB RAM for both phases. Work (training) ≈ 2.5–3.5 h, Judge (test evaluation) ≈ 0.3–0.5 h, startup ≈ 0.2 h; cycle ≈ 3.0–4.2 h → about 11–16 loops in 48 h (estimate from 1-GPU measurements, section 2; replaced by the 8-GPU measurement before submission). |

## 5. Requirements check (proposal-agent rubric, nine gates)

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
| 9. Task-generation readiness | Pending two items | Contributor email; measured 8-GPU cycle time (section 3 job). Baseline, evaluation mode, workload and lane are fixed in section 2. |

Compute check: within the normal reference (estimate of 11–16 loops in 48 h on 8 × H200).

## 6. Things to finish before submitting

1. Publish the release assets (problem files, checker cases, three adapters, evaluation rows) and pin the commit.
2. Run `cluster/openrsi_validate_8gpu.sh` on one 8 × H200 node: it validates the Work and Judge lane, measures the
   cycle time and gives SSPO seeds 1 and 2; then fill in s, m and G (section 3) and the measured times.
3. Contributor email.
