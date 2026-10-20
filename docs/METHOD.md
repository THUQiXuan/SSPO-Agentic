# Method details

## Deployed system and metric

For a problem x the agent is run k = 8 times. Each run's **representative program** y(τ) is the last version of
`solution.py` in the run that passes every public example (else the last version). The deployed **selector S**
drops programs that fail the frozen checker (Chk = 0), groups the rest by output signature, picks one of the largest
clusters uniformly at random and submits a uniformly random member. **select@k** = P(the submitted program passes
all hidden tests), computed exactly as the mean over all k-subsets of the n = 16 evaluation runs
(`selectrl/algolib/selector.py: exact_select_at_k`). **cov@k** = P(some run in the subset is correct and passes Chk).
The **selection gap** cov@k − select@k is the loss caused by the vote.

## Frozen execution checker

Per problem, a frozen Qwen3-8B (thinking off) writes 4 input generators, 4 brute-force solvers and 2 input validators
once (`selectrl/checker/gen.py`, `rbd/sspo/helpers.py`). `selectrl/checker/build.py` runs them to produce stress
inputs, small / edge inputs and confirmed references (≥ 2 brute-force solvers agree). Chk(y) = 1 iff y passes the
public examples, finishes every stress input within the time limit and matches every confirmed reference.
sig(y) = hash of y's normalized outputs on all check inputs. Hidden-test correctness g(y) is the reward oracle
during training and the score at evaluation; it is never visible to the selector.

## SSPO advantages

Classes of the n runs: C = {g = 1, Chk = 1}, T = {Chk = 0}, W* = {g = 0, Chk = 1, sig ∈ Σ⁺_x}, W_m = remaining wrong
programs that pass Chk, grouped by signature. Σ⁺_x holds signatures of programs known to be correct for x (the
reference solutions, then correct programs of earlier steps); it is fixed before the step's advantages.

    A_i = (k/n) [ (1 − α)(A_i^cov − b_i^cov) + α A_i^(S,U) ],   k = 8, α = 0.7

* A_i^cov = 1[i ∈ C] · C(n − n_C, k − 1) / C(n − 1, k − 1); b_i^cov = V(k) − V(k − 1) with V(j) the pass@j
  U-statistic of the other n − 1 runs (`selectrl/algolib/combin.py`).
* A_i^(S,U) = mean over (k − 1)-subsets K' of the other runs of J⁻(K' ∪ {i}) − J⁻(K'), with
  J⁻(K) = J_S(K) · 1[K ∩ W* = ∅]. Exact dynamic program over cluster sizes when C has one signature and no wrong
  cluster shares it (`selectrl/algolib/sspo_dp.py`), enumeration of all subsets otherwise. Not centered: a
  centered selection term would give positive credit to runs that fail the checker.

Baselines: Dr. GRPO, A_i = g_i − mean(g); PKPO pass@8, A_i = (k/n)(A_i^cov − b_i^cov) on f = g.
`tests/test_sel.py` checks the implementation against a direct subset-enumeration of these definitions.

## Training

Each step: 16 problems from the training pool × 16 runs; Σ⁺ update after the step; every assistant token of run i
receives A_i; PPO-clipped surrogate (0.2 / 0.28), constant normalizer (Dr. GRPO), 2 mini-batches, AdamW lr 1e-5 on a
LoRA (r = 64, α = 128) of all attention and MLP projections, no KL term, 30 steps. Groups whose advantages are all zero
are skipped. Rollouts are served by vLLM with the current adapter; with sleep mode the server releases its memory
during the update. The trainer aborts if the mean |log-prob| gap between serving and training exceeds 0.08 (it stayed
≤ 0.009 in all runs).

## Context handling

The model's full default window (40,960 tokens) is used; a turn may generate up to the remaining window. Command
outputs shown to the agent keep the first and last 5,000 characters. Training processes every sequence at full
length (a sequence longer than the packing budget is processed alone; nothing is truncated), with gradient
checkpointing and chunked LM-head log-probabilities.
