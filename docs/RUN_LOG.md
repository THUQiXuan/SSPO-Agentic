# Run log (seed 0, October 2026)

| stage | outcome |
|---|---|
| Problems | 768 training candidates (Codeforces rating ≤ 2200 or unknown), contest-disjoint from evaluation. |
| Checker | Helpers for about 2,050 problems by frozen Qwen3-8B; replies capped at 8,192 tokens (a few uncapped replies ran to the full window and blocked the server). |
| Training pool | Frozen agent, 8 runs per candidate: 231 problems solved 1–7 times (440 never, 97 always). |
| Gate on the training pool | select@8 72.0, pass@1 46.0 (selector gain +26.0); cov@8 81.0, selection gap 8.9 points; critical problems 16.9%. |
| First test set | The first test set (203 problems from the CodeContests test split and held-out contests, not selected by difficulty) had a selection gap of only 1.4 points because most problems were unsolved by the base model, so dev / test were re-selected with the training-pool rule. |
| Final dev / test | Pool of 1,288 problems from contests disjoint from all training candidates; frozen agent with an independent seed (777), 8 runs; kept 1–7 solves; split by contest: dev 65 / test 128 (114 contests). Evaluation uses seed 20261018. |
| Training | 30 steps per algorithm, one H200 each, vLLM sleep mode; jobs were preempted several times and resumed from the last finished step. |
| Environment fix | Commands longer than the 128 KiB argv limit are run from a script file (earlier they aborted the rollout). |
| Hardware finding | 48 GB cards and A100-80G stalled (KV cache full, ~150–420 tokens/s) on uncapped 40,960-token agent rollouts; H200 ran the same workload about 20× faster. |
| Result | Step 30 on the 128-problem test set: see the README table. Step 15: same ordering, differences within noise. |
