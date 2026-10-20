# Data and released assets

Attached to the GitHub release `v1.0` of this repository (checksums in `cluster/remote/assets.sha256`); extract every
archive at the repository root (`cluster/remote/fetch_assets.sh` downloads, verifies and extracts them):

| asset | content |
|---|---|
| `sel_problems.tar.gz` | `data/sel/train.jsonl.gz` (231-problem training pool, with reference solutions and hidden tests), `data/sel/dev.jsonl.gz` (65), `data/sel/test.jsonl.gz` (128) |
| `checker_cases.tar.gz` | `cache/checkers/cases/<pid>.json.gz` for the 424 problems above (check inputs, validity, confirmed references) |
| `adapters_step30.tar.gz` | `reference/adapters/{sel_sspo_s0,sel_drgrpo_s0,sel_pkpo_s0}/`: LoRA adapters (vLLM key layout) at step 30 |
| `eval_rows.tar.gz` | `reference/eval/{frozen,sel_sspo_s0,sel_drgrpo_s0,sel_pkpo_s0}/{dev,test}.{rows.jsonl,summary.json}`: per-problem rows of the reported results; `reference/train/<run>.metrics.jsonl`: per-step training metrics |

Problem records come from `deepmind/code_contests` (CC BY 4.0) through `selectrl/data/codecontests.py`; hidden tests
are the private plus generated tests (at most 40 per problem).
