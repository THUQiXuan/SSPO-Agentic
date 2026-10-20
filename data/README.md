# Data and released assets

Attached to the GitHub release `v1.0` of this repository:

| asset | content |
|---|---|
| `sel_problems.tar.gz` | `data/sel/train.jsonl.gz` (231-problem training pool, with reference solutions and hidden tests), `data/sel/dev.jsonl.gz` (65), `data/sel/test.jsonl.gz` (128), split statistics |
| `checker_cases.tar.gz` | `cache/checkers/cases/<pid>.json.gz` for the 424 problems above (inputs, validity, confirmed references) and the helper programs that produced them |
| `adapters_step30.tar.gz` | LoRA adapters (vLLM key layout) of `sel_sspo_s0`, `sel_drgrpo_s0`, `sel_pkpo_s0` at step 30 |
| `eval_rows.tar.gz` | per-problem rows (`*.rows.jsonl`) and summaries of the frozen model and the three adapters on dev and test |

Problem records come from `deepmind/code_contests` (CC BY 4.0) through `selectrl/data/codecontests.py`; hidden tests
are the private plus generated tests (at most 40 per problem). Extract the archives at the repository root.
