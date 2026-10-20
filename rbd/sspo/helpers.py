"""Checker helper programs written once per problem by a frozen model (SSPO second version: frozen Qwen3-8B).

    python -m rbd.sspo.helpers --servers S --model-path <Qwen3-8B snapshot> --problems F1 F2 ... [--cache cache]

Per problem 4 input generators, 4 brute-force solvers and 2 validators (prompts and sampling of
``selectrl.checker.gen``, thinking off); written to <cache>/checkers/artifacts/<pid>.json.gz (resumable). The cases
are then built on CPU with ``python -m selectrl.checker.build``.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json

from rbd.config import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=8192, help="cap of one helper reply (helper model only)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from rbd.serving.client import ServerPool
    from selectrl.checker import gen as gen_mod
    from selectrl.checker.gen import generate_artifacts

    # Helper programs are short; a reply that runs on for tens of thousands of tokens is degenerate and blocks the
    # server's KV cache. Cap the helper model's replies (the trained policy keeps its full context, uncapped).
    gen_mod.SAMPLING["max_tokens"] = a.max_tokens

    tok = AutoTokenizer.from_pretrained(a.model_path)

    def render(messages):
        txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return list(tok(txt, add_special_tokens=False)["input_ids"])

    probs, seen = [], set()
    for f in a.problems:
        with gzip.open(resolve(f), "rt", encoding="utf-8") as fh:
            for l in fh:
                p = json.loads(l)
                if p["pid"] not in seen:
                    seen.add(p["pid"])
                    probs.append(p)
    probs = sorted(probs, key=lambda p: p["pid"])[a.shard::a.nshards]
    pool = ServerPool.from_state(resolve(a.servers), max_inflight=128)
    out = resolve(a.cache) / "checkers" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(generate_artifacts(probs, out, pool, render, concurrency=a.concurrency))


if __name__ == "__main__":
    main()
