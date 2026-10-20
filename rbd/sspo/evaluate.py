"""Deployment evaluation of a policy as a multi-turn agent with the execution selector (select@k), and calibration.

    gen     python -m rbd.sspo.evaluate gen --servers S --split-file data/sel/test.jsonl.gz --n 16 \
                --adapter none|DIR --out runs/eval/<name>/test
            n agent runs per problem (seeds depend only on the problem and the run index, so every policy is
            evaluated on identical seeds), representative program y(tau) per run; resumable by chunk
            (<out>.samples/chunk_XXXX.jsonl.gz).
    score   python -m rbd.sspo.evaluate score --split-file F --out P
            frozen-checker signals of every y(tau) -> per-problem rows (<out>.rows.jsonl: unbiased pass@k, exact
            cov@k and select@k over all k-subsets, critical flag) and <out>.summary.json.
    active  python -m rbd.sspo.evaluate active --split-file data/sel/candidates.jsonl.gz --out P \
                --train-out data/sel/train.jsonl.gz
            calibration: keep candidates whose base pass count over the n runs is strictly between 0 and n.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from rbd.config import load, resolve
from rbd.utils import atomic_write_json, log

K_LIST = (1, 2, 4, 8, 16)


def _rows(path: Path) -> List[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_gz(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def _scorer(cfg: dict, problems: List[dict]):
    from rbd.sspo.signals import Scorer

    t = cfg["task"]
    return Scorer({p["pid"]: p for p in problems}, resolve(t.get("cache_root", "cache")),
                  resolve(t.get("cases_dir", "cache/checkers/cases")))


def gen(a: argparse.Namespace) -> None:
    os.environ["RBD_ENV_EVAL"] = "0"
    from rbd.eval.run_eval import _adapter, make_rollout
    from rbd.sspo.signals import solution_versions

    cfg = load(a.model, a.task, "sel_drgrpo")
    probs = _rows(resolve(a.split_file))[: a.max_tasks or None][a.shard::a.nshards]
    ro, pool = make_rollout(cfg, resolve(a.servers))
    adapter = _adapter(pool, a.adapter)
    scorer = _scorer(cfg, probs)
    out_dir = Path(f"{resolve(a.out)}.samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [probs[i:i + a.chunk] for i in range(0, len(probs), a.chunk)]
    for ci, chunk in enumerate(chunks):
        path = out_dir / (f"chunk_{ci:04d}.jsonl.gz" if a.nshards == 1 else f"chunk_s{a.shard:02d}_{ci:04d}.jsonl.gz")
        if path.exists():
            continue
        trajs = ro.run(chunk, a.n, seed=a.seed, adapter=adapter, group_prefix="eval")
        reps = scorer.representatives([(t.task_id, solution_versions(t)) for t in trajs])
        _write_gz(path, [{"pid": t.task_id, "k": t.sample_idx, "code": r, "finish": t.finish, "turns": len(t.turns),
                          "tokens": t.num_tokens(), "n_versions": len(solution_versions(t))}
                         for t, r in zip(trajs, reps)])
        log(f"[eval-gen] chunk {ci + 1}/{len(chunks)} ({len(trajs)} runs) -> {path.name}")
    atomic_write_json(Path(f"{resolve(a.out)}.gen_done{'' if a.nshards == 1 else f'_s{a.shard:02d}'}.json"),
                      {"problems": len(probs), "n": a.n, "adapter": a.adapter, "seed": a.seed})


def _samples(prefix: Path) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = {}
    for p in sorted(Path(f"{prefix}.samples").glob("chunk_*.jsonl.gz")):
        for r in _rows(p):
            by.setdefault(r["pid"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["k"])
    return by


def critical(g: List[int], chk: List[int], sig: List[int], k: int) -> int:
    """1 if some k-subset has checked coverage but the selector can fail on it (J_S < J_cov)."""
    from selectrl.algolib.selector import selector_success

    n = len(g)
    if not any(gi and ci for gi, ci in zip(g, chk)) or not any((not gi) and ci for gi, ci in zip(g, chk)):
        return 0
    for sub in itertools.combinations(range(n), min(k, n)):
        if any(g[i] and chk[i] for i in sub) and selector_success(sub, g, chk, sig) < 1.0:
            return 1
    return 0


def score(a: argparse.Namespace) -> dict:
    from rbd.sspo.advantages import local_sig_ids
    from selectrl.algolib.combin import pass_at_k
    from selectrl.algolib.selector import exact_cov_at_k, exact_select_at_k

    cfg = load(a.model, a.task, "sel_drgrpo")
    probs = {p["pid"]: p for p in _rows(resolve(a.split_file))}
    by = _samples(resolve(a.out))
    pids = [p for p in probs if p in by]
    scorer = _scorer(cfg, [probs[p] for p in pids])
    if a.require_cases:
        missing = [p for p in pids if not scorer.has_cases(p)]
        if missing:
            log(f"[eval-score] {len(missing)} problems without checker cases are left out, e.g. {missing[:5]}")
            pids = [p for p in pids if p not in set(missing)]
    items = [(p, r["code"]) for p in pids for r in by[p]]
    recs = scorer.score(items, require_cases=a.require_cases)
    rows, off = [], 0
    for p in pids:
        rs = by[p]
        rec = recs[off:off + len(rs)]
        off += len(rs)
        g = [int(x["g"]) for x in rec]
        chk = [int(x["chk"]) for x in rec]
        sig = local_sig_ids([x["sig"] for x in rec])
        n = len(g)
        row = {"pid": p, "contest_id": probs[p].get("contest_id", p), "n": n, "g": g, "chk": chk, "sig": sig,
               "mean_turns": float(np.mean([r["turns"] for r in rs])),
               "mean_tokens": float(np.mean([r["tokens"] for r in rs])),
               "context_hit": float(np.mean([r["finish"] == "context" for r in rs])),
               "submit": float(np.mean([r["finish"] == "submit" for r in rs])),
               "nocode": float(np.mean([r["code"] is None for r in rs]))}
        for k in K_LIST:
            if k <= n:
                row[f"pass@{k}"] = pass_at_k(n, sum(g), k)
                row[f"cov@{k}"] = exact_cov_at_k(g, chk, k)
                row[f"select@{k}"] = exact_select_at_k(g, chk, sig, k)
        row["gap@8"] = row.get("cov@8", 0.0) - row.get("select@8", 0.0)
        row["critical@8"] = critical(g, chk, sig, 8)
        rows.append(row)
    out = resolve(a.out)
    Path(f"{out}.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = [k for k in rows[0] if k.startswith(("pass@", "cov@", "select@", "gap@", "critical@", "mean_", "context",
                                                 "submit", "nocode"))] if rows else []
    summ = {"problems": len(rows), "runs": sum(r["n"] for r in rows),
            "chk_rate": float(np.mean([c for r in rows for c in r["chk"]])) if rows else 0.0,
            "false_reject_rate": float(np.mean([gi and not ci for r in rows for gi, ci in zip(r["g"], r["chk"])]))
            if rows else 0.0}
    summ.update({k: float(np.mean([r[k] for r in rows if k in r])) for k in keys})
    atomic_write_json(Path(f"{out}.summary.json"), summ)
    log(f"[eval-score] {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()})}")
    return summ


def active(a: argparse.Namespace) -> None:
    cfg = load(a.model, a.task, "sel_drgrpo")
    probs = _rows(resolve(a.split_file))
    by = _samples(resolve(a.out))
    scorer = _scorer(cfg, probs)
    items = [(p["pid"], r["code"]) for p in probs if p["pid"] in by for r in by[p["pid"]]]
    gs = scorer.hidden_labels(items)
    cnt: Dict[str, int] = {}
    nrun: Dict[str, int] = {}
    for (pid, _), g in zip(items, gs):
        cnt[pid] = cnt.get(pid, 0) + g
        nrun[pid] = nrun.get(pid, 0) + 1
    keep = [p for p in probs if p["pid"] in cnt and 0 < cnt[p["pid"]] < nrun[p["pid"]]]
    _write_gz(resolve(a.train_out), keep)
    hist: Dict[str, int] = {}
    for pid, c in cnt.items():
        hist[str(c)] = hist.get(str(c), 0) + 1
    stats = {"candidates": len(probs), "calibrated": len(cnt), "kept": len(keep), "histogram": hist,
             "base_pass1": float(np.mean([cnt[p] / nrun[p] for p in cnt])) if cnt else 0.0}
    atomic_write_json(Path(f"{resolve(a.train_out)}.stats.json"), stats)
    log(f"[active] {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "score", "active"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="qwen3_4b")
        p.add_argument("--task", default="code_sel")
        p.add_argument("--split-file", required=True)
        p.add_argument("--out", required=True)
        if name == "gen":
            p.add_argument("--servers", required=True)
            p.add_argument("--adapter", default="none")
            p.add_argument("--n", type=int, default=16)
            p.add_argument("--seed", type=int, default=20261018)
            p.add_argument("--chunk", type=int, default=32)
            p.add_argument("--max-tasks", type=int, default=0)
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--nshards", type=int, default=1)
        if name == "score":
            p.add_argument("--require-cases", type=int, default=1)
        if name == "active":
            p.add_argument("--train-out", required=True)
    a = ap.parse_args()
    {"gen": gen, "score": score, "active": active}[a.cmd](a)


if __name__ == "__main__":
    main()
