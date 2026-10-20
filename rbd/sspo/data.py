"""Benchmark subsets for the agentic selection experiment (CodeContests, contest-disjoint).

    python -m rbd.sspo.data prepare --src <selectrl data/problems dir> --out data/sel \
        [--n-test 200 --n-dev 96 --n-cand 960 --max-rating 2200]

Source: the filtered CodeContests problems of SelectRL-Bench (multi-answer statements removed, hidden tests =
private + generated tests, Python reference solutions kept as ``py_refs``), already split by contest into
train_candidates / val / test. Here:
  test       whole contests of the source test split in sha256 order until >= n_test problems,
  dev        whole contests of the source val split in sha256 order until >= n_dev problems,
  candidates n_cand problems of train_candidates (Codeforces rating <= max_rating or unknown) in sha256 order;
             the training pool is the calibrated subset of the candidates (rbd.sspo.evaluate active).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from rbd.config import resolve
from rbd.utils import atomic_write_json, log

SALT = "sspo-agentic-20261018"


def _rows(path: Path) -> List[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _h(x: str) -> str:
    return hashlib.sha256(f"{SALT}|{x}".encode()).hexdigest()


def by_contest(rows: List[dict], target: int) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("contest_id", r["pid"]))].append(r)
    out: List[dict] = []
    for cid in sorted(groups, key=_h):
        if len(out) >= target:
            break
        out += sorted(groups[cid], key=lambda r: r["pid"])
    return out


def _rating_ok(r: dict, max_rating: int) -> bool:
    d = str(r.get("difficulty", "unknown"))
    return not d.isdigit() or int(d) <= max_rating


def prepare(a: argparse.Namespace) -> None:
    src = Path(a.src)
    test = by_contest(_rows(src / "test.jsonl.gz"), a.n_test)
    dev = by_contest(_rows(src / "val.jsonl.gz"), a.n_dev)
    cand_all = [r for r in _rows(src / "train_candidates.jsonl.gz") if _rating_ok(r, a.max_rating)
                and r.get("public") and r.get("hidden")]
    cand = sorted(cand_all, key=lambda r: _h(r["pid"]))[: a.n_cand]
    for r in test + dev:
        r.pop("py_refs", None)
    out = resolve(a.out)
    _write(out / "test.jsonl.gz", test)
    _write(out / "dev.jsonl.gz", dev)
    _write(out / "candidates.jsonl.gz", cand)
    c = lambda rs: len({r.get("contest_id") for r in rs})  # noqa: E731
    ov = ({r.get("contest_id") for r in test} & {r.get("contest_id") for r in dev}) | \
         ({r.get("contest_id") for r in cand} & ({r.get("contest_id") for r in test} | {r.get("contest_id") for r in dev}))
    stats = {"test": len(test), "test_contests": c(test), "dev": len(dev), "dev_contests": c(dev),
             "candidates": len(cand), "candidate_contests": c(cand), "contest_overlap": len(ov),
             "max_rating": a.max_rating, "source": str(src)}
    atomic_write_json(out / "stats.json", stats)
    log(f"[data] {stats}")


def pool(a: argparse.Namespace) -> None:
    """Candidate pool for re-selecting dev / test with the training-pool rule (base agent solves 1-7 of 8):
    every problem of the source val and test splits, plus ``n_extra`` unused train candidates whose contests are
    disjoint from every calibrated training candidate (so from the training pool too)."""
    src = Path(a.src)
    base = _rows(src / "test.jsonl.gz") + _rows(src / "val.jsonl.gz")
    excl = {str(r.get("contest_id")) for r in _rows(resolve(a.exclude))}
    seen = {r["pid"] for r in base}
    extra = [r for r in _rows(src / "train_candidates.jsonl.gz")
             if str(r.get("contest_id")) not in excl and r["pid"] not in seen and _rating_ok(r, a.max_rating)
             and r.get("public") and r.get("hidden")]
    extra = sorted(extra, key=lambda r: _h("pool|" + r["pid"]))[: a.n_extra]
    rows = base + extra
    for r in rows:
        r.pop("py_refs", None)
    out = resolve(a.out)
    _write(out, rows)
    c_pool = {str(r.get("contest_id")) for r in rows}
    stats = {"pool": len(rows), "from_val_test": len(base), "extra_train_candidates": len(extra),
             "contests": len(c_pool), "contest_overlap_with_calibrated_candidates": len(c_pool & excl)}
    atomic_write_json(out.with_name(out.name.replace(".jsonl.gz", "") + "_stats.json"), stats)
    log(f"[data] pool {stats}")


def split_eval(a: argparse.Namespace) -> None:
    """Split the selected evaluation problems into dev / test by whole contests (sha256 order)."""
    rows = _rows(resolve(a.selected))
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("contest_id", r["pid"]))].append(r)
    dev: List[dict] = []
    test: List[dict] = []
    for cid in sorted(groups, key=lambda c: _h("split|" + c)):
        (dev if len(dev) < a.dev_frac * len(rows) else test).extend(sorted(groups[cid], key=lambda r: r["pid"]))
    out = resolve(a.out)
    _write(out / "dev.jsonl.gz", dev)
    _write(out / "test.jsonl.gz", test)
    c = lambda rs: len({r.get("contest_id") for r in rs})  # noqa: E731
    stats = {"selected": len(rows), "dev": len(dev), "dev_contests": c(dev), "test": len(test),
             "test_contests": c(test), "rule": "frozen base agent solves 1-7 of 8 calibration runs (seed 777)"}
    atomic_write_json(out / "eval_split_stats.json", stats)
    log(f"[data] eval split {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True)
    p.add_argument("--out", default="data/sel")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-dev", type=int, default=96)
    p.add_argument("--n-cand", type=int, default=960)
    p.add_argument("--max-rating", type=int, default=2200)
    q = sub.add_parser("pool")
    q.add_argument("--src", required=True)
    q.add_argument("--exclude", default="data/sel/candidates.jsonl.gz")
    q.add_argument("--out", default="data/sel2/pool.jsonl.gz")
    q.add_argument("--n-extra", type=int, default=600)
    q.add_argument("--max-rating", type=int, default=2200)
    t = sub.add_parser("split-eval")
    t.add_argument("--selected", required=True)
    t.add_argument("--out", default="data/sel")
    t.add_argument("--dev-frac", type=float, default=0.33)
    a = ap.parse_args()
    {"prepare": prepare, "pool": pool, "split-eval": split_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
