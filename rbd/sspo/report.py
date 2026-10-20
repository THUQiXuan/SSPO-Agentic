"""Results table and paired comparisons from scored evaluations (runs/eval/<name>/<split>.rows.jsonl).

    python -m rbd.sspo.report --split test --runs frozen=runs/eval/frozen sspo=runs/eval/sel_sspo_s0_40 ... \
        [--ref sspo] [--out runs/report_test.md]

Per run: mean over problems of pass@1, pass@8, cov@8, select@8, gap@8, chk rate, mean turns. Paired differences
of select@8 (and pass@1) between the reference run and every other run on the common problems, with a 95%
percentile bootstrap that resamples whole contests (10,000 draws), a two-sided bootstrap p-value and Holm correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from rbd.config import resolve

METRICS = ("pass@1", "pass@8", "cov@8", "select@8", "gap@8")


def load_rows(prefix: Path, split: str) -> Dict[str, dict]:
    """Rows of one split, or of several splits joined with '+' (e.g. dev+test; problem ids are disjoint)."""
    out: Dict[str, dict] = {}
    for sp in split.split("+"):
        p = Path(f"{prefix}/{sp}.rows.jsonl")
        out.update({r["pid"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())})
    return out


def paired(a: Dict[str, dict], b: Dict[str, dict], metric: str, n_boot: int = 10000, seed: int = 0) -> dict:
    pids = sorted(set(a) & set(b))
    d = np.array([a[p][metric] - b[p][metric] for p in pids], dtype=np.float64)
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(pids):
        groups.setdefault(str(a[p].get("contest_id", p)), []).append(i)
    idx = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[j] for j in pick])
        boots[t] = d[sel].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"n": len(pids), "contests": len(idx), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, p))}


def holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    out = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", required=True, help="label=runs/eval/<name>")
    ap.add_argument("--ref", default="sspo")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = {}
    for item in a.runs:
        lab, path = item.split("=", 1)
        runs[lab] = load_rows(resolve(path), a.split)
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    lines = [f"split {a.split}: {len(common)} common problems "
             f"({len({runs[next(iter(runs))][p].get('contest_id') for p in common})} contests); numbers in %", "",
             "| run | pass@1 | pass@8 | cov@8 | select@8 | gap@8 | Chk rate | turns |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lab, rows in runs.items():
        rs = [rows[p] for p in common]
        m = {k: 100 * np.mean([r[k] for r in rs]) for k in METRICS}
        chk = 100 * np.mean([c for r in rs for c in r["chk"]])
        turns = np.mean([r["mean_turns"] for r in rs])
        lines.append(f"| {lab} | {m['pass@1']:.2f} | {m['pass@8']:.2f} | {m['cov@8']:.2f} | {m['select@8']:.2f} | "
                     f"{m['gap@8']:.2f} | {chk:.1f} | {turns:.2f} |")
    if a.ref in runs:
        for metric in ("select@8", "pass@1"):
            others = [lab for lab in runs if lab != a.ref]
            res = [paired({p: runs[a.ref][p] for p in common}, {p: runs[o][p] for p in common}, metric)
                   for o in others]
            ph = holm([r["p"] for r in res])
            lines += ["", f"paired {metric}: {a.ref} minus other (points, 95% contest-cluster bootstrap, Holm)", "",
                      "| comparison | diff | 95% CI | p | p_holm |", "|---|---:|---|---:|---:|"]
            for o, r, h in zip(others, res, ph):
                lines.append(f"| {a.ref} - {o} | {100 * r['diff']:+.2f} | [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] | "
                             f"{r['p']:.3f} | {h:.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(resolve(a.out)).write_text(text)


if __name__ == "__main__":
    main()
