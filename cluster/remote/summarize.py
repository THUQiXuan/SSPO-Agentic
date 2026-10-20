"""Report of the 8-GPU validation run (cluster/openrsi_validate_8gpu.sh).

Writes runs/openrsi_validate/report.md, report.json and artifacts.tar.gz (step-30 adapters of the new runs,
per-problem rows, training metrics, timing, report; no raw completions). Compares the new runs with the released
reference rows (reference/eval/*, 1-GPU runs) on dev, test and dev+test, and the released SSPO adapter scored on
this machine with its released rows (cross-machine reproducibility of the evaluator).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics as st
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from rbd.sspo.report import holm, load_rows, paired  # noqa: E402

OUT = Path("runs/openrsi_validate")
SEEDS = os.environ.get("SEEDS", "0").split()
RUNS = {"frozen (released)": "reference/eval/frozen",
        "Dr. GRPO s0 (released)": "reference/eval/sel_drgrpo_s0",
        "PKPO s0 (released)": "reference/eval/sel_pkpo_s0",
        "SSPO s0 1-GPU (released rows)": "reference/eval/sel_sspo_s0",
        "SSPO s0 1-GPU (rescored here)": "runs/eval/val_released_s0_8g"}
NEW = {f"SSPO s{s} 8-GPU (trained here)": f"runs/eval/val_sspo_s{s}_8g" for s in SEEDS}
RUNS.update(NEW)
METRICS = ("pass@1", "cov@8", "select@8", "gap@8")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table(split: str) -> tuple[list[str], dict]:
    rows = {lab: load_rows(Path(path), split) for lab, path in RUNS.items()
            if all(Path(f"{path}/{sp}.rows.jsonl").exists() for sp in split.split("+"))}
    common = sorted(set.intersection(*(set(r) for r in rows.values())))
    res = {"problems": len(common), "runs": {}}
    lines = [f"**{split}**: {len(common)} problems; numbers in %", "",
             "| policy | pass@1 | cov@8 | select@8 | gap@8 | Chk rate | turns |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lab, r in rows.items():
        m = {k: 100 * st.mean(r[p][k] for p in common) for k in METRICS}
        m["chk"] = 100 * st.mean(c for p in common for c in r[p]["chk"])
        m["turns"] = st.mean(r[p]["mean_turns"] for p in common)
        res["runs"][lab] = m
        lines.append(f"| {lab} | {m['pass@1']:.2f} | {m['cov@8']:.2f} | {m['select@8']:.2f} | {m['gap@8']:.2f} | "
                     f"{m['chk']:.1f} | {m['turns']:.2f} |")
    res["paired"] = {}
    for new in [lab for lab in NEW if lab in rows]:
        others = [lab for lab in rows if lab != new and lab not in NEW]
        a = {p: rows[new][p] for p in common}
        ps = [paired(a, {p: rows[o][p] for p in common}, "select@8") for o in others]
        hs = holm([x["p"] for x in ps])
        lines += ["", f"paired select@8, {new} minus other (points; 95% contest-cluster bootstrap; Holm)", "",
                  "| comparison | diff | 95% CI | p | p (Holm) |", "|---|---:|---|---:|---:|"]
        for o, x, h in zip(others, ps, hs):
            res["paired"][f"{new} - {o}"] = {**x, "p_holm": h}
            lines.append(f"| − {o} | {100 * x['diff']:+.2f} | [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}] | "
                         f"{x['p']:.3f} | {h:.3f} |")
    return lines, res


def timing() -> tuple[list[str], dict]:
    last = {}
    p = OUT / "timing.tsv"
    for line in (p.read_text().splitlines() if p.exists() else []):
        name, ev, t = line.split("\t")
        last[(name, ev)] = int(t)  # the last attempt of a stage; earlier ones were interrupted
    res, lines = {}, ["| stage | wall time (min) |", "|---|---:|"]
    for (name, ev), t in sorted(last.items()):
        if ev.endswith("_end") and (name, ev.replace("_end", "_start")) in last:
            m = (t - last[(name, ev.replace("_end", "_start"))]) / 60
            res[f"{name} {ev[:-4]}"] = round(m, 1)
            lines.append(f"| {name} {ev[:-4]} (last attempt) | {m:.1f} |")
    for s in SEEDS:
        mp = Path(f"runs/sel_sspo_s{s}_8g/metrics.jsonl")
        if not mp.exists():
            continue
        m = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
        steps = [r["t_step_s"] for r in m]
        tr = {"steps": len(m), "sum_step_min": round(sum(steps) / 60, 1),
              "median_step_min": round(st.median(steps) / 60, 2),
              "median_rollout_min": round(st.median(r["t_rollout_s"] for r in m) / 60, 2),
              "median_score_min": round(st.median(r["t_score_s"] for r in m) / 60, 2),
              "max_mismatch_abs_lp": max((r.get("mismatch_abs_lp") or 0.0) for r in m),
              "mean_batch_sel8": st.mean(r["batch_sel8"] for r in m)}
        ref = Path("reference/train/sel_sspo_s0.metrics.jsonl")
        if ref.exists():
            rm = [json.loads(x) for x in ref.read_text().splitlines() if x.strip()]
            tr["reference_1gpu_mean_batch_sel8"] = st.mean(r["batch_sel8"] for r in rm)
            tr["step1_batch_sel8_here_vs_1gpu"] = [m[0]["batch_sel8"], rm[0]["batch_sel8"]]
        res[f"train seed {s}"] = tr
        lines.append(f"| Work: SSPO seed {s}, sum of {len(m)} steps (median step {tr['median_step_min']} min, "
                     f"rollout {tr['median_rollout_min']} min, scoring {tr['median_score_min']} min) | "
                     f"{tr['sum_step_min']} |")
    return lines, res


def rescore_check() -> dict:
    a, b = Path("reference/eval/sel_sspo_s0"), Path("runs/eval/val_released_s0_8g")
    if not all((d / "test.rows.jsonl").exists() for d in (a, b)):
        return {}
    ra, rb = load_rows(a, "test"), load_rows(b, "test")
    common = sorted(set(ra) & set(rb))
    d = [rb[p]["select@8"] - ra[p]["select@8"] for p in common]
    return {"problems": len(common), "select@8_released": 100 * st.mean(ra[p]["select@8"] for p in common),
            "select@8_rescored_here": 100 * st.mean(rb[p]["select@8"] for p in common),
            "mean_abs_per_problem_diff": 100 * st.mean(abs(x) for x in d),
            "identical_problems": sum(1 for x in d if abs(x) < 1e-12)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"host": platform.node(), "python": platform.python_version(),
           "packages": sh(f"{sys.executable} -c \"import vllm, torch, transformers, peft; "
                          f"print(vllm.__version__, torch.__version__, transformers.__version__, peft.__version__)\""),
           "gpus": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c"),
           "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
           "commit": sh("git rev-parse HEAD"), "dirty_files": sh("git status --porcelain --untracked-files=no"),
           "assets": {p.name: sha256(p) for p in sorted(Path("downloads").glob("*.tar.gz"))}}
    md = ["# 8-GPU validation run", "",
          f"host `{env['host']}`, commit `{env['commit']}`" + (" (local changes!)" if env["dirty_files"] else ""),
          f"GPUs: {env['gpus']}; CPU cores: {env['cpus']}; vllm / torch / transformers / peft: {env['packages']}", ""]
    out = {"env": env, "seeds": SEEDS}
    for split in ("test", "dev", "dev+test"):
        try:
            lines, res = table(split)
        except (FileNotFoundError, TypeError, ValueError) as e:
            lines, res = [f"**{split}**: unavailable ({e})"], {}
        md += lines + [""]
        out[split] = res
    rc = rescore_check()
    out["rescore_check"] = rc
    if rc:
        md += ["**Evaluator reproducibility** (released SSPO adapter, same run seeds, released rows vs scored here): "
               f"test select@8 {rc['select@8_released']:.2f} vs {rc['select@8_rescored_here']:.2f}; mean absolute "
               f"per-problem difference {rc['mean_abs_per_problem_diff']:.2f} points; "
               f"{rc['identical_problems']}/{rc['problems']} problems identical.", ""]
    tl, tr = timing()
    out["timing"] = tr
    md += ["**Timing**", ""] + tl + [""]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    (OUT / "report.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\n".join(md))
    with tarfile.open(OUT / "artifacts.tar.gz", "w:gz") as tar:
        for f in [OUT / "report.md", OUT / "report.json", OUT / "timing.tsv"]:
            if f.exists():
                tar.add(f)
        for s in SEEDS:
            for f in (Path(f"runs/sel_sspo_s{s}_8g/ckpt/rl_0030/adapter"), Path(f"runs/sel_sspo_s{s}_8g/metrics.jsonl")):
                if f.exists():
                    tar.add(f)
        for d in ["runs/eval/val_released_s0_8g"] + [f"runs/eval/val_sspo_s{s}_8g" for s in SEEDS]:
            for f in sorted(Path(d).glob("*.rows.jsonl")) + sorted(Path(d).glob("*.summary.json")):
                tar.add(f)
    print(f"artifacts: {OUT / 'artifacts.tar.gz'} sha256 {sha256(OUT / 'artifacts.tar.gz')}")


if __name__ == "__main__":
    main()
