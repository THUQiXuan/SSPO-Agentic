"""Rank-0 step construction for the selection-aware arms (drgrpo / pkpo / sspo on the same agent).

Per step: tasks_per_step problems x group_size multi-turn agent runs (the policy edits solution.py with bash
commands, may run the public examples, and submits; at most ``max_turns`` turns inside the model's full context
window). For every run the representative program y(tau) is scored by the frozen checker (g, Chk, sig); the arm's
advantages are computed per problem with Sigma^+_x fixed before the step (SSPO section 4), and Sigma^+_x is then
extended with this step's correct signatures. Every assistant token of a run receives the run's advantage.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from rbd.agent.loop import stable_seed
from rbd.config import resolve
from rbd.methods.build import StepBuilder, episode_seq
from rbd.sspo.advantages import group_advantages, group_stats
from rbd.sspo.signals import Scorer, solution_versions
from rbd.types import Trajectory
from rbd.utils import log


class SelStepBuilder(StepBuilder):
    def __init__(self, cfg: dict, run_dir, servers, seed: int, fmt, state: dict):
        os.environ["RBD_ENV_EVAL"] = "0"  # the env's own final-file reward is not used: g comes from y(tau)
        super().__init__(cfg, run_dir, servers, seed, fmt, state)
        self.arm = str(self.m["sel_arm"])
        self.algo = None
        if self.arm == "file":  # research-agent candidate (algorithms/*.py), see rbd/sspo/plugin.py
            from rbd.sspo.plugin import load_algorithm

            self.algo = load_algorithm(self.m["algorithm_file"])
        self.k = int(self.m.get("k_deploy", 8))
        self.alpha = float(self.m.get("alpha", 0.7))
        self.scorer = Scorer(self.by_pid, resolve(self.t.get("cache_root", "cache")),
                             resolve(self.t.get("cases_dir", "cache/checkers/cases")))
        missing = [p for p in self.by_pid if not self.scorer.has_cases(p)]
        self.require_cases = os.environ.get("RBD_SEL_ALLOW_NOCASES", "0") != "1"  # smoke tests only
        if missing and self.require_cases:
            raise RuntimeError(f"{len(missing)} training problems have no checker cases, e.g. {missing[:5]}")
        bs = state.get("builder") or {}
        if "sigma_plus" in bs:
            self.sigma_plus: Dict[str, Set[str]] = {p: set(v) for p, v in bs["sigma_plus"].items()}
        else:
            self.sigma_plus = self._init_sigma_plus()

    def _init_sigma_plus(self) -> Dict[str, Set[str]]:
        """Sigma^+_x before training: signatures of the reference solutions that are correct and pass the checker."""
        items = [(pid, code) for pid, p in self.by_pid.items() for code in (p.get("py_refs") or [])[:2]]
        recs = self.scorer.score(items, require_cases=self.require_cases) if items else []
        sp: Dict[str, Set[str]] = {p: set() for p in self.by_pid}
        for (pid, _), r in zip(items, recs):
            if r["g"] == 1 and r["chk"] == 1:
                sp[pid].add(r["sig"])
        log(f"[sel] Sigma^+ initialized from {len(items)} reference programs: "
            f"{sum(1 for v in sp.values() if v)}/{len(sp)} problems with a known-correct signature")
        return sp

    def state(self) -> dict:
        st = super().state()
        st["sigma_plus"] = {p: sorted(v) for p, v in self.sigma_plus.items()}
        return st

    def restore(self, st: dict) -> None:
        super().restore(st)
        if "sigma_plus" in st:
            self.sigma_plus = {p: set(v) for p, v in st["sigma_plus"].items()}

    def phases(self):
        return [("rl", int(self.m["steps"]))]

    def _build_rl(self, step: int, adapter: Optional[str]) -> Dict[str, Any]:
        G = int(self.m["group_size"])
        tasks = self._next_tasks(int(self.m["tasks_per_step"]))
        t0 = time.time()
        trajs = self.rollout.run(tasks, G, seed=stable_seed("rl", self.seed, step), adapter=adapter,
                                 group_prefix=f"s{step}")
        t_roll = time.time() - t0
        t1 = time.time()
        reps = self.scorer.representatives([(t.task_id, solution_versions(t)) for t in trajs])
        recs = self.scorer.score([(t.task_id, r) for t, r in zip(trajs, reps)], require_cases=self.require_cases)
        t_score = time.time() - t1

        groups: Dict[str, List[int]] = {}
        for i, t in enumerate(trajs):
            groups.setdefault(t.group_id, []).append(i)
        seqs, gid_of, stats, infos = [], [], [], []
        new_correct: Dict[str, Set[str]] = {}
        for gid, idx in groups.items():
            pid = trajs[idx[0]].task_id
            g = [recs[i]["g"] for i in idx]
            chk = [recs[i]["chk"] for i in idx]
            sigs = [recs[i]["sig"] for i in idx]
            sp = self.sigma_plus.setdefault(pid, set())  # fixed before this step's update
            stats.append(group_stats(g, chk, sigs, sp, self.k))
            if self.algo is not None:
                from rbd.sspo.plugin import call_algorithm

                extras = {"turns": [len(trajs[i].turns) for i in idx], "tokens": [trajs[i].num_tokens() for i in idx],
                          "finish": [trajs[i].finish for i in idx],
                          "n_versions": [len(solution_versions(trajs[i])) for i in idx]}
                ctx = {"k": self.k, "alpha": self.alpha, "n": len(idx), "step": step,
                       "total_steps": int(self.m["steps"])}
                adv, info = call_algorithm(self.algo, g, chk, sigs, sp, extras, ctx), {}
            else:
                adv, info = group_advantages(self.arm, g, chk, sigs, sp, self.k, self.alpha)
            infos.append(info)
            new_correct.setdefault(pid, set()).update(s for s, gi, ci in zip(sigs, g, chk) if gi == 1 and ci == 1)
            if np.all(np.abs(adv) < 1e-12):
                continue
            for i, a in zip(idx, adv):
                ids, pos, rlp = episode_seq(trajs[i])
                if pos:
                    seqs.append({"ids": ids, "pos": pos, "rlp": rlp, "adv": float(a), "kind": "grpo",
                                 "task": pid})
                    gid_of.append(gid)
        for pid, s in new_correct.items():  # Sigma^+_x is extended only after this step's advantages
            self.sigma_plus.setdefault(pid, set()).update(s)

        M = max(1, int(self.m.get("ppo_mini_batches", 1)))
        gids = sorted(set(gid_of))
        random.Random(stable_seed("mb", self.seed, step)).shuffle(gids)
        part = {g: i % M for i, g in enumerate(gids)}
        mbs: List[List[int]] = [[] for _ in range(M)]
        for i, g in enumerate(gid_of):
            mbs[part[g]].append(i)
        norm = max(1.0, len(tasks) * G / M * int(self.m.get("loss_norm_tokens", 4096)))

        def mean(key: str, rows: List[dict]) -> float:
            vals = [r[key] for r in rows if key in r]
            return round(float(np.mean(vals)), 5) if vals else 0.0

        metrics = {"reward": mean("pass1", stats), "seqs": len(seqs), "pivots": 0, "mixed_groups": len(gids),
                   "groups": len(groups), "chk_rate": round(float(np.mean([r["chk"] for r in recs])), 4),
                   "batch_pass1": mean("pass1", stats), "batch_cov8": mean("covk", stats),
                   "batch_sel8": mean("selk", stats), "batch_pass8": mean("passk", stats),
                   "impure_rate": mean("impure", stats), "n_C": mean("n_C", stats), "n_T": mean("n_T", stats),
                   "n_W": mean("n_W", stats), "n_W_star": mean("n_W_star", stats),
                   "max_wrong_cluster": mean("max_wrong_cluster", stats),
                   "sigma_plus_nonempty": round(float(np.mean([bool(self.sigma_plus.get(t["pid"])) for t in tasks])), 4),
                   "mean_turns": float(np.mean([len(t.turns) for t in trajs])),
                   "submit_rate": float(np.mean([t.finish == "submit" for t in trajs])),
                   "context_hit_rate": float(np.mean([t.finish == "context" for t in trajs])),
                   "nocode_rate": float(np.mean([r is None for r in reps])),
                   "mean_tokens": float(np.mean([t.num_tokens() for t in trajs])),
                   "max_tokens": int(max(t.num_tokens() for t in trajs)),
                   "adv_abs_mean": round(float(np.mean([abs(s["adv"]) for s in seqs])), 5) if seqs else 0.0,
                   "t_rollout_s": round(t_roll, 1), "t_score_s": round(t_score, 1)}
        if self.arm == "sspo" and infos:
            metrics["sel_term_mean"] = mean("sel_term_mean", infos)
            metrics["cov_term_mean"] = mean("cov_term_mean", infos)
        samples = self.run_dir / "steps" / f"rl_{step:04d}" / "episodes.jsonl.gz"
        samples.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(samples, "wt") as f:
            for t, r, rec in zip(trajs, reps, recs):
                f.write(json.dumps({"task": t.task_id, "k": t.sample_idx, "g": rec["g"], "chk": rec["chk"],
                                    "sig": rec["sig"], "finish": t.finish, "turns": len(t.turns),
                                    "tokens": t.num_tokens(), "code": r,
                                    "commands": [x.command for x in t.turns]}) + "\n")
        return {"seqs": seqs, "mini_batches": mbs, "norms": [norm] * M, "pivots": [], "metrics": metrics,
                "builder_state": self.state()}
