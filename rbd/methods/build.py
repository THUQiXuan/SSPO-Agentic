"""Rank-0 step construction for every method (what goes into ``steps/<phase>_XXXX/batch.pt``).

drgrpo     B0: multi-turn Dr. GRPO, A_i = r_i - mean(r) on every assistant token, constant normalizer.
rft_grpo   B1: phase "rft" = SFT on successful base-model episodes (8 per training task), then Dr. GRPO.
pivotopd   B2: Dr. GRPO + critical-step distillation: at the failed branch's divergence state (error erased) the
               student matches a teacher that is told the successful direction, on the successful action.
rbd        ours: Dr. GRPO + synthesized error -> feedback -> rollback -> correct-continuation trajectories with
               mixture-teacher KL on the pivot turns, decision-slot unlikelihood and recovery-value weights.
"""
from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rbd.agent.loop import Rollout, stable_seed
from rbd.config import resolve
from rbd.envs.factory import make_env_factory
from rbd.serving.client import ServerPool
from rbd.types import Trajectory
from rbd.utils import log


def read_tasks(path: str) -> List[dict]:
    with gzip.open(resolve(path), "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def episode_seq(t: Trajectory) -> Tuple[List[int], List[int], List[float]]:
    """Full token sequence of an episode (trailing observation dropped) and its assistant-token positions."""
    ids = list(t.prompt_ids)
    pos: List[int] = []
    rlp: List[float] = []
    for i, turn in enumerate(t.turns):
        pos += range(len(ids), len(ids) + len(turn.action_ids))
        rlp += turn.action_logprobs
        ids += turn.action_ids
        if i < len(t.turns) - 1:
            ids += turn.obs_ids
    return ids, pos, rlp


class StepBuilder:
    def __init__(self, cfg: dict, run_dir: Path, servers: Path, seed: int, fmt, state: dict):
        self.cfg, self.run_dir, self.seed, self.fmt = cfg, run_dir, seed, fmt
        self.m, self.t = cfg["method"], cfg["task"]
        self.pool = ServerPool.from_state(servers)
        self.make_env = make_env_factory(self.t)
        self.rollout = Rollout(self.pool, fmt, self.make_env, max_turns=int(self.t["max_turns"]),
                               l_ctx=self.pool.max_model_len)
        self.tasks = read_tasks(self.t["train"])
        self.by_pid = {t["pid"]: t for t in self.tasks}
        bs = state.get("builder") or {}
        self.cursor, self.epoch = int(bs.get("cursor", 0)), int(bs.get("epoch", 0))
        self.rft_data: Optional[List[dict]] = None

    # ---------------------------------------------------------------- bookkeeping
    def state(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch}

    def restore(self, st: dict) -> None:
        self.cursor, self.epoch = int(st["cursor"]), int(st["epoch"])

    def _next_tasks(self, k: int) -> List[dict]:
        out: List[dict] = []
        while len(out) < k:
            order = sorted(self.by_pid)
            random.Random(stable_seed("order", self.seed, self.epoch)).shuffle(order)
            take = order[self.cursor:self.cursor + k - len(out)]
            out += [self.by_pid[p] for p in take]
            self.cursor += len(take)
            if self.cursor >= len(order):
                self.epoch, self.cursor = self.epoch + 1, 0
        return out

    def phases(self) -> List[Tuple[str, int]]:
        ph: List[Tuple[str, int]] = []
        if self.m["name"] == "rft_grpo":
            data = self._rft_data()
            r = self.m["rft"]
            ph.append(("rft", max(1, math.ceil(len(data) / int(r["batch_seqs"]))) * int(r["epochs"])))
        ph.append(("rl", int(self.m["steps"])))
        return ph

    # ---------------------------------------------------------------- B1 data
    def _rft_data(self) -> List[dict]:
        if self.rft_data is not None:
            return self.rft_data
        path = self.run_dir / "rft_data.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                self.rft_data = json.load(f)
            return self.rft_data
        n = int(self.m["rft"]["samples_per_task"])
        trajs = self.rollout.run(self.tasks, n, seed=stable_seed("rft", self.seed), adapter=None, group_prefix="rft")
        data = []
        for t in trajs:
            if t.reward:
                ids, pos, rlp = episode_seq(t)
                data.append({"ids": ids, "pos": pos, "rlp": rlp, "adv": 1.0, "kind": "sft", "task": t.task_id})
        log(f"[rft] {len(data)} successful episodes from {len(trajs)} ({len(self.tasks)} tasks x {n})")
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
        self.rft_data = data
        return data

    # ---------------------------------------------------------------- steps
    def build(self, phase: str, step: int, adapter: Optional[str]) -> Dict[str, Any]:
        if phase == "rft":
            return self._build_rft(step)
        return self._build_rl(step, adapter)

    def _build_rft(self, step: int) -> Dict[str, Any]:
        data = self._rft_data()
        bsz = int(self.m["rft"]["batch_seqs"])
        per_epoch = max(1, math.ceil(len(data) / bsz))
        epoch, i = divmod(step - 1, per_epoch)
        order = list(range(len(data)))
        random.Random(stable_seed("rft-order", self.seed, epoch)).shuffle(order)
        seqs = [data[j] for j in order[i * bsz:(i + 1) * bsz]]
        n_tok = sum(len(s["pos"]) for s in seqs)
        return {"seqs": seqs, "mini_batches": [list(range(len(seqs)))], "norms": [max(1, n_tok)], "pivots": [],
                "metrics": {"seqs": len(seqs), "sft_tokens": n_tok}, "builder_state": self.state()}

    def _build_rl(self, step: int, adapter: Optional[str]) -> Dict[str, Any]:
        import time

        G = int(self.m["group_size"])
        tasks = self._next_tasks(int(self.m["tasks_per_step"]))
        t0 = time.time()
        trajs = self.rollout.run(tasks, G, seed=stable_seed("rl", self.seed, step), adapter=adapter,
                                 group_prefix=f"s{step}")
        t_roll = time.time() - t0
        groups: Dict[str, List[Trajectory]] = {}
        for t in trajs:
            groups.setdefault(t.group_id, []).append(t)
        seqs, gid_of = [], []
        for gid, g in groups.items():
            r = np.array([t.reward for t in g], dtype=np.float64)
            if r.min() == r.max():
                continue  # no gradient without a KL term
            for t, a in zip(g, r - r.mean()):
                ids, pos, rlp = episode_seq(t)
                if pos:
                    seqs.append({"ids": ids, "pos": pos, "rlp": rlp, "adv": float(a), "kind": "grpo",
                                 "task": t.task_id})
                    gid_of.append(gid)
        M = max(1, int(self.m.get("ppo_mini_batches", 1)))
        gids = sorted(set(gid_of))
        random.Random(stable_seed("mb", self.seed, step)).shuffle(gids)
        part = {g: i % M for i, g in enumerate(gids)}
        mbs: List[List[int]] = [[] for _ in range(M)]
        for i, g in enumerate(gid_of):
            mbs[part[g]].append(i)
        norm = max(1.0, len(tasks) * G / M * int(self.m.get("loss_norm_tokens", 4096)))
        t1 = time.time()
        pivots, pstats = [], {}
        if self.m["name"] in ("rbd", "pivotopd"):
            raise ValueError("pivot-synthesis methods are not part of this repository")
        rewards = [t.reward for t in trajs]
        metrics = {"reward": round(float(np.mean(rewards)), 4), "seqs": len(seqs), "pivots": len(pivots),
                   "mixed_groups": len(gids), "groups": len(groups),
                   "mean_turns": float(np.mean([len(t.turns) for t in trajs])),
                   "submit_rate": float(np.mean([t.finish == "submit" for t in trajs])),
                   "context_hit_rate": float(np.mean([t.finish == "context" for t in trajs])),
                   "mean_tokens": float(np.mean([t.num_tokens() for t in trajs])),
                   "t_rollout_s": round(t_roll, 1), "t_pivot_s": round(time.time() - t1, 1), **pstats}
        samples = self.run_dir / "steps" / f"rl_{step:04d}" / "episodes.jsonl.gz"
        samples.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(samples, "wt") as f:
            for t in trajs:
                f.write(json.dumps({"task": t.task_id, "k": t.sample_idx, "reward": t.reward, "finish": t.finish,
                                    "turns": [{"cmd": x.command, "rc": x.exit_code, "test": x.is_test,
                                               "wrote": x.wrote} for x in t.turns]}) + "\n")
        return {"seqs": seqs, "mini_batches": mbs, "norms": [norm] * M, "pivots": pivots, "metrics": metrics,
                "builder_state": self.state()}
