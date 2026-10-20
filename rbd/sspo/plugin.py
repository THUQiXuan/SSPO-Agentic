"""Algorithm plug-in interface: the single file a research agent edits.

An algorithm file defines

    def compute_advantages(group: dict, ctx: dict) -> numpy.ndarray   # shape (n,)

called once per training problem and step with the signals of its n agent runs (training split only):

  group["g"]           hidden-test correctness of each run's representative program y(tau), 0/1
  group["chk"]         frozen-checker verdict Chk(y), 0/1
  group["sig"]         execution-signature id (str); equal ids = identical outputs on all check inputs
  group["sigma_plus"]  set of signatures known to be correct for this problem before this step (Sigma^+_x)
  group["turns"], group["tokens"], group["finish"], group["n_versions"]   per-run episode statistics
  ctx = {"k": 8, "alpha": 0.7, "n": n, "step": s, "total_steps": T}

Every assistant token of run i receives advantage[i] (PPO-clipped surrogate, constant Dr. GRPO normalizer; fixed).
A group whose advantages are all zero is skipped. Non-finite values or a wrong shape make the candidate invalid.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Set

import numpy as np

from rbd.config import resolve


def load_algorithm(path: str | Path) -> ModuleType:
    p = resolve(path)
    spec = importlib.util.spec_from_file_location(f"sspo_algorithm_{p.stem}", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load algorithm file {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "compute_advantages", None)):
        raise AttributeError(f"{p} does not define compute_advantages(group, ctx)")
    return mod


def call_algorithm(mod: ModuleType, g: List[int], chk: List[int], sigs: List[str], sigma_plus: Set[str],
                   extras: Dict[str, list], ctx: Dict[str, int | float]) -> np.ndarray:
    group = {"g": list(g), "chk": list(chk), "sig": list(sigs), "sigma_plus": set(sigma_plus), **extras}
    adv = np.asarray(mod.compute_advantages(group, dict(ctx)), dtype=np.float64).reshape(-1)
    if adv.shape[0] != len(g) or not np.all(np.isfinite(adv)):
        raise ValueError(f"compute_advantages returned shape {adv.shape} / non-finite values for n={len(g)}")
    return adv
