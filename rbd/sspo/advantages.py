"""Group advantages of the selection-aware comparison (one group = the n runs of one problem in one step).

Arms (all on the same multi-turn agent, representative programs and loss):
  drgrpo  Dr. GRPO pass@1 (Liu et al., 2503.20783): A_i = g_i - mean(g), no std normalization.
  pkpo    pass@k with the PKPO binary closed form and leave-one-out baseline (Walder & Karkhanis, 2505.15201):
          A_i = (k/n) (A_i^cov - b_i^cov) on f = g.
  sspo    ours (SSPO second version, section 4): A_i = (k/n) [(1 - alpha)(A_i^cov - b_i^cov) + alpha A_i^(S,U)],
          coverage on C = {g = 1, Chk = 1}, uncentered leave-one-out credit of J^- (wrong programs whose signature is
          in Sigma^+_x count as failures), exact DP under cluster purity, subset enumeration otherwise; alpha = 0.7.
The factor k/n (Theorem 2(a)) is folded into the advantages; groups whose advantages are all zero are dropped.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from selectrl.algolib.combin import coverage_loo_advantages, pass_at_k
from selectrl.algolib.selector import classify_group, exact_cov_at_k, exact_select_at_k
from selectrl.algolib.sspo_dp import brute_force_selection_advantages, dp_selection_advantages

ARMS = ("drgrpo", "pkpo", "sspo")


def local_sig_ids(sigs: Sequence[str]) -> List[int]:
    ids: Dict[str, int] = {}
    return [ids.setdefault(s, len(ids)) for s in sigs]


def group_stats(g: Sequence[int], chk: Sequence[int], sigs: Sequence[str], sigma_plus: Set[str],
                k: int = 8) -> Dict[str, float]:
    """Class counts and the group's own pass@1 / cov@k / select@k (exact over all k-subsets of the n runs)."""
    n = len(g)
    ids = local_sig_ids(sigs)
    known = [1 if s in sigma_plus else 0 for s in sigs]
    cl = classify_group(g, chk, ids, known)
    kk = min(k, n)
    return {"n_C": cl.n_C, "n_T": cl.n_T, "n_W_star": cl.n_W_star, "n_W": int(sum(cl.n_W_clusters)),
            "max_wrong_cluster": max(cl.n_W_clusters, default=0), "impure": int(cl.impure),
            "false_reject": cl.n_false_reject, "pass1": float(np.mean(g)) if n else 0.0,
            "passk": pass_at_k(n, int(sum(g)), kk) if n else 0.0,
            "covk": exact_cov_at_k(g, chk, kk) if n else 0.0,
            "selk": exact_select_at_k(g, chk, ids, kk) if n else 0.0}


def group_advantages(arm: str, g: Sequence[int], chk: Sequence[int], sigs: Sequence[str], sigma_plus: Set[str],
                     k: int = 8, alpha: float = 0.7) -> Tuple[np.ndarray, Dict[str, float]]:
    g_a = np.asarray([int(x) for x in g], dtype=np.int64)
    chk_a = np.asarray([int(x) for x in chk], dtype=np.int64)
    n = len(g_a)
    info: Dict[str, float] = {}
    if n < 2:
        return np.zeros(n), info
    if arm == "drgrpo":
        adv = g_a.astype(np.float64) - g_a.mean()
    elif arm == "pkpo":
        kk = min(k, n - 1)
        a, b = coverage_loo_advantages(g_a, kk)
        adv = (kk / n) * (a - b)
    elif arm == "sspo":
        kk = min(k, n - 1)
        ids = local_sig_ids(sigs)
        known = [1 if s in sigma_plus else 0 for s in sigs]
        cl = classify_group(g_a, chk_a, ids, known)
        if cl.impure:
            sel, _, _ = brute_force_selection_advantages(g_a, chk_a, ids, known, kk)
        else:
            sel = dp_selection_advantages(cl, kk)
        in_c = np.zeros(n, dtype=np.int64)
        in_c[list(cl.c_idx)] = 1
        a, b = coverage_loo_advantages(in_c, kk)
        cov = a - b
        adv = (kk / n) * ((1.0 - alpha) * cov + alpha * np.asarray(sel, dtype=np.float64))
        info = {"sel_term_mean": float(np.mean(sel)), "cov_term_mean": float(np.mean(cov)),
                "sel_term_wrong_min": float(np.min(np.asarray(sel)[g_a == 0])) if (g_a == 0).any() else 0.0}
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return np.asarray(adv, dtype=np.float64), info
