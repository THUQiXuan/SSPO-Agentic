"""Reference algorithm: pass@k with the PKPO binary closed form and leave-one-out baseline (Walder & Karkhanis, 2025,
arXiv 2505.15201): A_i = (k/n)(A_i^cov - b_i^cov) on f = g.

Identical to the built-in arm used for the reported runs (rbd.sspo.advantages.group_advantages, arm "pkpo").
"""
from rbd.sspo.advantages import group_advantages


def compute_advantages(group, ctx):
    adv, _ = group_advantages("pkpo", group["g"], group["chk"], group["sig"], group["sigma_plus"],
                              k=int(ctx["k"]), alpha=float(ctx["alpha"]))
    return adv
