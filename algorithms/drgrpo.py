"""Reference algorithm: Dr. GRPO pass@1 (Liu et al., 2025, arXiv 2503.20783): A_i = g_i - mean(g), no std normalization.

Identical to the built-in arm used for the reported runs (rbd.sspo.advantages.group_advantages, arm "drgrpo").
"""
from rbd.sspo.advantages import group_advantages


def compute_advantages(group, ctx):
    adv, _ = group_advantages("drgrpo", group["g"], group["chk"], group["sig"], group["sigma_plus"],
                              k=int(ctx["k"]), alpha=float(ctx["alpha"]))
    return adv
