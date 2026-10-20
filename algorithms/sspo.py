"""Reference algorithm: SSPO (ours): selection-aware set policy optimization for the deployed execution selector.
A_i = (k/n) [(1 - alpha)(A_i^cov - b_i^cov) + alpha A_i^(S,U)] with coverage on C = {g = 1, Chk = 1} and the
uncentered leave-one-out credit of J^- (wrong programs whose signature is in Sigma^+ count as failures);
exact dynamic program under cluster purity, enumeration of all subsets otherwise.

Identical to the built-in arm used for the reported runs (rbd.sspo.advantages.group_advantages, arm "sspo").
"""
from rbd.sspo.advantages import group_advantages


def compute_advantages(group, ctx):
    adv, _ = group_advantages("sspo", group["g"], group["chk"], group["sig"], group["sigma_plus"],
                              k=int(ctx["k"]), alpha=float(ctx["alpha"]))
    return adv
