"""Candidate algorithm: the only file a research agent edits (see rbd/sspo/plugin.py for the interface).

It starts as a copy of the SSPO reference (algorithms/sspo.py). Any credit-assignment rule over the n runs of a
problem may be implemented here, e.g. a different selection objective, variance reduction, use of Sigma^+_x, or
run-level statistics such as turns and tokens. The training loop, sampling, loss form, checker, selector and
evaluation are fixed.
"""
from rbd.sspo.advantages import group_advantages


def compute_advantages(group, ctx):
    adv, _ = group_advantages("sspo", group["g"], group["chk"], group["sig"], group["sigma_plus"],
                              k=int(ctx["k"]), alpha=float(ctx["alpha"]))
    return adv
