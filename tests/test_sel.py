"""Selection-aware arms: advantages against direct definitions, and the representative-program rule."""
import itertools
import math
import random
from types import SimpleNamespace

import numpy as np

from rbd.sspo.advantages import group_advantages, local_sig_ids
from rbd.sspo.signals import solution_versions


def _js(sub, g, chk, sig):
    cl = {}
    for i in sub:
        if chk[i]:
            cl.setdefault(sig[i], []).append(i)
    if not cl:
        return 0.0
    v = max(len(m) for m in cl.values())
    win = [m for m in cl.values() if len(m) == v]
    return sum(sum(g[i] for i in m) / len(m) for m in win) / len(win)


def _spec_sspo(g, chk, sig, sigma_plus, k, alpha):
    """SSPO second version, section 4, written directly from the definitions (subset enumeration)."""
    n = len(g)
    wstar = {i for i in range(n) if g[i] == 0 and chk[i] == 1 and sig[i] in sigma_plus}
    jm = lambda sub: 0.0 if any(i in wstar for i in sub) else _js(sub, g, chk, sig)  # noqa: E731
    in_c = [1 if g[i] and chk[i] else 0 for i in range(n)]
    nc = sum(in_c)
    out = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        subs = list(itertools.combinations(others, k - 1))
        a_s = sum(jm(s + (i,)) - jm(s) for s in subs) / len(subs)
        a_cov = in_c[i] * math.comb(n - nc, k - 1) / math.comb(n - 1, k - 1)
        npc = nc - in_c[i]
        V = lambda j: 1 - math.comb(n - 1 - npc, j) / math.comb(n - 1, j)  # noqa: E731
        out.append((k / n) * ((1 - alpha) * (a_cov - (V(k) - V(k - 1))) + alpha * a_s))
    return np.array(out)


def _random_group(rng, n):
    g, chk, sig = [], [], []
    for _ in range(n):
        u = rng.random()
        if u < 0.3:
            g.append(1); chk.append(1); sig.append("A" if rng.random() < 0.9 else "B")
        elif u < 0.55:
            g.append(0); chk.append(1); sig.append(rng.choice("ABCD"))
        else:
            g.append(rng.choice([0, 0, 0, 1])); chk.append(0); sig.append("t%d" % rng.randint(0, 3))
    return g, chk, sig


def test_sspo_matches_definition():
    rng = random.Random(3)
    worst = 0.0
    for trial in range(120):
        n, k = rng.choice([(8, 4), (10, 5), (12, 6)]) if trial % 6 else (16, 8)
        g, chk, sig = _random_group(rng, n)
        sp = set(rng.sample(["A", "B", "C"], rng.randint(0, 2)))
        a, _ = group_advantages("sspo", g, chk, sig, sp, k=k, alpha=0.7)
        worst = max(worst, float(np.max(np.abs(a - _spec_sspo(g, chk, sig, sp, k, 0.7)))))
    assert worst < 1e-12, worst


def test_sspo_sign_properties():
    rng = random.Random(5)
    for _ in range(80):
        g, chk, sig = _random_group(rng, 16)
        c_sigs = {s for s, gi, ci in zip(sig, g, chk) if gi and ci}
        if len(c_sigs) > 1 or any(s in c_sigs for s, gi, ci in zip(sig, g, chk) if ci and not gi):
            continue  # Theorem 2's sign property assumes cluster purity
        a, _ = group_advantages("sspo", g, chk, sig, set(), k=8)
        for i in range(16):
            if g[i] == 0:  # wrong programs never get positive credit (no centering of the selection term)
                assert a[i] <= 1e-12


def test_pkpo_and_drgrpo():
    g = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a, _ = group_advantages("pkpo", g, [1] * 16, ["x"] * 16, set(), k=8)
    c, n, k = sum(g), 16, 8
    expect = []
    for gi in g:
        closed = gi * math.comb(n - c, k - 1) / math.comb(n, k)  # PKPO binary closed form
        npc = c - gi
        V = lambda j: 1 - math.comb(n - 1 - npc, j) / math.comb(n - 1, j)  # noqa: E731
        expect.append(closed - (k / n) * (V(k) - V(k - 1)))
    assert np.allclose(a, expect, atol=1e-12)
    d, _ = group_advantages("drgrpo", g, [1] * 16, ["x"] * 16, set())
    assert np.allclose(d, np.array(g) - np.mean(g))


def test_local_ids():
    assert local_sig_ids(["b", "a", "b", "c"]) == [0, 1, 0, 2]


def test_solution_versions_rule():
    stub = "import sys\n\n# write your solution here\n"
    turns = [SimpleNamespace(files_before={"solution.py": stub}),
             SimpleNamespace(files_before={"solution.py": "print(1)\n"}),
             SimpleNamespace(files_before={"solution.py": "print(1)\n"}),
             SimpleNamespace(files_before={"solution.py": "print(2)\n"})]
    t = SimpleNamespace(turns=turns, final_files={"solution.py": "print(3)\n"})
    assert solution_versions(t) == ["print(1)\n", "print(2)\n", "print(3)\n"]


def test_algorithm_files_match_builtin_arms():
    """algorithms/{sspo,drgrpo,pkpo}.py through the plug-in interface reproduce the built-in arms exactly."""
    from rbd.sspo.plugin import call_algorithm, load_algorithm

    rng = random.Random(11)
    ctx = {"k": 8, "alpha": 0.7, "n": 16, "step": 1, "total_steps": 30}
    for arm in ("sspo", "drgrpo", "pkpo"):
        mod = load_algorithm(f"algorithms/{arm}.py")
        for _ in range(20):
            g, chk, sig = _random_group(rng, 16)
            sp = set(rng.sample(["A", "B", "C"], rng.randint(0, 2)))
            extras = {"turns": [8] * 16, "tokens": [3000] * 16, "finish": ["max_turns"] * 16, "n_versions": [1] * 16}
            a = call_algorithm(mod, g, chk, sig, sp, extras, ctx)
            b, _ = group_advantages(arm, g, chk, sig, sp, k=8, alpha=0.7)
            assert np.array_equal(a, b)
    cand = load_algorithm("algorithms/candidate.py")
    g, chk, sig = _random_group(rng, 16)
    assert np.array_equal(call_algorithm(cand, g, chk, sig, set(), {}, ctx), group_advantages("sspo", g, chk, sig, set())[0])
