"""CPU tests of rbd.train.losses, including numeric checks of Theorem 3 and Propositions B, C, D."""
from __future__ import annotations

import math

import pytest
import torch

from rbd.train.losses import (aggregate_constant, aggregate_token_mean, drgrpo_token_loss, full_vocab_logprobs,
                              mix_teacher_rows, mixture_log_weights, mixture_teacher_logprobs, pivot_loss,
                              sft_token_loss, slot_unlikelihood, soft_cross_entropy_chunked)


def _rand_dist(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.log_softmax(torch.randn(*shape, generator=g, dtype=torch.float32) * 1.5, dim=-1)


# ---------------------------------------------------------------------------------------------- Dr. GRPO
def test_drgrpo_asymmetric_clipping() -> None:
    ratio = torch.tensor([1.5, 1.25, 0.5, 0.79, 0.5, 1.5, 1.0])
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 2.0])
    old = torch.zeros(7)
    logp = ratio.log().clone().requires_grad_(True)
    loss = drgrpo_token_loss(logp, old, adv, clip_low=0.2, clip_high=0.28)
    # A>0: capped at 1 + 0.28 (1.25 is inside the clip-higher bound); lower side never clips.
    # A<0: floored at 1 - 0.2 (0.79 and 0.5 clip); upper side never clips.
    expected = torch.tensor([-1.28, -1.25, -0.5, 0.8, 0.8, 1.5, -2.0])
    torch.testing.assert_close(loss.detach(), expected, rtol=0, atol=1e-6)
    loss.sum().backward()
    clipped = torch.tensor([True, False, False, True, True, False, False])
    assert torch.all(logp.grad[clipped] == 0)
    # d(-r A)/d logp = -r A on the unclipped tokens
    torch.testing.assert_close(logp.grad[~clipped], -(ratio * adv)[~clipped])
    # a symmetric 0.2 bound would clip 1.25 for A > 0
    sym = drgrpo_token_loss(ratio.log(), old, adv, clip_low=0.2, clip_high=0.2)
    assert sym[1].item() == pytest.approx(-1.2)


def test_aggregations_and_sft() -> None:
    tok = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert aggregate_constant(tok, 8.0).item() == pytest.approx(1.25)
    assert aggregate_constant(tok[:2], 8.0).item() + aggregate_constant(tok[2:], 8.0).item() == pytest.approx(1.25)
    assert aggregate_token_mean(tok).item() == pytest.approx(2.5)
    torch.testing.assert_close(sft_token_loss(torch.tensor([-0.5, -2.0])), torch.tensor([0.5, 2.0]))


# ---------------------------------------------------------------------------------- full-vocabulary pieces
def test_full_vocab_logprobs_matches_direct() -> None:
    g = torch.Generator().manual_seed(0)
    T, H, V = 37, 8, 50
    w = torch.randn(V, H, generator=g)
    h0 = torch.randn(T, H, generator=g)
    up = torch.randn(T, V, generator=g)
    h1, h2 = h0.clone().requires_grad_(True), h0.clone().requires_grad_(True)
    ours = full_vocab_logprobs(h1, w, chunk=8, grad=True)
    ref = torch.log_softmax(h2 @ w.T, dim=-1)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)
    (ours * up).sum().backward()
    (ref * up).sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad, rtol=1e-4, atol=1e-5)
    nograd = full_vocab_logprobs(h1, w, chunk=8, grad=False)
    assert not nograd.requires_grad
    torch.testing.assert_close(nograd, ref.detach(), rtol=1e-5, atol=1e-5)


def test_soft_cross_entropy_equals_kl_and_grad() -> None:
    g = torch.Generator().manual_seed(1)
    T, H, V = 29, 6, 40
    w = torch.randn(V, H, generator=g)
    h0 = torch.randn(T, H, generator=g)
    logq = _rand_dist(g, T, V)
    logq[3, 5] = float("-inf")  # zero teacher mass must not produce NaN
    logq[3] = torch.log_softmax(logq[3], dim=-1)
    h1, h2, h3 = (h0.clone().requires_grad_(True) for _ in range(3))
    kl, rows = soft_cross_entropy_chunked(h1, w, logq, chunk=7)
    ref_terms = torch.nn.functional.kl_div(torch.log_softmax(h2 @ w.T, -1), logq, log_target=True,
                                           reduction="none")
    ref_rows = torch.where(logq.isinf(), torch.zeros_like(ref_terms), ref_terms).sum(-1)  # 0 log 0 = 0
    torch.testing.assert_close(rows, ref_rows.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(kl, ref_rows.sum(), rtol=1e-4, atol=1e-4)
    kl.backward()
    ref_rows.sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad, rtol=1e-4, atol=1e-5)
    # callable target (re-evaluated in backward) gives the same value and gradient
    calls = []

    def fn(s: int, e: int) -> torch.Tensor:
        calls.append((s, e))
        return logq[s:e]

    kl3, _ = soft_cross_entropy_chunked(h3, w, fn, chunk=7)
    kl3.backward()
    torch.testing.assert_close(kl3, kl)
    torch.testing.assert_close(h3.grad, h1.grad)
    assert len(calls) == 2 * math.ceil(T / 7)  # forward + checkpoint recomputation


# ------------------------------------------------------------------------------------ mixture teacher
def _brute_mixture(lps: list, y: list, prior: list, starts: list) -> torch.Tensor:
    T, V = lps[0].shape
    out = torch.zeros(T, V, dtype=torch.float64)
    for k in range(T):
        s = max(st for st in starts if st <= k)
        wts = []
        for u, lp in enumerate(lps):
            like = 1.0
            for j in range(s, k):
                like *= math.exp(lp[j, y[j]].item())
            wts.append(prior[u] * like)
        z = sum(wts)
        for v in range(V):
            out[k, v] = sum(wts[u] / z * math.exp(lps[u][k, v].item()) for u in range(len(lps)))
    return out.log()


def test_mixture_posterior_brute_force() -> None:
    g = torch.Generator().manual_seed(2)
    U, T, V = 3, 4, 5
    lps = [_rand_dist(g, T, V) for _ in range(U)]
    y = [1, 3, 0, 4]
    prior = [0.5, 0.3, 0.2]
    y_t = torch.tensor(y)
    ours = mixture_teacher_logprobs(lps, y_t, prior=torch.tensor(prior), span_starts=[0, 2])
    ref = _brute_mixture(lps, y, prior, [0, 2])
    torch.testing.assert_close(ours.double(), ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours.exp().sum(-1), torch.ones(T), rtol=0, atol=1e-6)
    # posterior restarts at each span start: weights at rows 0 and 2 are the prior
    w = mixture_log_weights(lps, y_t, torch.tensor(prior), [2]).exp()
    torch.testing.assert_close(w[:, 0], torch.tensor(prior))
    torch.testing.assert_close(w[:, 2], torch.tensor(prior))
    # lazy row access equals the dense result
    torch.testing.assert_close(mix_teacher_rows(lps, w.log(), 1, 3), ours[1:3])
    # uniform prior default and no restart
    torch.testing.assert_close(mixture_teacher_logprobs(lps, y_t).double(),
                               _brute_mixture(lps, y, [1 / 3] * 3, [0]), rtol=1e-5, atol=1e-6)


def test_mixture_single_teacher_is_identity() -> None:
    g = torch.Generator().manual_seed(3)
    lp = _rand_dist(g, 4, 5)
    out = mixture_teacher_logprobs([lp], torch.tensor([0, 1, 2, 3]), span_starts=[2])
    torch.testing.assert_close(out, lp)


# ------------------------------------------------------- Theorem 3 / Propositions B, C (numeric checks)
def _minimize(q: torch.Tensor, ul_of_logp, beta: float, gamma: float, steps: int = 4000) -> torch.Tensor:
    """argmin_p beta*KL(q||p) + gamma*UL over the simplex, by gradient descent on logits routed through our
    losses (identity LM head, so the "hidden state" is the logit vector)."""
    V = q.shape[0]
    z = torch.zeros(1, V, requires_grad=True)
    eye = torch.eye(V)
    lr = 1.0 / (beta + gamma)
    for _ in range(steps):
        kl, _ = soft_cross_entropy_chunked(z, eye, q.log()[None], chunk=1)
        loss = beta * kl + gamma * ul_of_logp(torch.log_softmax(z[0], -1))
        (gz,) = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z -= lr * gz
    return torch.softmax(z.detach()[0], -1)


@pytest.mark.parametrize("beta,gamma", [(0.2, 0.5), (1.0, 0.0), (0.5, 3.0)])
def test_theorem3_closed_form(beta: float, gamma: float) -> None:
    g = torch.Generator().manual_seed(4)
    V, c = 7, 2
    q = torch.softmax(torch.randn(V, generator=g), -1)
    eps_q = q[c].item()
    p = _minimize(q, lambda lp: slot_unlikelihood(lp[c]), beta, gamma)
    p_c = beta * eps_q / (beta + gamma)
    p_rest = q * (beta + gamma - beta * eps_q) / ((beta + gamma) * (1 - eps_q))
    expected = p_rest.clone()
    expected[c] = p_c
    torch.testing.assert_close(p, expected, rtol=0, atol=2e-5)
    if (beta, gamma) == (0.2, 0.5):
        assert p[c].item() == pytest.approx(2 * eps_q / 7, abs=2e-5)


def test_pivot_loss_reaches_theorem3_fixed_point() -> None:
    g = torch.Generator().manual_seed(5)
    V, c, beta, gamma = 6, 4, 0.2, 0.5
    q = torch.softmax(torch.randn(V, generator=g), -1)
    z = torch.zeros(1, V, requires_grad=True)
    eye = torch.eye(V)
    for _ in range(4000):
        loss, _ = pivot_loss(z, eye, q.log()[None], (0, c), beta, gamma)
        (gz,) = torch.autograd.grad(loss, z)
        with torch.no_grad():
            z -= gz / (beta + gamma)
    _, stats = pivot_loss(z, eye, q.log()[None], (0, c), beta, gamma)
    assert stats["slot_teacher_prob"] == pytest.approx(q[c].item(), rel=1e-5)
    assert stats["slot_prob"] == pytest.approx(beta * q[c].item() / (beta + gamma), abs=2e-5)
    _, no_slot = pivot_loss(z, eye, q.log()[None], None, beta, gamma)
    assert no_slot["ul"] == 0.0 and math.isnan(no_slot["slot_prob"])


def test_proposition_b1_event_version() -> None:
    g = torch.Generator().manual_seed(6)
    V, beta, gamma = 8, 0.3, 0.7
    E = torch.tensor([1, 3, 6])
    inE = torch.zeros(V, dtype=torch.bool)
    inE[E] = True
    q = torch.softmax(torch.randn(V, generator=g), -1)
    p = _minimize(q, lambda lp: slot_unlikelihood(torch.logsumexp(lp[E], 0)), beta, gamma)
    qE = q[inE].sum().item()
    assert p[inE].sum().item() == pytest.approx(beta * qE / (beta + gamma), abs=2e-5)
    torch.testing.assert_close(p[inE] / p[inE].sum(), q[inE] / q[inE].sum(), rtol=0, atol=2e-5)
    torch.testing.assert_close(p[~inE] / p[~inE].sum(), q[~inE] / q[~inE].sum(), rtol=0, atol=2e-5)


def test_proposition_b2_gradient_identity() -> None:
    g = torch.Generator().manual_seed(7)
    theta = torch.randn(5, generator=g, requires_grad=True)
    logP = torch.log_softmax(theta, -1)[2]
    (gl,) = torch.autograd.grad(logP, theta, retain_graph=True)
    (gu,) = torch.autograd.grad(slot_unlikelihood(logP), theta)
    P = logP.exp().item()
    torch.testing.assert_close(gu, P / (1 - P) * gl)


def test_proposition_c_retry_chain_monte_carlo() -> None:
    g = torch.Generator().manual_seed(8)
    V, c, k, beta, gamma, N = 6, 0, 3, 0.2, 0.5, 200_000
    q = torch.tensor([0.35, 0.25, 0.15, 0.1, 0.1, 0.05])
    good = torch.tensor([False, True, True, False, False, False])  # direction set G
    eps_q = q[c].item()
    nu = beta * eps_q / (beta + gamma)
    p = q * (1 - nu) / (1 - eps_q)  # Theorem 3 fixed point: proportional to q outside a-
    p[c] = nu
    s_bar = q[good].sum().item() / (1 - eps_q)
    draws = torch.multinomial(p, N * k, replacement=True, generator=g).view(N, k)
    not_repeat = draws != c
    first = torch.where(not_repeat.any(1), not_repeat.float().argmax(1), torch.full((N,), -1))
    has = first >= 0
    passed = torch.zeros(N, dtype=torch.bool)
    passed[has] = good[draws[has, first[has]]]
    rho_mc = passed.float().mean().item()
    rho = s_bar * (1 - (beta * eps_q / (beta + gamma)) ** k)
    se = math.sqrt(rho * (1 - rho) / N)
    assert abs(rho_mc - rho) < 4 * se
    # rho_inf = s_bar is independent of nu; the gain over gamma = 0 is bounded by s_bar * eps_q^k
    gain = s_bar * (eps_q ** k - nu ** k)
    assert 0 < gain <= s_bar * eps_q ** k


# ------------------------------------------------------------------------------------ UL stability
def test_slot_unlikelihood_stability() -> None:
    # p near 0: UL ~ p, gradient d UL / d logp = p / (1 - p) ~ p
    lp = torch.tensor([math.log(1e-12), math.log(1e-4), math.log(0.5)], requires_grad=True)
    ul = slot_unlikelihood(lp)
    ref = -torch.log1p(-lp.detach().double().exp())
    torch.testing.assert_close(ul.double(), ref, rtol=1e-5, atol=1e-18)
    ul.sum().backward()
    P = lp.detach().double().exp()
    torch.testing.assert_close(lp.grad.double(), P / (1 - P), rtol=1e-5, atol=1e-18)
    # p near 1 in fp32: exact below the truncation, capped at -log(eps) above it, always finite
    near = torch.tensor([math.log1p(-1e-4), math.log1p(-2e-6), math.log1p(-1e-9), 0.0], requires_grad=True)
    ul = slot_unlikelihood(near, eps=1e-6)
    assert torch.isfinite(ul).all()
    assert ul[0].item() == pytest.approx(-math.log(1e-4), rel=1e-3)
    assert ul[1].item() == pytest.approx(-math.log(2e-6), rel=1e-3)
    assert ul[2].item() == pytest.approx(-math.log(1e-6), rel=1e-3)
    assert ul[3].item() == pytest.approx(-math.log(1e-6), rel=1e-3)
    ul.sum().backward()
    assert torch.isfinite(near.grad).all()
    assert near.grad[2].item() == 0.0 and near.grad[3].item() == 0.0  # truncated region
    assert near.grad[0].item() == pytest.approx((1 - 1e-4) / 1e-4, rel=1e-2)
    # p = 0 exactly
    zero = torch.tensor([float("-inf")], requires_grad=True)
    out = slot_unlikelihood(zero)
    out.sum().backward()
    assert out.item() == 0.0 and zero.grad.item() == 0.0
