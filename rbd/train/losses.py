"""Loss functions of rollback-recovery distillation (Section 4.1 of the method document).

    L(theta) = L_DrGRPO + lambda_piv * sum_{tau in P} w(tau) * [beta_bar * sum_t KL_t + gamma_rep * UL_s]

Everything here is pure torch and operates on already-gathered tensors: hidden states of the scored positions,
the (frozen) LM-head weight and teacher log-probabilities. No model is loaded here.

Memory model (V = vocabulary size, H = hidden size, T = number of scored positions, c = chunk rows):
  * a dense fp32 (T, V) tensor costs 4*T*V bytes, e.g. 0.99 GB for T = 1000 at V = 248,320 (Qwen3.5) and
    0.61 GB at V = 151,936 (Qwen3);
  * ``soft_cross_entropy_chunked`` never holds the student's (T, V) log-probs with grad: per chunk it has
    ~5 transient fp32 (c, V) tensors (target, logits, q, two products), ~2.5 GB peak for c = 512 at
    V = 248,320 (~1.6 GB at V = 151,936; halve c to halve it), and autograd stores only the chunk inputs plus a
    (c,) output, the logits being recomputed in backward;
  * ``mixture_teacher_logprobs`` needs the U teacher tensors (U * 4*T*V bytes if fp32, half if bf16) plus
    ~2 fp32 (rows, V) working tensors; ``mix_teacher_rows`` lets the KL consume the mixture lazily so the dense
    (T, V) mixture is never built.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rbd.train.model import chunked_token_logprobs

TargetFn = Callable[[int, int], torch.Tensor]  # (start, end) -> (end - start, V) teacher log-probs
Target = Union[torch.Tensor, TargetFn]


# ---------------------------------------------------------------------------------------------- Dr. GRPO
def drgrpo_token_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
                      clip_low: float = 0.2, clip_high: float = 0.28) -> torch.Tensor:
    """Per-token PPO-clipped surrogate with asymmetric (DAPO clip-higher) bounds, Section 4.1 Dr. GRPO term.

    loss_t = -min(r_t * A_t, clip(r_t, 1 - clip_low, 1 + clip_high) * A_t),  r_t = exp(logp_t - old_logp_t).
    ``adv`` is broadcast against ``logp`` (pass the sequence advantage A_i = r_i - mean(r) expanded to tokens).
    """
    ratio = torch.exp(logp - old_logp.detach())
    adv = adv.detach()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * adv
    return -torch.minimum(unclipped, clipped)


def aggregate_constant(tok_loss: torch.Tensor, norm: float) -> torch.Tensor:
    """Dr. GRPO aggregation: sum of token losses over a constant normalizer (|X| * G * L_max in Section 4.1).

    Because the normalizer does not depend on the batch, summing this over micro-batches reproduces the full-step
    loss exactly under gradient accumulation.
    """
    return tok_loss.sum() / norm


def aggregate_token_mean(tok_loss: torch.Tensor) -> torch.Tensor:
    """Token-mean aggregation (GRPO/DAPO style) over the given tokens, kept for ablations."""
    return tok_loss.mean()


def sft_token_loss(logp: torch.Tensor) -> torch.Tensor:
    """Per-token negative log-likelihood, used for the rejection-sampling fine-tuning baseline."""
    return -logp


# ---------------------------------------------------------------------------------- full-vocabulary pieces
def _chunk_full_logprobs(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(F.linear(h, w).float(), dim=-1)


def full_vocab_logprobs(hidden: torch.Tensor, head_weight: torch.Tensor, chunk: int = 512,
                        grad: bool = True) -> torch.Tensor:
    """(T, V) fp32 log-softmax rows of the student at the given positions.

    The output itself is dense (4*T*V bytes, plus the same again for its gradient in backward); chunking with
    activation checkpointing only removes the per-chunk logits/softmax intermediates from the autograd graph.
    Use ``soft_cross_entropy_chunked`` for training on long spans.
    """
    outs = []
    with nullcontext() if grad else torch.no_grad():
        for i in range(0, hidden.shape[0], chunk):
            hc = hidden[i:i + chunk]
            if grad and hc.requires_grad:
                outs.append(checkpoint(_chunk_full_logprobs, hc, head_weight, use_reentrant=False))
            else:
                outs.append(_chunk_full_logprobs(hc, head_weight))
    return torch.cat(outs) if outs else hidden.new_zeros(0, head_weight.shape[0], dtype=torch.float32)


def _chunk_kl(h: torch.Tensor, w: torch.Tensor, target: Target, start: int, end: int) -> torch.Tensor:
    """Row-wise KL(q || p) for one chunk: sum_v q (log q - z) + logsumexp(z) * sum_v q, z = student logits."""
    logq = (target(start, end) if callable(target) else target).float()
    z = F.linear(h, w).float()
    q = logq.exp()
    neg_entropy = torch.where(q > 0, q * logq, torch.zeros_like(q)).sum(-1)
    return neg_entropy - (q * z).sum(-1) + torch.logsumexp(z, dim=-1) * q.sum(-1)


def soft_cross_entropy_chunked(hidden: torch.Tensor, head_weight: torch.Tensor, target: Target,
                               chunk: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward KL sum_t sum_v q_t(v) (log q_t(v) - log p_t(v)), the KL_t term of Section 4.1 / Proposition D.

    hidden:      (T, H) student hidden states whose next-token distributions are matched
    head_weight: (V, H) LM-head weight
    target:      (T, V) teacher log-probs, or a deterministic callable (start, end) -> (end - start, V) log-probs;
                 a callable is evaluated inside the checkpointed chunk, so it is re-run in backward and its output
                 is never kept alive by autograd (e.g. a lazy mixture via ``mix_teacher_rows``)
    Returns (scalar KL sum with grad, per-position KL (T,) detached).
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    rows = []
    for i in range(0, hidden.shape[0], chunk):
        j = min(i + chunk, hidden.shape[0])
        hc = hidden[i:j]
        tc: Target = target if callable(target) else target[i:j].detach()
        if hc.requires_grad:
            kl = checkpoint(_chunk_kl, hc, head_weight, tc, i, j, use_reentrant=False)
        else:
            kl = _chunk_kl(hc, head_weight, tc, i, j)
        total = total + kl.sum()
        rows.append(kl.detach())
    per_pos = torch.cat(rows) if rows else hidden.new_zeros(0, dtype=torch.float32)
    return total, per_pos


# ------------------------------------------------------------------------------------ mixture teacher
def mixture_log_weights(teacher_logprobs_list: Sequence[torch.Tensor], y_ids: torch.Tensor,
                        prior: Optional[torch.Tensor] = None,
                        span_starts: Optional[Sequence[int]] = None) -> torch.Tensor:
    """(U, T) log posterior weights log mu_k(u) of Proposition D, mu_k(u) ∝ mu(u) * prod_{j<k} q_u(y_j).

    The product runs over earlier tokens of the same action span only: the posterior restarts at every index in
    ``span_starts`` (index 0 always starts a span). ``prior`` is mu(u) as (U,) probabilities (uniform if None).
    """
    U, T = len(teacher_logprobs_list), y_ids.shape[0]
    if U == 0 or T == 0:
        raise ValueError("need at least one teacher and one position")
    tok = torch.stack([lp.gather(1, y_ids[:, None])[:, 0].float() for lp in teacher_logprobs_list])  # (U, T)
    reset = torch.zeros(T, dtype=torch.bool, device=y_ids.device)
    reset[0] = True
    for s in span_starts or ():
        if not 0 <= s < T:
            raise ValueError(f"span start {s} outside [0, {T})")
        reset[s] = True
    # exclusive prefix sum of tok within spans: cum[k] - cum[start(k)], cum = inclusive cumsum shifted by one
    cum = torch.cat([tok.new_zeros(U, 1), tok.cumsum(1)], dim=1)  # (U, T + 1), cum[:, k] = sum_{j<k}
    pos = torch.arange(T, device=y_ids.device)
    start_of = torch.cummax(torch.where(reset, pos, torch.zeros_like(pos)), dim=0).values  # span start of each k
    prefix = cum[:, :T] - cum.gather(1, start_of[None].expand(U, T))
    log_prior = (torch.full((U,), -math.log(U), device=tok.device) if prior is None
                 else torch.as_tensor(prior, dtype=torch.float32, device=tok.device).log())
    return torch.log_softmax(log_prior[:, None] + prefix, dim=0)


def mix_teacher_rows(teacher_logprobs_list: Sequence[torch.Tensor], log_weights: torch.Tensor,
                     start: int, end: int) -> torch.Tensor:
    """Rows [start, end) of log q_bar = logsumexp_u (log mu_k(u) + log q_u), in fp32, accumulated one teacher at a
    time so only two (end - start, V) working tensors exist besides the inputs."""
    out: Optional[torch.Tensor] = None
    for u, lp in enumerate(teacher_logprobs_list):
        term = log_weights[u, start:end, None] + lp[start:end].float()
        out = term if out is None else torch.logaddexp(out, term)
    assert out is not None, "at least one teacher is required"
    return out


def mixture_teacher_logprobs(teacher_logprobs_list: Sequence[torch.Tensor], y_ids: torch.Tensor,
                             prior: Optional[torch.Tensor] = None,
                             span_starts: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Dense (T, V) log q_bar of the posterior-weighted mixture teacher (Proposition D, chain-rule form).

    teacher_logprobs_list[u]: (T, V) log-probs of teacher u (privileged hint u) at the positions of the target
    action tokens ``y_ids``; row k is the distribution of y_k given the prefix y_{<k}. Memory: the U inputs
    (U * T * V floats, all needed because the weights at row k depend on every teacher's likelihood of y_{<k};
    4 GB for U = 4, T = 1000, V = 248,320 in fp32, half in bf16) plus ~2 fp32 (T, V) working tensors and the
    output. With U = 1 this returns teacher 0 unchanged (up to fp32 casting).
    """
    log_w = mixture_log_weights(teacher_logprobs_list, y_ids, prior, span_starts)
    return mix_teacher_rows(teacher_logprobs_list, log_w, 0, y_ids.shape[0])


# ------------------------------------------------------------------------------------ slot unlikelihood
def slot_unlikelihood(logp_slot_token: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """UL_s = -log(1 - p) with p = exp(logp) truncated to 1 - eps (Theorem 3; Proposition B (2)).

    Uses the stable log1mexp split: log(-expm1(logp)) for logp > -ln 2, log1p(-exp(logp)) otherwise. Its
    gradient w.r.t. logp is p / (1 - p) (Proposition B (2)); it is zero inside the truncated region p > 1 - eps.
    """
    lp = logp_slot_token.clamp(max=math.log1p(-eps))
    log1m = torch.where(lp > -math.log(2.0), torch.log(-torch.expm1(lp)), torch.log1p(-torch.exp(lp)))
    return -log1m


# ------------------------------------------------------------------------------------------ pivot loss
def pivot_loss(student_hidden_at_spans: torch.Tensor, head_weight: torch.Tensor, mixture_target: Target,
               slot: Optional[Tuple[int, int]], beta_bar: float, gamma_rep: float,
               chunk: int = 512) -> Tuple[torch.Tensor, Dict[str, float]]:
    """beta_bar * sum_t KL_t + gamma_rep * UL_s for one PivotSample (bracket of Section 4.1; Theorem 3).

    student_hidden_at_spans: (T, H) student hidden states predicting the tokens of ``kl_spans`` (actions at t_piv
                             and t_piv + 1), concatenated in span order
    mixture_target:          (T, V) log q_bar rows aligned with those positions, or a lazy callable (start, end)
    slot:                    (row in student_hidden_at_spans whose next-token distribution is h_s, token id c-_s),
                             or None when the edit field could not be parsed (KL only, algorithm line 8)
    The caller applies lambda_piv and the recovery weight w(tau). Stats: kl (sum), kl_mean, ul, slot_prob
    (student p(c-_s)) and slot_teacher_prob (q_bar(c-_s) = eps_q; Theorem 3 predicts p* = beta eps_q/(beta+gamma)).
    """
    kl, kl_rows = soft_cross_entropy_chunked(student_hidden_at_spans, head_weight, mixture_target, chunk)
    loss = beta_bar * kl
    stats: Dict[str, float] = {"kl": float(kl.detach()),
                               "kl_mean": float(kl_rows.mean()) if kl_rows.numel() else 0.0,
                               "ul": 0.0, "slot_prob": float("nan"), "slot_teacher_prob": float("nan")}
    if slot is not None:
        row, tok = int(slot[0]), int(slot[1])
        h = student_hidden_at_spans[row:row + 1]
        y = torch.tensor([tok], dtype=torch.long, device=h.device)
        logp = chunked_token_logprobs(h, y, head_weight, grad=True)[0]
        ul = slot_unlikelihood(logp)
        loss = loss + gamma_rep * ul
        tgt_row = mixture_target(row, row + 1) if callable(mixture_target) else mixture_target[row:row + 1]
        stats.update(ul=float(ul.detach()), slot_prob=float(logp.detach().exp()),
                     slot_teacher_prob=float(tgt_row[0, tok].detach().float().exp()))
    return loss, stats
