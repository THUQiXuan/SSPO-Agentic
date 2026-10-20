"""Training entry point (one process per training GPU, launched by torchrun from scripts/train.sh).

    torchrun --nproc_per_node G -m rbd.train.main --run-dir runs/<name> --model qwen35_9b --task code_mt \
        --method rbd --servers runs/<name>/servers.json [--seed 0]

Rank 0 runs the multi-turn rollouts against the vLLM servers, computes Dr. GRPO advantages and the method's extra
samples (rejection-sampling SFT data, critical-step distillation items or synthesized pivot trajectories), and
writes ``runs/<name>/steps/step_XXXX/batch.pt``; every rank trains on a token-balanced shard. Re-running the same
command resumes from ``runs/<name>/resume`` (state after the last finished step).
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from rbd.config import load, model_path, resolve
from rbd.utils import atomic_write_json, log


class Dist:
    def __init__(self) -> None:
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.cpu_group = None
        if self.world > 1:
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6), device_id=self.device)
            self.cpu_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=6))

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier(group=self.cpu_group)

    def allreduce_(self, t: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(t)
        return t


def append_metrics(path: Path, row: dict) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    rows = [r for r in rows if r.get("step") != row["step"] or r.get("phase") != row.get("phase")] + [row]
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


def balance(costs: List[int], world: int) -> List[List[int]]:
    from rbd.train.model import balance_across_ranks

    return balance_across_ranks(costs, world)


# ---------------------------------------------------------------------------------------------- updates


def _sync_and_step(model, optimizer, d: Dist, max_grad_norm: float) -> float:
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.trainable]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    d.allreduce_(flat)
    off = 0
    for p in model.trainable:
        p.grad = flat[off:off + p.numel()].view_as(p).to(p.dtype)
        off += p.numel()
    gn = float(torch.nn.utils.clip_grad_norm_(model.trainable, max_grad_norm))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return gn


def _pivot_backward(model, item: dict, d: Dist, pad_id: int, scale: float, beta_bar: float, gamma: float,
                    stats: Dict[str, float]) -> None:
    """Mixture-teacher KL on the pivot turns + decision-slot unlikelihood for one pivot sample (Section 4.1)."""
    from rbd.train.losses import (full_vocab_logprobs, mix_teacher_rows, mixture_log_weights, slot_unlikelihood,
                                  soft_cross_entropy_chunked)
    from rbd.train.model import SeqBatch, hidden_at

    W = model.head_w
    pos = [j for a, b in item["kl_spans"] for j in range(a, b)]
    starts, acc = [], 0
    for a, b in item["kl_spans"]:
        starts.append(acc)
        acc += b - a
    y = torch.tensor([item["ids"][j] for j in pos], device=d.device)
    teacher = []
    n_u = len(item["teacher_ids"])
    store = torch.bfloat16 if 4.0 * len(pos) * W.shape[0] * n_u > 6e9 else torch.float32  # memory guard
    with model.base_only(), torch.no_grad():
        for tids, tspans in zip(item["teacher_ids"], item["teacher_spans"]):
            tpos = [j for a, b in tspans for j in range(a, b)]
            hb = hidden_at(model, SeqBatch([tids], [tpos], pad_id, d.device), grad=False)
            teacher.append(full_vocab_logprobs(hb, W, grad=False).to(store))
    # mixture teacher q_bar (Proposition D) built lazily per chunk from the posterior weights
    lw = mixture_log_weights(teacher, y, prior=None, span_starts=starts)
    h = hidden_at(model, SeqBatch([item["ids"]], [pos], pad_id, d.device), grad=True)
    kl, _ = soft_cross_entropy_chunked(h, W, lambda s, e: mix_teacher_rows(teacher, lw, s, e))
    loss = beta_bar * kl
    if item.get("slot") is not None and gamma > 0:
        spos, tok = item["slot"]
        r = pos.index(spos)
        lp_row = torch.log_softmax(torch.nn.functional.linear(h[r:r + 1], W).float(), -1)[0, tok]
        ul = slot_unlikelihood(lp_row)
        loss = loss + gamma * ul
        stats["slot_prob"] += float(lp_row.exp())
        stats["ul"] += float(ul)
        stats["n_slot"] += 1
    (scale * item["weight"] * loss).backward()
    stats["kl"] += float(kl) / max(1, len(pos))
    stats["n_pivot"] += 1


def train_on_batch(batch: dict, model, optimizer, d: Dist, mcfg: dict, micro_tokens: int, pad_id: int) -> dict:
    """Dr. GRPO / SFT sequences in PPO mini-batches; pivot items are added to the first mini-batch."""
    from rbd.train.losses import drgrpo_token_loss, sft_token_loss
    from rbd.train.model import SeqBatch, pack_micro_batches

    seqs = batch["seqs"]
    M = max(1, len(batch["mini_batches"]))
    mine = []
    for idx in batch["mini_batches"]:
        shards = balance([len(seqs[i]["ids"]) for i in idx], d.world)
        mine.append([idx[j] for j in shards[d.rank]])

    def micro(idx: List[int]) -> List[List[int]]:
        lens = [(len(seqs[i]["ids"]), 0) for i in idx]
        return [[idx[j] for j in mb] for mb in pack_micro_batches(lens, micro_tokens)]

    def mk(idx: List[int]) -> SeqBatch:
        return SeqBatch([seqs[i]["ids"] for i in idx], [seqs[i]["pos"] for i in idx], pad_id, d.device)

    old: Dict[int, torch.Tensor] = {}
    if M > 1:
        model.train(False)
        for m_idx in mine:
            for mb in micro(m_idx):
                b = mk(mb)
                lp = model.response_logprobs(b, grad=False)
                off = 0
                for j, i in enumerate(mb):
                    L = int(b.seq_len[j])
                    old[i] = lp[off:off + L]
                    off += L
        model.train(True)
    model.train(True)
    st = {"loss": 0.0, "tokens": 0, "mismatch": 0.0, "pre_tokens": 0, "kl": 0.0, "ul": 0.0, "slot_prob": 0.0,
          "n_pivot": 0, "n_slot": 0, "grad_norm": 0.0, "updates": 0}
    piv = batch.get("pivots") or []
    piv_mine = [piv[i] for i in balance([len(p["ids"]) for p in piv], d.world)[d.rank]] if piv else []
    pcfg = mcfg.get("pivot", {})
    for m, m_idx in enumerate(mine):
        norm = float(batch["norms"][m])
        for mb in micro(m_idx):
            b = mk(mb)
            logp = model.response_logprobs(b, grad=True)
            cat = lambda dct: torch.cat([dct[i] for i in mb])  # noqa: E731
            old_lp = cat(old) if M > 1 else logp.detach()
            adv = torch.tensor([seqs[i]["adv"] for i in mb], device=d.device, dtype=torch.float32)[b.seq_index]
            kinds = [seqs[i]["kind"] for i in mb]
            if all(k == "sft" for k in kinds):
                tok = sft_token_loss(logp)
            else:
                tok = drgrpo_token_loss(logp, old_lp, adv, mcfg.get("clip_low", 0.2), mcfg.get("clip_high", 0.28))
            loss = tok.sum() / norm
            loss.backward()
            st["loss"] += float(loss)
            st["tokens"] += b.num_tokens
            if m == 0:
                rlp = torch.tensor([x for i in mb for x in seqs[i]["rlp"]], device=d.device, dtype=torch.float32)
                st["mismatch"] += float((logp.detach() - rlp).abs().sum())
                st["pre_tokens"] += b.num_tokens
        if m == 0 and piv_mine:
            scale = float(pcfg.get("lambda", 1.0)) / max(1, len(piv))
            for item in piv_mine:
                _pivot_backward(model, item, d, pad_id, scale, float(pcfg.get("beta_bar", 0.2)),
                                float(pcfg.get("gamma_rep", 0.0)), st)
        st["grad_norm"] += _sync_and_step(model, optimizer, d, float(mcfg.get("max_grad_norm", 1.0)))
        st["updates"] += 1
    t = torch.tensor([st[k] for k in st], device=d.device, dtype=torch.float64)
    d.allreduce_(t)
    agg = dict(zip(st, t.tolist()))
    out = {"loss": agg["loss"], "train_tokens": int(agg["tokens"]),
           "mismatch_abs_lp": agg["mismatch"] / max(1.0, agg["pre_tokens"]),
           "grad_norm": agg["grad_norm"] / max(1.0, agg["updates"]), "optimizer_updates": int(agg["updates"] / d.world),
           "pivot_items": int(agg["n_pivot"]), "pivot_kl_per_token": agg["kl"] / max(1.0, agg["n_pivot"])}
    if agg["n_slot"]:
        out.update({"slot_prob": agg["slot_prob"] / agg["n_slot"], "slot_ul": agg["ul"] / agg["n_slot"]})
    return out


# ---------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="code_mt")
    ap.add_argument("--method", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-tokens", type=int, default=int(os.environ.get("RBD_MICRO_TOKENS", "8192")))
    ap.add_argument("--steps", type=int, default=0, help="override the method's number of steps (smoke tests)")
    ap.add_argument("--tasks-per-step", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=0, help="override the task's turn budget T_max (smoke tests)")
    args = ap.parse_args()

    from rbd.train.ckpt import export_for_vllm, load_resume, prune_adapters, restore_weights, save_resume
    from rbd.train.model import PolicyModel

    d = Dist()
    cfg = load(args.model, args.task, args.method)
    mcfg, tcfg, modcfg = cfg["method"], cfg["task"], cfg["model"]
    for k, a in (("steps", args.steps), ("tasks_per_step", args.tasks_per_step), ("group_size", args.group_size)):
        if a:
            mcfg[k] = a
    if args.max_turns:
        tcfg["max_turns"] = args.max_turns
    run_dir = resolve(args.run_dir)
    if d.rank == 0:
        (run_dir / "steps").mkdir(parents=True, exist_ok=True)
        (run_dir / "adapters").mkdir(exist_ok=True)
        man = run_dir / "manifest.json"
        if not man.exists():
            atomic_write_json(man, {"run": run_dir.name, "model": modcfg["name"], "task": tcfg["name"],
                                    "method": mcfg["name"], "seed": args.seed, "config_hash": cfg["hash"],
                                    "config": cfg, "world_size": d.world,
                                    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    d.barrier()
    if (run_dir / "done").exists():
        return
    torch.manual_seed(args.seed)
    mpath = model_path(modcfg)
    model = PolicyModel(mpath, modcfg["lora"], d.device)
    optimizer = torch.optim.AdamW(model.trainable, lr=float(mcfg["lr"]), betas=(0.9, 0.99), weight_decay=0.0)
    from rbd.agent.chat import ChatFormat

    fmt = ChatFormat(mpath, enable_thinking=bool(modcfg.get("enable_thinking", False)))
    pad_id = fmt.tok.pad_token_id if fmt.tok.pad_token_id is not None else fmt.im_end
    state = load_resume(run_dir) or {}
    if state:
        restore_weights(model.peft, optimizer, run_dir, d.device)
    builder = None
    adapter = None
    if d.rank == 0:
        if mcfg.get("sel_arm"):
            from rbd.sspo.builder import SelStepBuilder as StepBuilder
        else:
            from rbd.methods.build import StepBuilder

        builder = StepBuilder(cfg, run_dir, resolve(args.servers), args.seed, fmt, state)
        if state:
            adapter = f"resume_{state['step']:04d}"
            builder.pool.load_adapter(adapter, run_dir / "resume" / "adapter_vllm")

    phases = builder.phases() if builder else None
    phases = _bcast(d, phases)
    for phase, n_steps in phases:
        done_steps = int(state.get(f"{phase}_steps", 0))
        for step in range(done_steps + 1, n_steps + 1):
            t0 = time.time()
            bpath = run_dir / "steps" / f"{phase}_{step:04d}" / "batch.pt"
            if d.rank == 0:
                if bpath.exists():
                    batch = torch.load(bpath, weights_only=False)
                    builder.restore(batch["builder_state"])
                else:
                    batch = builder.build(phase, step, adapter)
                    bpath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(batch, bpath.with_suffix(".tmp"))
                    os.replace(bpath.with_suffix(".tmp"), bpath)
                log(f"[train] {phase} {step}/{n_steps}: " + " ".join(f"{k} {v}" for k, v in batch["metrics"].items()
                                                                    if k in ("reward", "seqs", "pivots", "t_rollout_s")))
            sleep_mode = os.environ.get("RBD_SLEEP_MODE", "0") == "1"
            if sleep_mode and d.rank == 0:
                builder.pool.sleep(1)
            d.barrier()
            if d.rank != 0:
                batch = torch.load(bpath, weights_only=False)
            upd = train_on_batch(batch, model, optimizer, d, mcfg, args.micro_tokens, pad_id) \
                if (batch["seqs"] or batch.get("pivots")) else {"optimizer_updates": 0}
            if sleep_mode:
                torch.cuda.empty_cache()
                d.barrier()
                if d.rank == 0:
                    builder.pool.wake()
            mm = float(upd.get("mismatch_abs_lp", 0.0))
            if phase == "rl" and step >= 2 and mm > float(os.environ.get("RBD_MAX_LP_MISMATCH", "0.08")):
                raise RuntimeError(f"inference/training log-prob gap {mm:.3f}: the served LoRA is not the trained "
                                   "policy (adapter key layout / vLLM version)")
            d.barrier()
            if d.rank == 0:
                name = f"{phase}_{step:04d}"
                served = export_for_vllm(model.peft, run_dir / "adapters" / name, model.vllm_key_map,
                                         modcfg["lora"]["target_modules"])
                if step % int(mcfg.get("save_every", 10)) == 0 or step == n_steps:
                    export_for_vllm(model.peft, run_dir / "ckpt" / name / "adapter", model.vllm_key_map,
                                    modcfg["lora"]["target_modules"])
                builder.pool.swap_adapter(name, served, adapter)
                adapter = name
                state = dict(state, **{f"{phase}_steps": step, "step": step, "builder": builder.state()})
                save_resume(run_dir, model.peft, optimizer, state, vllm_export=served)
                prune_adapters(run_dir / "adapters", keep=2)
                append_metrics(run_dir / "metrics.jsonl", {"phase": phase, "step": step, **batch["metrics"], **upd,
                                                           "t_step_s": round(time.time() - t0, 1)})
            d.barrier()
    if d.rank == 0:
        atomic_write_json(run_dir / "done", {"finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    d.barrier()


def _bcast(d: Dist, obj: Any) -> Any:
    if d.world == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=d.cpu_group)
    return box[0]


if __name__ == "__main__":
    main()
