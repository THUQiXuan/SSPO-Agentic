"""Rollout construction helpers shared by the evaluation and calibration commands (rbd.sspo.evaluate)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rbd.config import model_path, resolve


def make_rollout(cfg: dict, servers: Path, system_prompt: Optional[str] = None):
    """Multi-turn agent rollout driver bound to the running vLLM servers (model default context window)."""
    from rbd.agent.chat import ChatFormat
    from rbd.agent.loop import SYSTEM_PROMPT, Rollout
    from rbd.envs.factory import make_env_factory
    from rbd.serving.client import ServerPool

    pool = ServerPool.from_state(servers)
    fmt = ChatFormat(model_path(cfg["model"]), enable_thinking=bool(cfg["model"].get("enable_thinking", False)))
    ro = Rollout(pool, fmt, make_env_factory(cfg["task"]), max_turns=int(cfg["task"]["max_turns"]),
                 l_ctx=pool.max_model_len, system_prompt=system_prompt or SYSTEM_PROMPT)
    return ro, pool


def _adapter(pool, path: str) -> Optional[str]:
    """Load a LoRA adapter directory on every server; ``none`` evaluates the base model."""
    if not path or path == "none":
        return None
    name = "eval_" + str(abs(hash(str(resolve(path)))) % 10**8)
    pool.load_adapter(name, resolve(path))
    return name
