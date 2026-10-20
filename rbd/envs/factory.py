"""Environment factory from the task config (configs/tasks/*.yaml)."""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Dict

from rbd.config import resolve
from rbd.envs.base import Env


def make_env_factory(tcfg: Dict) -> Callable[[dict], Env]:
    kind = tcfg["env"]
    timeout = float(tcfg.get("command_timeout_s", 30))
    if kind == "repo":
        from rbd.envs.repo_env import CodeRepoEnv

        return lambda task: CodeRepoEnv(task, cmd_timeout_s=timeout)
    raise ValueError(f"unknown env {kind!r}")
