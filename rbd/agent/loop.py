"""Multi-turn rollouts: many episodes in parallel against the per-GPU vLLM servers.

Each episode: prompt -> [sample one assistant message (token ids + log-probs) -> extract the single bash
command -> execute it in the environment (thread pool) -> append the observation tokens] until the agent
submits, the turn budget T_max is reached or the context window L_ctx (the model's default context) is full.
The file-system state is recorded around every turn for pivot synthesis.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from rbd.agent.chat import ChatFormat
from rbd.envs.base import SUBMIT_TOKEN, Env
from rbd.serving.client import SampleRequest, ServerPool
from rbd.types import Trajectory, Turn
from rbd.utils import log

SYSTEM_PROMPT = (
    "You are a software engineer working in a git repository through a bash shell. In every reply, think briefly "
    "about the next step and then give exactly ONE bash command in a single ```bash code block; it is executed and "
    "you receive its return code and output. Edit files with shell commands (for example cat > file <<'EOF'). "
    "Run the tests to check your work. If a change turns out to be wrong you may revert it and try another "
    "approach. When you are done, reply with the command `echo " + SUBMIT_TOKEN + "`."
)
ROLLBACK_PROMPT_SUFFIX = (
    " When a test fails right after an edit, first decide whether the edit itself is wrong; if it is, restore the "
    "edited files with `git checkout -- <files>` before trying a different fix."
)
_BLOCK = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.S)
FORMAT_ERROR = ("<returncode>-1</returncode>\n<output>\nFormat error: reply with exactly one ```bash code block "
                "containing one command.\n</output>")


def extract_command(text: str) -> Optional[str]:
    """The single bash command of an assistant message; None unless there is exactly one non-empty block."""
    blocks = [b.strip() for b in _BLOCK.findall(text or "")]
    blocks = [b for b in blocks if b]
    return blocks[0] if len(blocks) == 1 else None


class Rollout:
    def __init__(self, pool: ServerPool, fmt: ChatFormat, make_env: Callable[[dict], Env], max_turns: int,
                 l_ctx: int, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1,
                 system_prompt: str = SYSTEM_PROMPT, env_threads: int = 64, max_inflight: int = 256):
        self.pool, self.fmt, self.make_env = pool, fmt, make_env
        self.max_turns, self.l_ctx = max_turns, l_ctx
        self.sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k)
        self.system_prompt = system_prompt
        self.exec = ThreadPoolExecutor(env_threads)
        self.max_inflight = max_inflight

    async def _episode(self, session, sems, task: dict, group_id: str, k: int, seed: int,
                       adapter: Optional[str], start: Optional[dict] = None) -> Trajectory:
        """One episode; ``start`` = {"ids", "prompt_ids", "turns", "files"} resumes from a recorded state."""
        loop = asyncio.get_running_loop()
        env = self.make_env(task)
        try:
            task_msg = await loop.run_in_executor(self.exec, env.reset)
            if start is None:
                prompt = self.fmt.prompt(self.system_prompt, task_msg)
                ids, turns = list(prompt), []
            else:
                await loop.run_in_executor(self.exec, env.restore_files, start["files"])
                prompt, ids, turns = list(start["prompt_ids"]), list(start["ids"]), list(start["turns"])
            finish = "max_turns"
            for t in range(len(turns), self.max_turns):
                room = self.l_ctx - len(ids)
                if room <= 1:
                    finish = "context"
                    break
                files_before = await loop.run_in_executor(self.exec, env.tracked_files)
                fs_before = await loop.run_in_executor(self.exec, env.fs_state)
                req = SampleRequest(ids, 1, room, self.sampling["temperature"], self.sampling["top_p"],
                                    self.sampling["top_k"], (seed * 1000003 + t) & 0x7FFFFFFF, adapter,
                                    (group_id, k, t))
                ch = (await self.pool._sample_one(session, sems, req))[0]
                cmd = extract_command(ch["text"])
                if cmd is None:
                    obs, rc, done, is_test = FORMAT_ERROR, None, False, False
                else:
                    res = await loop.run_in_executor(self.exec, env.step, cmd)
                    obs, rc, done, is_test = res.obs_text, res.exit_code, res.done, res.is_test
                fs_after = await loop.run_in_executor(self.exec, env.fs_state)
                obs_ids = self.fmt.observation(obs, ch["token_ids"])
                turns.append(Turn(idx=t, action_text=ch["text"], action_ids=ch["token_ids"],
                                  action_logprobs=ch["logprobs"], command=cmd, obs_text=obs, obs_ids=obs_ids,
                                  exit_code=rc, is_test=is_test, fs_before=fs_before, fs_after=fs_after,
                                  files_before=files_before))
                ids += ch["token_ids"] + obs_ids
                if done:
                    finish = "submit"
                    break
            reward = await loop.run_in_executor(self.exec, env.evaluate)
            final_files = await loop.run_in_executor(self.exec, env.tracked_files)
            return Trajectory(task_id=task["pid"], group_id=group_id, sample_idx=k, prompt_ids=prompt,
                              turns=turns, reward=int(reward), finish=finish, final_files=final_files)
        finally:
            await loop.run_in_executor(self.exec, env.close)

    async def _run(self, jobs: Sequence[tuple], adapter: Optional[str], progress_s: float,
                   starts: Optional[Sequence[Optional[dict]]] = None) -> List[Trajectory]:
        sems = [asyncio.Semaphore(self.max_inflight) for _ in self.pool.urls]
        out: List[Optional[Trajectory]] = [None] * len(jobs)
        t0, last, done = time.time(), time.time(), 0
        async with self.pool.session() as session:
            async def one(i: int, job: tuple) -> None:
                nonlocal done, last
                task, gid, k, seed = job
                out[i] = await self._episode(session, sems, task, gid, k, seed, adapter,
                                             starts[i] if starts else None)
                done += 1
                if time.time() - last > progress_s:
                    last = time.time()
                    log(f"[rollout] {done}/{len(jobs)} episodes ({time.time() - t0:.0f}s)")
            await asyncio.gather(*(one(i, j) for i, j in enumerate(jobs)))
        return [t for t in out if t is not None]

    def run(self, tasks: Sequence[dict], group_size: int, seed: int, adapter: Optional[str] = None,
            group_prefix: str = "g", progress_s: float = 60.0) -> List[Trajectory]:
        """G episodes per task; trajectories ordered by (task, sample)."""
        jobs = [(task, f"{group_prefix}:{task['pid']}", k, stable_seed(seed, task["pid"], k))
                for task in tasks for k in range(group_size)]
        return asyncio.run(self._run(jobs, adapter, progress_s))

    def run_from(self, items: Sequence[tuple], seed: int, adapter: Optional[str] = None,
                 progress_s: float = 60.0) -> List[Trajectory]:
        """Episodes resumed from recorded states: items = [(task, start_dict, tag)] (injection evaluation and the
        rollback probe of the recovery weights)."""
        jobs = [(task, f"from:{tag}", 0, stable_seed(seed, tag)) for task, _, tag in items]
        return asyncio.run(self._run(jobs, adapter, progress_s, starts=[st for _, st, _ in items]))


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
