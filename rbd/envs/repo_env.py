"""Competitive-programming task as a local git repository (``CodeRepoEnv``).

Workspace layout (one temporary base directory per episode, under ``$RBD_WORKDIR`` if set)::

    base/repo/   the git repository the agent works in (cwd of every command)
        TASK.md, solution.py, run_tests.py, tests/public/NN.in|NN.out, .gitignore
    base/home/   $HOME of the commands        base/tmp/  $TMPDIR of the commands
    base/bin/    ``python`` / ``python3`` wrappers around the current interpreter
    base/.out    combined stdout + stderr of the last command

Keeping HOME, TMPDIR and the output file outside the repository means they never show up as untracked files,
so ``fs_state`` sees exactly what the agent changed in the repository (condition A1 of Theorem 1: the
transition depends on the file-system state and the command only).

Every command runs as ``bash -c`` in its own session with CPU / file-size / address-space limits
(RLIMIT_AS is not enforceable on macOS), single-threaded numeric libraries, pinned git identity and dates,
no bytecode writes, and inside ``unshare -rn`` (no network) when available. The switch is the sandbox pool's
``RBD_UNSHARE_NET`` (auto / 1 / 0). Hidden-test judging goes through ``rbd.sandbox.pool.run_prog``.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from rbd.envs.base import SUBMIT_TOKEN, Env
from rbd.sandbox.exec_worker import THREAD_ENV_VARS
from rbd.sandbox.pool import _unshare_available, run_prog
from rbd.types import FsState, StepResult

OUTPUT_LIMIT = 10000  # characters of command output shown to the agent
GIT_DATE = "2000-01-01T00:00:00+0000"
STUB = "import sys\n\n# write your solution here\n"
GITIGNORE = "__pycache__/\n.pytest_cache/\n"
FSIZE_LIMIT = 64 << 20
DEFAULT_MEM_MB = 4096
GIT_TIMEOUT_S = 60.0
TIMEOUT_RC = 124
_READ_CAP = 1 << 20
_TEST_RE = re.compile(r"\brun_tests\.py\b")
_EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}

RUN_TESTS = r'''"""Run solution.py on every public example in tests/public; prints PASS / FAIL per test, exits 1 on failure."""
import math
import subprocess
import sys
from pathlib import Path

TIME_LIMIT_S = __TIME_LIMIT__


def same(got, exp):
    g = [line.rstrip() for line in got.strip().splitlines()]
    e = [line.rstrip() for line in exp.strip().splitlines()]
    if g == e:
        return True
    gt, et = got.split(), exp.split()
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if not (math.isfinite(fa) and math.isfinite(fb)) or abs(fa - fb) > 1e-6 * max(1.0, abs(fb)):
            return False
    return True


def head(text, n=5):
    lines = text.rstrip("\n").splitlines() or [""]
    out = "\n".join("    " + line[:200] for line in lines[:n])
    return out + ("\n    ..." if len(lines) > n else "")


def main():
    cases = sorted(Path("tests/public").glob("*.in"))
    failed = 0
    for inp in cases:
        exp = inp.with_suffix(".out").read_text()
        try:
            r = subprocess.run([sys.executable, "solution.py"], input=inp.read_text(), capture_output=True,
                               text=True, timeout=TIME_LIMIT_S)
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"FAIL {inp.stem}: time limit exceeded ({TIME_LIMIT_S} s)")
            continue
        if r.returncode != 0:
            failed += 1
            tail = "\n".join(r.stderr.rstrip().splitlines()[-5:])
            print(f"FAIL {inp.stem}: runtime error (exit code {r.returncode})\n{head(tail)}")
        elif same(r.stdout, exp):
            print(f"PASS {inp.stem}")
        else:
            failed += 1
            print(f"FAIL {inp.stem}: wrong answer\n  expected:\n{head(exp)}\n  got:\n{head(r.stdout)}")
    print(f"passed {len(cases) - failed}/{len(cases)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
'''

TASK_MESSAGE = """You are working in a git repository (the current directory) that contains a competitive-programming \
task. Solve it by editing solution.py: the program reads the input from stdin and writes the answer to stdout.

Repository layout:
  TASK.md          the problem statement (reproduced below)
  solution.py      your solution (currently a stub) -- edit this file
  tests/public/    public examples NN.in / NN.out
  run_tests.py     runs solution.py on the public examples: python run_tests.py

Hidden tests decide the result, so make the solution correct and efficient (time limit {tl:g} s per test).
Reply with exactly one bash command per turn in a ```bash block. When you are done, run:
echo {submit}

# Problem statement

{question}
"""


def _blob_sha(data: bytes) -> str:
    """Git blob id of ``data`` (same value ``git hash-object`` prints)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _excluded(path: str) -> bool:
    return any(part in _EXCLUDED_PARTS for part in path.split("/"))


def _set_limits(cpu_s: int, mem_bytes: int) -> None:
    """``preexec_fn`` of agent commands: CPU, address-space, file-size and core-dump limits."""
    for lim, val in ((resource.RLIMIT_CPU, (cpu_s, cpu_s + 1)), (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
                     (resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT)), (resource.RLIMIT_CORE, (0, 0))):
        try:
            resource.setrlimit(lim, val)
        except (ValueError, OSError):  # RLIMIT_AS cannot be lowered on macOS
            pass


def _net_prefix() -> List[str]:
    """``unshare -rn`` when network isolation is requested / available (same switch as the sandbox pool)."""
    mode = os.environ.get("RBD_UNSHARE_NET", "auto").strip().lower()
    return ["unshare", "-rn"] if mode == "1" or (mode == "auto" and _unshare_available()) else []


def truncate_output(text: str, limit: int = OUTPUT_LIMIT, skipped: int = 0) -> str:
    """Keep the first and last ``limit // 2`` characters with a marker in between."""
    if len(text) <= limit and not skipped:
        return text
    half = limit // 2
    cut = len(text) - 2 * half + skipped
    return f"{text[:half]}\n<... {cut} characters truncated ...>\n{text[-half:]}"


def format_obs(rc: int, output: str) -> str:
    """Observation shown to the agent."""
    return f"<returncode>{rc}</returncode>\n<output>\n{output}</output>"


class CodeRepoEnv(Env):
    """Task dict: {"pid", "question", "public": [[in, out], ...], "hidden": [[in, out], ...], "time_limit_s"}."""

    def __init__(self, task: Dict[str, Any], cmd_timeout_s: float = 30.0, mem_mb: Optional[int] = None,
                 workdir: Optional[str] = None) -> None:
        self.task = task
        self.cmd_timeout_s = float(cmd_timeout_s)
        self.mem_mb = int(mem_mb or os.environ.get("RBD_ENV_MEM_MB", DEFAULT_MEM_MB))
        self.workdir = workdir or os.environ.get("RBD_WORKDIR") or None
        self.time_limit_s = float(task.get("time_limit_s", 6.0))
        self.base: Optional[Path] = None
        self.repo: Optional[Path] = None
        self._env: Dict[str, str] = {}

    # ------------------------------------------------------------ workspace

    def _make_env(self, base: Path) -> Dict[str, str]:
        env = {
            "PATH": f"{base / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(base / "home"), "TMPDIR": str(base / "tmp"), "LANG": "C.UTF-8", "TERM": "dumb",
            "PAGER": "cat", "GIT_PAGER": "cat", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1", "GIT_AUTHOR_NAME": "rbd", "GIT_AUTHOR_EMAIL": "rbd@localhost",
            "GIT_COMMITTER_NAME": "rbd", "GIT_COMMITTER_EMAIL": "rbd@localhost", "GIT_AUTHOR_DATE": GIT_DATE,
            "GIT_COMMITTER_DATE": GIT_DATE, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
        }
        for var in THREAD_ENV_VARS:
            env[var] = "1"
        return env

    def _git(self, *args: str) -> bytes:
        assert self.repo is not None, "reset() first"
        r = subprocess.run(["git", *args], cwd=self.repo, env=self._env, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=GIT_TIMEOUT_S)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace')[-500:]}")
        return r.stdout

    def reset(self) -> str:
        """Fresh repository with the task files in one deterministic commit; returns the task message."""
        self.close()
        if self.workdir:
            os.makedirs(self.workdir, exist_ok=True)
        pid = re.sub(r"[^\w.-]", "_", str(self.task.get("pid", "task")))[:40]
        base = Path(tempfile.mkdtemp(prefix=f"rbd_{pid}_", dir=self.workdir))
        repo = base / "repo"
        for d in ("repo", "home", "tmp", "bin"):
            (base / d).mkdir()
        for name in ("python", "python3"):
            wrapper = base / "bin" / name
            wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            wrapper.chmod(0o755)
        self.base, self.repo, self._env = base, repo, self._make_env(base)
        files = {"TASK.md": self.task["question"].rstrip() + "\n", "solution.py": STUB, ".gitignore": GITIGNORE,
                 "run_tests.py": RUN_TESTS.replace("__TIME_LIMIT__", repr(self.time_limit_s))}
        for k, (inp, out) in enumerate(self.task.get("public", []), 1):
            files[f"tests/public/{k:02d}.in"] = inp
            files[f"tests/public/{k:02d}.out"] = out
        for rel, text in files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
        self._git("-c", "init.defaultBranch=main", "init", "-q")
        self._git("config", "core.autocrlf", "false")
        self._git("add", "-A")
        self._git("commit", "-q", "--no-verify", "-m", "task")
        return TASK_MESSAGE.format(tl=self.time_limit_s, submit=SUBMIT_TOKEN, question=self.task["question"].strip())

    def close(self) -> None:
        if self.base is not None:
            shutil.rmtree(self.base, ignore_errors=True)
        self.base = self.repo = None

    # ------------------------------------------------------------ acting

    def step(self, command: str) -> StepResult:
        """Run one command (A1: the result depends only on the workspace and the command)."""
        assert self.base is not None and self.repo is not None, "reset() first"
        if SUBMIT_TOKEN in command:
            return StepResult(obs_text=format_obs(0, ""), exit_code=0, done=True, is_test=False)
        out_path = self.base / ".out"
        limits = functools.partial(_set_limits, max(1, int(self.cmd_timeout_s + 1)), self.mem_mb << 20)
        timed_out = False
        # The command goes through a script file outside the repository: a single argv string is limited to
        # 128 KiB on Linux, and agents sometimes write very long heredocs.
        script = self.base / ".cmd.sh"
        script.write_text(command, encoding="utf-8", errors="replace")
        with open(out_path, "wb") as out:
            try:
                proc = subprocess.Popen([*_net_prefix(), "bash", str(script)], cwd=self.repo, env=self._env,
                                        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=True, preexec_fn=limits)
            except OSError as e:
                return StepResult(obs_text=format_obs(126, f"command could not be started: {e}\n"), exit_code=126,
                                  done=False, is_test=bool(_TEST_RE.search(command)))
            try:
                rc = proc.wait(timeout=self.cmd_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # stray background processes of the command
            except (ProcessLookupError, PermissionError):
                pass
            if timed_out:
                proc.wait()
                rc = TIMEOUT_RC
        with open(out_path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            skipped = 0
            if size <= 2 * _READ_CAP:
                data = f.read()
            else:
                data = f.read(_READ_CAP)
                f.seek(size - _READ_CAP)
                data += f.read()
                skipped = size - 2 * _READ_CAP
        text = truncate_output(data.decode("utf-8", "replace"), skipped=skipped)
        if timed_out:
            text += f"\n<command timed out after {self.cmd_timeout_s:g} s and was killed>\n"
        rc = 128 - rc if rc < 0 else rc  # killed by a signal
        return StepResult(obs_text=format_obs(rc, text), exit_code=rc, done=False,
                          is_test=bool(_TEST_RE.search(command)))

    # ------------------------------------------------------------ state

    def _stage(self) -> tuple[bytes, List[str]]:
        raw = self._git("ls-files", "--stage", "-z")
        paths = [e.split(b"\t", 1)[1].decode("utf-8", "replace") for e in raw.split(b"\0") if b"\t" in e]
        return raw, [p for p in dict.fromkeys(paths) if not _excluded(p)]

    def _content(self, rel: str) -> Optional[bytes]:
        assert self.repo is not None
        p = self.repo / rel
        if p.is_symlink():
            return os.readlink(p).encode("utf-8", "replace")
        if p.is_file():
            return p.read_bytes()
        return None

    def fs_state(self) -> FsState:
        """Tracked contents (git blob ids), untracked non-ignored files, and a hash of the staged index entries."""
        raw, tracked = self._stage()
        tr = {p: _blob_sha(c) for p in tracked if (c := self._content(p)) is not None}
        others = self._git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        un: Dict[str, str] = {}
        for e in others:
            p = e.decode("utf-8", "replace")
            if p and not _excluded(p) and (c := self._content(p)) is not None:
                un[p] = _blob_sha(c)
        return FsState(tracked=tr, untracked=un, index=hashlib.sha1(raw).hexdigest())

    def tracked_files(self) -> Dict[str, str]:
        _, tracked = self._stage()
        return {p: c.decode("utf-8", "replace") for p in tracked if (c := self._content(p)) is not None}

    def restore_files(self, files: Dict[str, str]) -> None:
        """Write ``files`` and delete tracked files missing from the mapping (untracked files and index untouched)."""
        assert self.repo is not None
        _, tracked = self._stage()
        for rel in tracked:
            if rel not in files and (self.repo / rel).exists():
                (self.repo / rel).unlink()
        for rel, text in files.items():
            p = self.repo / rel
            data = text.encode("utf-8")
            if self._content(rel) != data:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)

    def evaluate(self) -> int:
        """1 iff solution.py passes every hidden test (public tests when no hidden tests exist)."""
        assert self.repo is not None
        if os.environ.get("RBD_ENV_EVAL", "1") == "0":  # selection-aware arms score y(tau) themselves
            return 0
        sol = self.repo / "solution.py"
        if not sol.is_file():
            return 0
        tests = self.task.get("hidden") or self.task.get("public") or []
        if not tests:
            return 0
        res = run_prog(sol.read_text(encoding="utf-8", errors="replace"), [t[0] for t in tests], self.time_limit_s,
                       expected=[t[1] for t in tests], early_stop=True)
        return int(len(res) == len(tests) and all(r[0] == "pass" for r in res))
