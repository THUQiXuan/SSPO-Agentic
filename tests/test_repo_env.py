"""CodeRepoEnv: reset determinism, observation format, fs_state bookkeeping, restore and hidden-test evaluation."""
from __future__ import annotations

import subprocess
import time

import pytest

from rbd.envs.base import SUBMIT_TOKEN
from rbd.envs.repo_env import CodeRepoEnv
from rbd.sandbox.pool import shutdown_pool

TASK = {"pid": "sum", "question": "Read two integers a and b and print a + b.",
        "public": [["1 2\n", "3\n"], ["5 7\n", "12\n"]],
        "hidden": [["10 20\n", "30\n"], ["-1 1\n", "0\n"], ["100 -50\n", "50\n"]], "time_limit_s": 2.0}
RIGHT = "cat > solution.py <<'EOF'\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\nEOF"
WRONG = RIGHT.replace("a + b", "a - b")


@pytest.fixture(scope="module", autouse=True)
def _pool():
    yield
    shutdown_pool()


@pytest.fixture
def env():
    e = CodeRepoEnv(TASK)
    e.reset()
    yield e
    e.close()


def _head(e: CodeRepoEnv) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=e.repo, capture_output=True, text=True).stdout.strip()


def test_reset_layout_and_determinism(env):
    other = CodeRepoEnv(TASK)
    msg = other.reset()
    try:
        assert SUBMIT_TOKEN in msg and "solution.py" in msg and "python run_tests.py" in msg
        assert "print a + b" in msg
        files = other.tracked_files()
        assert set(files) == {".gitignore", "TASK.md", "solution.py", "run_tests.py", "tests/public/01.in",
                              "tests/public/01.out", "tests/public/02.in", "tests/public/02.out"}
        assert _head(other) == _head(env)  # fixed author / date: identical commits
        assert other.fs_state().key() == env.fs_state().key()
        assert env.fs_state().untracked == {}
    finally:
        other.close()


def test_step_observation_format(env):
    r = env.step("echo hi; echo err >&2")
    assert r.obs_text == "<returncode>0</returncode>\n<output>\nhi\nerr\n</output>"
    assert (r.exit_code, r.done, r.is_test) == (0, False, False)
    r = env.step("exit 3")
    assert r.exit_code == 3 and r.obs_text.startswith("<returncode>3</returncode>")
    r = env.step("python -c \"print('x' * 20000)\"")
    assert "characters truncated" in r.obs_text and len(r.obs_text) < 10200
    assert env.step(f"echo {SUBMIT_TOKEN}").done


def test_step_timeout():
    e = CodeRepoEnv(TASK, cmd_timeout_s=1.0)
    e.reset()
    try:
        t0 = time.monotonic()
        r = e.step("echo start; sleep 20")
        assert time.monotonic() - t0 < 5
        assert r.exit_code == 124 and "timed out" in r.obs_text and "start" in r.obs_text
    finally:
        e.close()


def test_run_tests_and_evaluate(env):
    r = env.step("python run_tests.py")
    assert r.is_test and r.exit_code == 1 and "FAIL 01" in r.obs_text and "passed 0/2" in r.obs_text
    assert env.evaluate() == 0
    env.step(WRONG)
    r = env.step("python run_tests.py")
    assert r.exit_code == 1 and "wrong answer" in r.obs_text and "expected" in r.obs_text
    assert env.evaluate() == 0
    env.step(RIGHT)
    r = env.step("python run_tests.py")
    assert r.exit_code == 0 and "PASS 01" in r.obs_text and "passed 2/2" in r.obs_text
    assert env.evaluate() == 1


def test_fs_state_tracked_untracked_index(env):
    s0 = env.fs_state()
    env.step(WRONG)
    s1 = env.fs_state()
    assert s0.diff_paths(s1) == {"tracked": ["solution.py"], "untracked": [], "index": []}
    env.step("mkdir -p __pycache__ .pytest_cache && echo x > __pycache__/a.pyc && echo y > .pytest_cache/v")
    assert env.fs_state().key() == s1.key()
    env.step("echo note > notes.txt")
    s2 = env.fs_state()
    assert s1.diff_paths(s2) == {"tracked": [], "untracked": ["notes.txt"], "index": []}
    env.step("git add solution.py")
    s3 = env.fs_state()
    assert s2.diff_paths(s3) == {"tracked": [], "untracked": [], "index": ["<index>"]}


def test_restore_files(env):
    files0, key0 = env.tracked_files(), env.fs_state().key()
    env.step(WRONG + "\nrm TASK.md")
    assert "TASK.md" not in env.fs_state().tracked
    env.restore_files(files0)
    assert env.fs_state().key() == key0
    env.restore_files({k: v for k, v in files0.items() if k != "TASK.md"})
    assert "TASK.md" not in env.fs_state().tracked and "TASK.md" not in env.tracked_files()
