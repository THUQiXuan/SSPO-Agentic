"""Representative program of an agent episode and its execution signals (g, Chk, sig).

SSPO second version, notation: the representative program y(tau) of run tau is the last program of the run that
passes every public example; if none does, the last program. Here a "program" is a version of ``solution.py``
in the episode's repository (the state after every turn, then the final state). Signals of y(tau), from the frozen
execution checker of ``selectrl.checker`` (helpers written once per problem by a frozen model):

  g(y)    1 iff y passes every hidden test (reward oracle; never visible to the selector),
  Chk(y)  1 iff y passes all public examples, finishes every stress input in time and matches every confirmed
          brute-force reference,
  sig(y)  hash of y's normalized outputs on all check inputs (what the deployed selector votes on).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rbd.envs.repo_env import STUB
from selectrl.checker.build import CHECKER_VERSION, load_cases
from selectrl.checker.cache import ExecCache
from selectrl.checker.chk import NOCODE_KEY, code_hash, label_program_on_hidden
from selectrl.sandbox import pmap, run_prog

SOLUTION = "solution.py"


def solution_versions(traj) -> List[str]:
    """Distinct consecutive versions of solution.py after each turn (stub and empty files dropped)."""
    raw = [(turn.files_before or {}).get(SOLUTION) for turn in traj.turns[1:]]
    raw.append((traj.final_files or {}).get(SOLUTION))
    out: List[str] = []
    for v in raw:
        if v is None or not v.strip() or v.strip() == STUB.strip():
            continue
        if out and out[-1] == v:
            continue
        out.append(v)
    return out


class Scorer:
    """Representative-program choice and cached execution signals for a fixed set of problems."""

    def __init__(self, problems: Dict[str, dict], cache_root: str | Path, cases_dir: str | Path):
        self.problems = problems
        self.cases_dir = Path(cases_dir)
        self.exec = ExecCache(cache_root, CHECKER_VERSION)
        self._cases: Dict[str, Optional[dict]] = {}
        self._pub: Dict[Tuple[str, str], bool] = {}
        self._hidden: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------ cases
    def cases(self, pid: str) -> Optional[dict]:
        if pid not in self._cases:
            self._cases[pid] = load_cases(self.cases_dir, pid)
        return self._cases[pid]

    def has_cases(self, pid: str) -> bool:
        return self.cases(pid) is not None

    # ------------------------------------------------------------------ representative program
    def _public_pass(self, item: Tuple[str, str]) -> bool:
        pid, code = item
        prob = self.problems[pid]
        pub = prob.get("public") or []
        if not pub:
            return False
        res = run_prog(code, [t[0] for t in pub], float(prob.get("time_limit_s", 6.0)),
                       expected=[t[1] for t in pub], key_only=True)
        return len(res) == len(pub) and all(r[0] == "pass" for r in res)

    def representatives(self, items: Sequence[Tuple[str, List[str]]]) -> List[Optional[str]]:
        """y(tau) for every (pid, versions) item: last version passing all public examples, else the last one."""
        todo: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for pid, vers in items:
            for v in vers:
                key = (pid, code_hash(v))
                if key[1] != NOCODE_KEY and key not in self._pub and key not in todo:
                    todo[key] = (pid, v)
        for key, ok in zip(todo, pmap(self._public_pass, list(todo.values()))):
            self._pub[key] = bool(ok)
        reps: List[Optional[str]] = []
        for pid, vers in items:
            vers = [v for v in vers if code_hash(v) != NOCODE_KEY]
            if not vers:
                reps.append(None)
                continue
            passing = [v for v in vers if self._pub.get((pid, code_hash(v)), False)]
            reps.append(passing[-1] if passing else vers[-1])
        return reps

    # ------------------------------------------------------------------ signals
    def score(self, items: Sequence[Tuple[str, Optional[str]]], require_cases: bool = True) -> List[Dict[str, Any]]:
        """Execution records {g, chk, sig, public_pass, ...} of (pid, code) items (cached by code hash)."""
        trip = []
        for pid, code in items:
            cases = self.cases(pid)
            if cases is None:
                if require_cases:
                    raise RuntimeError(f"no checker cases for {pid}; build them before scoring")
                cases = {"pid": pid, "checker_version": CHECKER_VERSION, "inputs": []}
            trip.append((self.problems[pid], cases, code))
        return self.exec.evaluate_many(trip)

    def hidden_labels(self, items: Sequence[Tuple[str, Optional[str]]]) -> List[int]:
        """g only (no checker cases needed), used to calibrate the training pool before the checker exists."""
        todo: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for pid, code in items:
            key = (pid, code_hash(code))
            if key[1] != NOCODE_KEY and key not in self._hidden and key not in todo:
                todo[key] = (pid, code)
        res = pmap(lambda it: int(label_program_on_hidden(it[1], self.problems[it[0]])["all"]), list(todo.values()))
        for key, g in zip(todo, res):
            self._hidden[key] = int(g)
        return [0 if code_hash(c) == NOCODE_KEY else self._hidden[(p, code_hash(c))] for p, c in items]
