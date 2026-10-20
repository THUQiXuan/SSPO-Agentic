"""Core data contract shared by environments, the agent loop, pivot synthesis, losses and the trainer.

A *trajectory* is a multi-turn episode of a shell agent working in a git repository. Every turn the model emits
one assistant message containing exactly one bash command; the environment executes it and returns an
observation. The file-system state of the repository is recorded before and after every turn so that pivot
synthesis can locate where a failed branch diverged from a successful one and verify the conditions of the
deterministic-rollback isomorphism (Theorem 1 of the method document).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FsState:
    """Repository state modulo timestamps / stat caches.

    tracked:   path -> sha1 of the content of every git-tracked file (missing files are absent)
    untracked: path -> sha1 of every untracked, non-ignored file
    index:     sha1 of the git index entries (``git ls-files --stage``), i.e. what ``git checkout --`` restores
    """

    tracked: Dict[str, str]
    untracked: Dict[str, str]
    index: str

    def key(self) -> str:
        """Single hash identifying the state (used for equality tests such as f-_{t*} == f+_{t+})."""
        import hashlib

        h = hashlib.sha1()
        for part in (sorted(self.tracked.items()), sorted(self.untracked.items()), self.index):
            h.update(repr(part).encode())
        return h.hexdigest()[:16]

    def diff_paths(self, other: "FsState") -> Dict[str, List[str]]:
        """Paths whose tracked / untracked content differs between two states, and whether the index changed."""
        def d(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
            return sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))
        return {"tracked": d(self.tracked, other.tracked), "untracked": d(self.untracked, other.untracked),
                "index": ["<index>"] if self.index != other.index else []}


@dataclass
class StepResult:
    obs_text: str  # observation shown to the agent (already formatted: return code + output)
    exit_code: Optional[int]  # None when no command could be executed (format error)
    done: bool  # the agent submitted or the episode cannot continue
    is_test: bool  # the command ran the task's test entry point (public tests / repository tests)


@dataclass
class Turn:
    idx: int
    action_text: str  # assistant message text (what the model generated, special tokens stripped)
    action_ids: List[int]  # sampled token ids of the assistant message (including the end-of-turn token)
    action_logprobs: List[float]  # sampling-time log-probabilities of action_ids
    command: Optional[str]  # the extracted bash command (None if the message had no valid command block)
    obs_text: str
    obs_ids: List[int]  # tokens appended after the action: end of turn + observation message + next prompt
    exit_code: Optional[int]
    is_test: bool
    fs_before: FsState
    fs_after: FsState
    files_before: Dict[str, str] = field(default_factory=dict)  # full text of tracked files before the turn

    @property
    def wrote(self) -> bool:
        return self.fs_before.key() != self.fs_after.key()


@dataclass
class Trajectory:
    task_id: str
    group_id: str  # same id for the G samples of one task in one step
    sample_idx: int
    prompt_ids: List[int]  # system + task message + first generation prompt
    turns: List[Turn]
    reward: int  # hidden-test reward in {0, 1}
    finish: str  # "submit" | "max_turns" | "context" | "format"
    final_files: Dict[str, str] = field(default_factory=dict)  # tracked files at the end
    meta: Dict[str, Any] = field(default_factory=dict)

    def num_tokens(self) -> int:
        return len(self.prompt_ids) + sum(len(t.action_ids) + len(t.obs_ids) for t in self.turns)

    def token_layout(self) -> List[tuple]:
        """[(kind, turn_idx, start, end)] spans over the concatenated sequence; kind in {prompt, action, obs}."""
        out, pos = [("prompt", -1, 0, len(self.prompt_ids))], len(self.prompt_ids)
        for t in self.turns:
            out.append(("action", t.idx, pos, pos + len(t.action_ids)))
            pos += len(t.action_ids)
            out.append(("obs", t.idx, pos, pos + len(t.obs_ids)))
            pos += len(t.obs_ids)
        return out

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PivotSample:
    """A synthesized error -> feedback -> rollback -> correct-continuation trajectory (Section 4 of the method).

    ids: full token sequence of the spliced trajectory (failure prefix, pivot turn, success suffix)
    kl_spans: [(start, end)] token spans of the actions at t_piv and t_piv + 1 (positions whose next-token
              distribution is matched to the mixture teacher)
    teacher_ids: per success direction u, the token sequence with the privileged hint inserted, plus the matching
              spans, so that teacher log-probs line up position by position with ``kl_spans``
    slot: (position in ids, token id c-_s) of the decision slot for the unlikelihood term, or None
    weight: recovery-value weight w(tau) before normalization
    """

    task_id: str
    ids: List[int]
    kl_spans: List[tuple]
    teacher_ids: List[List[int]]
    teacher_spans: List[List[tuple]]
    slot: Optional[tuple]
    weight: float
    info: Dict[str, Any] = field(default_factory=dict)
