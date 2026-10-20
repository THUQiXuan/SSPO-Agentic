"""Environment interface: a git repository the agent edits through one bash command per turn."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from rbd.types import FsState, StepResult

SUBMIT_TOKEN = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class Env(ABC):
    """One episode. Implementation: ``repo_env.CodeRepoEnv`` (competitive programming in a local git repo,
    sandboxed processes)."""

    task: Dict[str, Any]

    @abstractmethod
    def reset(self) -> str:
        """Create a fresh workspace and return the task message shown to the agent."""

    @abstractmethod
    def step(self, command: str) -> StepResult:
        """Execute one bash command in the workspace (the submit token ends the episode)."""

    @abstractmethod
    def fs_state(self) -> FsState:
        """Current repository state (tracked contents, untracked files, index)."""

    @abstractmethod
    def tracked_files(self) -> Dict[str, str]:
        """Text of every tracked file (small repositories only; SWE envs return the dirty files)."""

    @abstractmethod
    def restore_files(self, files: Dict[str, str]) -> None:
        """Overwrite tracked files with the given contents (re-creates a recorded state for off-line checks)."""

    @abstractmethod
    def evaluate(self) -> int:
        """Hidden-test reward of the current workspace, in {0, 1}."""

    @abstractmethod
    def close(self) -> None:
        """Remove the workspace."""
