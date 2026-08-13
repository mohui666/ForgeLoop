from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RunMode(str, Enum):
    GOAL = "goal"
    TASK = "task"
    PLAN = "plan"
    BUILD = "build"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    summary: str
    evidence: str
    trajectory_path: Path
    budget: dict
    stop_reason: str = "unknown"
    model: str | None = None
    provider: str | None = None
    delivery: dict | None = None
    provider_reliability: dict | None = None
    conversation: tuple[dict, ...] = ()
