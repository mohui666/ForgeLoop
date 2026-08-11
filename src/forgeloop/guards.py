from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall

LONG_HORIZON_GUARD_SCHEMA = "forgeloop.long-horizon-guards.v1"


@dataclass(frozen=True)
class RepeatedActionEvidence:
    """Evidence for one contiguous, observation-equivalent action streak."""

    action_fingerprint: str
    observation_fingerprint: str
    streak: int
    proven_repeat: bool


class RepeatedActionDetector:
    """Detect only consecutive actions that produce no new observable state.

    Similar actions outside the contiguous window are deliberately unrelated. A
    repeat is proven only when the canonical tool call and its visible result are
    identical, the current call leaves the workspace unchanged, and the workspace
    still matches the state left by the previous action.
    """

    def __init__(self) -> None:
        self._last_action = ""
        self._last_observation = ""
        self._last_after_fingerprint = ""
        self._last_workspace_unchanged = False
        self._streak = 0

    def observe(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
    ) -> RepeatedActionEvidence:
        action = _digest({"name": call.name, "arguments": call.arguments})
        observation = _digest({"ok": result.ok, "output": result.output})
        proven_repeat = bool(
            self._streak
            and action == self._last_action
            and observation == self._last_observation
            and self._last_workspace_unchanged
            and before_fingerprint == after_fingerprint
            and before_fingerprint == self._last_after_fingerprint
        )
        self._streak = self._streak + 1 if proven_repeat else 1
        self._last_action = action
        self._last_observation = observation
        self._last_after_fingerprint = after_fingerprint
        self._last_workspace_unchanged = before_fingerprint == after_fingerprint
        return RepeatedActionEvidence(
            action_fingerprint=action,
            observation_fingerprint=observation,
            streak=self._streak,
            proven_repeat=proven_repeat,
        )


def terminal_attempt_fingerprint(
    kind: str,
    payload: Any,
    *,
    current_fingerprint: str,
    controller_state: str = "",
) -> str:
    """Canonical fingerprint for a terminal request and its unchanged state."""

    return _digest(
        {
            "kind": kind,
            "payload": payload,
            "workspace": current_fingerprint,
            "controller_state": controller_state,
        }
    )


def guard_semantics(
    *,
    max_repeated_tool_calls: int,
    max_repeated_errors: int,
    max_no_progress_steps: int,
) -> dict[str, Any]:
    """Serializable semantics and resolved thresholds for trajectory provenance."""

    return {
        "schema_version": LONG_HORIZON_GUARD_SCHEMA,
        "repeated_action": {
            "window": "contiguous",
            "action_fingerprint": "canonical_tool_name_and_arguments_sha256",
            "observation_fingerprint": "tool_ok_and_output_sha256",
            "workspace_progress": "git_progress_fingerprint_before_and_after",
            "warning_streak": 2,
            "configured_repeat_limit": max_repeated_tool_calls,
            "hard_stop_streak": max_repeated_tool_calls + 1,
        },
        "repeated_error": {
            "terminal": False,
            "advisory_streak": max_repeated_errors,
        },
        "mutation_no_progress": {
            "terminal": False,
            "advisory_streak": max_no_progress_steps,
        },
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
