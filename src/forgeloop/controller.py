from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from forgeloop.agent_types import RunStatus
from forgeloop.policy import PolicyIdentity
from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall
from forgeloop.workspace import Workspace


CONTROLLER_V1_ID = "forgeloop.controller.v1"


@dataclass(frozen=True)
class ControllerRecovery:
    strategy: str
    trigger: str
    feedback: str


@dataclass(frozen=True)
class ControllerTerminal:
    status: RunStatus
    summary: str
    evidence: str
    stop_reason: str


class ControllerV1:
    """Small deterministic stability layer around the existing AgentLoop protocol."""

    identity = CONTROLLER_V1_ID

    def __init__(
        self,
        *,
        repeat_recovery_at: int = 2,
        edit_failure_recovery_at: int = 2,
        no_progress_recovery_at: int = 6,
    ) -> None:
        self.repeat_recovery_at = repeat_recovery_at
        self.edit_failure_recovery_at = edit_failure_recovery_at
        self.no_progress_recovery_at = no_progress_recovery_at
        self._initial_fingerprint = ""
        self._calls: Counter[str] = Counter()
        self._recoveries: Counter[str] = Counter()
        self._consecutive_edit_failures = 0
        self._no_progress_actions = 0
        self._terminal_recoveries = 0
        self._action_required = False

    def start(self, workspace: Workspace) -> None:
        self._initial_fingerprint = workspace.git_progress_fingerprint()

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
    ) -> tuple[ControllerRecovery, ...]:
        recoveries: list[ControllerRecovery] = []
        signature = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        self._calls[signature] += 1
        if self._calls[signature] == self.repeat_recovery_at:
            recoveries.append(
                self._recovery(
                    "repeated_action",
                    f"{call.name} repeated with identical arguments",
                    "Controller v1 detected an identical repeated action. Use the "
                    "latest observation, then choose a materially different inspect "
                    "or edit action instead of repeating it.",
                )
            )

        if before_fingerprint == after_fingerprint:
            self._no_progress_actions += 1
        else:
            self._no_progress_actions = 0
            self._action_required = False

        if not result.ok:
            recoveries.append(
                self._recovery(
                    "tool_error_feedback",
                    f"{call.name} returned an error",
                    "Controller v1: the ERROR tool observation immediately above is "
                    "authoritative. Correct the arguments or inspect the target state "
                    "before trying another action.",
                )
            )

        if call.name == "apply_patch":
            self._action_required = False
            if result.ok:
                self._consecutive_edit_failures = 0
            else:
                self._consecutive_edit_failures += 1
                if (
                    self._consecutive_edit_failures
                    == self.edit_failure_recovery_at
                ):
                    recoveries.append(
                        self._recovery(
                            "edit_failure_reinspect",
                            "consecutive apply_patch failures",
                            "Controller v1 recovery: stop patching from stale context. "
                            "Re-read the exact target region, then construct a smaller "
                            "patch from the current file contents.",
                        )
                    )

        if self._no_progress_actions == self.no_progress_recovery_at:
            recoveries.append(
                self._recovery(
                    "no_progress_reinspect",
                    f"{self._no_progress_actions} actions without Git-visible progress",
                    "Controller v1 detected prolonged exploration without repository "
                    "progress. Re-inspect only the evidence needed for the next minimal "
                    "edit, make the edit, and verify it. If the task is genuinely "
                    "blocked, use finish with blocked and concrete evidence.",
                )
            )
        if self._no_progress_actions == self.no_progress_recovery_at + 2:
            self._action_required = True
            recoveries.append(
                self._recovery(
                    "no_progress_action_required",
                    "two additional inspections after no-progress recovery",
                    "Controller v1: the recovery inspection window is exhausted and "
                    "there is still no Git-visible progress. The next action must be "
                    "a minimal apply_patch/edit, or an explicit finish with blocked or "
                    "failed and concrete evidence. Do not perform another broad "
                    "inspection.",
                )
            )
        return tuple(recoveries)

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        if not self._action_required or self._has_progress(current_fingerprint):
            return None
        if call.name in {"apply_patch", "finish"}:
            return None
        return self._recovery(
            "exploration_action_blocked",
            f"{call.name} attempted after the action-required recovery",
            "Controller v1 blocked further exploration because the repository still "
            "has no Git-visible progress. Use apply_patch for the smallest supported "
            "edit now, or call finish with blocked/failed and concrete evidence.",
        )

    def review_final(
        self, content: str | None, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal:
        has_progress = self._has_progress(current_fingerprint)
        if self._terminal_recoveries == 0:
            self._terminal_recoveries += 1
            if has_progress:
                return self._recovery(
                    "missing_explicit_finish",
                    "model returned text after modifying the repository",
                    "Controller v1 requires an explicit terminal decision. Verify the "
                    "current changes, then call finish with status, summary, and "
                    "evidence. Do not respond with a plain final message.",
                )
            return self._recovery(
                "exploration_without_change",
                "model returned text without repository progress",
                "Controller v1 rejected an exploratory final response because no "
                "Git-visible change exists. Inspect the relevant code, make and verify "
                "the requested edit, then call finish. If truly blocked, call finish "
                "with blocked and concrete evidence.",
            )
        reason = "controller_missing_finish" if has_progress else "controller_no_change_final"
        return ControllerTerminal(
            RunStatus.FAILED,
            "Controller v1 stopped after the model again ended without explicit finish.",
            (content or "No final content was provided.").strip(),
            reason,
        )

    def review_finish(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal | None:
        status = str(call.arguments.get("status", "failed"))
        if status != RunStatus.COMPLETED.value or self._has_progress(current_fingerprint):
            return None
        if self._terminal_recoveries == 0:
            self._terminal_recoveries += 1
            return self._recovery(
                "finish_without_change",
                "finish(completed) requested without repository progress",
                "Controller v1 rejected finish(completed) because no Git-visible "
                "change exists. Inspect and implement the requested modification, "
                "verify it, then call finish again. Use blocked or failed only when "
                "supported by concrete evidence.",
            )
        return ControllerTerminal(
            RunStatus.BLOCKED,
            "Controller v1 stopped repeated finish(completed) without a repository change.",
            "No Git-visible modification was produced.",
            "controller_finish_without_change",
        )

    def summary(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "recoveries": dict(sorted(self._recoveries.items())),
            "strategies_triggered": sorted(self._recoveries),
        }

    def _has_progress(self, current_fingerprint: str) -> bool:
        return bool(
            self._initial_fingerprint
            and current_fingerprint != self._initial_fingerprint
        )

    def _recovery(
        self, strategy: str, trigger: str, feedback: str
    ) -> ControllerRecovery:
        self._recoveries[strategy] += 1
        return ControllerRecovery(strategy, trigger, feedback)


def controller_for_policy(policy: Any) -> ControllerV1 | None:
    if not isinstance(policy, PolicyIdentity):
        return None
    if str(policy.serving_config.get("controller") or "").lower() != "v1":
        return None
    return ControllerV1()


__all__ = [
    "CONTROLLER_V1_ID",
    "ControllerRecovery",
    "ControllerTerminal",
    "ControllerV1",
    "controller_for_policy",
]
