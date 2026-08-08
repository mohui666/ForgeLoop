from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from forgeloop.security import EvidenceSanitizer, SecretRedactor

EFFECT_SCHEMA_VERSION = "forgeloop.effect.v1"
EFFECT_TYPES = {
    "file.read",
    "file.write",
    "file.delete",
    "shell.exec",
    "git.change",
    "test.run",
    "http.request",
    "policy.violation",
}


@dataclass(frozen=True)
class EffectDraft:
    type: str
    target: str
    action: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EFFECT_TYPES:
            raise ValueError(f"Unsupported effect type: {self.type}")


@dataclass(frozen=True)
class EffectContext:
    step: int
    tool_call_id: str


class EffectTrajectory(Protocol):
    run_id: str
    redactor: SecretRedactor

    def append(self, event_type: str, payload: Any) -> None: ...


class EffectRecorder:
    """Attach sanitized tool-side effects to the existing trajectory stream."""

    def __init__(self, trajectory: EffectTrajectory, workspace_root: Path) -> None:
        self.trajectory = trajectory
        self.sanitizer = EvidenceSanitizer(
            redactor=trajectory.redactor,
            local_roots=(workspace_root,),
        )
        self._sequence = 0

    def record(
        self,
        draft: EffectDraft,
        *,
        context: EffectContext,
        tool_name: str,
    ) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "schema_version": EFFECT_SCHEMA_VERSION,
            "event_id": f"eff_{self.trajectory.run_id[:8]}_{self._sequence:04d}",
            "trajectory_id": self.trajectory.run_id,
            "step": context.step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": draft.type,
            "tool_name": tool_name,
            "tool_call_id": context.tool_call_id,
            "target": draft.target,
            "action": draft.action,
            "result": draft.result,
            "risk": draft.risk or {"level": "low", "flags": []},
            "evidence": draft.evidence,
        }
        sanitized = self.sanitizer.sanitize(event)
        self.trajectory.append("effect", sanitized)
        return sanitized


def summarize_effects(
    events: list[dict[str, Any]], *, legacy: bool = False
) -> dict[str, Any]:
    by_type = Counter(str(event.get("type") or "unknown") for event in events)
    status_counts = Counter(
        str((event.get("result") or {}).get("status") or "unknown") for event in events
    )
    flags: set[str] = set()
    modified_files: set[str] = set()
    max_risk = "none"
    risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    for event in events:
        effect_type = str(event.get("type") or "")
        target = str(event.get("target") or "")
        risk = event.get("risk") or {}
        level = str(risk.get("level") or "low").lower()
        if risk_order.get(level, 0) > risk_order.get(max_risk, 0):
            max_risk = level
        flags.update(str(flag) for flag in risk.get("flags") or [] if flag)
        if effect_type == "policy.violation":
            flags.add("policy_violation")
        if effect_type in {"file.write", "file.delete"} and target:
            modified_files.add(target)
        if effect_type == "git.change":
            flags.update(
                str(flag)
                for flag in (event.get("action") or {}).get("safety_flags") or []
                if flag
            )
            modified_files.update(
                str(path)
                for path in (event.get("action") or {}).get("changed_paths") or []
                if path
            )
    return {
        "status": (
            "legacy_no_effect_events"
            if legacy and not events
            else "recorded"
            if events
            else "recorded_empty"
        ),
        "total": len(events),
        "by_type": dict(sorted(by_type.items())),
        "result_statuses": dict(sorted(status_counts.items())),
        "max_risk": max_risk,
        "risk_flags": sorted(flags),
        "modified_files": sorted(modified_files),
    }


__all__ = [
    "EFFECT_SCHEMA_VERSION",
    "EFFECT_TYPES",
    "EffectContext",
    "EffectDraft",
    "EffectRecorder",
    "summarize_effects",
]
