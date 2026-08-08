from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from forgeloop.model_capabilities import ModelCapability
from forgeloop.security import SecretRedactor
from forgeloop.types import Message


@dataclass(frozen=True)
class ContextBudget:
    context_window: int | None
    reserved_output: int | None
    thinking_tool_reserve: int | None
    safety_margin: int | None
    usable_context: int | None
    auto_compact_threshold: int | None


def context_budget(
    capability: ModelCapability, thinking_level: str = "auto"
) -> ContextBudget:
    window = capability.context_window
    if window is None:
        return ContextBudget(None, None, None, None, None, None)
    # These are explicit ForgeLoop safety policies, not claimed model facts.
    # Provider/registry metadata supplies the actual window and output ceiling.
    output = capability.max_output_tokens
    if output is None:
        return ContextBudget(window, None, None, None, None, None)
    tool_reserve = max(2_048, int(window * 0.03))
    if capability.thinking and thinking_level != "auto":
        tool_reserve += max(2_048, int(window * 0.03))
    safety = max(2_048, int(window * 0.03))
    usable = max(0, window - output - tool_reserve - safety)
    threshold = int(usable * 0.85)
    return ContextBudget(window, output, tool_reserve, safety, usable, threshold)


def estimate_tokens(messages: Sequence[Message]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, chars // 4) if chars else 0


def compact_messages(
    messages: Sequence[Message],
    *,
    keep_recent: int = 8,
    redactor: SecretRedactor | None = None,
    context_state: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[list[Message], dict[str, int]]:
    """Deterministically compress older conversational turns without a model call."""
    source = [dict(message) for message in messages]
    before = estimate_tokens(source)
    if len(source) <= keep_recent and not force:
        return source, {"before_tokens": before, "after_tokens": before, "compacted": 0}
    if force and len(source) > 1:
        keep_recent = min(keep_recent, max(1, len(source) // 3))
    older, recent = source[:-keep_recent], source[-keep_recent:]
    lines: list[str] = []
    for message in older:
        role = str(message.get("role", "unknown"))
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        lines.append(f"{role}: {content[:600]}")
    state = context_state or {}
    sections: list[str] = [
        "ForgeLoop compacted context (authoritative working summary):"
    ]
    labels = (
        ("original_task", "Original task"),
        ("latest_constraints", "Latest user constraints"),
        ("current_plan", "Current plan"),
        ("diff_summary", "Current diff summary"),
        ("key_locations", "Key code locations"),
        ("test_evidence", "Key tests/errors"),
        ("completed", "Completed"),
        ("pending", "Pending"),
        ("tool_evidence", "Important tool evidence"),
    )
    for key, label in labels:
        value = state.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            rendered = "; ".join(str(item) for item in value[-12:])
        else:
            rendered = str(value)
        sections.append(f"{label}: {rendered[:4000]}")
    if lines:
        sections.append("Earlier turns:\n" + "\n".join(lines))
    summary = "\n".join(sections)
    if redactor:
        summary = redactor.redact_text(summary)
    result: list[Message] = [{"role": "system", "content": summary}, *recent]
    return result, {
        "before_tokens": before,
        "after_tokens": estimate_tokens(result),
        "compacted": len(older),
    }
