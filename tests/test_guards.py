from forgeloop.guards import RepeatedActionDetector
from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall


def test_repeated_action_requires_same_contiguous_observation_and_workspace() -> None:
    detector = RepeatedActionDetector()
    read_a = ToolCall("1", "read_file", {"path": "a.py", "start_line": 1})
    read_b = ToolCall("2", "read_file", {"path": "b.py", "start_line": 1})

    first = detector.observe(
        read_a,
        ToolResult(True, "A = 1"),
        before_fingerprint="tree-1",
        after_fingerprint="tree-1",
    )
    changed_observation = detector.observe(
        read_a,
        ToolResult(True, "A = 2"),
        before_fingerprint="tree-1",
        after_fingerprint="tree-1",
    )
    different_scope = detector.observe(
        read_b,
        ToolResult(True, "B = 1"),
        before_fingerprint="tree-1",
        after_fingerprint="tree-1",
    )
    workspace_change = detector.observe(
        read_b,
        ToolResult(True, "B = 1"),
        before_fingerprint="tree-1",
        after_fingerprint="tree-2",
    )
    after_progress = detector.observe(
        read_b,
        ToolResult(True, "B = 1"),
        before_fingerprint="tree-2",
        after_fingerprint="tree-2",
    )
    proven = detector.observe(
        read_b,
        ToolResult(True, "B = 1"),
        before_fingerprint="tree-2",
        after_fingerprint="tree-2",
    )

    assert first.streak == 1
    assert changed_observation.streak == 1
    assert different_scope.streak == 1
    assert workspace_change.streak == 1
    assert after_progress.streak == 1
    assert proven.proven_repeat is True
    assert proven.streak == 2
