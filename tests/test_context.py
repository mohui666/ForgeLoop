from __future__ import annotations

from forgeloop.context import prepare_agent_context


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object"},
        },
    }
]


def _turn(
    index: int,
    name: str,
    arguments: dict,
    output: str,
    *,
    reasoning: str = "",
    ok: bool = True,
) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"{'OK' if ok else 'ERROR'}\n{output}",
        },
    ]


def test_agent_context_compacts_stale_outputs_and_reasoning() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "keep this exact task"},
    ]
    for index in range(8):
        messages.extend(
            _turn(
                index,
                "list_files",
                {"path": f"old-{index}"},
                f"STALE-{index}-" + "x" * 4_000,
                reasoning=f"OLD-REASONING-{index}-" + "r" * 2_000,
            )
        )

    compacted, report = prepare_agent_context(
        messages,
        TOOLS,
        base_message_count=2,
        compact_threshold_tokens=1,
    )

    rendered = str(compacted)
    assert report["applied"] is True
    assert report["after_estimated_tokens"] < report["before_estimated_tokens"]
    assert compacted[1] == messages[1]
    assert "list_files" in compacted[2]["content"]
    assert "old-0" in compacted[2]["content"]
    assert "STALE-0-" in compacted[2]["content"]
    assert "OLD-REASONING-0" not in rendered
    assert "OLD-REASONING-7" in rendered
    assert "STALE-7-" in rendered


def test_agent_context_preserves_current_change_and_failure_evidence() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task-contract"},
    ]
    messages.extend(
        _turn(
            0,
            "read_file",
            {"path": "src/current.py", "start_line": 10, "end_line": 30},
            "CURRENT-SOURCE\n" + "s" * 3_000,
        )
    )
    messages.extend(
        _turn(1, "list_files", {"path": "old"}, "OBSOLETE-LIST\n" + "x" * 4_000)
    )
    messages.extend(
        _turn(
            2,
            "apply_patch",
            {
                "path": "src/current.py",
                "old_text": "before",
                "new_text": "CURRENT-PATCH",
            },
            "Updated src/current.py",
        )
    )
    messages.extend(
        _turn(
            3,
            "shell",
            {"command": "pytest tests/test_current.py -q"},
            "exit_code: 1\nLATEST-TEST-FAILURE",
            ok=False,
        )
    )
    messages.extend(
        _turn(4, "git_diff", {}, "LATEST-DIFF\n+CURRENT-PATCH")
    )
    for index in range(5, 11):
        messages.extend(
            _turn(index, "search_files", {"pattern": str(index)}, "recent")
        )

    compacted, report = prepare_agent_context(
        messages,
        TOOLS,
        base_message_count=2,
        compact_threshold_tokens=1,
    )

    rendered = str(compacted)
    assert report["applied"] is True
    assert "task-contract" in rendered
    assert "CURRENT-SOURCE" in rendered
    assert "CURRENT-PATCH" in rendered
    assert "LATEST-TEST-FAILURE" in rendered
    assert "LATEST-DIFF" in rendered
    assert "OBSOLETE-LIST" in rendered
    assert "x" * 2_000 not in rendered
    assert report["preserved_reasons"]["current_patch"] == 1
    assert report["preserved_reasons"]["latest_failed_test"] == 1
