from __future__ import annotations

import json
import os
import sys
from collections import UserDict
from pathlib import Path
from typing import Any

import pytest

from forgeloop.agent import FINISH_SCHEMA
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.workspace import Workspace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from sft_data import (  # noqa: E402
    balance_by_task,
    render_assistant_only,
    shorten_tool_observations,
)


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | UserDict:
        text = "<system-tools>" + json.dumps(tools, sort_keys=True) + "</system-tools>"
        for message in messages:
            role = message["role"]
            if role == "assistant":
                text += "<assistant>" + str(message.get("content") or "")
                text += json.dumps(message.get("tool_calls") or [], sort_keys=True)
                text += "</assistant>"
            else:
                text += f"<{role}>{message.get('content', '')}</{role}>"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return UserDict({"input_ids": [ord(character) for character in text]})
        return text

    @staticmethod
    def __call__(
        text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list[Any]]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_render_assistant_only_masks_context_and_preserves_tool_calls() -> None:
    messages = [
        {"role": "system", "content": "system context"},
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "I will inspect.",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}],
        },
        {"role": "tool", "content": "file contents"},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [
                {
                    "function": {
                        "name": "finish",
                        "arguments": {"status": "completed"},
                    }
                }
            ],
        },
    ]
    tools = [{"function": {"name": "read_file"}, "type": "function"}]

    rendered = render_assistant_only(messages, tools, FakeTokenizer())
    target = "".join(
        chr(token)
        for token, label in zip(rendered["input_ids"], rendered["labels"], strict=True)
        if label != -100
    )

    assert "I will inspect." in target
    assert '"name": "read_file"' in target
    assert '"name": "finish"' in target
    assert "system context" not in target
    assert "fix it" not in target
    assert "file contents" not in target
    assert rendered["assistant_messages"] == 2
    assert rendered["assistant_target_tokens"] > 0
    assert rendered["masked_tokens"] > 0


def test_balance_by_task_keeps_two_shortest_samples_per_task() -> None:
    samples = [
        {"id": "a-long", "task_id": "a", "tokens": 30},
        {"id": "a-short", "task_id": "a", "tokens": 10},
        {"id": "a-medium", "task_id": "a", "tokens": 20},
        {"id": "b-only", "task_id": "b", "tokens": 40},
    ]

    selected, excluded = balance_by_task(samples, 2)

    assert [sample["id"] for sample in selected] == [
        "a-short",
        "a-medium",
        "b-only",
    ]
    assert excluded == [
        {
            "id": "a-long",
            "task_id": "a",
            "tokens": 30,
            "reason": "task_sample_cap",
        }
    ]


def test_tool_observation_shortening_never_changes_assistant_targets() -> None:
    messages = [
        {"role": "assistant", "content": "target", "tool_calls": []},
        {"role": "tool", "content": "A" * 500 + "B" * 500},
    ]

    shortened, stats = shorten_tool_observations(messages, 20)

    assert messages[1]["content"] == "A" * 500 + "B" * 500
    assert shortened[0] == messages[0]
    assert shortened[1]["content"].startswith("A" * 20)
    assert shortened[1]["content"].endswith("B" * 20)
    assert "omitted 960 chars" in shortened[1]["content"]
    assert stats == {"changed_messages": 1, "omitted_chars": 960}


@pytest.mark.skipif(os.name != "nt", reason="Snapshot targets Windows LocalRuntime")
def test_training_tool_snapshot_matches_live_inference_schemas(tmp_path: Path) -> None:
    snapshot = json.loads(
        (
            Path(__file__).parents[1] / "training" / "qwen3.5-tools-windows-local.json"
        ).read_text(encoding="utf-8")
    )
    live = [
        *build_default_tools(Workspace(tmp_path), LocalRuntime()).schemas(),
        FINISH_SCHEMA,
    ]

    assert snapshot == live
    assert [schema["function"]["name"] for schema in snapshot] == [
        "read_file",
        "search_files",
        "apply_patch",
        "shell",
        "validate",
        "list_files",
        "git_diff",
        "git_inspect",
        "finish",
    ]
