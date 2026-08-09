from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return list(value)


def render_assistant_only(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Render the upstream chat template and label assistant output spans only."""

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _input_ids(encoded)
    template_ids = _input_ids(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if input_ids != template_ids:
        raise ValueError("Tokenizer ids differ from apply_chat_template ids")

    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        generation_prefix = tokenizer.apply_chat_template(
            messages[:index],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_prefix = tokenizer.apply_chat_template(
            messages[: index + 1],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not rendered.startswith(assistant_prefix):
            raise ValueError("Assistant prefix does not match full chat rendering")
        if not assistant_prefix.startswith(generation_prefix):
            raise ValueError("Generation prefix does not match assistant rendering")
        spans.append((len(generation_prefix), len(assistant_prefix)))
    if not spans:
        raise ValueError("Conversation contains no assistant messages")

    labels = [-100] * len(input_ids)
    boundary_tokens = 0
    target_counts = [0] * len(spans)
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if end <= start:
            continue
        contained = False
        overlaps = False
        for span_index, (span_start, span_end) in enumerate(spans):
            if start >= span_start and end <= span_end:
                labels[token_index] = input_ids[token_index]
                target_counts[span_index] += 1
                contained = True
                break
            if start < span_end and end > span_start:
                overlaps = True
        if overlaps and not contained:
            boundary_tokens += 1
    if not all(target_counts):
        raise ValueError("At least one assistant message has no target tokens")

    tool_schema = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "tokens": len(input_ids),
        "assistant_target_tokens": sum(target_counts),
        "assistant_messages": len(spans),
        "masked_tokens": len(input_ids) - sum(target_counts),
        "boundary_tokens_excluded": boundary_tokens,
        "tool_count": len(tools),
        "tool_schema_sha256": hashlib.sha256(tool_schema.encode("utf-8")).hexdigest(),
    }


def shorten_tool_observations(
    messages: list[dict[str, Any]], keep_chars_per_side: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if keep_chars_per_side < 1:
        raise ValueError("keep_chars_per_side must be positive")
    shortened = deepcopy(messages)
    changed_messages = 0
    omitted_chars = 0
    for message in shortened:
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            continue
        minimum = keep_chars_per_side * 2 + 80
        if len(content) <= minimum:
            continue
        omitted = len(content) - keep_chars_per_side * 2
        marker = (
            f"\n...[tool observation shortened for SFT; omitted {omitted} chars]...\n"
        )
        message["content"] = (
            content[:keep_chars_per_side]
            + marker
            + content[-keep_chars_per_side:]
        )
        changed_messages += 1
        omitted_chars += omitted
    return shortened, {
        "changed_messages": changed_messages,
        "omitted_chars": omitted_chars,
    }


def balance_by_task(
    samples: list[dict[str, Any]], max_samples_per_task: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_samples_per_task < 1:
        raise ValueError("max_samples_per_task must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        task_id = str(sample.get("task_id") or "")
        if not task_id:
            raise ValueError(f"Sample has no task_id: {sample.get('id')}")
        grouped[task_id].append(sample)

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        ranked = sorted(
            grouped[task_id],
            key=lambda item: (int(item["tokens"]), str(item["id"])),
        )
        selected.extend(ranked[:max_samples_per_task])
        excluded.extend(
            {
                "id": item["id"],
                "task_id": task_id,
                "tokens": item["tokens"],
                "reason": "task_sample_cap",
            }
            for item in ranked[max_samples_per_task:]
        )
    return selected, excluded
