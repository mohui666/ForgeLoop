from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sft_data import (
    balance_by_task,
    render_assistant_only,
    shorten_tool_observations,
)


SECRET_PATTERN = re.compile(
    r"authorization:\s*bearer|api[_-]?key|password|"
    r"(?:^|\W)secret(?:\W|$)|sk-[a-z0-9_-]{10,}|hf_[a-z0-9]{10,}",
    re.IGNORECASE,
)
PACKAGE_NAMES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface-hub",
    "liger-kernel",
    "peft",
    "safetensors",
    "sentencepiece",
    "torch",
    "transformers",
    "trl",
)


class ConversationDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.samples[index]["input_ids"], dtype=torch.long)
        labels = torch.tensor(self.samples[index]["labels"], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in features],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned ForgeLoop Qwen3.5-4B QLoRA/SFT job."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def explicit_completed_finish(messages: list[dict[str, Any]]) -> bool:
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if not assistants:
        return False
    for call in assistants[-1].get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "finish":
            continue
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False
        return arguments.get("status") == "completed"
    return False


def audit_and_tokenize(
    dataset_path: Path,
    tokenizer: Any,
    tool_schemas: list[dict[str, Any]],
    max_length: int,
    require_finish: bool,
    max_samples_per_task: int,
    excluded_task_ids: set[str],
    tool_observation_keep_chars_per_side: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_lines = dataset_path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise ValueError("SFT export is empty")
    records = [json.loads(line) for line in raw_lines]
    ids = [str(record.get("id") or "") for record in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("SFT sample ids must be non-empty and unique")
    raw_text = dataset_path.read_text(encoding="utf-8")
    if SECRET_PATTERN.search(raw_text):
        raise ValueError("SFT export matched a credential-like pattern")

    conversation_hashes: set[str] = set()
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    all_lengths: list[int] = []
    source_models: Counter[str] = Counter()
    source_policies: Counter[str] = Counter()
    source_tool_schema_hashes: Counter[str] = Counter()
    shortened_tool_messages = 0
    shortened_tool_chars = 0
    training_tools_json = json.dumps(
        tool_schemas, ensure_ascii=False, sort_keys=True
    )
    training_tool_schema_sha256 = hashlib.sha256(
        training_tools_json.encode("utf-8")
    ).hexdigest()
    for record in records:
        if record.get("schema_version") != "forgeloop.sft.conversation.v1":
            raise ValueError(f"Unsupported SFT sample schema: {record.get('id')}")
        metadata = record.get("metadata") or {}
        if metadata.get("verifier_passed") is not True:
            raise ValueError(f"Non-verified sample reached SFT export: {record['id']}")
        messages = record.get("messages") or []
        training_messages, shortening = shorten_tool_observations(
            messages, tool_observation_keep_chars_per_side
        )
        shortened_tool_messages += shortening["changed_messages"]
        shortened_tool_chars += shortening["omitted_chars"]
        recorded_tools = record.get("tools") or []
        if not recorded_tools:
            raise ValueError(f"SFT sample has no tool schemas: {record['id']}")
        recorded_tools_json = json.dumps(
            recorded_tools, ensure_ascii=False, sort_keys=True
        )
        source_tool_schema_hashes[
            hashlib.sha256(recorded_tools_json.encode("utf-8")).hexdigest()
        ] += 1
        canonical = json.dumps(
            {"messages": messages, "tools": tool_schemas},
            ensure_ascii=False,
            sort_keys=True,
        )
        conversation_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if conversation_hash in conversation_hashes:
            raise ValueError(f"Duplicate conversation: {record['id']}")
        conversation_hashes.add(conversation_hash)
        training_conversation = json.dumps(
            {"messages": training_messages, "tools": tool_schemas},
            ensure_ascii=False,
            sort_keys=True,
        )
        training_conversation_hash = hashlib.sha256(
            training_conversation.encode("utf-8")
        ).hexdigest()
        source_models[str(metadata.get("model") or "unknown")] += 1
        policy = metadata.get("policy_identity") or {}
        source_policies[str(policy.get("policy_id") or "teacher_legacy")] += 1

        rendered = render_assistant_only(
            training_messages, tool_schemas, tokenizer
        )
        length = int(rendered["tokens"])
        all_lengths.append(length)
        reason = None
        task_id = str(metadata.get("task_id") or "")
        if not task_id:
            raise ValueError(f"SFT sample has no task_id: {record['id']}")
        if task_id in excluded_task_ids:
            reason = "frozen_holdout_task"
        elif require_finish and not explicit_completed_finish(messages):
            reason = "missing_explicit_completed_finish"
        elif length > max_length:
            reason = "over_max_length"
        if reason:
            excluded.append(
                {
                    "id": record["id"],
                    "task_id": task_id,
                    "tokens": length,
                    "reason": reason,
                }
            )
            continue
        usage = metadata.get("usage") or {}
        eligible.append(
            {
                "id": record["id"],
                "task_id": task_id,
                "repo": metadata.get("repo"),
                "base_sha": metadata.get("base_sha"),
                "source_trajectory_id": metadata.get("trajectory_id"),
                "source_model": metadata.get("model"),
                "source_policy": policy.get("policy_id") or "teacher_legacy",
                "source_steps": usage.get("steps"),
                "source_tool_calls": usage.get("tool_calls"),
                "tokens": length,
                **rendered,
                "source_conversation_sha256": conversation_hash,
                "training_conversation_sha256": training_conversation_hash,
                "shortened_tool_messages": shortening["changed_messages"],
                "shortened_tool_chars": shortening["omitted_chars"],
            }
        )
    selected, capped = balance_by_task(eligible, max_samples_per_task)
    excluded.extend(capped)
    if not selected:
        raise ValueError("No complete samples fit the configured maximum length")
    lengths = sorted(all_lengths)
    selected_lengths = sorted(sample["tokens"] for sample in selected)
    audit = {
        "exported_samples": len(records),
        "unique_ids": len(set(ids)),
        "unique_conversations": len(conversation_hashes),
        "secret_pattern_matches": 0,
        "render_tool_definitions": True,
        "chat_template": "upstream_tokenizer_apply_chat_template",
        "loss_scope": "assistant_natural_language_and_tool_calls_only",
        "non_assistant_label_id": -100,
        "assistant_mask_alignment": "character_offsets_from_chat_template_prefixes",
        "assistant_boundary_tokens": "excluded_when_crossing_target_boundary",
        "truncated_samples": 0,
        "max_length": max_length,
        "max_samples_per_task": max_samples_per_task,
        "tool_observation_shortening": {
            "strategy": "preserve_head_and_tail_with_explicit_marker",
            "keep_chars_per_side": tool_observation_keep_chars_per_side,
            "changed_messages_across_export": shortened_tool_messages,
            "omitted_chars_across_export": shortened_tool_chars,
            "assistant_targets_changed": False,
            "tool_schemas_changed": False,
        },
        "excluded_task_ids": sorted(excluded_task_ids),
        "all_length_tokens": {
            "min": min(lengths),
            "median": lengths[len(lengths) // 2],
            "max": max(lengths),
        },
        "selected_samples": len(selected),
        "selected_tasks": len({sample["task_id"] for sample in selected}),
        "selected_assistant_target_tokens": sum(
            sample["assistant_target_tokens"] for sample in selected
        ),
        "selected_masked_tokens": sum(sample["masked_tokens"] for sample in selected),
        "selected_boundary_tokens_excluded": sum(
            sample["boundary_tokens_excluded"] for sample in selected
        ),
        "selected_tool_schema_hashes": sorted(
            {sample["tool_schema_sha256"] for sample in selected}
        ),
        "training_tool_count": len(tool_schemas),
        "training_tool_names": [
            str(schema.get("function", {}).get("name")) for schema in tool_schemas
        ],
        "training_tool_schema_sha256": training_tool_schema_sha256,
        "source_recorded_tool_schema_hashes": dict(
            sorted(source_tool_schema_hashes.items())
        ),
        "selected_length_tokens": {
            "min": min(selected_lengths),
            "median": selected_lengths[len(selected_lengths) // 2],
            "max": max(selected_lengths),
        },
        "selected_source_models": dict(
            sorted(Counter(sample["source_model"] for sample in selected).items())
        ),
        "selected_source_policies": dict(
            sorted(Counter(sample["source_policy"] for sample in selected).items())
        ),
        "export_source_models": dict(sorted(source_models.items())),
        "export_source_policies": dict(sorted(source_policies.items())),
        "selected": [
            {
                key: value
                for key, value in sample.items()
                if key not in {"input_ids", "labels"}
            }
            for sample in selected
        ],
        "excluded": sorted(excluded, key=lambda item: (item["reason"], item["id"])),
    }
    return selected, audit


def aggregate_adapter_revision(adapter_dir: Path) -> tuple[str, dict[str, str]]:
    hashes = {
        path.relative_to(adapter_dir).as_posix(): sha256_file(path)
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), hashes


def nvidia_snapshot() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    set_seed(int(config["training"]["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=config["base_revision"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tool_schema_path = (
        args.config.parent / str(config["tool_schema_path"])
    ).resolve()
    tool_schemas = json.loads(tool_schema_path.read_text(encoding="utf-8"))
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise ValueError("Tool schema snapshot must be a non-empty list")
    selected, audit = audit_and_tokenize(
        args.dataset,
        tokenizer,
        tool_schemas,
        int(config["max_length"]),
        bool(config["require_explicit_completed_finish"]),
        int(config["max_samples_per_task"]),
        set(config["excluded_task_ids"]),
        int(config["tool_observation_keep_chars_per_side"]),
    )
    write_json(output / "dataset_audit.json", audit)
    provenance = {
        "schema_version": "forgeloop.sft.provenance.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_export": args.dataset.name,
        "source_export_sha256": sha256_file(args.dataset),
        "source_samples": audit["exported_samples"],
        "selected_samples": audit["selected_samples"],
        "selected_sample_ids": [sample["id"] for sample in selected],
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "tokenizer_revision": config["base_revision"],
        "tool_schema_path": str(tool_schema_path),
        "tool_schema_sha256": sha256_file(tool_schema_path),
        "selection": {
            "classification": "sft_candidate export only",
            "explicit_completed_finish": True,
            "render_tool_definitions": True,
            "chat_template": "upstream_tokenizer_apply_chat_template",
            "loss_scope": "assistant_natural_language_and_tool_calls_only",
            "non_assistant_label_id": -100,
            "max_samples_per_task": config["max_samples_per_task"],
            "task_balance_order": "shortest_rendered_tokens_then_sample_id",
            "excluded_task_ids": sorted(config["excluded_task_ids"]),
            "tool_observation_shortening": {
                "strategy": "preserve_head_and_tail_with_explicit_marker",
                "keep_chars_per_side": config[
                    "tool_observation_keep_chars_per_side"
                ],
                "assistant_targets_changed": False,
                "tool_schemas_changed": False,
            },
            "maximum_tokens": config["max_length"],
            "overlength_strategy": config["overlength_strategy"],
            "truncation": False,
        },
        "selected_tasks": audit["selected_tasks"],
        "selected_task_ids": sorted({sample["task_id"] for sample in selected}),
        "selected_source_trajectories": [
            {
                "sample_id": sample["id"],
                "task_id": sample["task_id"],
                "repo": sample["repo"],
                "base_sha": sample["base_sha"],
                "trajectory_id": sample["source_trajectory_id"],
            }
            for sample in selected
        ],
    }
    write_json(output / "dataset_provenance.json", provenance)
    write_json(output / "training_config.json", config)
    environment = {
        "created_at": datetime.now(UTC).isoformat(),
        "python_packages": {
            name: importlib.metadata.version(name) for name in PACKAGE_NAMES
        },
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": nvidia_snapshot(),
    }
    write_json(output / "environment.json", environment)

    quantization = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=str(quantization["type"]),
        bnb_4bit_use_double_quant=bool(quantization["double_quantization"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        revision=config["base_revision"],
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            task_type="CAUSAL_LM",
            target_modules=list(lora["target_modules"]),
        ),
    )
    apply_liger_kernel_to_qwen3_5(
        model=model.get_base_model(),
        fused_linear_cross_entropy=True,
        rms_norm=False,
        swiglu=False,
    )
    model.config.use_cache = False
    trainable, total = model.get_nb_trainable_parameters()

    training = config["training"]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"),
            num_train_epochs=float(training["epochs"]),
            per_device_train_batch_size=int(training["micro_batch_size"]),
            gradient_accumulation_steps=int(
                training["gradient_accumulation_steps"]
            ),
            learning_rate=float(training["learning_rate"]),
            lr_scheduler_type=str(training["scheduler"]),
            warmup_ratio=float(training["warmup_ratio"]),
            optim=str(training["optimizer"]),
            max_grad_norm=float(training["max_grad_norm"]),
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_strategy="steps",
            logging_steps=1,
            logging_nan_inf_filter=False,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=int(training["seed"]),
            data_seed=int(training["seed"]),
        ),
        train_dataset=ConversationDataset(selected),
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_revision, adapter_hashes = aggregate_adapter_revision(adapter_dir)
    losses = [
        float(item["loss"])
        for item in trainer.state.log_history
        if item.get("loss") is not None
    ]
    metrics = {
        "schema_version": "forgeloop.sft.metrics.v1",
        "train_metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "losses": losses,
        "initial_logged_loss": losses[0] if losses else None,
        "final_logged_loss": losses[-1] if losses else None,
        "loss_finite": bool(losses) and all(torch.isfinite(torch.tensor(losses))),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / total,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "gpu_after_training": nvidia_snapshot(),
        "adapter_revision": adapter_revision,
        "adapter_files_sha256": adapter_hashes,
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
