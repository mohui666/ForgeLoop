from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a Qwen3.5 PEFT adapter into upstream trainable weights."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision,
    )
    tokenizer.save_pretrained(args.output)
    manifest = {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": args.base_revision,
        "adapter": str(args.adapter.resolve()),
        "dtype": "bfloat16",
        "merged_at": datetime.now(UTC).isoformat(),
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
