# Qwen3.5-4B SFT v1 closed loop

Date: 2026-08-09

This run completed ForgeLoop's first real, minimal
Dataset -> Train -> Redeploy -> Evaluate loop. It used one training run and one
evaluation attempt per task. It did not change AgentLoop, the Dataset classifier,
the Verifier, or the tasks, and it did not start RL.

## Dataset audit

The fresh ForgeLoop Dataset build contained 90 samples: 78 `sft_candidate`, 10
`successful_but_inefficient`, and 2 `infrastructure_failure`. Only the strict
`sft_candidate` export was considered. The audit found 78 unique ids, 78 unique
exact conversations, and no secret-pattern match.

Training selected 48 real conversations after rendering with the upstream Qwen
chat template and applying the 2,048-token limit without truncation. All 48 end
with an explicit `finish(status=completed)`. They comprise 47
`deepseek/deepseek-v4-flash` teacher trajectories and one qualifying
`qwen3.5-4b-local` trajectory. Static tool definitions were omitted from the
rendered training text to fit this first 8 GB run; recorded user, assistant,
tool-call, and tool-result messages were preserved. Selected lengths were
1,158/1,636/2,033 tokens (min/median/max).

The ignored local audit and provenance artifacts are:

- `.forgeloop/training/qwen3.5-4b-sft-v1/dataset_audit.json`
- `.forgeloop/training/qwen3.5-4b-sft-v1/dataset_provenance.json`

## Training

The trainable source was `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, not the Ollama Q4 artifact.
Training ran once in WSL2 on an RTX 4060 Laptop 8 GB with:

- 4-bit NF4 base loading, BF16 compute, double quantization;
- LoRA rank 4, alpha 8, dropout 0.05 on q/k/v/o and gate/up/down projections;
- sequence length 2,048, micro-batch 1, gradient accumulation 8;
- one epoch, six optimizer steps, learning rate 2e-4, cosine schedule;
- paged AdamW 8-bit, gradient checkpointing, seed 42;
- Liger fused linear cross entropy and full-conversation causal LM loss.

The ordinary full-vocabulary loss path exceeded available memory at the longest
sequence because Qwen's 248k-token logits materialized about 1.9 GB. The fused
loss path avoided those logits. PyTorch recorded 6,447.54 MiB peak allocated and
6,614 MiB peak reserved; `nvidia-smi` observed 7,642 MiB in use and 98% GPU
utilization during training.

Training took 565.85 seconds. Mean train loss was 1.69156. Logged losses were
2.37295, 2.39933, 1.85913, 1.28813, 1.19782, and 1.03202; all were finite and
the run converged normally for this small one-epoch objective.

Local artifacts:

- adapter: `.forgeloop/training/qwen3.5-4b-sft-v1/adapter`
- trainer checkpoint/state: `.forgeloop/training/qwen3.5-4b-sft-v1/trainer`
- metrics/config/environment: `.forgeloop/training/qwen3.5-4b-sft-v1/*.json`
- adapter revision: `ea73f2c44e68dcf10c8c381a662888ed284953f1cb84f3b9e4156201db0308c3`
- adapter safetensors SHA-256: `75f580d345b5fb83c7b8e88c5049a63c7bab395ac048087788d68d03ff7cc5cf`

The tracked training inputs are `training/qwen3.5-4b-sft-v1.json`,
`training/requirements-qwen35-sft.txt`, and
`scripts/train_qwen35_qlora.py`.

## Redeployment

The adapter was merged into the same pinned upstream BF16 base with
`scripts/merge_qwen35_adapter.py`. Because Ollama 0.32.6 on Windows could not
directly convert the PEFT adapter, and its experimental int4 importer attempted
to load MLX, the merged model was converted and quantized with upstream
llama.cpp. Conversion must exclude the MTP draft head:

```bash
python convert_hf_to_gguf.py <merged-bf16> \
  --outfile qwen3.5-4b-sft-v1-bf16.gguf \
  --outtype bf16 --no-nextn
llama-quantize qwen3.5-4b-sft-v1-bf16.gguf \
  qwen3.5-4b-sft-v1-q4_k_m.gguf Q4_K_M
```

The actual first conversion exposed incorrect MTP metadata (33 blocks and one
NextN layer despite only 32 exported trunk blocks). Correcting those values to
the upstream and official Ollama Base values is equivalent to `--no-nextn` and
produced deployed GGUF SHA-256
`adb1ca229a9caa2e260b5fe3600b10f7375044bbcbfcc49ddb25390ee02a7f35`.

The local Modelfile imports that GGUF, sets `num_ctx 8192`, temperature 0.2, and
top-p 0.95. Redeploy with:

```powershell
ollama create qwen3.5-4b-sft-v1 -f Modelfile
uv run forgeloop task "Fix the failing test" `
  --policy-manifest qwen3.5-4b-sft-v1 `
  --runtime local
```

The live OpenAI-compatible probe returned a correctly structured `finish` call
with real usage. Ollama reported 8,192 context, 100% GPU placement, and about
3,753 MiB runtime GPU memory. ForgeLoop records local API cost as zero while
preserving provider usage and wall time.

## Base versus SFT

All tasks used the same fixtures and each policy ran each task once.

| Task | Base verifier / finish | SFT verifier / finish | Base steps/calls/tools | SFT steps/calls/tools | Base in/out | SFT in/out | Base/SFT seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| root-cause-cache | PASS / yes | PASS / no | 9/9/9 | 9/9/8 | 18,334/1,408 | 17,700/985 | 37.287/26.788 |
| datetime-api | PASS / no | PASS / yes | 9/9/8 | 11/11/11 | 17,565/1,296 | 24,496/1,445 | 33.837/33.757 |
| retry-boundary | PASS / no | PASS / yes | 6/6/5 | 6/6/6 | 11,832/1,192 | 11,760/1,000 | 28.711/21.964 |
| cross-file-config | PASS / no | PASS / yes | 12/12/12 | 9/9/11 | 23,400/1,156 | 17,200/1,030 | 33.671/24.274 |

Verifier remained 4/4 and correct explicit termination improved from 1/4 to
3/4. The SFT total was 71,156 input and 4,460 output tokens, 75,616 total,
with 106.783 seconds total wall time and zero API cost.

Replay and explain succeeded for all four SFT trajectories. No trajectory hit a
`repeated_tool_error`, `no_progress`, or malformed-tool-call terminal condition.
`datetime-api` had non-identical shell/cwd errors before recovery;
`cross-file-config` had redundant reads. The SFT evaluation Dataset classified
`datetime-api` and `retry-boundary` as `sft_candidate`, `cross-file-config` as
`successful_but_inefficient` for redundant tool calls, and `root-cause-cache` as
`successful_but_inefficient` for terminal state `failed`. There were no model or
infrastructure failures by the existing classifier.

The observed protocol improvement is real but incomplete: SFT preserved solving
ability and raised correct `finish` use by two tasks, while the Base policy's one
previously correct termination moved to a different task set and one SFT task
still ended with `model_final_message`.
