# Qwen3.5-4B SFT v2

Date: 2026-08-09

## Conclusion

SFT v2 corrected the two known v1 recipe defects: loss is computed only on
assistant natural-language/tool-call tokens, and training uses the same eight
tool schemas and upstream Qwen chat template as Windows LocalRuntime inference.
It also caps each task at two trajectories. One training run and one v2 attempt
per frozen holdout task were performed; v1 and the holdout set remain unchanged.

The frozen holdout does **not** show an improvement over v1. Verifier PASS fell
from v1's 4/6 to 3/6, explicit `finish` remained 3/6, and verifier-PASS plus
`finish` fell from 3/6 to 2/6. V2 equals Base on verifier PASS (3/6) and on
correct `finish` (2/6). It therefore introduced meaningful new regressions and
does not meet the gate for RL.

> 闭环已完成，但泛化 improvement 仍未验证。

No AgentLoop, Dataset classifier, Verifier, task, prompt, benchmark, SFT v1, or
holdout definition was changed. No RL/GRPO was started, and neither training nor
evaluation was repeated to improve the result.

## Dataset and recipe

The source was the frozen strict `sft_candidate` export: 78 unique sample ids
and conversations, SHA-256
`1778099b89aba57810382df462351b6efed0d688067f00c8d9937f48af074612`.
The audit found no credential-pattern match. The final set contains **17 real
trajectories from 10 tasks**, all with verifier PASS and explicit
`finish(status=completed)`. There are 17 unique source trajectories. Every task
contributes at most two samples; within a task, the shortest eligible rendered
samples win, then sample id breaks ties.

The six frozen holdout task ids are explicitly excluded. After schema-aware
rendering, 43 samples exceeded 2,500 tokens and were excluded, one lacked an
explicit completed `finish`, and four were removed by the task cap. No sample
was silently truncated. Long tool observations are shortened to a 120-character
head and tail with an explicit omission marker; assistant targets and tool
schemas are never shortened. The selected lengths are 2,209 / 2,338 / 2,491
tokens (min/median/max), with 7,757 assistant target tokens, 32,405 masked
context tokens, and 90 boundary tokens excluded from loss.

Selected task ids are `condition-boundary`, `dict-missing`, `email-regression`,
`mutable-default`, `off-by-one`, `parameter-validation`, `path-suffix`,
`safe-ratio`, `stable-sort`, and `string-normalization`. Full sample ids,
repo/base SHA, and source trajectory ids are preserved in the ignored local
artifacts:

- `.forgeloop/training/qwen3.5-4b-sft-v2/dataset_audit.json`
- `.forgeloop/training/qwen3.5-4b-sft-v2/dataset_provenance.json`

### Assistant-only loss

`scripts/sft_data.py` renders the complete conversation with the pinned upstream
Qwen tokenizer's `apply_chat_template`. For each assistant message, it renders
the preceding conversation with `add_generation_prompt=True` to locate the
assistant start and renders the prefix including that message to locate its end.
Tokenizer offset mappings then set labels only for tokens wholly contained in
those assistant spans; all system, user, tool-result/observation, and other
context labels are `-100`. Tokens crossing a target boundary are also masked.

The implementation verifies that offset-tokenized ids exactly match
`apply_chat_template(..., tokenize=True)`. Because the whole serialized
assistant span is targeted, both natural language and Qwen's tool-call/function
JSON remain learnable. Regression tests verify that user/system/tool text is
masked while assistant text and tool calls remain labeled.

### Inference-matched tools

Training passes `tools=` directly to the upstream Qwen chat template; it does
not construct a synthetic prompt. The frozen schema snapshot contains the live
Windows LocalRuntime schemas for `read_file`, `search_files`, `apply_patch`,
`shell`, `list_files`, `git_diff`, and `git_inspect`, plus ForgeLoop's `finish`.
A Windows regression test reconstructs the inference tool set and requires
exact equality with `training/qwen3.5-tools-windows-local.json`.

The tracked recipe is `training/qwen3.5-4b-sft-v2.json`. The rendered schema
identity is
`55a888a66136a058eec9ef92ea6e462930de6278051a78ccd011f5e1a49e4131`.

## Training

The trainable source was `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, not an Ollama quantized artifact.
The single WSL2 QLoRA run used:

- NF4 4-bit base, double quantization, BF16 compute;
- LoRA rank 4, alpha 8, dropout 0.05 on q/k/v/o and gate/up/down projections;
- maximum sequence 2,500, micro-batch 1, gradient accumulation 8;
- two epochs, six optimizer steps, learning rate 2e-4, cosine schedule, warmup 0.1;
- paged AdamW 8-bit, max grad norm 0.3, gradient checkpointing, seed 42;
- Liger fused linear cross entropy with the assistant-only labels above.

There are 5,308,416 trainable parameters out of 4,211,059,712 (0.1261%). The
run took 526.622 seconds (8m46.6s). Logged losses were 0.41584, 0.42276,
0.40040, 0.25906, 0.28836, and 0.22904; mean train loss was 0.33591 and all
values were finite. PyTorch measured 6,949.38 MiB peak allocated and 7,148 MiB
peak reserved. `nvidia-smi` observed approximately 7,827-7,830 MiB in use and
100% GPU utilization during training.

Artifacts:

- Windows adapter mirror: `.forgeloop/training/qwen3.5-4b-sft-v2/adapter`
- WSL artifact root: `/home/mohui666/forgeloop-artifacts/qwen3.5-4b-sft-v2`
- adapter revision: `2f8f34073355bbd7eecc46576fe36adc9608b92f144a94abffc8dd0d68278561`
- adapter safetensors SHA-256: `6f3d80a140171114af7dd56d38d8ba36fa17bbfb63c0d3bd8027256052016418`
- deployed Q4_K_M GGUF SHA-256/model revision:
  `67116bcdf1c60649dc88cfe53439588a002debc8fd0f4437c13c5b9428858def`

## Redeployment

The adapter was merged into the same pinned upstream BF16 base, converted by
llama.cpp with `--no-nextn`, quantized once to Q4_K_M, and imported using the
local Modelfile (8192 context, temperature 0.2, top-p 0.95):

```powershell
ollama create qwen3.5-4b-sft-v2 -f .forgeloop/training/qwen3.5-4b-sft-v2/Modelfile
uv run forgeloop task "Fix the failing test" `
  --policy-manifest qwen3.5-4b-sft-v2 `
  --runtime local
```

Ollama 0.32.6 assigned model id `dcaf19b8ec99`. A real request through
`http://127.0.0.1:11434/v1` returned a valid structured
`finish({"summary":"probe-ok"})` tool call with 274 input and 60 output tokens.
`ollama ps` reported 8192 context and 100% GPU placement. The bundled policy
manifest records zero local API cost while ForgeLoop retains real usage and
latency.

## Frozen holdout comparison

Base and v1 values below are the frozen results from
`qwen3.5-4b-sft-v1-holdout.md`; they were not rerun. Each v2 task ran exactly
once with the same LocalRuntime, prompt, verifier, budgets, 8192/2048 limits,
temperature 0.2, top-p 0.95, and non-streaming generation. V2's
`duration-parser` usage is reconstructed from 11 successful model responses
because request 12 ended in a redacted `ModelProviderError`.

| Task | Policy | Verifier / explicit finish | Terminal / stop | Steps/model/tool | Input/output | Seconds | Dataset |
|---|---|---|---|---:|---:|---:|---|
| `duration-parser` | Base | FAIL / no | failed / final message | 9/9/8 | 35,797/4,318 | 87.723 | `model_failure` |
|  | v1 | PASS / no | failed / orchestration error | 12/12/11 | 48,955/4,033 | 99.844 | `successful_but_inefficient` |
|  | v2 | FAIL / no | failed / orchestration error | 12/12/11 | 36,623/3,542 | 96.201 | `infrastructure_failure` |
| `query-encoding` | Base | PASS / yes | completed / finish | 7/7/7 | 12,681/1,083 | 23.174 | `sft_candidate` |
|  | v1 | FAIL / no | blocked / repeated tool call | 5/5/5 | 8,038/352 | 9.709 | `model_failure` |
|  | v2 | PASS / no | blocked / repeated error | 11/11/11 | 29,646/2,688 | 51.781 | `successful_but_inefficient` |
| `atomic-reservation` | Base | PASS / no | completed / final message | 8/8/7 | 16,529/1,376 | 28.507 | `sft_candidate` |
|  | v1 | PASS / yes | completed / finish | 7/7/8 | 13,486/934 | 21.392 | `sft_candidate` |
|  | v2 | PASS / yes | completed / finish | 11/11/11 | 23,906/1,697 | 33.688 | `successful_but_inefficient` |
| `dependency-order` | Base | FAIL / no | failed / final message | 16/16/15 | 62,092/5,347 | 104.154 | `model_failure` |
|  | v1 | not reached / no | interrupted infrastructure failure | 5/5/5 | 9,240/833 | >780 | `infrastructure_failure` |
|  | v2 | FAIL / yes | completed / finish | 8/8/8 | 20,327/1,659 | 32.077 | `model_failure` |
| `deep-merge` | Base | FAIL / no | failed / final message | 13/13/12 | 48,020/3,682 | 79.645 | `model_failure` |
|  | v1 | PASS / yes | completed / finish | 12/12/12 | 39,025/3,050 | 65.177 | `successful_but_inefficient` |
|  | v2 | FAIL / no | failed / final message | 11/11/10 | 32,739/4,817 | 84.948 | `model_failure` |
| `interval-merge` | Base | PASS / yes | completed / finish | 8/8/8 | 16,747/1,312 | 28.575 | `sft_candidate` |
|  | v1 | PASS / yes | completed / finish | 10/10/10 | 23,570/1,829 | 36.956 | `sft_candidate` |
|  | v2 | PASS / yes | completed / finish | 11/11/11 | 31,040/2,085 | 44.102 | `successful_but_inefficient` |

| Aggregate | Base | v1 | v2 |
|---|---:|---:|---:|
| Verifier PASS | 3/6 | 4/6 (one verifier not reached) | 3/6 |
| Explicit `finish` | 2/6 | 3/6 | 3/6 |
| Verifier PASS + `finish` | 2/6 | 3/6 | 2/6 |
| Completed terminal | 3/6 | 3/6 | 3/6 |
| Dataset `sft_candidate` | 3 | 2 | 0 |
| Dataset successful but inefficient | 0 | 2 | 3 |
| Dataset model failure | 3 | 1 | 2 |
| Dataset infrastructure failure | 0 | 1 | 1 |

V2 recorded/reconstructed 174,281 input and 16,488 output tokens, 64 steps,
64 model calls, 62 tool calls, 342.797 seconds, and zero local API cost. All six
v2 trajectories passed offline replay/explain and produced 79 Effect Events.

No v2 trajectory contained a malformed tool call or hit `no_progress`.
`query-encoding` stopped on repeated failed patch attempts; `atomic-reservation`
repeated a safety-blocked shell listing; `interval-merge` made redundant calls.
These three otherwise verifier-PASS runs were consequently classified as
`successful_but_inefficient`. Tool calling remained structurally valid, but its
behavioral stability did not improve.

The main paired changes versus v1 are mixed and net negative: v2 recovers
`query-encoding`, obtains a real result for `dependency-order`, and preserves
`atomic-reservation`/`interval-merge`; it regresses `duration-parser` and
`deep-merge`, and makes all three successful runs inefficient. This is new
holdout degradation, so the evidence does not support moving to RL yet.
