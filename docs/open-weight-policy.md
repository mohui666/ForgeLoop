# Open-weight Policy Stage A

ForgeLoop's first trainable deployment target is
`Qwen/Qwen3.5-9B`, pinned by the checked-in
bundled `forgeloop.policy.v1` manifest (`qwen3.5-9b`).
It is a base policy: no ForgeLoop SFT or RL has been applied yet.

## Why this model

Qwen3.5-9B is an Apache-2.0, dense 9B-class model released in 2026. The
official model card describes coding and agent capabilities, OpenAI-compatible
tool calling, a native 262,144-token context, reasoning output, and support in
Transformers, vLLM, and SGLang. It is materially newer than Qwen2.5-Coder while
remaining much easier to tune than Qwen3-Coder-Next (80B total) or the 30B/35B
MoE alternatives.

ForgeLoop deliberately serves the text-only language model with a 131,072-token
limit and a 32,768-token output cap. The smaller serving limit is an explicit
deployment profile, not a claim about the native model limit. The manifest is
the source of truth for the deployed capability.

## Hardware envelope

The numbers below are planning estimates; sequence length, batch size, optimizer,
attention implementation, and checkpointing change them substantially.

| Workload | Practical GPU target |
| --- | --- |
| BF16 vLLM, 131K profile | 48GB GPU (L40S/A40) recommended |
| BF16 vLLM, reduced 32K profile | 24GB may fit with little concurrency |
| 4-bit inference, reduced context | 12–16GB; not the canonical rollout profile |
| QLoRA SFT, 4K–8K sequences | 24GB minimum, 48GB recommended |
| BF16 LoRA SFT / longer sequences | 48–80GB |
| Full-parameter Adam SFT | multi-GPU; outside the intended first experiment |
| Later LoRA GRPO/RLVR | 80GB or multiple GPUs is the realistic starting point |

The current development machine has an RTX 4060 Laptop GPU with 8GB VRAM. It
cannot run the canonical BF16 vLLM profile, and this checkout currently has no
Docker CLI. Stage A therefore validates the provider/tool protocol with a local
test server boundary, but does not label those tests as a real Qwen rollout.

## Start vLLM

Use a Linux CUDA host. Install a Qwen3.5-capable vLLM build, then launch the
pinned weights:

```bash
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name forgeloop-qwen35-9b \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --dtype bfloat16
```

If the endpoint is protected, place its credential only in the process
environment:

```powershell
$env:FORGELOOP_SELF_HOSTED_API_KEY = "..."
```

An unprotected local vLLM endpoint needs no credential; ForgeLoop supplies the
non-secret placeholder `EMPTY` to the OpenAI-compatible client.

## ForgeLoop usage

Run one normal task through the existing LiteLLM `openai/` route:

```powershell
uv run forgeloop task "Fix the failing test and verify the change" `
  --workspace C:\path\to\repo `
  --policy-manifest qwen3.5-9b `
  --api-base http://GPU_HOST:8000/v1
```

Run three existing real-swe tasks once each, without creating a new benchmark:

```powershell
uv run forgeloop eval `
  --suite real-swe `
  --stage c `
  --task more-itertools-empty-interleave `
  --task pygments-raw-token-error-color `
  --task click-optional-metavar-brackets `
  --live `
  --runtime docker `
  --repeats 1 `
  --policy-manifest qwen3.5-9b `
  --api-base http://GPU_HOST:8000/v1 `
  --output-dir .forgeloop/eval-runs/qwen35-base
```

The run is intentionally not retried to improve a score. PASS and FAIL results
both become trajectory evidence and can be consumed normally:

```powershell
uv run forgeloop dataset build `
  --source .forgeloop/eval-runs/qwen35-base `
  --output .forgeloop/dataset/qwen35-base
uv run forgeloop dataset inspect --dataset .forgeloop/dataset/qwen35-base
```

## Identity and adapter boundary

The manifest records:

- policy stage (`base`, with `sft` and `rl` reserved for later checkpoints);
- base model and immutable model revision;
- tokenizer and immutable tokenizer revision;
- LiteLLM route and inference backend;
- non-secret serving configuration;
- generation configuration;
- deployed capabilities and their evidence sources.

`LiteLLMProvider` applies the generation profile and consumes the standard
OpenAI-compatible tool-call response. `AgentLoop` remains responsible only for
the same provider-neutral tool loop as before. Qwen3.5 reasoning is parsed by
vLLM for accounting but omitted from subsequent message history, matching the
model card's multi-turn guidance.

The identity is stored in `run_started`, every eval task record and summary, the
internal dataset sample, and SFT metadata. Legacy trajectories remain readable
and are marked `legacy_model_only`; ForgeLoop does not invent missing revisions.

Policy manifests reject credential-shaped keys. API keys stay in the process
environment and the existing trajectory/dataset sanitizers continue to redact
provider credentials and local paths.

## Live rollout gate

Before calling a rollout complete, verify all of the following on the GPU host:

1. `read_file`, `search_files`, `apply_patch`, `shell`, `git_diff`, test commands,
   and `finish` arrive as structured tool calls.
2. The real-swe Docker verifier runs without Harness/Environment failures.
3. The trajectory includes tool observations, Effect Events, token/latency data,
   terminal state, verifier result, and the pinned policy identity.
4. Dataset build preserves the same model and tokenizer revisions.

Do not substitute a hosted API model for this gate.

## Primary references

- [Qwen3.5-9B official model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [vLLM automatic tool-calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM Qwen3 reasoning parser](https://docs.vllm.ai/en/stable/api/vllm/reasoning/qwen3_reasoning_parser/)
