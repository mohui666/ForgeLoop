# Local open-weight policy

ForgeLoop has one active open-weight policy: `qwen3.5-4b-local`. It runs the
Ollama `qwen3.5:4b` artifact through Ollama's OpenAI-compatible endpoint and the
existing LiteLLM/`ModelProvider` boundary. It is a base policy: ForgeLoop has not
applied SFT or RL.

The previous `qwen3.5-9b`/vLLM manifest remains checked in only so historical
policy provenance can still be interpreted. It is not a bundled active policy
and is not part of the current rollout path.

## Pinned local deployment

The bundled `forgeloop.policy.v1` manifest pins the deployment observed from
Ollama:

| Field | Value |
| --- | --- |
| Policy id | `qwen3.5-4b-local` |
| Ollama model | `qwen3.5:4b` |
| Ollama digest | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` |
| Parameters / quantization | 4.7B / Q4_K_M |
| Native context | 262,144 tokens |
| ForgeLoop serving profile | 8,192 context / 2,048 max output |
| API base | `http://127.0.0.1:11434/v1` |
| Local API cost | $0.00 |

The smaller serving context is an explicit 8GB-GPU profile, not a claim about
the model's native limit. Provider-reported tokens and measured request latency
remain in trajectories and eval records even though API cost is zero.

## Install and start Ollama

On Windows:

```powershell
winget install --id Ollama.Ollama --exact `
  --accept-package-agreements --accept-source-agreements
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "8192", "User")
ollama pull qwen3.5:4b
Invoke-RestMethod http://127.0.0.1:11434/api/version
ollama list
```

The desktop installer normally starts Ollama automatically. Restart Ollama after
setting `OLLAMA_CONTEXT_LENGTH`, then confirm `ollama ps` reports `CONTEXT 8192`
during a request. If the version endpoint is unavailable, start `ollama serve`
as a background process and retry the endpoint before running ForgeLoop.

## ForgeLoop usage

Run a normal bounded coding task with the manifest defaults:

```powershell
uv run forgeloop task "Fix the failing test and verify the change" `
  --workspace C:\path\to\repo `
  --policy-manifest qwen3.5-4b-local
```

The manifest route is `openai/qwen3.5:4b`; LiteLLM sends it to Ollama's
OpenAI-compatible `/v1/chat/completions` API. The unprotected local endpoint uses
the non-secret placeholder key `EMPTY`. A protected non-local compatible
endpoint can instead read `FORGELOOP_SELF_HOSTED_API_KEY` from the process
environment; credentials never belong in policy manifests.

Run one deterministic, verifier-backed safe coding task locally:

```powershell
uv run forgeloop eval `
  --stage a `
  --live `
  --runtime local `
  --repeats 1 `
  --policy-manifest qwen3.5-4b-local `
  --output-dir .forgeloop/eval-runs/qwen35-4b-local
```

`LocalRuntime` is the supported fallback when Docker is unavailable. It uses the
same `AgentLoop`, tools, verifier, trajectory, Effect Event, replay/explain, and
dataset paths as the Docker runtime.

Inspect the resulting evidence:

```powershell
uv run forgeloop trace replay .forgeloop/eval-runs/qwen35-4b-local/<run>/trajectories/<trajectory>.jsonl
uv run forgeloop trace explain .forgeloop/eval-runs/qwen35-4b-local/<run>/trajectories/<trajectory>.jsonl
uv run forgeloop dataset build `
  --source .forgeloop/eval-runs/qwen35-4b-local `
  --output .forgeloop/dataset/qwen35-4b-local
uv run forgeloop dataset inspect --dataset .forgeloop/dataset/qwen35-4b-local
```

## Identity and adapter boundary

The manifest records the base model and artifact digest, tokenizer artifact,
Ollama backend, LiteLLM route, non-secret serving profile, generation profile,
and deployed capabilities. The identity is copied into `run_started`, eval task
records and summaries, dataset samples, and future framework-neutral exports.

`LiteLLMProvider` applies the serving/generation profile, normalizes standard
OpenAI-compatible tool calls, records real usage and latency, applies the
explicit zero-cost local policy, and bypasses Windows environment/system proxies
only for the loopback endpoint. `AgentLoop` stays provider-neutral and is not
special-cased for Ollama or Qwen.

Historical trajectories remain readable. Records that predate policy identity
continue to be marked `legacy_model_only`; ForgeLoop does not fabricate missing
revisions.

## Live rollout gate

Do not call the local policy validated until all of the following hold:

1. The exact digest above is installed and Ollama reports the model loaded.
2. `ollama ps` reports the intended GPU/CPU split during a real request.
3. A real coding task emits structured tool calls and a verifier result.
4. Its trajectory contains observations, Effect Events, usage, latency, terminal
   state, final diff, and the pinned policy identity.
5. Replay and explain can consume the trajectory.
6. Dataset build preserves the policy identity and produces an explicit
   classification.

Do not substitute a hosted model or a mocked tool loop for this gate.
