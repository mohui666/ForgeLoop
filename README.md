# ForgeLoop

English | [简体中文](README.zh-CN.md)

ForgeLoop is infrastructure for running, evaluating, and iterating on coding agents. It includes a CLI agent loop, provider-neutral model boundary, Docker runtime, trajectory recording, deterministic evaluation, a small real-repository SWE Environment Foundry, and an interactive Textual/Rich TUI. LiteLLM provides the replaceable multi-provider adapter.

The project is intentionally small and currently validated on a fixed 30-task smoke suite plus an eight-task real-repository suite. It is a practical foundation for continued evaluation and training-data work, not a claim of benchmark leadership or production-grade sandbox isolation.

This phase intentionally does **not** include a web UI, general-purpose sandbox platform, container pool, multi-agent orchestration, RAG, an inference server implementation, large benchmark infrastructure, SFT/RL, dashboards, or distributed execution. It includes one fixed smoke eval, a narrowly scoped per-task Docker runtime, a small curated Environment Foundry path, and an adapter for an externally served open-weight base policy.

## Quick start

Python 3.10+ and `uv` are recommended.

```powershell
uv sync --extra dev
uv run forgeloop
```

The no-argument command opens the conversational TUI. It intentionally does not
show a repository list, recent-Session dashboard, or file tree at startup. Type plain text to talk
to the coding Agent and `/help` for ForgeLoop controls. The first conversation
creates a Session and binds the current directory when it is a Git repository;
otherwise ForgeLoop asks for a project path.

界面只保留欢迎信息、连续对话、折叠 Tool Call、输入框和一行轻状态；没有 Dashboard、侧栏、文件树或常驻详情面板。`Ctrl+P` 打开命令面板，`Ctrl+C` 请求安全中断当前 Agent，`Esc` 关闭临时面板，`Ctrl+D` 保存退出。输入框支持上下键历史和 Tab 命令补全。

在 TUI 中使用以下流程配置：

```text
/api
/model
/thinking
```

`/api` 统一管理 Provider、API Key、Base URL、Connection Test 和删除配置；
API Key stores through the operating-system credential store.
It is never written to config, Session, context, log, or trajectory files. The
usual provider environment variables remain supported.

`/model` 只显示配置完整且可用的 Provider。进入 Provider 后优先显示按
Provider + Base URL 隔离的本地缓存，可以 Refresh Models 调用真实 Provider
模型接口；刷新失败时旧缓存仍可用，也可手动输入真实 Model ID。ForgeLoop
内部生成 LiteLLM canonical route，但普通用户只看到 Provider 与 Model ID。
Preflight 在 AgentLoop 前阻止缺失或不匹配的路由、凭据与 endpoint。
Connection Test performs a real forced tool call, verifying authentication,
model access, and tool-calling support together.

`/thinking` 根据当前模型 capability 只显示真实支持的 Auto / Low / Medium /
High / Max 子集。切换模型后会重新解析能力；不兼容的 Session thinking 设置
回退到 Auto。Model capability 按 Provider metadata、本地缓存、ForgeLoop
已验证 registry、unknown 的顺序解析，未知限制不会被伪造。

`/context` 显示当前模型 context limit、usable context、当前估算用量、reserved
output、thinking/tool reserve、自动压缩阈值与压缩次数。阈值跟随模型动态变化；
切换到较小上下文模型前会先压缩，仍无法安全容纳时会阻止切换。

Useful interactive controls include `/plan`, `/build`, `/status`, `/diff`,
`/undo`, `/test`, `/lint`, `/build run`, `/context`, `/compact`, `/cost`,
`/sessions`, `/resume`, and `/new`. Plan mode is read-only. Build mode creates a
Git-object checkpoint before each Agent turn so `/undo` can restore both the
pre-existing worktree and index state.

`/sessions`, `/diff`, `/status`, `/context`, `/cost`, `/api`, `/model`, and
`/help` use temporary TUI panels. Long tool stdout/stderr is collapsed by default;
failed tools keep a concise error summary visible. `/details` exposes the most
recent redacted provider diagnostic without dumping a LiteLLM traceback into the
conversation.

Global non-secret Provider/API defaults and budgets live in
`~/.forgeloop/config.json`; model metadata lives in `~/.forgeloop/model_cache.json`.
Repository path, conversation, diff/checkpoint state,
context, selected Provider/Model, thinking, mode, runtime, usage, and trajectory
references are stored per Session. Model cache data is non-secret and isolated by
Provider + Base URL + Model.

Project-level `FORGELOOP.md` and `AGENTS.md` files are loaded into each Agent turn.
File tools stay inside the selected repository, sensitive credential files are
blocked, and host Shell calls reject known destructive and credential-exposing
patterns. Token, cost, step, tool-call, model-call, and timeout budgets are
enforced by the existing `AgentLoop` budget boundary.

The original automation entries are unchanged:

```powershell
$env:OPENAI_API_KEY = "..."
uv run forgeloop task "修复 AuthService 中的空指针错误" --model openai/gpt-4.1
```

Goal Mode accepts an outcome and lets the agent decompose it:

```powershell
uv run forgeloop goal "让这个项目的所有测试通过" --model anthropic/claude-sonnet-4-5
```

Task Mode stays within a bounded engineering request:

```powershell
uv run forgeloop task "修复 AuthService 中的空指针错误" --model openai/gpt-4.1
```

Provider credentials follow LiteLLM's standard environment variables. `--api-base` supports OpenAI-compatible endpoints, while `--model` (or `FORGELOOP_MODEL`) chooses the provider/model route.

## Open-weight base policy

The only active open-weight policy is `qwen3.5-4b-local`: Ollama's
`qwen3.5:4b`, served at `http://127.0.0.1:11434/v1` through the existing
LiteLLM/OpenAI-compatible provider boundary. The checked-in manifest pins the
Ollama digest, tokenizer artifact, backend, serving profile, generation profile,
and model capabilities:

```powershell
uv run forgeloop task "Fix the failing test" `
  --policy-manifest qwen3.5-4b-local
```

The prior 9B/vLLM manifest is retained only for historical provenance
compatibility. The first independently deployable SFT policy is bundled as
`qwen3.5-4b-sft-v1`; `qwen3.5-4b-local` remains the active Base policy. See
[docs/open-weight-policy.md](docs/open-weight-policy.md) for Base setup and
[docs/qwen3.5-4b-sft-v1.md](docs/qwen3.5-4b-sft-v1.md) for the first complete
Dataset -> Train -> Redeploy -> Evaluate run.

## Budgets and records

Every run enforces step, model-call, tool-call, wall-clock, and reported-token limits. An optional cost limit is available through `--max-cost-usd`. Run `uv run forgeloop goal --help` for all options.

Append-only JSONL trajectories from automation runs are written under `.forgeloop/runs/` by default. Interactive trajectories are grouped by Session under `~/.forgeloop/trajectories/`. They contain normalized model requests/responses, tool calls, observations, budget snapshots, and the terminal result. API keys are never placed in run configuration or trajectory events.

Verifier-backed eval trajectories can be converted into traceable, classified
training records and a framework-neutral SFT conversation JSONL:

```powershell
uv run forgeloop dataset build
uv run forgeloop dataset inspect
uv run forgeloop dataset export
```

The default SFT export contains only verified, efficient candidates and excludes
all infrastructure failures. See [docs/dataset.md](docs/dataset.md) for the
schema, classifications, provenance, filtering, and sanitization guarantees.

Tool-side file, shell, Git, test, and safety effects are recorded as structured
trajectory evidence. Replay and deterministic explanation are offline CLI
operations and never re-execute the original side effects:

```powershell
uv run forgeloop trace replay <trajectory-id-or-path>
uv run forgeloop trace explain <trajectory-id-or-path>
```

See [docs/observability.md](docs/observability.md) for the Effect Event schema,
TraceSeal design lineage, evidence safety, analysis rules, and compatibility.

## Local execution warning

Local runtime Shell commands run directly on the host in an independent PowerShell or `/bin/sh` process. File tools cannot escape the selected workspace. ForgeLoop strips credential-like environment variables from child processes and blocks known dangerous command shapes, but this is a guardrail rather than an OS sandbox. Use ForgeLoop only in a trusted workspace and review changes with `/diff`.

Agent and workflow execution runs in Textual thread workers so the terminal stays
responsive. Ctrl+C is cooperative: it records the interrupt immediately and ends
the turn after the currently active synchronous provider or tool call returns.
The Session, checkpoint, trajectory, and modifications already made remain usable.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run forgeloop --help
```

See [docs/design.md](docs/design.md) for the architecture assessment, reference projects, and extension boundaries.

## Fixed smoke eval

The bundled `python-smoke-v1` suite contains 30 verifier-driven tasks across easy, medium, and hard tiers. Eval is dry-run by default and uses staged execution to prevent accidental model spend. Live runs use three independent attempts per task by default; use `--repeats 2` for two:

```powershell
uv run forgeloop eval --stage a

# Only when ready for one real canary task:
$env:AGENT_TEMP_KEY = [Environment]::GetEnvironmentVariable("agent_temp_key", "Machine")
uv run forgeloop eval --stage a --live --repeats 3
Remove-Item Env:AGENT_TEMP_KEY
```

Run the same canary in one disposable Docker container per task with:

```powershell
$env:AGENT_TEMP_KEY = [Environment]::GetEnvironmentVariable("agent_temp_key", "Machine")
uv run forgeloop eval --stage a --live --runtime docker --repeats 3
Remove-Item Env:AGENT_TEMP_KEY
```

Stage `a` selects one canary, `b` selects three additional task types, and `c` runs the complete suite. See [docs/eval.md](docs/eval.md) for task definitions, reproducibility guarantees, result fields, and DeepSeek V4 Flash configuration.

## Real-repository Foundry

Build the curated eight-task Stage B suite from fixed public Git commits, including two independent Docker FAIL-to-PASS validations per task:

```powershell
uv run forgeloop foundry build
uv run forgeloop eval --suite real-swe --runtime docker
uv run forgeloop eval --suite real-swe --runtime docker --live --repeats 2
```

The build refuses to publish a partial suite. Generated fixtures and hidden gold patches live under `.forgeloop/foundry/real-swe`; only fixture contents are copied into Agent workspaces. The default command performs no model calls; add `--live` only when an evaluated model run is intended. See [docs/foundry.md](docs/foundry.md) for task provenance, screening records, the trust boundary, and extension criteria.
