# ForgeLoop

English | [简体中文](README.zh-CN.md)

ForgeLoop is a compact platform for building, running, and evaluating coding
agents on executable software-engineering tasks.

It provides:

- a provider-neutral `AgentLoop` with tool calling and explicit budgets;
- interactive TUI, task, and goal workflows over the same core;
- local and Docker runtimes with safety boundaries;
- trajectories, effect events, replay, and deterministic explanations;
- deterministic long-trajectory context compaction with per-call source metrics;
- verifier-backed internal and DeepSWE evaluation;
- traceable dataset export for future training loops.

## Quick start

ForgeLoop requires Python 3.10+. [`uv`](https://docs.astral.sh/uv/) is
recommended.

```powershell
uv sync --extra dev
uv run forgeloop
```

The default command opens the conversational TUI. Configure a provider with
`/api`, choose a model with `/model`, and use `/help` to list commands.

Common controls:

- `/plan` and `/build` switch between read-only planning and editing;
- `/status`, `/diff`, `/test`, and `/undo` inspect or validate work;
- `/context`, `/compact`, and `/cost` show runtime usage;
- `Ctrl+C` interrupts the active turn; `Ctrl+D` saves and exits.

API keys are stored in the operating-system credential store and are not
written to sessions, logs, or trajectories.

## CLI automation

Run a bounded task:

```powershell
$env:OPENAI_API_KEY = "..."
uv run forgeloop task "Fix the failing test" --model openai/gpt-4.1
```

Run an outcome-oriented goal:

```powershell
uv run forgeloop goal "Make all tests pass" --model anthropic/claude-sonnet-4-5
```

OpenAI-compatible endpoints are supported with `--api-base`. Use
`--policy-manifest` for a reproducible bundled or custom policy.

## Policies

The active local open-weight policy is `qwen3.5-4b-local`, served by Ollama at
`http://127.0.0.1:11434/v1`:

```powershell
uv run forgeloop task "Fix the failing test" `
  --policy-manifest qwen3.5-4b-local
```

The current V4-Flash Controller policy is
`deepseek-v4-flash-controller-v1.3-simplified`. Historical Base, SFT, and
Controller manifests remain available for provenance and comparison.

See [local policy setup](docs/open-weight-policy.md),
[Controller v1.3 Simplified](docs/hybrid-controller-v1.3-simplified.md), and
[Agent Context Efficiency v1](docs/agent-context-efficiency-v1.md).

## Evaluation

List or run the internal verifier-backed smoke suite:

```powershell
uv run forgeloop eval --stage a
uv run forgeloop eval --stage a --live --repeats 1
```

Check and run the frozen 20-task DeepSWE Eval v2 subset:

```powershell
uv sync --extra dev --extra deepswe
uv run forgeloop deepswe check
uv run forgeloop deepswe run --policy qwen3.5-4b-local
```

Live eval commands call paid or local model endpoints. Dry-run commands do not.
See [internal eval](docs/eval.md), [DeepSWE Eval v2](docs/deepswe-eval-v2.md),
and [Environment Foundry](docs/foundry.md).

## Trajectories and datasets

Runs record model usage, tool calls, observations, effects, Git state, verifier
results, and terminal reasons.

```powershell
uv run forgeloop trace replay <trajectory-id-or-path>
uv run forgeloop trace explain <trajectory-id-or-path>
uv run forgeloop dataset build
uv run forgeloop dataset inspect
uv run forgeloop dataset export
```

See [observability](docs/observability.md) and [dataset format](docs/dataset.md).

## Safety

Local runtime commands execute on the host and are not an OS sandbox. Use it
only in trusted repositories and review changes with `/diff`. Use Docker for
isolated or external tasks. File tools remain inside the selected workspace,
sensitive paths are blocked, and step, token, call, cost, and time budgets are
enforced.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

Architecture and extension boundaries are documented in
[docs/design.md](docs/design.md). ForgeLoop is licensed under MIT.
