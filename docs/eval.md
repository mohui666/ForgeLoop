# Fixed real-model smoke eval

ForgeLoop ships one deliberately small suite: `python-smoke-v1`. It reuses the production `AgentLoop`, tools, budgets, workspace, runtime, and trajectory. The harness adds only fixture reset, objective verification, aggregation, and JSON/JSONL persistence.

## Tasks

| Difficulty | Tasks | Capabilities |
| --- | ---: | --- |
| easy | 10 | conditions, None/missing values, validation, datetime API, off-by-one, mutable defaults, normalization, dictionaries, safe arithmetic, status boundaries |
| medium | 14 | parsers, cache root cause, multi-file API/config changes, regression tests, refactors, paths, sorting, URL encoding, retry/TTL, serialization, recursive structures |
| hard | 6 | missing-spec blocking, timezone semantics, atomic rollback, dependency ordering/cycles, non-aliasing deep merge, interval algorithms |

Twenty-nine tasks begin with a failing verifier. The blocked task begins with passing baseline tests and is considered correctly blocked only when the Agent returns `blocked`, the verifier still passes, and the Git diff remains empty. Exact task descriptions, tags, fixed SHAs, and verifier commands live in `src/forgeloop/eval_suite/smoke/tasks.json`.

## Stable schema

The suite is a single `forgeloop.eval.v2` JSON document. Each task has:

- `id`, `description`, `fixture`, and a fixed `base_commit`;
- `mode`, `timeout_seconds`, `expected_outcome`, `tags`, `stage`, and `difficulty`;
- one verifier `command` and verifier timeout.

No benchmark DSL or database is involved.

## Reset and isolation

For every task the runner copies immutable fixture bytes into a new isolated directory, initializes Git with fixed identity/date and `core.autocrlf=false`, creates the baseline commit, and rejects the task unless its SHA exactly matches the manifest. It also requires a clean initial status. Workspaces are separate, and are removed after diff/result capture unless `--keep-workspaces` is set.

A SHA mismatch or reset error is recorded as `environment_eval_failure`, never as model failure.

## Local and Docker execution

The same smoke task can use either runtime without changing `AgentLoop`:

```powershell
uv run forgeloop eval --stage a --live --runtime local
uv run forgeloop eval --stage a --live --runtime docker
```

`DockerRuntime` builds the fixed `forgeloop-eval:py312` image on first use. Each task starts one `--rm`, network-disabled container with its own freshly reset fixture mounted at `/workspace`. Read, search, patch, shell, Git, and verifier operations all cross the `Runtime` boundary. The container is forcibly removed in success and error paths before the next task starts; the disposable host staging directory is then removed by the existing Eval runner.

This phase intentionally provides no container pool, remote executor, Kubernetes integration, or general-purpose sandbox policy.

## Repeated attempts and metrics

The CLI runs three independent attempts per selected task by default. Use
`--repeats 1` for a one-shot engineering rollout or `--repeats 2` for two
attempts; 1 through 3 are accepted. Each attempt gets a fresh fixed-SHA
workspace and, with Docker, a fresh container. Repeatable `--task` options can
select a small named subset from an existing stage without creating or changing
benchmark tasks.

```powershell
uv run forgeloop eval --stage c --live --runtime docker --repeats 3
```

- `Pass@1`: fraction of unique tasks whose first attempt matches the expected outcome.
- `Pass@3`: fraction with at least one matching outcome in three attempts; it is `null`/`N/A` for two-attempt runs.
- A correctly blocked task contributes to Pass@k but not to `Solved`.
- `Cost per Solved` and `Tokens per Solved`: total consumption across every attempt divided by unique solvable tasks completed at least once.
- Failure categories remain attempt-level so intermittent model, harness, and environment failures are not hidden by a later successful retry.

The summary also contains the same metrics split by `easy`, `medium`, and `hard`.

## DeepSeek V4 Flash mapping

The real-model eval uses the official **DeepSeek V4 Flash** release:

- API model: `deepseek-v4-flash`
- LiteLLM route: `deepseek/deepseek-v4-flash`
- API base: `https://api.deepseek.com`
- request configuration: `reasoning_effort="max"` and `extra_body={"thinking": {"type": "enabled"}}`

DeepSeek V4 Flash supports thinking and non-thinking modes; this eval explicitly uses thinking mode at maximum effort. ForgeLoop omits temperature, preserves `reasoning_content` across tool-call turns, and uses the native OpenAI-style function tool schema.

The pinned LiteLLM registry contains an exact `deepseek/deepseek-v4-flash` entry. ForgeLoop uses LiteLLM's response cost calculation and records its source as `litellm_calculated`; it still returns `unknown` instead of guessing if exact pricing or usage is unavailable.

## Credential handling

The eval reads only process environment variable `AGENT_TEMP_KEY` (Windows names are case-insensitive). It passes the value directly as LiteLLM's `api_key`, never as a prompt or serialized configuration. A lightweight exact-value and Authorization-header redactor protects trajectories and infrastructure error messages.

If the variable was added at Machine scope after the terminal started, map it into the current process without printing it:

```powershell
$env:AGENT_TEMP_KEY = [Environment]::GetEnvironmentVariable("agent_temp_key", "Machine")
uv run forgeloop eval --stage a --live
Remove-Item Env:AGENT_TEMP_KEY
```

## Results

Each eval run writes:

- `tasks.jsonl`: one machine-readable result per task;
- `summary.json`: aggregate metrics plus embedded task records;
- `trajectories/*.jsonl`: normal Agent events followed by verifier and final-diff events.

For cache-stability checks, pass `--min-warm-cache-hit-rate 0.98`. The gate uses
the provider-reported, token-weighted warm reusable-prefix metric, excludes each
trajectory's cold start and legitimate compaction/prefix resets, records its
threshold and verdict in `summary.json`, and exits 3 on failure. It does not
change the selected tasks or any AgentLoop budget.

Task success is verifier-driven. Records include terminal and stop state, failure category, verifier output, Git SHA/dirty state, steps/calls, normalized usage, cost source, wall time, final status/diff, and trajectory path.

Unknown token or cost values remain `null`; aggregate totals become unknown if any included attempt is unknown. `Cost / Solved` and `Tokens / Solved` are `null` when solved is zero.
