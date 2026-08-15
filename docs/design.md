# Phase-one design assessment

## What was reused and what was learned

- [LiteLLM](https://github.com/BerriAI/litellm) is reused as the provider compatibility layer. ForgeLoop converts its responses immediately into internal dataclasses, so LiteLLM can later be replaced without changing the loop or tools.
- [mini-SWE-agent](https://mini-swe-agent.com/latest/) demonstrates that a linear loop, independent command execution, and a full trajectory are strong baselines. ForgeLoop keeps those properties but uses explicit native tools instead of bash-only interaction.
- [SWE-agent](https://swe-agent.com/0.7/background/architecture/) and its Agent-Computer Interface support a small, deliberate set of repository tools. ForgeLoop starts with five tools rather than a general plugin framework.
- [OpenHands](https://docs.openhands.dev/sdk/arch/agent) validates Action → Observation and a replaceable runtime boundary. Its stateless event architecture, Docker service, event bus, and security services are valuable at larger scale but excessive for phase one.
- [Aider](https://aider.chat/docs/git.html) validates diff-first review and careful Git handling. ForgeLoop exposes status/diff but deliberately avoids automatic commits and repository maps in this phase.

ForgeLoop is not a wrapper around any one agent. It owns the state machine, protocol, boundaries, records, and stopping rules while delegating only provider protocol normalization to LiteLLM.

## Components

```text
Textual TUI / Headless CLI (Session / Goal / Task)
        |
PresentationController -- ProviderConfig / ModelCache
        |                         |
    AgentLoop -------- BudgetState + Controller / ContextBudget
      |   |  \-------- RunDelivery (optional terminal boundary)
      |   |   +------- TrajectoryStore (append-only JSONL)
      |   +----------- ToolRegistry
      |                   |-- workspace file tools
      |                   |-- shell -> Runtime
      |                   `-- git diff
      `--------------- ModelProvider
                          `-- LiteLLMProvider (phase one)
```

The loop is intentionally synchronous and linear:

1. Check wall-clock, step, call, and between-response token budgets.
2. Ask the model for the next action using native tool schemas.
3. Normalize and record the model response.
4. Execute tool calls and append observations to both context and trajectory.
5. Continue until `finish`, a controller terminal, an unrecoverable error, or a
   budget guard stops the run.
6. Invoke the configured `RunDelivery` boundary. DeepSWE uses this boundary to
   create and verify the real base-to-HEAD commit consumed by its collector.

See [Execution Closure v3](execution-closure-v3-2026-08-15.md) for the current
validation, worktree-review, explicit-finalization, and patch-delivery state
machine. [Closure v2](execution-closure-v2.md) remains documented for historical
policy replay.

The Textual TUI is a presentation/controller layer around the same loop. Global
configuration owns non-secret Provider/API endpoints and defaults; secrets stay in
the OS credential store. A Session owns its repository, selected Provider/Model,
thinking level, conversation context, Plan/Build mode, runtime, checkpoints, usage
totals, and trajectory references. It does not introduce another agent state machine.

Synchronous AgentLoop turns run in a Textual thread Worker. A small event observer
reports model/tool lifecycle states to the presentation layer; it does not alter
tool execution or trajectory semantics. Tool output widgets show one-line state
and error summaries with stdout/stderr in a collapsed detail region. Modal screens
are transient viewers/selectors, not persistent IDE panes.

Provider preflight is outside AgentLoop. It combines the selected Provider and a
bare Model into a canonical LiteLLM route, validates required credentials and API
base configuration, and rejects mismatches before a run/checkpoint begins. The API
probe forces a nonce-bearing tool call so a successful result demonstrates auth,
model access, and tool-calling capability.

`ModelCache` keys records by Provider + normalized Base URL + Model. Refresh uses
the Provider model-list endpoint and only replaces the route cache after a valid
response, so a network failure leaves the old cache usable. `CapabilityResolver`
merges each field in this order: live Provider metadata, local capability cache,
small explicitly verified ForgeLoop registry, then unknown. Unknown context/tool/
thinking/streaming values stay unknown.

`ContextBudget` is derived for the current Session model from context window minus
reserved output, ForgeLoop thinking/tool reserve policy, and safety margin. Auto
compact uses a percentage of the resulting usable context rather than a global
128K-style constant. Model switching performs this check and compaction before any
request reaches the Provider.

## Mode boundary

Goal and Task modes use the same mechanics. Only policy differs:

- Goal Mode may decompose and revise a plan to reach the requested outcome.
- Task Mode must stay within the explicit engineering task and avoid unrelated improvements.

Keeping mode policy in prompts avoids duplicated state machines and makes mode behavior easy to evaluate later.

Interactive Plan mode additionally removes mutating and Shell tools from the
registry, making the read-only boundary structural rather than prompt-only.
Interactive Build mode retains the existing tools and creates a Git checkpoint
before the turn. Repeated identical calls, repeated identical failures, and
mutation steps with no Git-visible progress terminate with an explicit blocked
result before the normal budgets are exhausted.

## Extension seams, not phase-one features

- `Runtime` can be implemented by Docker without changing tool or loop semantics.
- `ModelProvider` can gain direct provider adapters or an inference gateway.
- JSONL action/observation events can be transformed into evaluation, SFT, or RL datasets.
- `TrajectoryStore` can be replaced by an event sink for batch evaluation.
- Tool schemas and terminal status are explicit enough for deterministic replay and scoring.

No abstractions for multi-agent scheduling, RAG, dashboards, Kubernetes, or Environment Foundry are introduced until a real second consumer exists.
