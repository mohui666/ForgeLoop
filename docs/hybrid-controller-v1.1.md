# Hybrid Controller v1.1

Hybrid Controller v1.1 keeps DeepSeek V4-Flash as the only coding model and
adds a local finite-state classifier around the existing deterministic
Controller v1. It does not change `AgentLoop`'s tool protocol, DeepSWE's patch
collector/verifier, or any frozen task.

The bundled main policy is `deepseek-v4-flash-hybrid-controller-v1.1`:

- coding policy: `deepseek/deepseek-v4-flash`, thinking level `max`;
- controller policy: `qwen2.5:1.5b-instruct` through Ollama's native
  `http://127.0.0.1:11434/api/chat` structured-output route;
- hard constraints: the unchanged deterministic Controller v1 plus the normal
  AgentLoop budget, timeout, provider/tool-error, Git-progress, and terminal
  boundaries.

No planner, critic, reflection pass, SFT/RL, multi-agent flow, or free-form
supervisor prompt is involved.

## Local model and probe

Install Ollama, then pull and probe the pinned controller artifact:

```powershell
ollama pull qwen2.5:1.5b-instruct
uv run forgeloop controller probe
```

The selected official Ollama artifact is Qwen2.5 1.54B, Q4_K_M, approximately
986 MB, model ID `65ec06548149`, with backing blob
`sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4`.
It is small enough to remain resident on the 8 GB RTX 4060 alongside normal
desktop workloads. The controller pins `num_ctx=2048`, `num_predict=64`,
temperature 0, a 15-second request timeout, and a 30-minute Ollama keep-alive.
Its local API cost is recorded as zero while token counts and latency remain in
the trajectory.

Qwen2.5 1.5B was retained because its five-state structured probe succeeded
with a smaller 986 MB Q4_K_M artifact. A local Llama 3.2 1B candidate was also
checked during selection, but its constrained output collapsed all probe cases
to `finalize/finalize`; it was rejected and removed rather than retained as a
second Controller policy.

On 2026-08-09 the real five-case probe returned the exact sequence:

```text
needs_inspection   -> explore/inspect
inspected_no_diff  -> implement/edit
modified_untested  -> verify/test
tests_failed       -> verify/replan
tests_passed       -> finalize/finalize
```

The aggregate local usage was 816 input tokens, 92 output tokens, 1.904 seconds,
and $0. Ollama reported 100% GPU placement with a 2,048-token context; total
system GPU memory in use during the probe was about 3.0 GB of 8,188 MiB.

## Controlled decision boundary

After each executed tool, ForgeLoop derives a compact progress signal from:

- the last four tool categories and success/change booleans;
- whether a source diff exists;
- focused test status (`unknown`, `pass`, or `fail`);
- remaining step, model-call, tool-call, time, and token budgets.

No task text, repository path, source text, tool arguments, raw observation, or
full conversation is sent to the controller model. The model can return only:

```text
state = explore | implement | verify | finalize
next_action = inspect | edit | test | replan | finalize
```

Ollama receives a JSON Schema whose `oneOf` branches permit only the five valid
state/action pairs. ForgeLoop then performs a second semantic check against the
compact progress signal. HTTP failures, malformed JSON, schema failures, and
semantic mismatches produce `controller_policy_fallback`; they never abort the
Agent and never replace deterministic Controller v1.

The model output itself is never forwarded to V4-Flash. ForgeLoop maps a valid
enum pair to one of five fixed guidance strings. Only stage changes inject
guidance, avoiding prompt spam. During `verify` and `finalize`, broad
`list_files` and `search_files` actions are rejected; targeted reads, edits,
tests, Git diff/commit commands, and `finish` remain available. All classifier
decisions, fallbacks, local usage, transitions, and fixed recoveries are normal
trajectory events.

## Running V4-Flash with Hybrid v1.1

```powershell
$env:DEEPSEEK_API_KEY = "..."
uv run forgeloop policy probe `
  --policy deepseek-v4-flash-hybrid-controller-v1.1
uv run forgeloop task "Fix the failing test" `
  --policy-manifest deepseek-v4-flash-hybrid-controller-v1.1
```

The interactive app selects this Hybrid policy automatically for the canonical
`deepseek/deepseek-v4-flash` route. The historical deterministic-only
`deepseek-v4-flash-controller-v1` manifest remains bundled for provenance and
comparison.

## Real validation on 2026-08-09

### Frozen `query-encoding`

The final unchanged LocalRuntime attempt passed all 3 verifier tests and ended
`completed/model_finish_tool`. V4-Flash used 6 model calls, 8 tool calls, 13,863
input tokens, 1,019 output tokens, and 20.34 seconds; provider cost was
$0.00064560. The local controller made 7 decisions using 1,156 input tokens,
126 output tokens, 3.029 seconds, and $0, with no fallback.

The recorded transition path was:

```text
explore/inspect -> implement/edit -> verify/test -> finalize/finalize
```

An earlier implementation check exposed two Controller-policy integration
problems: independent enum constraints permitted contradictory pairs, and the
previous state could anchor the 1.5B classifier after the source changed. The
final implementation constrains pair branches, omits that anchor from the
model input, and treats a progress/decision mismatch as a safe fallback.

### DeepSWE `oxvg-structural-selector-preservation`

One final attempt ran through the pinned DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, official Pier 0.3.0 Docker
environment, unchanged patch collector, and official verifier. It failed
honestly:

- terminal: `failed/controller_no_change_final`;
- Agent: 9 model calls, 11 tool calls, 125,383 input tokens, 23,276 output
  tokens, 366.348 seconds, $0.01010943;
- patch collector: `no_patch_collected`;
- verifier: P2P 62/62, F2P 0/6, partial 0.9117647058823529, binary reward 0;
- attribution: `model_failure`, not runtime, provider, or environment failure.

Hybrid Controller participated: its first shell observation retained
`explore/inspect`, the next successful listing moved to `implement/edit`, and
the fixed implement guidance was returned to V4-Flash. V4-Flash nevertheless
continued inspecting and never edited. The deterministic layer then triggered
`no_progress_reinspect`, `no_progress_action_required`, three
`exploration_action_blocked` recoveries, and `exploration_without_change` before
the explicit terminal failure. The controller made 8 valid local decisions,
used 1,327 input and 156 output tokens over 4.244 seconds, and had zero fallback.

This run improves attribution and pushes the Agent into an explicit implement
phase earlier, but it does **not** demonstrate a solved DeepSWE task or a
committed patch. Hybrid v1.1 cannot force V4-Flash to invent a correct edit;
the remaining failure is the coding model ignoring controlled edit guidance on
this hard repository task.
