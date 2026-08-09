# Hybrid Controller implement readiness v1

This change adds one deterministic readiness boundary before the existing Edit
Intent Handoff. The bundled policy is
`deepseek-v4-flash-edit-intent-readiness-v1`; the historical
`deepseek-v4-flash-edit-intent-v1` policy remains available unchanged.

V4-Flash remains the coding model, local `qwen2.5:1.5b-instruct` remains only a
state classifier, and deterministic Controller v1 retains all hard limits.
There is no new model, prompt planner, SFT/RL, stronger implement tool gate, or
DeepSWE verifier/collector change.

## Readiness rule

An `explore -> implement` classifier decision is actionable only when both are
true:

1. a successful `read_file` observation contains non-empty source content for a
   recognized source-code path; and
2. that concrete path is available as a candidate target file.

Directory listings, `search_files` results, Git status/history/diff metadata,
README files, and generic shell output do not satisfy the rule. A premature
classifier decision keeps the effective state at `explore`, records
`implement_readiness_blocked`, and gives V4 the fixed instruction to read the
single most relevant source file without broadening the search.

After readiness is satisfied, ForgeLoop records
`implement_readiness_satisfied` and uses the existing Edit Intent Handoff and
Hybrid v1.2 state-aware tool gating without modification.

The compact classifier input now includes a structured
`implementation_readiness` object with:

- `source_content_read` and `source_files_read`;
- `candidate_target_files`;
- `saw_test_evidence` and `saw_error_evidence`;
- `has_diff`, `has_intent`, and `ready`.

Only paths and booleans are included; repository contents are not sent to the
1.5B classifier. The same evidence is retained in Controller decisions and the
terminal summary.

## Run

```powershell
$env:DEEPSEEK_API_KEY = "..."
ollama pull qwen2.5:1.5b-instruct

uv run forgeloop eval --stage c --task query-encoding --live `
  --repeats 1 `
  --policy-manifest deepseek-v4-flash-edit-intent-readiness-v1

uv run forgeloop deepswe run `
  --task oxvg-structural-selector-preservation `
  --policy deepseek-v4-flash-edit-intent-readiness-v1
```

## One-shot validation on 2026-08-09

### Frozen `query-encoding`

The single run passed its verifier and ended
`completed/model_finish_tool`:

- 7 model calls, 10 tool calls, 22,826 total tokens;
- 20,925 input and 1,901 output tokens;
- 32.51 seconds, provider cost $0.00165294;
- 2 premature implement decisions blocked after listing/search;
- `urls.py` was then read and recorded before the handoff;
- valid intent target: `urls.py`;
- actual progression: source read -> intent -> patch -> passing test -> diff ->
  explicit finish.

This preserves the frozen regression and proves the required trajectory order.

### DeepSWE `oxvg-structural-selector-preservation`

The single run used pinned DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, the official Docker
environment, and the unchanged patch collector/verifier. It failed honestly:

- terminal: `failed/controller_invalid_edit_intent`;
- 11 model calls, 14 tool calls, 186,868 total tokens;
- 183,271 input and 3,597 output tokens;
- Agent wall time 200.757 seconds, provider cost $0.0144071;
- 6 premature implement decisions were held in `explore`;
- V4 then read `crates/oxvg_ast/src/style.rs`, which caused
  `implement_readiness_satisfied` before `edit_intent_requested`;
- handoff context then read `selectors.rs` and `collapse_groups.rs`, but V4 did
  not submit `submit_edit_intent`;
- after 3 bounded context actions and 1 focused replan, the second invalid
  handoff action terminated the run;
- no intent was accepted, no patch/test action occurred, and no patch was
  collected;
- official verifier: P2P 62/62, F2P 0/6, partial 0.9117647058823529, reward 0.

The readiness defect is fixed: the trajectory proves V4 read real related Rust
source before implementation was offered. It did not solve the downstream model
failure. Although the Controller stopped before the 200,000-token ceiling, the
run still consumed 186,868 tokens because V4 ignored several focused readiness
instructions before reading source. This result does not support a solve-rate or
efficiency improvement claim.
