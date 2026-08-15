# Offline Reliability Hardening

Date: 2026-08-15

## Outcome

This offline-only pass closes four reliability gaps around Execution Closure v3,
runtime evidence, delivery, and repository search. It used no provider request,
DeepSWE LLM job, network lookup, or external API.

## Base-aware review

The AgentLoop already fixed its progress fingerprint and patch delivery to the
Git commit present at run start. `git_diff` was still HEAD-relative, however. If
the model committed its own source change before validation, the worktree became
clean and the review tool returned `No changes.` with `review_scope=worktree`.
Closure v3 could therefore accept a review that did not contain the patch later
delivered from run base to HEAD.

The ToolRegistry now binds the stable run-start `base_head` to context-aware
tools. An unfiltered, non-cached `git_diff` compares that base directly with the
current worktree, so its tracked evidence includes model commits, staged edits,
and unstaged edits in one final-tree patch. Untracked files retain the existing
synthetic new-file evidence. Metadata records `review_base` and
`tracked_diff_layers=["base_to_worktree"]`.

A deterministic integration test performs:

```text
source edit -> model-owned Git commit -> validation PASS
-> base-to-worktree review -> explicit finish -> delivery
```

The review observation contains both committed source changes and delivery
correctly records that ForgeLoop did not create a second commit.

## Truncated command evidence

Structured `stdout_truncated` and `stderr_truncated` flags now cross the
remaining control boundaries:

- `GitPatchDelivery` rejects truncated branch, status, HEAD, staging/commit, or
  full-patch identity output instead of parsing a partial control value;
- `VerifierResult` preserves both flags without changing the official
  exit-code-driven pass/fail decision;
- Environment Foundry records runtime truncation and its own 4,000-character
  evidence excerpting.

Fault-injection coverage exercises both streams for branch, status, HEAD, and
patch identity. Collection fails closed with no patch identity when any control
output is incomplete.

## Search parity

Local repository search now observes its timeout deadline while traversing
candidates and lines. A timeout returns no partial matches and records
`SearchResult(error="Search timed out", timed_out=True)`.

The Docker search program now applies the same path-level credential exclusions
as local search, preventing recursive root searches from returning `.env`,
credential JSON, private-key files, or `.git` content. The parity test executes
the container search program locally against a deterministic fixture, so Docker
and network access are not required.

## Verification

- focused offline reliability tests: 34 passed;
- full pytest: 206 passed, 4 skipped;
- Ruff lint and format check: PASS;
- `git diff --check`: PASS;
- sdist and wheel: PASS;
- external model calls, provider usage, and API cost: zero.

## Remaining limits

- Local search deadlines are cooperative around traversal and line matching; a
  single blocking filesystem read or pathological regular-expression operation
  cannot be pre-empted inside the current in-process implementation.
- POSIX local shell timeouts still need process-group termination coverage so a
  grandchild cannot outlive the timed-out shell.
- Pier structured consumers such as remote JSON search/list parsing should gain
  explicit incomplete-output rejection.
- Agent-facing shell output carries truncation flags in metadata, but the text
  observation does not yet call out markerless truncation explicitly.

