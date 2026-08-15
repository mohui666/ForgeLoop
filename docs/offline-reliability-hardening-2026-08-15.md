# Offline Reliability Hardening

Date: 2026-08-15

## Outcome

This offline-only work closes reliability gaps around Execution Closure v3,
runtime evidence, delivery, repository search, process termination, and Pier
result parsing. It used no provider request, DeepSWE LLM job, network lookup,
or external API.

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

Local repository search now runs behind a subprocess boundary. The parent owns
the timeout and can terminate a blocked filesystem read or pathological regular
expression instead of depending on cooperative checks inside the search loop. A
timeout returns no partial matches and records
`SearchResult(error="Search timed out", timed_out=True)`.

The Docker search program now applies the same path-level credential exclusions
as local search, preventing recursive root searches from returning `.env`,
credential JSON, private-key files, or `.git` content. The parity test executes
the container search program locally against a deterministic fixture, so Docker
and network access are not required.

## Follow-up runtime hardening

Local POSIX commands now start in a new session. On timeout ForgeLoop kills the
entire process group, preventing a shell grandchild from surviving the command
boundary and modifying the workspace later. The Windows `taskkill /T` path is
unchanged and now has an explicit fallback when `taskkill` itself fails.

Pier path-kind, search, and file-list consumers reject structured stdout or
stderr truncation before parsing results. They also validate the expected JSON
shape, fail closed on non-zero path probes, and filter sensitive paths from
search/list observations. Deterministic fault injection covers both streams and
freezes the existing 40,000-character Pier truncation representation.

Shell, validation, and Git inspection observations now append an explicit
warning when either runtime stream is structurally truncated. This warning is
independent of a textual truncation marker, while exit-code success semantics
remain unchanged.

## Verification

- focused second-pass reliability tests: 34 passed, 1 skipped;
- full pytest: 223 passed, 5 skipped;
- Ruff lint (full tree) and format check (changed Python files): PASS;
- `git diff --check`: PASS;
- sdist and wheel: PASS;
- external model calls, provider usage, and API cost: zero.

## Remaining limits

- The standalone search program intentionally duplicates the sensitive-path
  constants used by the host security module. A future change should derive
  both representations from one policy source so the local, Docker, and Pier
  filters cannot drift.
- Pier filters remote search/list results again on the host. Moving the same
  filter into the remote program would avoid transporting sensitive path names
  to the host process at all.
- Third-party Runtime implementations can still construct `CommandResult`
  without setting structured truncation flags. Core runtimes set the flags, but
  the default remains `False` for compatibility.
- The subprocess boundary adds one Python-process startup to each local search;
  correctness and hard timeout enforcement currently take priority over that
  small fixed cost.
