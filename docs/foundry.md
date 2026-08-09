# Minimal Environment Foundry

ForgeLoop's Stage B Foundry turns a manually curated list of real Python bug-fix commits into the existing `forgeloop.eval.v2` task format. It reuses the normal Agent loop, verifier, Docker runtime, trajectory, budgets, and result accounting.

## Current eight-task suite

| Task | Source | Base SHA | Fix SHA | Difficulty | Verifier focus |
| --- | --- | --- | --- | --- | --- |
| `more-itertools-empty-interleave` | more-itertools PR 1193 | `5d946b3590bfe92f1465c1b9b9830dd434745c84` | `f51a53bfd2fe9504063a33ef5f4a73e30d82d0e2` | easy | empty `interleave_evenly` |
| `pygments-raw-token-error-color` | Pygments PR 3215 | `6a7aa837d5001dd199f0094165d85568faa1142a` | `2f0d713b396de53e8780c114b00a7adad99cbffc` | easy | bytes/text formatter boundary |
| `click-optional-metavar-brackets` | Click PR 3578 | `8929d392781c8113bc569f388c15c47b94f86581` | `762c97eef7c1b3779678992f26a553a2a8c80793` | medium | optional metavar rendering |
| `more-itertools-empty-numeric-range-reversed` | more-itertools PR 1153 | `247e15b3a489d5805375c95dfa79486c9bd0eb1b` | `edb3346f835ca917efbfda5e2d6664ab952da369` | medium | reversed empty range |
| `more-itertools-repeat-with-iterators` | more-itertools PR 1125 | `fdbd43c569feb4388302ddee0551d18edd39fadc` | `be5793a55f58ee29521d8cbc9f14019f5833454e` | medium | iterator reuse in product helpers |
| `h11-empty-chunk-data` | h11 commit | `e2a752b310dd5acd95151d2693f981f4dd4711c1` | `28f3715f63ff65a1f28d30af2ecfc2d4d298bd0e` | medium | empty chunk serialization |
| `pygments-regexopt-deduplicate` | Pygments PR 3190 | `53750ec8def276668ce4ce6b565a70105b5085e3` | `64226095eb736dbf23e16786f04c82f3a3a86b92` | medium | regex input deduplication |
| `more-itertools-product-repeat-with-iterators` | more-itertools PR 1117 | `b110a77364ec0ed5bb06194b591f97a54591dbc9` | `073d23421b60e05a99872e7b174902f4aa151f31` | medium | iterator repeat during indexing |

Exact verifier commands, fixture commits, Docker image IDs, source URLs, patch hashes, and per-repeat stdout/stderr are recorded in the generated `tasks.json` and `artifacts/<task-id>/metadata.json` files.

## Build flow

The curated catalog at `src/forgeloop/foundry_assets/catalogs/stage_b.json` provides source provenance, explicit test and solution paths, focused verifier, difficulty, and every rejected candidate with a reason. `forgeloop foundry build` then:

1. resolves the exact fix commit and requires exactly one parent;
2. extracts separate test and implementation patches from explicit paths;
3. exports the base and applies only the test patch to the Agent fixture;
4. builds the digest-pinned Python 3.12 image with pinned pytest;
5. twice copies a fresh fixture and requires `base + test patch` to fail;
6. applies hidden gold outside the container and requires a fresh container to pass;
7. fingerprints the bug fixture and records validation evidence;
8. atomically publishes `tasks.json` only after every task is accepted.

Candidate discovery, issue-description cleanup, patch-path selection, verifier selection, and difficulty labels remain manual. Commit/base checks, patch extraction, fixture export, Docker validation, repeat checks, hashing, screening-count validation, and suite generation are automated.

## Isolation and hidden gold

Runtime containers are network-disabled, disposable, and mount only a fresh task fixture at `/workspace`. Gold and test patches plus build evidence live in `artifacts/<task-id>/`; Agent-visible source lives in `fixtures/<task-id>/`. EvalRunner copies only the fixture.

The Stage B acceptance audit found no `gold.patch`, `solution.patch`, artifact pointer, or gold metadata in any fixture. No Agent trajectory is created during Foundry validation. The shared validation image is `forgeloop-real-swe:stage-b-py312`, recorded by immutable image ID in each task.

## Screening and quality result

Stage B inspected 37 candidate commits, accepted 8, and rejected 29: a Candidate-to-Valid-Task yield of 21.62%. Every rejection is recorded in the catalog with repository, commit, reason code, and concrete reason. The main groups were missing or inseparable regression tests, broad patches, legacy runner conflicts, platform/environment sensitivity, optional dependency expansion, nondeterministic concurrency, and bases that already passed.

All eight accepted tasks produced two independent `1 -> 0` verifier transitions with no timeout: 16 failing base validations and 16 passing hidden-gold validations. No flaky verifier or container-state contamination was observed. A build aborts rather than publishing when any task unexpectedly passes before gold, fails after gold, times out, has a patch error, has multiple parents, or needs missing project-specific infrastructure.

## Integration and next expansion

The CLI resolves `--suite real-swe` to this generated manifest and creates `DockerRuntime` from task metadata. `AgentLoop`, tool routing, `EvalRunner`, `Verifier`, trajectory, and cost/token accounting are unchanged.

This curated suite is now frozen as a local regression asset rather than a path
to a large in-house benchmark. Broader daily executable evaluation uses the
fixed external [DeepSWE Eval v2 subset](deepswe-eval-v2.md), reusing upstream
tasks, images, and verifiers. No crawler, new Foundry task batch, container pool,
distributed builder, or RL layer is added in this phase.
