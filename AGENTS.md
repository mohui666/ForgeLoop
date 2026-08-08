# ForgeLoop Project Charter

## Mission

ForgeLoop is an application-first, engineering-first Coding Agent platform. Its long-term goal is to become a genuinely usable coding agent that can continuously improve from executable software-engineering experience.

The target loop is:

`real repositories / real tasks -> agent execution -> trajectory collection -> executable environments from Environment Foundry -> dataset curation -> SFT / RLVR / GRPO -> improved policy -> redeploy into ForgeLoop`

Every major implementation decision should move ForgeLoop toward this closed loop.

## Product principle

ForgeLoop is not primarily a paper project or a benchmark-optimization exercise. Benchmarks, eval suites, failure analysis, and metrics exist to improve the product, validate regressions, create training signal, and demonstrate that the learning loop works.

Prefer:

- real usability over experiment-only machinery;
- executable verification over subjective scoring;
- reusable infrastructure over one-off benchmark code;
- real repository tasks over toy tasks;
- measurable end-to-end improvement over isolated feature count;
- simple, maintainable architecture over unnecessary framework complexity.

## Priority order

When choosing what to build next, prefer work in roughly this order:

1. Keep the Coding Agent reliable and usable on real repositories.
2. Preserve deterministic Runtime, Verifier, Trajectory, Eval, and safety boundaries.
3. Improve Environment Foundry as a source of diverse executable SWE training environments.
4. Turn successful and failed trajectories into high-quality training data.
5. Integrate a trainable open-weight policy and complete an SFT loop.
6. Add verifier-driven RLVR / GRPO and redeploy the improved policy into ForgeLoop.
7. Measure whether the new policy is actually better on held-out real tasks.

## Architectural constraint

Keep the core boundaries reusable:

- `AgentLoop`
- `ModelProvider`
- `Runtime`
- tools
- Session / Context management
- Trajectory
- Verifier / Eval
- Environment Foundry
- training-data export and future training pipeline

Interactive TUI and headless automation are interfaces over the same core. Do not duplicate core agent, runtime, evaluation, or trajectory logic inside presentation layers.

## Anti-goals

Do not divert the roadmap into features that do not materially advance the mission. In particular, do not add large side systems merely because other coding agents have them.

Examples that require a concrete product need before implementation include:

- multi-agent orchestration;
- IDE-like file-tree UI;
- web dashboards;
- generic RAG/vector databases;
- distributed infrastructure;
- benchmark-specific hacks;
- large experimental taxonomies or repeated evaluation runs with no engineering decision attached.

## Definition of progress

A milestone is valuable when it improves one or more of these properties:

- ForgeLoop can solve more real coding work reliably;
- environments are easier to construct and reproduce;
- trajectories are more useful for training;
- failures are attributable to model vs harness vs environment;
- training can produce a new policy that is deployed back into ForgeLoop;
- the improvement can be demonstrated with reproducible executable evaluation.

If a proposed task does not clearly support one of these outcomes, question whether it belongs on the current roadmap.
