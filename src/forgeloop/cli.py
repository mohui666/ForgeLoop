from __future__ import annotations

import os
from pathlib import Path

import typer

from forgeloop.agent import AgentLoop, RunMode, RunResult, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.dataset import (
    DatasetBuilder,
    DatasetError,
    export_dataset,
    inspect_dataset,
)
from forgeloop.evals import (
    EvalRunner,
    EvalSuite,
    EvalTask,
    default_suite_path,
    resolve_suite_path,
)
from forgeloop.foundry import FoundryBuilder, FoundryError, default_catalog_path
from forgeloop.interactive import run_interactive
from forgeloop.models import LiteLLMProvider
from forgeloop.policy import PolicyIdentity, PolicyManifestError
from forgeloop.runtime import DockerRuntime, LocalRuntime
from forgeloop.security import SecretRedactor
from forgeloop.tools import build_default_tools
from forgeloop.trace import (
    TraceError,
    explain_trajectory,
    replay_trajectory,
    resolve_trajectory,
)
from forgeloop.trajectory import TrajectoryStore
from forgeloop.workspace import Workspace

app = typer.Typer(
    name="forgeloop",
    help="A small, provider-neutral CLI coding agent.",
    no_args_is_help=False,
    add_completion=False,
)
foundry_app = typer.Typer(
    help="Build a small curated eval suite from real Python bug-fix commits.",
    no_args_is_help=True,
)
app.add_typer(foundry_app, name="foundry")
dataset_app = typer.Typer(
    help="Build, inspect, and export traceable training data from trajectories.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")
trace_app = typer.Typer(
    help="Replay and deterministically explain recorded Agent trajectories.",
    no_args_is_help=True,
)
app.add_typer(trace_app, name="trace")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Start the interactive coding Agent when no automation command is supplied."""
    if ctx.invoked_subcommand is None:
        run_interactive()


DEFAULT_EVAL_SUITE = default_suite_path()
DEFAULT_EVAL_OUTPUT = Path(".forgeloop/eval-runs")
EVAL_SUITE_OPTION = typer.Option(
    DEFAULT_EVAL_SUITE, "--suite", help="Eval suite JSON file."
)
EVAL_OUTPUT_OPTION = typer.Option(DEFAULT_EVAL_OUTPUT, help="Eval artifacts directory.")
DEFAULT_DATASET_SOURCE = Path(".forgeloop/eval-runs")
DEFAULT_DATASET_OUTPUT = Path(".forgeloop/dataset")


def _resolved_trace(reference: str) -> Path:
    try:
        return resolve_trajectory(reference)
    except (TraceError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@trace_app.command("replay")
def trace_replay_command(
    reference: str = typer.Argument(..., help="Trajectory id or JSONL path."),
) -> None:
    """Render an offline evidence timeline without re-executing side effects."""
    path = _resolved_trace(reference)
    try:
        typer.echo(replay_trajectory(path), nl=False)
    except (TraceError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@trace_app.command("explain")
def trace_explain_command(
    reference: str = typer.Argument(..., help="Trajectory id or JSONL path."),
) -> None:
    """Explain recorded outcome, repetition, safety, and termination evidence."""
    path = _resolved_trace(reference)
    try:
        typer.echo(explain_trajectory(path), nl=False)
    except (TraceError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@dataset_app.command("build")
def dataset_build_command(
    source: Path = typer.Option(
        DEFAULT_DATASET_SOURCE,
        help="Directory containing eval run tasks.jsonl and trajectories.",
    ),
    output: Path = typer.Option(
        DEFAULT_DATASET_OUTPUT,
        help="Directory for index.jsonl and manifest.json.",
    ),
    suite: list[Path] | None = typer.Option(
        None,
        "--suite",
        help="Additional task suite metadata file; may be repeated.",
    ),
) -> None:
    """Build the internal dataset index without modifying source trajectories."""
    try:
        result = DatasetBuilder(source, output, suite_paths=suite or ()).build()
    except (DatasetError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Dataset: {result.index_path}")
    typer.echo(f"Samples: {result.samples}")
    typer.echo(f"Classifications: {result.classifications}")
    typer.echo(f"Skipped source records: {result.skipped}")


@dataset_app.command("inspect")
def dataset_inspect_command(
    dataset: Path = typer.Option(
        DEFAULT_DATASET_OUTPUT,
        help="Dataset directory or index.jsonl path.",
    ),
) -> None:
    """Show sample counts, classifications, sources, and basic usage statistics."""
    try:
        stats = inspect_dataset(dataset)
    except (DatasetError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Samples: {stats['samples']}")
    typer.echo(f"Classifications: {stats['classifications']}")
    typer.echo(f"Sources: {stats['sources']}")
    typer.echo(f"Models: {stats['models']}")
    typer.echo(f"Policy Stages: {stats['policy_stages']}")
    typer.echo(f"Inference Backends: {stats['inference_backends']}")
    typer.echo(f"Repositories: {stats['repositories']}")
    typer.echo(f"Effect Events: {stats['effect_events']}")
    typer.echo(f"Effect Types: {stats['effect_types']}")
    typer.echo(f"Effect Coverage: {stats['effect_statuses']}")
    typer.echo(f"Safety Flags: {stats['safety_flags']}")
    typer.echo(f"Total Tokens: {_format_count(stats['total_tokens'])}")
    typer.echo(f"Total Cost: {_format_optional_money(stats['total_cost_usd'])}")
    typer.echo(f"Average Steps: {stats['average_steps']:.2f}")


@dataset_app.command("export")
def dataset_export_command(
    dataset: Path = typer.Option(
        DEFAULT_DATASET_OUTPUT,
        help="Dataset directory or index.jsonl path.",
    ),
    output: Path = typer.Option(
        Path(".forgeloop/dataset/sft.jsonl"),
        help="Destination JSONL path.",
    ),
    export_format: str = typer.Option(
        "sft",
        "--format",
        help="sft exports only SFT candidates; internal exports curated samples.",
    ),
    include_infrastructure: bool = typer.Option(
        False,
        help="Include infrastructure failures in internal exports.",
    ),
) -> None:
    """Export sanitized JSONL through a framework-neutral adapter."""
    try:
        count, classifications = export_dataset(
            dataset,
            output,
            export_format=export_format.lower(),
            include_infrastructure=include_infrastructure,
        )
    except (DatasetError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Exported: {count}")
    typer.echo(f"Classifications: {dict(classifications)}")
    typer.echo(f"Output: {output.expanduser().resolve()}")


@foundry_app.command("build")
def foundry_build_command(
    catalog: Path = typer.Option(
        default_catalog_path(),
        help="Curated source-commit catalog.",
    ),
    output: Path = typer.Option(
        Path(".forgeloop/foundry/real-swe"),
        help="Fresh output directory for the generated suite.",
    ),
    cache: Path = typer.Option(
        Path(".forgeloop/foundry/cache"),
        help="Reusable bare-ish source clone cache.",
    ),
    validation_repeats: int = typer.Option(2, min=2, max=3),
) -> None:
    """Extract patches and require repeated Docker FAIL-to-PASS validation."""
    try:
        result = FoundryBuilder(
            catalog,
            output,
            cache,
            repeats=validation_repeats,
        ).build()
    except (FoundryError, OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Built: {result.suite_path}")
    typer.echo(f"Accepted: {result.accepted} | Filtered: {result.filtered}")
    typer.echo(f"Docker: {result.image} ({result.image_id})")


def _execute(
    mode: RunMode,
    request: str,
    model: str | None,
    workspace_path: Path,
    api_base: str | None,
    max_steps: int,
    max_model_calls: int,
    max_tool_calls: int,
    timeout_seconds: float,
    max_tokens: int,
    max_cost_usd: float,
    trajectory_dir: Path | None,
    policy_manifest: Path | None,
) -> None:
    policy = _load_policy(policy_manifest)
    selected_model = (
        model
        or os.getenv("FORGELOOP_MODEL")
        or (policy.litellm_model if policy else None)
    )
    if not selected_model:
        raise typer.BadParameter(
            "Set --model or FORGELOOP_MODEL (for example: openai/gpt-4.1)."
        )
    workspace = Workspace(workspace_path)
    output_dir = trajectory_dir or (workspace.root / ".forgeloop" / "runs")
    limits = BudgetLimits(
        max_steps=max_steps,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_seconds=timeout_seconds,
        max_tokens=max_tokens or None,
        max_cost_usd=max_cost_usd or None,
    )
    if policy and selected_model != policy.litellm_model:
        raise typer.BadParameter(
            f"--model must match policy manifest route {policy.litellm_model}"
        )
    resolved_api_base = api_base or (
        str(policy.serving_config.get("api_base") or "") if policy else None
    )
    if policy:
        if not resolved_api_base:
            raise typer.BadParameter(
                "Self-hosted policy requires --api-base or serving_config.api_base."
            )
        policy = policy.with_serving_config(api_base=resolved_api_base)
    self_hosted_key = (
        os.getenv("FORGELOOP_SELF_HOSTED_API_KEY") or "EMPTY" if policy else None
    )
    provider = LiteLLMProvider(
        model=selected_model,
        api_base=resolved_api_base,
        api_key=self_hosted_key,
        policy=policy,
    )
    runtime = LocalRuntime()
    agent = AgentLoop(
        provider=provider,
        tools=build_default_tools(workspace, runtime),
        workspace=workspace,
        trajectory=TrajectoryStore(
            output_dir,
            redactor=SecretRedactor(
                ()
                if not self_hosted_key or self_hosted_key == "EMPTY"
                else (self_hosted_key,)
            ),
        ),
        limits=limits,
    )
    typer.echo(
        f"ForgeLoop {mode.value} | model={selected_model} | workspace={workspace.root}"
    )
    result = agent.run(mode, request)
    _print_result(result)
    if result.status is not RunStatus.COMPLETED:
        raise typer.Exit(code=2)


def _print_result(result: RunResult) -> None:
    typer.echo(f"\nResult: {result.status.value}")
    typer.echo(f"Summary: {result.summary}")
    if result.evidence:
        typer.echo(f"Evidence: {result.evidence}")
    usage = result.budget["usage"]
    typer.echo(f"\nModel: {result.model or 'N/A'}")
    typer.echo(f"Provider: {result.provider or 'N/A'}")
    typer.echo(f"Steps: {usage['steps']}")
    typer.echo(f"Model Calls: {usage['model_calls']}")
    typer.echo(f"Tool Calls: {usage['tool_calls']}")
    typer.echo(f"\nInput Tokens: {_format_count(usage['input_tokens'])}")
    typer.echo(f"Output Tokens: {_format_count(usage['output_tokens'])}")
    typer.echo(f"Total Tokens: {_format_count(usage['total_tokens'])}")
    typer.echo(f"Cached Tokens: {_format_count(usage['cached_tokens'])}")
    typer.echo(f"Reasoning Tokens: {_format_count(usage['reasoning_tokens'])}")
    cost = usage["cost_usd"]
    typer.echo(f"\nCost: {'unknown' if cost is None else f'${cost:.6f}'}")
    typer.echo(f"Cost Source: {', '.join(usage['cost_sources']) or 'unknown'}")
    typer.echo(f"Wall Time: {usage['elapsed_seconds']}s")
    typer.echo(f"Stop Reason: {result.stop_reason}")
    typer.echo(f"Trajectory: {result.trajectory_path}")


def _format_count(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


def _format_optional_money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.6f}"


def _format_average(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.2f}"


def _load_policy(path: Path | None) -> PolicyIdentity | None:
    if path is None:
        return None
    try:
        return PolicyIdentity.load(path)
    except PolicyManifestError as exc:
        raise typer.BadParameter(str(exc)) from exc


COMMON = {
    "model": typer.Option(
        None, "--model", "-m", help="LiteLLM model id, usually provider/model."
    ),
    "workspace_path": typer.Option(
        Path("."), "--workspace", "-w", help="Workspace root."
    ),
    "api_base": typer.Option(
        None, "--api-base", help="Optional OpenAI-compatible API base."
    ),
    "max_steps": typer.Option(30, min=1, help="Maximum agent loop steps."),
    "max_model_calls": typer.Option(30, min=1, help="Maximum model calls."),
    "max_tool_calls": typer.Option(80, min=1, help="Maximum tool calls."),
    "timeout_seconds": typer.Option(900.0, min=1, help="Whole-run wall-clock limit."),
    "max_tokens": typer.Option(
        200_000, min=0, help="Reported token limit; 0 disables it."
    ),
    "max_cost_usd": typer.Option(
        0.0, min=0, help="Reported cost limit in USD; 0 disables it."
    ),
    "trajectory_dir": typer.Option(
        None, help="Output directory; defaults to .forgeloop/runs."
    ),
    "policy_manifest": typer.Option(
        None,
        "--policy-manifest",
        help="Policy JSON path or bundled id such as qwen3.5-9b.",
    ),
}


@app.command()
def goal(
    request: str = typer.Argument(..., help="Final outcome the agent should achieve."),
    model: str | None = COMMON["model"],
    workspace_path: Path = COMMON["workspace_path"],
    api_base: str | None = COMMON["api_base"],
    max_steps: int = COMMON["max_steps"],
    max_model_calls: int = COMMON["max_model_calls"],
    max_tool_calls: int = COMMON["max_tool_calls"],
    timeout_seconds: float = COMMON["timeout_seconds"],
    max_tokens: int = COMMON["max_tokens"],
    max_cost_usd: float = COMMON["max_cost_usd"],
    trajectory_dir: Path | None = COMMON["trajectory_dir"],
    policy_manifest: Path | None = COMMON["policy_manifest"],
) -> None:
    """Run autonomously toward an outcome without broadening into unrelated goals."""
    _execute(
        RunMode.GOAL,
        request,
        model,
        workspace_path,
        api_base,
        max_steps,
        max_model_calls,
        max_tool_calls,
        timeout_seconds,
        max_tokens,
        max_cost_usd,
        trajectory_dir,
        policy_manifest,
    )


@app.command()
def task(
    request: str = typer.Argument(..., help="Bounded software-engineering task."),
    model: str | None = COMMON["model"],
    workspace_path: Path = COMMON["workspace_path"],
    api_base: str | None = COMMON["api_base"],
    max_steps: int = COMMON["max_steps"],
    max_model_calls: int = COMMON["max_model_calls"],
    max_tool_calls: int = COMMON["max_tool_calls"],
    timeout_seconds: float = COMMON["timeout_seconds"],
    max_tokens: int = COMMON["max_tokens"],
    max_cost_usd: float = COMMON["max_cost_usd"],
    trajectory_dir: Path | None = COMMON["trajectory_dir"],
    policy_manifest: Path | None = COMMON["policy_manifest"],
) -> None:
    """Run a bounded coding task and avoid unrelated work."""
    _execute(
        RunMode.TASK,
        request,
        model,
        workspace_path,
        api_base,
        max_steps,
        max_model_calls,
        max_tool_calls,
        timeout_seconds,
        max_tokens,
        max_cost_usd,
        trajectory_dir,
        policy_manifest,
    )


@app.command("eval")
def eval_command(
    suite_path: Path = EVAL_SUITE_OPTION,
    stage: str = typer.Option(
        "a", "--stage", help="a=one canary, b=three varied tasks, c=full suite."
    ),
    task_ids: list[str] | None = typer.Option(
        None,
        "--task",
        help="Run only these existing task ids after stage selection; may be repeated.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="LiteLLM provider/model id.",
    ),
    output_dir: Path = EVAL_OUTPUT_OPTION,
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        help="OpenAI-compatible endpoint; policy manifest default is used if omitted.",
    ),
    policy_manifest: Path | None = typer.Option(
        None,
        "--policy-manifest",
        help="Policy JSON path or bundled id such as qwen3.5-9b.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Actually call the model. Without this flag, only list selected tasks.",
    ),
    reasoning_effort: str = typer.Option(
        "max", help="DeepSeek V4 reasoning effort: high or max."
    ),
    max_steps: int = typer.Option(30, min=1),
    max_model_calls: int = typer.Option(30, min=1),
    max_tool_calls: int = typer.Option(80, min=1),
    max_tokens: int = typer.Option(500_000, min=1),
    keep_workspaces: bool = typer.Option(False, help="Keep isolated task workspaces."),
    runtime_name: str = typer.Option(
        "local", "--runtime", help="Execution runtime: local or docker."
    ),
    docker_image: str = typer.Option(
        "forgeloop-eval:py312",
        "--docker-image",
        help="Docker image used by --runtime docker.",
    ),
    repeats: int = typer.Option(
        3, "--repeats", min=1, max=3, help="Independent attempts per task."
    ),
) -> None:
    """Run the fixed, verifier-driven software-engineering smoke eval."""
    policy = _load_policy(policy_manifest)
    selected_model = model or (
        policy.litellm_model if policy else "deepseek/deepseek-v4-flash"
    )
    if policy and selected_model != policy.litellm_model:
        raise typer.BadParameter(
            f"--model must match policy manifest route {policy.litellm_model}"
        )
    if not policy and reasoning_effort not in {"high", "max"}:
        raise typer.BadParameter("--reasoning-effort must be high or max")
    if runtime_name not in {"local", "docker"}:
        raise typer.BadParameter("--runtime must be local or docker")
    try:
        suite = EvalSuite.load(resolve_suite_path(suite_path))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        tasks = suite.select_stage(stage)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if task_ids:
        requested = set(task_ids)
        available = {task.id for task in tasks}
        missing = sorted(requested - available)
        if missing:
            raise typer.BadParameter(
                "Task ids are not in the selected stage: " + ", ".join(missing)
            )
        tasks = tuple(task for task in tasks if task.id in requested)
    typer.echo(
        f"Suite: {suite.id} | stage={stage.lower()} | tasks={len(tasks)} | "
        f"repeats={repeats}"
    )
    typer.echo(
        f"Model: {selected_model} | policy={policy.policy_id if policy else 'legacy'}"
    )
    for task_item in tasks:
        typer.echo(f"- {task_item.id}: {task_item.description}")
    if not live:
        typer.echo("\nDry run only. Add --live to execute real model calls.")
        return

    if policy:
        self_hosted_key = os.getenv("FORGELOOP_SELF_HOSTED_API_KEY") or "EMPTY"
        redactor = SecretRedactor(
            () if self_hosted_key == "EMPTY" else (self_hosted_key,)
        )
        resolved_api_base = api_base or str(policy.serving_config.get("api_base") or "")
        if not resolved_api_base:
            raise typer.BadParameter(
                "Self-hosted policy requires --api-base or serving_config.api_base."
            )
        policy = policy.with_serving_config(api_base=resolved_api_base)
        provider = LiteLLMProvider(
            model=selected_model,
            api_base=resolved_api_base,
            api_key=self_hosted_key,
            policy=policy,
        )
    else:
        api_key = os.getenv("AGENT_TEMP_KEY") or os.getenv("agent_temp_key")
        if not api_key:
            raise typer.BadParameter(
                "AGENT_TEMP_KEY is not available in this process. Map the system environment "
                "variable into the current shell before running."
            )
        redactor = SecretRedactor((api_key,))
        provider = LiteLLMProvider(
            model=selected_model,
            api_base=api_base or "https://api.deepseek.com",
            api_key=api_key,
            extra={
                "reasoning_effort": reasoning_effort,
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        )

    def docker_runtime_for_task(task: EvalTask) -> DockerRuntime:
        return DockerRuntime(
            image=task.docker_image or docker_image,
            dockerfile=task.dockerfile,
            build_context=task.docker_build_context,
        )

    runner = EvalRunner(
        provider=provider,
        limits=BudgetLimits(
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            max_seconds=max(task_item.timeout_seconds for task_item in tasks),
            max_tokens=max_tokens,
            max_cost_usd=None,
        ),
        output_root=output_dir,
        redactor=redactor,
        keep_workspaces=keep_workspaces,
        runtime_factory=LocalRuntime if runtime_name == "local" else None,
        task_runtime_factory=(
            docker_runtime_for_task if runtime_name == "docker" else None
        ),
    )
    summary, run_dir = runner.run(
        suite, tasks, repeats=repeats, stop_on_systemic_failure=True
    )
    typer.echo("\nEval Result")
    typer.echo(f"Tasks: {summary.tasks}")
    typer.echo(f"Attempts: {summary.attempts}")
    typer.echo(f"Planned Attempts: {summary.planned_attempts}")
    if summary.stopped_early:
        typer.echo(f"Stopped Early: {summary.stop_reason}")
    typer.echo(f"Solved: {summary.solved}")
    typer.echo(f"Failed: {summary.failed}")
    typer.echo(f"Blocked: {summary.blocked}")
    typer.echo(f"Budget Exceeded: {summary.budget_exceeded}")
    typer.echo(f"Pass Rate: {summary.pass_rate:.1%}")
    typer.echo(f"Pass@1: {summary.pass_at_1:.1%}")
    typer.echo(
        "Pass@3: "
        + (f"{summary.pass_at_3:.1%}" if summary.pass_at_3 is not None else "N/A")
    )
    typer.echo(f"Total Input Tokens: {_format_count(summary.total_input_tokens)}")
    typer.echo(f"Total Output Tokens: {_format_count(summary.total_output_tokens)}")
    typer.echo(f"Total Tokens: {_format_count(summary.total_tokens)}")
    typer.echo(
        f"Average Tokens / Task: {_format_average(summary.average_tokens_per_task)}"
    )
    typer.echo(
        "Average Tokens / Solved: "
        f"{_format_average(summary.average_tokens_per_solved_task)}"
    )
    typer.echo(f"Tokens / Solved: {_format_average(summary.tokens_per_solved_task)}")
    typer.echo(f"Total Cost: {_format_optional_money(summary.total_cost_usd)}")
    typer.echo(
        f"Average Cost / Task: {_format_optional_money(summary.average_cost_per_task_usd)}"
    )
    cost_per_solved = (
        "N/A"
        if summary.solved == 0
        else _format_optional_money(summary.cost_per_solved_task_usd)
    )
    typer.echo(f"Cost / Solved: {cost_per_solved}")
    typer.echo(f"Average Steps: {summary.average_steps:.2f}")
    typer.echo(f"Average Model Calls: {summary.average_model_calls:.2f}")
    typer.echo(f"Average Tool Calls: {summary.average_tool_calls:.2f}")
    typer.echo(f"Average Wall Time: {summary.average_wall_time_seconds:.2f}s")
    typer.echo(f"Failure Categories: {summary.failure_categories}")
    typer.echo(f"Difficulty Metrics: {summary.difficulty_metrics}")
    typer.echo(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    app()
