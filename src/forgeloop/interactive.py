from __future__ import annotations

import getpass
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forgeloop.agent import AgentLoop, RunMode, RunResult
from forgeloop.budget import BudgetLimits
from forgeloop.controller import controller_for_policy
from forgeloop.config import (
    PROVIDERS,
    ConfigError,
    ConfigStore,
    CredentialStore,
    provider_api_base,
    set_provider_api_base,
)
from forgeloop.context import compact_messages, context_budget, estimate_tokens
from forgeloop.gitops import CheckpointManager, GitError, git_output, is_git_repo
from forgeloop.instructions import load_project_instructions
from forgeloop.models import LiteLLMProvider
from forgeloop.policy import PolicyIdentity
from forgeloop.model_capabilities import CapabilityResolver, ModelCache, ModelCapability
from forgeloop.provider_config import (
    PreflightError,
    canonical_model_route,
    configured_provider_names,
    fetch_provider_models,
    model_name_from_route,
    preflight_provider,
    test_provider_api,
)
from forgeloop.runtime import DockerRuntime, LocalRuntime, Runtime
from forgeloop.security import SecretRedactor
from forgeloop.sessions import Session, SessionStore
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.workflows import run_workflow
from forgeloop.workspace import Workspace


HELP = """ForgeLoop commands:
  /help                  Show this help
  /status                Session, repository, and recent Git status
  /diff                  Current Git diff
  /undo                  Restore the last Build checkpoint
  /test                  Run the detected test workflow
  /lint                  Run the detected lint workflow
  /plan [request]        Switch to read-only Plan mode
  /build [request|run]   Switch to Build mode, or run the build workflow
  /context               Show context size and project instructions
  /compact               Compact older conversation context now
  /model                 Select a configured Provider and Model
  /thinking              Select model-supported reasoning effort
  /api                   Configure/test/delete Provider API settings
  /runtime [local|docker] Show or change this Session runtime
  /cost                  Show Session token/cost/step usage
  /config                Show non-secret global and Session configuration
  /sessions              List saved Sessions
  /resume [id]           Resume a Session (latest when id is omitted)
  /new [path]            Start a new Session for a Git repository
  /exit                  Save and exit

Plain text is sent to the coding Agent. Slash commands control ForgeLoop.
"""


class InteractiveCLI:
    def __init__(
        self,
        *,
        cwd: Path | None = None,
        config_store: ConfigStore | None = None,
        credential_store: CredentialStore | None = None,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
        read_secret: Callable[[str], str] = getpass.getpass,
    ) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        self.config_store = config_store or ConfigStore()
        self.config = self.config_store.load()
        self.credentials = credential_store or CredentialStore()
        self.redactor = SecretRedactor.from_environment()
        self.sessions = SessionStore(self.config_store.home, redactor=self.redactor)
        self.model_cache = ModelCache(self.config_store.home)
        self.capabilities = CapabilityResolver(self.model_cache)
        self.session: Session | None = None
        self.read = read
        self.write = write
        self.read_secret = read_secret
        self.running = True

    def run(self) -> None:
        self.write("ForgeLoop — type /help for commands")
        while self.running:
            try:
                value = self.read("forgeloop> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.write("")
                break
            if not value:
                continue
            self.handle(value)
        if self.session:
            self.sessions.save(self.session)

    def handle(self, value: str) -> None:
        if value.startswith("/"):
            command, _, argument = value[1:].partition(" ")
            self._command(command.lower(), argument.strip())
            return
        self._chat(value)

    def _command(self, command: str, argument: str) -> None:
        handlers = {
            "help": lambda: self.write(HELP.rstrip()),
            "status": self._status,
            "diff": self._diff,
            "undo": self._undo,
            "test": lambda: self._workflow("test"),
            "lint": lambda: self._workflow("lint"),
            "context": self._context,
            "compact": self._compact,
            "cost": self._cost,
            "sessions": self._sessions,
            "exit": self._exit,
        }
        if command in handlers:
            if argument and command not in {"help"}:
                self.write(f"/{command} does not accept an argument")
            else:
                handlers[command]()
            return
        if command == "plan":
            self._set_mode("plan", argument)
        elif command == "build":
            if argument == "run":
                self._workflow("build")
            else:
                self._set_mode("build", argument)
        elif command == "model":
            self._model(argument)
        elif command == "thinking":
            self._thinking(argument)
        elif command == "api":
            self._api(argument)
        elif command == "runtime":
            self._runtime(argument)
        elif command == "config":
            self._config(argument)
        elif command == "resume":
            self._resume(argument)
        elif command == "new":
            self._new(argument)
        else:
            self.write(f"Unknown command: /{command}. Use /help.")

    def _ensure_session(self) -> bool:
        if self.session:
            return True
        repo = self.cwd
        if not is_git_repo(repo):
            selected = self.read(
                "Current directory is not a Git repo. Project path (blank to cancel): "
            ).strip()
            if not selected:
                return False
            repo = Path(selected).expanduser().resolve()
        return self._create_session(repo)

    def _create_session(self, repo: Path) -> bool:
        if not repo.is_dir() or not is_git_repo(repo):
            self.write(f"Not a Git repository: {repo}")
            return False
        self.session = Session.create(repo)
        if self.config.model:
            self.session.provider = self.config.provider
            self.session.model = model_name_from_route(
                self.config.provider, self.config.model
            )
        self.sessions.save(self.session)
        self.write(f"Session {self.session.id[:8]} created for {repo}")
        return True

    def _chat(self, request: str) -> None:
        self.chat(request)

    def chat(
        self,
        request: str,
        *,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        emit_output: bool = True,
    ) -> RunResult | None:
        if not self._ensure_session() or not self.session:
            return None
        repo = Path(self.session.repo)
        selected_provider, selected_model = self.current_model()
        try:
            route = preflight_provider(
                self.config,
                self.credentials,
                provider=selected_provider,
                model=selected_model,
            )
        except PreflightError as exc:
            self.write(str(exc))
            return None
        if (
            self.config.provider == route.provider
            and self.config.model != route.canonical_model
        ):
            self.config.model = route.canonical_model
            self.config_store.save(self.config)
        self.redactor = SecretRedactor.from_environment(route.api_key)
        self.sessions.redactor = self.redactor
        request = self.redactor.redact_text(request)
        budget = self.current_context_budget()
        if (
            budget.auto_compact_threshold is not None
            and estimate_tokens(self.session.conversation)
            >= budget.auto_compact_threshold
        ):
            self._compact(automatic=True)
        if (
            budget.usable_context is not None
            and estimate_tokens(self.session.conversation) >= budget.usable_context
        ):
            self.write("当前 Session 即使压缩后仍超过模型安全上下文，已阻止调用。")
            return None
        checkpoint = None
        if self.session.mode == "build":
            try:
                checkpoint = CheckpointManager(
                    repo, self.config_store.home, self.session.id
                ).create()
                self.session.checkpoints.append(checkpoint.id)
            except GitError as exc:
                self.write(f"Cannot create safe checkpoint: {exc}")
                return None
        runtime = self._make_runtime(self.session.runtime)
        workspace = Workspace(repo)
        try:
            runtime.start(repo)
            trajectory = TrajectoryStore(
                self.config_store.home / "trajectories" / self.session.id,
                redactor=self.redactor,
            )
            limits = BudgetLimits(
                max_steps=self.config.max_steps,
                max_model_calls=self.config.max_model_calls,
                max_tool_calls=self.config.max_tool_calls,
                max_seconds=self.config.timeout_seconds,
                max_tokens=self.config.max_tokens or None,
                max_cost_usd=self.config.max_cost_usd or None,
            )
            policy = (
                PolicyIdentity.load("deepseek-v4-flash-edit-intent-v1")
                if route.canonical_model == "deepseek/deepseek-v4-flash"
                else None
            )
            provider = LiteLLMProvider(
                model=route.canonical_model,
                api_base=route.api_base,
                api_key=route.api_key,
                thinking_level=self.session.thinking,
                policy=policy,
            )
            mode = RunMode.PLAN if self.session.mode == "plan" else RunMode.BUILD
            observed: list[tuple[str, dict[str, Any]]] = []

            def collect_event(name: str, payload: dict[str, Any]) -> None:
                observed.append((name, self.redactor.redact(payload)))
                if event_sink:
                    event_sink(name, payload)

            agent = AgentLoop(
                provider,
                build_default_tools(workspace, runtime, read_only=mode is RunMode.PLAN),
                workspace,
                trajectory,
                limits,
                event_sink=collect_event,
                cancel_check=cancel_check,
                controller=controller_for_policy(policy),
            )
            result = agent.run(
                mode,
                request,
                context_messages=tuple(self.session.conversation),
                instructions=load_project_instructions(repo),
            )
        finally:
            runtime.close()
        self.session.conversation = [dict(message) for message in result.conversation]
        self.session.trajectories.append(str(result.trajectory_path))
        self.session.last_summary = result.summary
        self._update_context_state(request, result, observed)
        self._record_usage(result.budget["usage"])
        self.sessions.save(self.session)
        if emit_output:
            self.write(result.summary)
            if result.evidence:
                self.write(f"Evidence: {result.evidence}")
            if result.status.value != "completed":
                self.write(f"Stopped: {result.status.value} ({result.stop_reason})")
        return result

    def _status(self) -> None:
        if not self._ensure_session() or not self.session:
            return
        repo = Path(self.session.repo)
        try:
            status = git_output(repo, "status", "--short", "--branch").rstrip()
        except GitError as exc:
            status = str(exc)
        try:
            log = git_output(repo, "log", "-n", "5", "--oneline", "--decorate").rstrip()
        except GitError:
            log = ""
        self.write(
            f"Session: {self.session.id[:8]} | mode={self.session.mode} | "
            f"runtime={self.session.runtime}\nRepo: {repo}\n{status or 'Working tree clean.'}"
        )
        if log:
            self.write("Recent commits:\n" + log)

    def _diff(self) -> None:
        if not self._ensure_session() or not self.session:
            return
        try:
            repo = Path(self.session.repo)
            output = git_output(repo, "status", "--short")
            output += git_output(repo, "diff", "--stat")
            output += git_output(repo, "diff")
            staged = git_output(repo, "diff", "--cached")
            if staged:
                output += "\nStaged diff:\n" + staged
            self.write(output.rstrip() or "No changes.")
        except GitError as exc:
            self.write(str(exc))

    def _undo(self) -> None:
        if not self.session or not self.session.checkpoints:
            self.write("No checkpoint is available for this Session.")
            return
        checkpoint_id = self.session.checkpoints[-1]
        try:
            CheckpointManager(
                Path(self.session.repo), self.config_store.home, self.session.id
            ).undo(checkpoint_id)
        except GitError as exc:
            self.write(f"Undo failed: {exc}")
            return
        self.session.checkpoints.pop()
        self.sessions.save(self.session)
        self.write(f"Restored checkpoint {checkpoint_id[:8]}.")

    def _workflow(self, kind: str) -> None:
        if not self._ensure_session() or not self.session:
            return
        runtime = self._make_runtime(self.session.runtime)
        repo = Path(self.session.repo)
        try:
            runtime.start(repo)
            result = run_workflow(runtime, repo, kind, self.config.timeout_seconds)
            output = result.stdout + (
                ("\nstderr:\n" + result.stderr) if result.stderr else ""
            )
            self.write(output.rstrip())
            self.write(f"{kind}: exit {result.exit_code}")
        except (ValueError, OSError) as exc:
            self.write(str(exc))
        finally:
            runtime.close()

    def _set_mode(self, mode: str, request: str) -> None:
        if not self._ensure_session() or not self.session:
            return
        self.session.mode = mode
        self.sessions.save(self.session)
        self.write(f"Mode: {mode}")
        if request:
            self._chat(request)

    def _context(self) -> None:
        if not self._ensure_session() or not self.session:
            return
        instructions = load_project_instructions(Path(self.session.repo))
        names = [
            name
            for name in ("FORGELOOP.md", "AGENTS.md")
            if (Path(self.session.repo) / name).is_file()
        ]
        budget = self.current_context_budget()
        usage = estimate_tokens(self.session.conversation)

        def show(value: int | None) -> str:
            return "unknown" if value is None else f"{value:,}"

        self.write(
            f"Model context limit: {show(budget.context_window)}\n"
            f"Usable context: {show(budget.usable_context)}\n"
            f"Current usage: ~{usage:,}\n"
            f"Reserved output: {show(budget.reserved_output)}\n"
            f"Thinking/tool reserve: {show(budget.thinking_tool_reserve)}\n"
            f"Safety margin: {show(budget.safety_margin)}\n"
            f"Auto compact threshold: {show(budget.auto_compact_threshold)}\n"
            f"Compactions: {self.session.compact_count}\n"
            f"Instructions: {', '.join(names) or 'none'} ({len(instructions):,} chars)"
        )

    def _compact(self, automatic: bool = False) -> None:
        if not self.session:
            if not automatic:
                self.write("No active Session.")
            return
        compacted, stats = compact_messages(
            self.session.conversation,
            redactor=self.redactor,
            context_state=self.session.context_state,
            force=True,
        )
        self.session.conversation = compacted
        if stats["compacted"]:
            self.session.compact_count += 1
        self.sessions.save(self.session)
        label = "Auto-compact" if automatic else "Compact"
        self.write(
            f"{label}: {stats['before_tokens']:,} -> {stats['after_tokens']:,} estimated tokens; "
            f"{stats['compacted']} messages compacted."
        )

    def _model(self, value: str) -> None:
        if not value:
            provider, model = self.current_model()
            self.write(
                f"Model: {PROVIDERS.get(provider, {}).get('label', provider)} / {model or 'not configured'}"
            )
            return
        available = configured_provider_names(self.config, self.credentials)
        provider, slash, model = value.partition("/")
        if not slash:
            if len(available) != 1:
                self.write(
                    "请通过 /model 选择 Provider；手动输入需使用 provider/model-id。"
                )
                return
            provider, model = available[0], provider
        if provider not in available:
            self.write(f"Provider {provider} 尚未完整配置，请先使用 /api。")
            return
        try:
            self.select_model(provider, model)
        except PreflightError as exc:
            self.write(str(exc))

    def select_model(self, provider: str, model: str) -> None:
        canonical = canonical_model_route(provider, model)
        bare = model_name_from_route(provider, canonical)
        base = provider_api_base(self.config, provider)
        capability = self.capabilities.resolve(provider, base, bare)
        new_budget = context_budget(
            capability, self.session.thinking if self.session else "auto"
        )
        if self.session and new_budget.auto_compact_threshold is not None:
            usage = estimate_tokens(self.session.conversation)
            if usage >= new_budget.auto_compact_threshold:
                self._compact(automatic=True)
                usage = estimate_tokens(self.session.conversation)
            if (
                new_budget.usable_context is not None
                and usage >= new_budget.usable_context
            ):
                raise PreflightError(
                    f"切换已阻止：压缩后约 {usage:,} tokens，仍超过新模型可用上下文 "
                    f"{new_budget.usable_context:,}。"
                )
        old_thinking = self.session.thinking if self.session else "auto"
        if self.session:
            self.session.provider = provider
            self.session.model = bare
            if old_thinking not in capability.thinking_levels:
                self.session.thinking = "auto"
                if old_thinking != "auto":
                    self.write("新模型不支持当前 thinking 等级，已回退到 Auto。")
            self.sessions.save(self.session)
        self.config.provider = provider
        self.config.model = canonical
        self.config.api_base = self.config.provider_configs.get(provider, {}).get(
            "api_base", ""
        )
        self.config_store.save(self.config)
        self.model_cache.remember_manual(provider, base, bare)
        self.write(f"Model: {PROVIDERS[provider]['label']} / {bare}")
        if capability.context_window is None:
            self.write("该模型的 context capability 未知；ForgeLoop 不会伪造限制。")

    def _thinking(self, value: str) -> None:
        if not self.session:
            self.write("Thinking 属于 Session；首次对话或 /new 后再设置。")
            return
        capability = self.current_capability()
        levels = capability.thinking_levels
        if not value:
            self.write(
                f"Thinking: {self.session.thinking} | available: "
                f"{', '.join(levels) if levels else 'unknown'}"
            )
            return
        value = value.lower()
        if value not in levels:
            self.write(
                f"当前模型不支持 thinking: {value}。可选：{', '.join(levels) or 'none'}"
            )
            return
        self.session.thinking = value
        self.sessions.save(self.session)
        self.write(f"Thinking: {value}")

    def _api(self, value: str) -> None:
        provider, _, rest = value.partition(" ")
        if not provider:
            lines = []
            for name, metadata in PROVIDERS.items():
                try:
                    key_set = bool(self.credentials.get_api_key(name))
                except ConfigError:
                    key_set = False
                configured = name in self.config.provider_configs or key_set
                if configured:
                    lines.append(
                        f"{metadata['label']}: key={'configured' if key_set else 'not required/not set'} | "
                        f"base={provider_api_base(self.config, name) or 'not configured'}"
                    )
            self.write(
                "\n".join(lines)
                or "No Provider configured. Use /api <provider> key|base."
            )
            return
        provider = provider.lower()
        if provider not in PROVIDERS:
            self.write(f"Unknown Provider: {provider}")
            return
        action, _, argument = rest.partition(" ")
        if action == "key":
            secret = argument or self.read_secret(f"API key for {provider}: ")
            try:
                self.credentials.set_api_key(provider, secret)
                self.config.provider_configs.setdefault(provider, {})
                self.config_store.save(self.config)
                self.write("API key stored in the OS credential store.")
            except ConfigError as exc:
                self.write(str(exc))
        elif action == "test":
            selected_provider, selected_model = self.current_model()
            model = argument or (
                selected_model if selected_provider == provider else ""
            )
            try:
                test_provider_api(
                    self.config,
                    self.credentials,
                    provider=provider,
                    model=model,
                    thinking_level=self.session.thinking if self.session else "auto",
                    timeout_seconds=min(45, self.config.timeout_seconds),
                )
                settings = self.config.provider_configs.setdefault(provider, {})
                settings["connection_ok"] = True
                self.config_store.save(self.config)
                self.write(f"API test passed: {provider} / {model} | tool calling=ok")
            except (PreflightError, RuntimeError) as exc:
                self.write(f"API test failed: {exc}")
        elif action == "base":
            set_provider_api_base(self.config, provider, argument)
            self.config_store.save(self.config)
            self.write(
                f"API base: {provider_api_base(self.config, provider) or 'not configured'}"
            )
        elif action == "delete":
            try:
                self.credentials.delete_api_key(provider)
                self.config.provider_configs.pop(provider, None)
                if self.config.provider == provider:
                    self.config.model = ""
                    self.config.api_base = ""
                self.config_store.save(self.config)
                self.write(f"Provider config removed: {provider}")
            except ConfigError as exc:
                self.write(str(exc))
        else:
            self.write(
                "Usage: /api <provider> [key [secret]|base URL|test [model]|delete]"
            )

    def usable_providers(self) -> list[str]:
        return configured_provider_names(self.config, self.credentials)

    def cached_models(self, provider: str) -> list[str]:
        return self.model_cache.models(
            provider, provider_api_base(self.config, provider)
        )

    def refresh_models(self, provider: str) -> list[str]:
        models = fetch_provider_models(
            self.config,
            self.credentials,
            provider,
            cache=self.model_cache,
            timeout_seconds=min(30.0, self.config.timeout_seconds),
        )
        self.config_store.save(self.config)
        return models

    def current_model(self) -> tuple[str, str]:
        if self.session and self.session.provider and self.session.model:
            return self.session.provider, self.session.model
        return (
            self.config.provider,
            model_name_from_route(self.config.provider, self.config.model),
        )

    def current_capability(self) -> ModelCapability:
        provider, model = self.current_model()
        if not provider or not model:
            return ModelCapability()
        return self.capabilities.resolve(
            provider, provider_api_base(self.config, provider), model
        )

    def current_context_budget(self):
        thinking = self.session.thinking if self.session else "auto"
        return context_budget(self.current_capability(), thinking)

    def set_api_key(self, provider: str, secret: str) -> None:
        if provider not in PROVIDERS:
            raise ConfigError(f"Unknown Provider: {provider}")
        self.credentials.set_api_key(provider, secret)
        self.config.provider_configs.setdefault(provider, {})
        self.config_store.save(self.config)

    def set_api_base(self, provider: str, value: str) -> None:
        if provider not in PROVIDERS:
            raise ConfigError(f"Unknown Provider: {provider}")
        set_provider_api_base(self.config, provider, value)
        self.config_store.save(self.config)

    def delete_provider_config(self, provider: str) -> bool:
        self.credentials.delete_api_key(provider)
        self.config.provider_configs.pop(provider, None)
        if self.config.provider == provider:
            self.config.model = ""
            self.config.api_base = ""
        self.config_store.save(self.config)
        return bool(self.credentials.get_api_key(provider))

    def _update_context_state(
        self,
        request: str,
        result: RunResult,
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        if not self.session:
            return
        state = dict(self.session.context_state)
        state.setdefault("original_task", request)
        constraints = list(state.get("latest_constraints", []))
        constraints.append(request[:2_000])
        state["latest_constraints"] = constraints[-4:]
        if self.session.mode == "plan":
            state["current_plan"] = result.summary[:4_000]
        evidence = list(state.get("tool_evidence", []))
        tests = list(state.get("test_evidence", []))
        locations = list(state.get("key_locations", []))
        for name, payload in events:
            if name != "tool_finished":
                continue
            tool = str(payload.get("name", "tool"))
            ok = bool(payload.get("ok"))
            output = " ".join(str(payload.get("output", "")).split())[:800]
            evidence.append(f"{tool} {'ok' if ok else 'failed'}: {output}")
            if tool in {"shell", "run_tests"} or not ok:
                tests.append(f"{tool} {'ok' if ok else 'failed'}: {output}")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, dict) and arguments.get("path"):
                locations.append(str(arguments["path"]))
        state["tool_evidence"] = evidence[-12:]
        state["test_evidence"] = tests[-8:]
        state["key_locations"] = list(dict.fromkeys(locations))[-12:]
        state["completed"] = [result.summary[:2_000]]
        state["pending"] = (
            [] if result.status.value == "completed" else [result.stop_reason]
        )
        try:
            state["diff_summary"] = git_output(
                Path(self.session.repo), "diff", "--stat"
            ).strip()[:4_000]
        except GitError:
            pass
        self.session.context_state = self.redactor.redact(state)

    def _runtime(self, value: str) -> None:
        if not self._ensure_session() or not self.session:
            return
        if not value:
            self.write(f"Runtime: {self.session.runtime}")
            return
        if value not in {"local", "docker"}:
            self.write("Runtime must be local or docker.")
            return
        if value == "docker" and not DockerRuntime.available():
            self.write("Docker runtime is not available.")
            return
        self.session.runtime = value
        self.sessions.save(self.session)
        self.write(f"Runtime: {value}")

    def _cost(self) -> None:
        if not self.session:
            self.write("No active Session.")
            return
        usage = self.session.usage
        cost = usage.get("cost_usd")
        self.write(
            f"Steps: {usage.get('steps', 0)} | model calls: {usage.get('model_calls', 0)} | "
            f"tool calls: {usage.get('tool_calls', 0)}\nTokens: {usage.get('total_tokens', 'unknown')} | "
            f"Cost: {'unknown' if cost is None else f'${cost:.6f}'}"
        )

    def _show_config(self) -> None:
        provider, model = self.current_model()
        session = (
            f"session={self.session.id[:8]} repo={self.session.repo} mode={self.session.mode} "
            f"runtime={self.session.runtime} thinking={self.session.thinking}"
            if self.session
            else "session=none"
        )
        self.write(
            f"provider={provider}\nmodel={model or 'not configured'}\n"
            f"api_base={provider_api_base(self.config, provider) or 'not configured'}\n"
            f"budgets: steps={self.config.max_steps} model_calls={self.config.max_model_calls} "
            f"tool_calls={self.config.max_tool_calls} tokens={self.config.max_tokens} "
            f"cost={self.config.max_cost_usd or 'disabled'} timeout={self.config.timeout_seconds}s\n{session}"
        )

    def _config(self, value: str) -> None:
        if not value:
            self._show_config()
            return
        key, _, raw = value.partition(" ")
        allowed: dict[str, type] = {
            "max_steps": int,
            "max_model_calls": int,
            "max_tool_calls": int,
            "max_tokens": int,
            "max_cost_usd": float,
            "timeout_seconds": float,
        }
        converter = allowed.get(key)
        if converter is None or not raw:
            self.write(
                "Usage: /config <max_steps|max_model_calls|max_tool_calls|max_tokens|max_cost_usd|timeout_seconds> <value>"
            )
            return
        try:
            converted = converter(raw)
        except ValueError:
            self.write(f"Invalid value for {key}: {raw}")
            return
        if converted < 0 or (key not in {"max_cost_usd"} and converted == 0):
            self.write(f"{key} must be positive (max_cost_usd may be 0 to disable).")
            return
        setattr(self.config, key, converted)
        self.config_store.save(self.config)
        self.write(f"{key}={converted}")

    def _sessions(self) -> None:
        items = self.sessions.list()
        if not items:
            self.write("No saved Sessions.")
            return
        self.write(
            "\n".join(
                f"{'*' if self.session and item.id == self.session.id else ' '} "
                f"{item.id[:8]} {item.mode:5} {item.updated_at[:19]} {item.repo}"
                for item in items
            )
        )

    def _resume(self, value: str) -> None:
        try:
            if value:
                session = self.sessions.load(value)
            else:
                items = self.sessions.list()
                if not items:
                    self.write("No saved Sessions.")
                    return
                session = items[0]
        except ValueError as exc:
            self.write(str(exc))
            return
        if not Path(session.repo).is_dir() or not is_git_repo(Path(session.repo)):
            self.write(f"Session repository is unavailable: {session.repo}")
            return
        self.session = session
        self.write(f"Resumed {session.id[:8]} | {session.repo} | mode={session.mode}")

    def _new(self, value: str) -> None:
        repo = Path(value).expanduser().resolve() if value else self.cwd
        if not is_git_repo(repo) and not value:
            selected = self.read("Git project path (blank to cancel): ").strip()
            if not selected:
                return
            repo = Path(selected).expanduser().resolve()
        self._create_session(repo)

    def _exit(self) -> None:
        if self.session:
            self.sessions.save(self.session)
        self.running = False

    def _make_runtime(self, name: str) -> Runtime:
        return DockerRuntime() if name == "docker" else LocalRuntime()

    def _record_usage(self, usage: dict) -> None:
        if not self.session:
            return
        totals = dict(self.session.usage)
        for key in (
            "steps",
            "model_calls",
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = usage.get(key)
            if value is not None:
                totals[key] = int(totals.get(key, 0)) + int(value)
        cost = usage.get("cost_usd")
        if cost is None:
            totals.setdefault("cost_usd", None)
        elif "cost_usd" not in totals:
            totals["cost_usd"] = float(cost)
        elif totals["cost_usd"] is not None:
            totals["cost_usd"] = float(totals.get("cost_usd", 0.0)) + float(cost)
        self.session.usage = totals


def run_interactive() -> None:
    from forgeloop.tui import run_tui

    run_tui()
