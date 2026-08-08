from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from forgeloop.cli import app
from forgeloop.config import ConfigStore
from forgeloop.context import compact_messages
from forgeloop.gitops import CheckpointManager
from forgeloop.interactive import InteractiveCLI
from forgeloop.models import LiteLLMProvider
from forgeloop.security import SecretRedactor, ShellSafetyPolicy
from forgeloop.sessions import Session, SessionStore
from forgeloop.types import ModelResponse, ModelUsage, ToolCall


class FakeCredentials:
    def __init__(self, secret: str = "test-secret-value") -> None:
        self.secret = secret

    def get_api_key(self, provider: str) -> str:
        del provider
        return self.secret

    def set_api_key(self, provider: str, value: str) -> None:
        del provider
        self.secret = value

    def delete_api_key(self, provider: str) -> None:
        del provider
        self.secret = ""


def init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def commit_all(path: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ForgeLoop Test",
            "-c",
            "user.email=test@forgeloop.local",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )


def test_no_argument_cli_starts_without_repo_or_recent_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGELOOP_HOME", str(tmp_path / "state"))
    launched: list[bool] = []
    monkeypatch.setattr("forgeloop.cli.run_interactive", lambda: launched.append(True))
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert launched == [True]


def test_first_chat_creates_persistent_session_and_keeps_secret_out(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "sample.txt").write_text("before\n", encoding="utf-8")
    config_store = ConfigStore(tmp_path / "state")
    config = config_store.load()
    config.model = "openai/mock-model"
    config_store.save(config)
    seen_tools: list[str] = []

    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, timeout_seconds
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    "done",
                    "finish",
                    {"status": "completed", "summary": "Done", "evidence": "mock"},
                ),
            ),
            usage=ModelUsage(10, 5, 0.001),
        )

    monkeypatch.setattr(LiteLLMProvider, "complete", complete)
    output: list[str] = []
    cli = InteractiveCLI(
        cwd=repo,
        config_store=config_store,
        credential_store=FakeCredentials(),
        write=output.append,
    )
    assert cli.session is None
    cli.handle("Please inspect the project")

    assert cli.session is not None
    assert cli.session.repo == str(repo.resolve())
    assert "apply_patch" in seen_tools
    assert cli.session.conversation[-1]["content"] == "Done"
    persisted = (
        config_store.home.read_text(encoding="utf-8")
        if config_store.home.is_file()
        else ""
    )
    artifacts = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in config_store.home.rglob("*")
        if path.is_file()
    )
    assert "test-secret-value" not in persisted + artifacts


def test_plan_mode_exposes_only_read_only_tools(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    config_store = ConfigStore(tmp_path / "state")
    config = config_store.load()
    config.model = "openai/mock-model"
    config_store.save(config)

    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, timeout_seconds
        names = {tool["function"]["name"] for tool in tools}
        assert "apply_patch" not in names
        assert "shell" not in names
        return ModelResponse(content="Read-only plan", usage=ModelUsage(1, 1, 0.0))

    monkeypatch.setattr(LiteLLMProvider, "complete", complete)
    cli = InteractiveCLI(
        cwd=repo,
        config_store=config_store,
        credential_store=FakeCredentials(),
        write=lambda value: None,
    )
    cli.handle("/plan")
    cli.handle("Plan the change")
    assert cli.session and cli.session.mode == "plan"


def test_checkpoint_undo_restores_preexisting_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    checkpoint = manager.create()
    (repo / "tracked.txt").write_text("after\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    manager.undo(checkpoint.id)

    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "before\n"
    assert not (repo / "new.txt").exists()


def test_compaction_and_session_redaction(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": f"request {index}"} for index in range(12)]
    compacted, stats = compact_messages(messages, keep_recent=4)
    assert stats["compacted"] == 8
    assert compacted[0]["role"] == "system"

    secret = "sk-sensitive-test"
    store = SessionStore(tmp_path, redactor=SecretRedactor((secret,)))
    session = Session.create(tmp_path)
    session.conversation = [{"role": "user", "content": secret}]
    store.save(session)
    raw = store.path_for(session.id).read_text(encoding="utf-8")
    assert secret not in raw
    assert "[REDACTED]" in raw
    assert json.loads(raw)["repo"] == str(tmp_path.resolve())


def test_dangerous_shell_commands_are_blocked() -> None:
    policy = ShellSafetyPolicy()
    assert policy.rejection("git reset --hard HEAD")
    assert policy.rejection("Remove-Item x -Recurse -Force")
    assert policy.rejection("Get-ChildItem Env:")
    assert policy.rejection("pytest") is None


def test_session_cost_accumulates_from_first_known_usage(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    cli = InteractiveCLI(
        cwd=repo,
        config_store=ConfigStore(tmp_path / "state"),
        credential_store=FakeCredentials(),
        write=lambda value: None,
    )
    assert cli._create_session(repo)
    cli._record_usage({"cost_usd": 0.001, "steps": 1})
    cli._record_usage({"cost_usd": 0.002, "steps": 2})
    assert cli.session and cli.session.usage["cost_usd"] == 0.003
    assert cli.session.usage["steps"] == 3


def test_real_repo_conversation_regression_flow(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "real-repo"
    init_repo(repo)
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (repo / "answer.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (repo / "test_answer.py").write_text(
        "from answer import answer\n\ndef test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    commit_all(repo)
    config_store = ConfigStore(tmp_path / "state")
    config = config_store.load()
    config.model = "openai/mock-model"
    config_store.save(config)
    responses = iter(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "edit",
                        "apply_patch",
                        {
                            "path": "answer.py",
                            "old_text": "return 1",
                            "new_text": "return 2",
                        },
                    ),
                ),
                usage=ModelUsage(5, 2, 0.001),
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "done",
                        "finish",
                        {
                            "status": "completed",
                            "summary": "Changed answer",
                            "evidence": "answer.py updated",
                        },
                    ),
                ),
                usage=ModelUsage(5, 2, 0.001),
            ),
            ModelResponse(
                content="Session resumed and context retained.",
                usage=ModelUsage(2, 2, 0.0),
            ),
        ]
    )

    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, tools, timeout_seconds
        return next(responses)

    monkeypatch.setattr(LiteLLMProvider, "complete", complete)
    output: list[str] = []
    first = InteractiveCLI(
        cwd=repo,
        config_store=config_store,
        credential_store=FakeCredentials(),
        write=output.append,
    )
    first.handle("Change answer to two")
    assert "return 2" in (repo / "answer.py").read_text(encoding="utf-8")
    first.handle("/test")
    assert any("test: exit 0" in line for line in output)
    first.handle("/diff")
    assert any("return 2" in line for line in output)
    first.handle("/undo")
    assert "return 1" in (repo / "answer.py").read_text(encoding="utf-8")
    first.handle("/compact")
    assert any(line.startswith("Compact:") for line in output)
    session_id = first.session.id if first.session else ""
    first.handle("/exit")

    second = InteractiveCLI(
        cwd=repo,
        config_store=config_store,
        credential_store=FakeCredentials(),
        write=output.append,
    )
    second.handle(f"/resume {session_id[:8]}")
    second.handle("Continue working")
    assert second.session and second.session.id == session_id
    assert (
        second.session.conversation[-1]["content"]
        == "Session resumed and context retained."
    )
