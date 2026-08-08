from pathlib import Path

from typer.testing import CliRunner

from forgeloop.cli import app
from forgeloop.models import LiteLLMProvider
from forgeloop.types import ModelResponse, ModelUsage, ToolCall


def test_task_cli_runs_with_explicit_model(monkeypatch, tmp_path: Path) -> None:
    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, tools, timeout_seconds
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    "done",
                    "finish",
                    {
                        "status": "completed",
                        "summary": "Checked",
                        "evidence": "Mock run",
                    },
                ),
            ),
            usage=ModelUsage(1, 1, 0.0),
        )

    monkeypatch.setattr(LiteLLMProvider, "complete", complete)
    result = CliRunner().invoke(
        app,
        [
            "task",
            "Check the project",
            "--model",
            "mock/model",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Result: completed" in result.output
    assert list((tmp_path / ".forgeloop" / "runs").glob("*.jsonl"))
