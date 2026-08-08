from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from forgeloop.config import ConfigStore
from forgeloop.interactive import InteractiveCLI
from forgeloop.models import LiteLLMProvider
from forgeloop.tui import (
    SLASH_COMMANDS,
    ComposerInput,
    ForgeLoopTUI,
    SelectionPanel,
    TextPanel,
    ToolCallView,
    ValuePanel,
)
from forgeloop.types import ModelResponse, ModelUsage, ToolCall


class Credentials:
    def get_api_key(self, provider: str) -> str:
        del provider
        return "test-secret"

    def set_api_key(self, provider: str, value: str) -> None:
        del provider, value

    def delete_api_key(self, provider: str) -> None:
        del provider


def make_controller(
    tmp_path: Path, *, git: bool = False, model: bool = False
) -> InteractiveCLI:
    repo = tmp_path / "repo"
    repo.mkdir()
    if git:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    store = ConfigStore(tmp_path / "state")
    config = store.load()
    if model:
        config.provider = "deepseek"
        config.model = "deepseek-v4-flash"
    store.save(config)
    return InteractiveCLI(
        cwd=repo,
        config_store=store,
        credential_store=Credentials(),
        write=lambda value: None,
    )


def test_tui_starts_without_repo_or_recent_session_dashboard(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen_text = " ".join(
                str(widget.render()) for widget in app.query("Static")
            )
            assert "Repo list" not in screen_text
            assert "Recent Sessions" not in screen_text
            assert "未配置" in screen_text
            assert "会话" not in screen_text
            assert "步骤" not in screen_text
            assert "费用" not in screen_text
            prompt = app.query_one("#prompt", ComposerInput)
            assert prompt.region.height == 4
            assert app.query_one("#composer").region.height == 7

    asyncio.run(exercise())


def test_help_modal_and_command_completion(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt")
            prompt.text = "/he"
            await pilot.press("tab")
            assert prompt.text == "/help "
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TextPanel)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, TextPanel)

    asyncio.run(exercise())


def test_command_candidates_filter_and_keep_composer_focus(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path))
        async with app.run_test(size=(120, 30)) as pilot:
            prompt = app.query_one("#prompt", ComposerInput)
            prompt.text = "/"
            await pilot.pause()
            panel = app.query_one("#command-suggestions")
            assert panel.display
            assert app.command_matches == list(SLASH_COMMANDS)
            body = app.query_one("#command-suggestions-body")
            assert "/help" in str(body.render())
            assert "/exit" in str(body.render())
            assert "查看分类帮助" in str(body.render())
            assert prompt.has_focus
            assert app.query_one("#welcome").display
            assert app.query_one("#statusbar").region.bottom <= 30

            await pilot.press("up")
            await pilot.pause()
            assert app.selected_command_suggestion() == "/exit"
            assert panel.scroll_y > 0

            await pilot.press("down", "down", "tab")
            assert prompt.text == "/status "
            await pilot.pause()
            assert not panel.display

            prompt.text = "/thi"
            await pilot.pause()
            assert app.command_matches == ["/thinking"]
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TextPanel)

    asyncio.run(exercise())


def test_multiline_composer_and_conversation_transition(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path))
        async with app.run_test(size=(160, 40)) as pilot:
            prompt = app.query_one("#prompt", ComposerInput)
            prompt.text = "first line"
            prompt.move_cursor((0, len(prompt.text)))
            await pilot.press("shift+enter")
            await pilot.press(*"second")
            assert prompt.text == "first line\nsecond"

            app.add_message("user", "Please fix this issue")
            app._start_agent_turn()
            await pilot.pause()
            app.add_message("agent", "Fixed and verified.")
            await pilot.pause()
            turns = list(app.query(".turn"))
            assert [turn.has_class("user-turn") for turn in turns] == [True, False]
            assert turns[1].has_class("agent-turn")
            assert turns[0].region.y >= 0
            assert turns[1].region.y > turns[0].region.y
            assert not app.query_one("#welcome").display
            assert app.query_one("#composer").region.y > turns[1].region.y

    asyncio.run(exercise())


def test_tui_agent_runs_in_worker_and_renders_tool_status(
    monkeypatch, tmp_path: Path
) -> None:
    controller = make_controller(tmp_path, git=True, model=True)
    repo = controller.cwd
    (repo / "answer.py").write_text("value = 1\n", encoding="utf-8")
    responses = iter(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "edit",
                        "apply_patch",
                        {
                            "path": "answer.py",
                            "old_text": "value = 1",
                            "new_text": "value = 2",
                        },
                    ),
                ),
                usage=ModelUsage(3, 2, 0.001),
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "done",
                        "finish",
                        {"status": "completed", "summary": "修改完成"},
                    ),
                ),
                usage=ModelUsage(3, 2, 0.001),
            ),
        ]
    )

    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, tools, timeout_seconds
        return next(responses)

    monkeypatch.setattr(LiteLLMProvider, "complete", complete)

    async def exercise() -> None:
        app = ForgeLoopTUI(controller)
        async with app.run_test(size=(120, 40)) as pilot:
            app.start_request("把 value 改成 2")
            for _ in range(100):
                await pilot.pause(0.02)
                if app.agent_worker is None:
                    break
            assert app.agent_worker is None
            assert "value = 2" in (repo / "answer.py").read_text(encoding="utf-8")
            assert controller.config.model == "deepseek/deepseek-v4-flash"
            views = list(app.query(ToolCallView))
            assert len(views) == 1
            assert "✓ 编辑完成" in str(views[0].query_one(".tool-summary").render())
            assert views[0].query_one("Collapsible").collapsed
            assert views[0].parent.parent.has_class("agent-turn")
            assert not app.query_one("#welcome").display
            app.handle_command("diff", "")
            await pilot.pause()
            assert isinstance(app.screen, TextPanel)

    asyncio.run(exercise())


def test_non_git_first_prompt_asks_for_project(tmp_path: Path) -> None:
    controller = make_controller(tmp_path)
    git_repo = tmp_path / "selected"
    git_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=git_repo, check=True)

    async def exercise() -> None:
        app = ForgeLoopTUI(controller)
        async with app.run_test(size=(120, 40)) as pilot:
            app.start_request("检查项目")
            await pilot.pause()
            assert isinstance(app.screen, ValuePanel)
            app.pending_request = None
            await app.screen.dismiss(str(git_repo))
            await pilot.pause()
            assert controller.session is not None
            assert controller.session.repo == str(git_repo.resolve())

    asyncio.run(exercise())


def test_ctrl_c_sets_cooperative_interrupt(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path, git=True))
        async with app.run_test(size=(120, 40)) as pilot:
            app.agent_worker = type("Worker", (), {"is_finished": False})()
            app.action_interrupt()
            await pilot.pause()
            assert app.cancel_event.is_set()

    asyncio.run(exercise())


def test_ctrl_c_exits_when_agent_is_idle(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path, git=True))
        async with app.run_test(size=(120, 40)) as pilot:
            exit_calls: list[bool] = []
            app.action_save_exit = lambda: exit_calls.append(True)  # type: ignore[method-assign]
            await pilot.press("ctrl+c")
            assert exit_calls == [True]

    asyncio.run(exercise())


@pytest.mark.parametrize("size", [(120, 30), (160, 40), (200, 50)])
def test_responsive_content_column_and_model_modal(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    async def exercise() -> None:
        app = ForgeLoopTUI(make_controller(tmp_path))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            column = app.query_one("#content-column").region
            welcome = app.query_one("#welcome").region
            composer = app.query_one("#composer").region
            status = app.query_one("#statusbar").region
            assert column.width <= 112
            assert abs((size[0] - column.width) // 2 - column.x) <= 2
            assert welcome.x == composer.x == status.x == column.x
            assert welcome.width == composer.width == status.width == column.width
            assert composer.y == welcome.bottom + 2
            assert column.height < size[1]

            original = composer
            app.handle_command("model", "")
            await pilot.pause()
            assert isinstance(app.screen, SelectionPanel)
            assert app.query_one("#composer").region == original

            await app.screen.dismiss(None)
            app.add_message("user", "Please fix the validation failure.")
            await pilot.pause()
            app._start_agent_turn()
            await pilot.pause()
            app.add_message(
                "agent",
                "Searching\n\nReading\n\nEditing\n\nRunning tests\n\nFixed and verified.",
            )
            await pilot.pause()
            assert app.query_one("#composer").region.bottom < size[1]
            assert app.query_one("#statusbar").region.bottom <= size[1]

    asyncio.run(exercise())
