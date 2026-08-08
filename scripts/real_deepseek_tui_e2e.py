from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path

from forgeloop.config import ConfigStore, CredentialStore
from forgeloop.interactive import InteractiveCLI
from forgeloop.tui import ForgeLoopTUI, TextPanel, ToolCallView


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


async def wait_for_worker(app: ForgeLoopTUI, *, timeout: float = 240.0) -> None:
    started = time.monotonic()
    while app.agent_worker is not None:
        if time.monotonic() - started > timeout:
            raise TimeoutError("TUI worker did not finish")
        await asyncio.sleep(0.05)


async def run() -> dict:
    with tempfile.TemporaryDirectory(prefix="forgeloop-real-tui-") as raw:
        root = Path(raw)
        repo = root / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (repo / "calculator.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a - b\n",
            encoding="utf-8",
        )
        (repo / "test_calculator.py").write_text(
            "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        git(repo, "add", "-A")
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=ForgeLoop E2E",
                "-c",
                "user.email=e2e@forgeloop.local",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

        store = ConfigStore(root / "state")
        config = store.load()
        config.provider = "openai"
        config.model = ""
        config.provider_configs = {"deepseek": {}}
        config.max_steps = 16
        config.max_model_calls = 16
        config.max_tool_calls = 30
        config.timeout_seconds = 240
        store.save(config)
        credentials = CredentialStore()
        controller = InteractiveCLI(
            cwd=repo,
            config_store=store,
            credential_store=credentials,
            write=lambda value: None,
        )

        app = ForgeLoopTUI(controller)
        async with app.run_test(size=(140, 44)):
            app.handle_command("new", "")
            app._launch_model_refresh("deepseek")
            await wait_for_worker(app, timeout=90)
            if len(app.screen_stack) > 1:
                app.pop_screen()
            cached_models = controller.cached_models("deepseek")
            if "deepseek-v4-flash" not in cached_models:
                raise RuntimeError(f"Refresh Models missed V4 Flash: {cached_models}")
            app._select_model("deepseek", "deepseek-v4-flash")
            assert controller.config.model == "deepseek/deepseek-v4-flash"
            app.handle_command("thinking", "high")
            assert controller.session and controller.session.thinking == "high"

            app._launch_api_test("deepseek")
            await wait_for_worker(app, timeout=90)
            screen_text = "\n".join(str(widget.render()) for widget in app.query(".system"))
            if "API test ✓" not in screen_text:
                raise RuntimeError("TUI API test did not pass: " + screen_text[-1000:])

            app.start_request(
                "修复 calculator.py：add(2, 3) 应返回 5。先读取相关文件，做最小修改，"
                "运行 pytest 验证，然后使用 finish 总结。"
            )
            await wait_for_worker(app)
            changed = "return a + b" in (repo / "calculator.py").read_text(encoding="utf-8")
            if not changed:
                raise RuntimeError("Agent did not make the expected repository change")

            app.handle_command("test", "")
            await wait_for_worker(app, timeout=120)
            screen_text = "\n".join(str(widget.render()) for widget in app.query(".system"))
            if "test: exit 0" not in screen_text:
                raise RuntimeError("TUI test workflow did not pass")

            app.handle_command("diff", "")
            await asyncio.sleep(0.05)
            if not isinstance(app.screen, TextPanel) or "return a + b" not in app.screen.content:
                raise RuntimeError("Diff Viewer did not show the repository change")
            app.pop_screen()

            app.handle_command("context", "")
            await asyncio.sleep(0.05)
            if not isinstance(app.screen, TextPanel) or "1,000,000" not in app.screen.content:
                raise RuntimeError("Context panel did not use DeepSeek model capability")
            context_panel = app.screen.content
            app.pop_screen()

            app.handle_command("compact", "")
            await asyncio.sleep(0.05)
            compact_count = controller.session.compact_count if controller.session else 0
            if compact_count < 1:
                raise RuntimeError("Manual compact did not update Session state")
            session_id = controller.session.id if controller.session else ""
            first_messages = len(controller.session.conversation) if controller.session else 0
            first_tools = len(list(app.query(ToolCallView)))
            app.action_save_exit()

        resumed = InteractiveCLI(
            cwd=repo,
            config_store=store,
            credential_store=credentials,
            write=lambda value: None,
        )
        second = ForgeLoopTUI(resumed)
        async with second.run_test(size=(140, 44)):
            second.handle_command("resume", session_id[:8])
            second._select_model("deepseek", "deepseek-v4-flash")
            assert resumed.session and resumed.session.model == "deepseek-v4-flash"
            second.handle_command("plan", "")
            second.start_request(
                "继续这个 Session：读取 calculator.py 并简要说明当前实现，不要修改文件。"
            )
            await wait_for_worker(second)
            resumed_messages = len(resumed.session.conversation) if resumed.session else 0
            second.action_save_exit()

        return {
            "api_test": True,
            "models_refreshed": cached_models,
            "canonical_route": controller.config.model,
            "thinking": controller.session.thinking if controller.session else None,
            "changed": changed,
            "test_passed": True,
            "diff_viewer": True,
            "compact_command": True,
            "compact_count": compact_count,
            "context_panel": context_panel,
            "continued_model": resumed.session.model if resumed.session else None,
            "session_id": session_id[:8],
            "first_messages": first_messages,
            "resumed_messages": resumed_messages,
            "tool_views": first_tools,
            "resumed": resumed_messages > first_messages,
            "usage": controller.session.usage if controller.session else {},
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, sort_keys=True))
