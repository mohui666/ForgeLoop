from __future__ import annotations

import logging
import re
import threading
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Collapsible, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from forgeloop import __version__
from forgeloop.config import ConfigError, PROVIDERS, provider_api_base
from forgeloop.context import estimate_tokens
from forgeloop.gitops import is_git_repo
from forgeloop.interactive import InteractiveCLI
from forgeloop.provider_config import test_provider_api


COMMAND_DESCRIPTIONS = {
    "/help": "查看分类帮助",
    "/status": "查看会话与 Git 状态",
    "/diff": "查看当前代码差异",
    "/undo": "回退最近一次检查点",
    "/test": "运行测试工作流",
    "/lint": "运行代码检查",
    "/plan": "切换到只读规划模式",
    "/build": "切换到修改模式",
    "/context": "查看上下文详情",
    "/compact": "立即压缩上下文",
    "/model": "选择可用模型",
    "/thinking": "设置推理强度",
    "/api": "配置模型 API",
    "/runtime": "选择运行环境",
    "/cost": "查看令牌与费用",
    "/config": "查看或修改配置",
    "/sessions": "打开会话列表",
    "/resume": "恢复已有会话",
    "/new": "新建会话",
    "/exit": "保存并退出",
}

SLASH_COMMANDS = tuple(COMMAND_DESCRIPTIONS)


HELP_ZH = """# ForgeLoop 帮助

## 会话
`/sessions` 会话面板　`/resume [id]` 恢复　`/new [path]` 新建　`/compact` 压缩上下文　`/exit` 保存退出

## 智能体
`/plan [任务]` 只读规划　`/build [任务]` 修改模式　`/context` 上下文状态　`/cost` 令牌与费用

## 代码
`/status` Git 状态　`/diff` 差异查看器　`/undo` 回退最近检查点　`/test` 测试　`/lint` 检查　`/build run` 构建

## 模型
`/api` 配置服务商 / API 密钥 / 基础 URL / 连接测试 / 删除配置　`/model` 只显示可用服务商与缓存模型　`/thinking` 选择当前模型支持的推理等级

## 运行环境
`/runtime local|docker`　`/config [预算项 值]`

快捷键：`Ctrl+P` 命令面板　`Ctrl+C` 运行时中断 / 空闲时退出　`Esc` 关闭面板　`Ctrl+D` 保存退出
"""


class AgentEvent(Message):
    def __init__(self, event_type: str, payload: dict[str, Any]) -> None:
        super().__init__()
        self.event_type = event_type
        self.payload = payload


class ControllerOutput(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ComposerSubmitted(Message):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value


class ComposerInput(TextArea):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_history: list[str] = []
        self.prompt_history_index = 0

    def remember(self, value: str) -> None:
        if value and (not self.prompt_history or self.prompt_history[-1] != value):
            self.prompt_history.append(value)
        self.prompt_history_index = len(self.prompt_history)

    async def _on_key(self, event: events.Key) -> None:
        value = self.text
        suggestions_visible = bool(
            getattr(self.app, "command_suggestions_visible", False)
        )
        if event.key == "enter":
            if value.startswith("/") and " " not in value:
                matches = [
                    command for command in SLASH_COMMANDS if command.startswith(value)
                ]
                match = self.app.selected_command_suggestion()
                if match not in matches:
                    match = matches[0] if matches else None
                if suggestions_visible or len(matches) == 1:
                    value = match or value
            if value.strip():
                self.post_message(ComposerSubmitted(value.strip()))
            event.prevent_default()
            event.stop()
            return
        if event.key in {"shift+enter", "ctrl+j"}:
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            event.prevent_default()
            event.stop()
            return
        if event.key in {"up", "down"} and suggestions_visible:
            self.app.move_command_suggestion(-1 if event.key == "up" else 1)
            event.prevent_default()
            event.stop()
            return
        if event.key == "escape" and suggestions_visible:
            self.app.hide_command_suggestions()
            event.prevent_default()
            event.stop()
            return
        if event.key == "up" and "\n" not in value and self.prompt_history:
            self.prompt_history_index = max(0, self.prompt_history_index - 1)
            self.text = self.prompt_history[self.prompt_history_index]
            lines = self.text.split("\n")
            self.move_cursor((len(lines) - 1, len(lines[-1])))
            event.prevent_default()
            event.stop()
            return
        if event.key == "down" and "\n" not in value and self.prompt_history:
            self.prompt_history_index = min(
                len(self.prompt_history), self.prompt_history_index + 1
            )
            self.text = (
                ""
                if self.prompt_history_index == len(self.prompt_history)
                else self.prompt_history[self.prompt_history_index]
            )
            lines = self.text.split("\n")
            self.move_cursor((len(lines) - 1, len(lines[-1])))
            event.prevent_default()
            event.stop()
            return
        if event.key == "tab" and value.startswith("/") and "\n" not in value:
            matches = [
                command for command in SLASH_COMMANDS if command.startswith(value)
            ]
            match = self.app.selected_command_suggestion()
            if match not in matches:
                match = matches[0] if matches else None
            if match:
                self.text = match + " "
                self.move_cursor((0, len(self.text)))
            event.prevent_default()
            event.stop()
            return
        await super()._on_key(event)

    def on_focus(self) -> None:
        if self.parent:
            self.parent.parent.add_class("composer-focused")

    def on_blur(self) -> None:
        if self.parent:
            self.parent.parent.remove_class("composer-focused")


class TextPanel(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "关闭", show=False)]

    def __init__(self, title: str, content: str, *, diff: bool = False) -> None:
        super().__init__()
        self.panel_title = title
        self.content = content
        self.diff = diff

    def compose(self) -> ComposeResult:
        body: Static | Markdown
        if self.diff:
            body = Static(Text.from_ansi(self.content), id="panel-body")
        else:
            body = Markdown(self.content, id="panel-body")
        with Container(id="modal-card"):
            yield Static(self.panel_title, classes="modal-title")
            with VerticalScroll(id="modal-scroll"):
                yield body
            yield Static("Esc 关闭", classes="modal-hint")

    def action_dismiss(self) -> None:
        self.dismiss()


class SelectionPanel(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(self, title: str, options: Iterable[tuple[str, str]]) -> None:
        super().__init__()
        self.panel_title = title
        self.options = list(options)

    def compose(self) -> ComposeResult:
        with Container(id="selection-card"):
            yield Static(self.panel_title, classes="modal-title")
            yield OptionList(
                *(Option(label, id=value) for value, label in self.options),
                id="selection-list",
            )
            yield Static("Enter 选择 · Esc 取消", classes="modal-hint")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ValuePanel(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def __init__(self, title: str, placeholder: str, *, password: bool = False) -> None:
        super().__init__()
        self.panel_title = title
        self.placeholder = placeholder
        self.password = password

    def compose(self) -> ComposeResult:
        with Container(id="value-card"):
            yield Static(self.panel_title, classes="modal-title")
            yield Input(
                placeholder=self.placeholder, password=self.password, id="value-input"
            )
            yield Static("Enter 确认 · Esc 取消", classes="modal-hint")

    def on_mount(self) -> None:
        self.query_one("#value-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ToolCallView(Vertical):
    def __init__(self, call_id: str, title: str) -> None:
        super().__init__(classes="tool-call")
        self.call_id = re.sub(r"[^a-zA-Z0-9_-]", "-", call_id)
        self.initial_title = title

    def compose(self) -> ComposeResult:
        yield Static(
            f"● {self.initial_title}",
            classes="tool-summary",
            id=f"tool-label-{self.call_id}",
        )
        details = Collapsible(
            Static("等待结果…", markup=False, id=f"tool-detail-{self.call_id}"),
            title="详情",
            collapsed=True,
            id=f"tool-more-{self.call_id}",
        )
        details.display = False
        yield details

    def finish(self, *, ok: bool, summary: str, detail: str) -> None:
        icon = "✓" if ok else "✗"
        if self.initial_title == "正在运行测试":
            summary = "测试通过" if ok else "测试失败"
        self.query_one(f"#tool-label-{self.call_id}", Static).update(
            f"{icon} {summary}"
        )
        self.query_one(f"#tool-detail-{self.call_id}", Static).update(
            detail or "无输出"
        )
        collapsible = self.query_one(f"#tool-more-{self.call_id}", Collapsible)
        collapsible.title = "详情"
        collapsible.display = bool(detail.strip()) and (not ok or len(detail) > 400)
        if not ok:
            self.add_class("tool-error")


class ForgeLoopTUI(App[None]):
    TITLE = "ForgeLoop"
    SUB_TITLE = "编程智能体"
    COMMAND_PALETTE_BINDING = "ctrl+p"
    BINDINGS = [
        Binding("ctrl+c", "interrupt", "中断或退出", show=False, priority=True),
        Binding("escape", "close_panel", "关闭面板", show=False, priority=True),
        Binding("ctrl+d", "save_exit", "保存退出", show=False, priority=True),
    ]

    CSS = """
    Screen { background: #0d0f12; color: #e4e7eb; }
    #viewport { width: 100%; height: 1fr; align-horizontal: center; padding: 1 2 0 2; }
    #content-column { width: 112; max-width: 100%; height: auto; max-height: 100%; }
    #conversation { width: 100%; height: auto; max-height: 60%; scrollbar-size: 1 1; }
    #feed { width: 100%; height: auto; }
    #welcome {
        width: 100%; height: 10; margin: 0 0 1 0; padding: 1 2;
        border: round #3d4654; background: #15181d; color: #d8dde5;
    }
    .turn { width: 100%; height: auto; margin: 0 0 1 0; padding: 0 1; }
    .turn-label { height: 1; margin: 0 0 1 0; text-style: bold; color: #f2f4f7; }
    .message-body { width: 100%; height: auto; color: #d7dce4; }
    .user-turn { padding: 1 1; background: #141820; border-left: thick #718cff; }
    .agent-turn { padding: 1 1 0 1; }
    .agent-body { width: 100%; height: auto; }
    .system { width: 100%; height: auto; margin: 0 0 1 0; padding: 0 1; color: #aab3c0; }
    .error { color: #ff9b9b; border-left: thick #d65757; }
    .turn-meta { height: auto; margin: 1 0 0 0; color: #7f8998; }
    .tool-call { width: 100%; height: auto; margin: 0 0 1 0; color: #bac3d0; }
    .tool-summary { height: 1; }
    .tool-error .tool-summary { color: #ff9c9c; }
    Collapsible { width: 100%; padding: 0; border: none; background: #111419; }
    CollapsibleTitle { color: #7f8998; }
    #command-suggestions {
        width: 100%; height: auto; max-height: 7; margin: 1 0 0 0; padding: 0 1;
        border: round #3d4654; background: #12161c; color: #bdc6d3;
        scrollbar-size: 1 1;
    }
    #command-suggestions-body { width: 100%; height: auto; }
    #composer {
        width: 100%; height: 7; margin: 1 0 0 0; padding: 0 1;
        border: round #4a5566; background: #15181d;
    }
    #composer.composer-focused { border: round #7896ff; background: #171b22; }
    #composer-row { width: 100%; height: 4; }
    #prompt-glyph { width: 3; height: 4; padding: 0; color: #8ea5ff; text-style: bold; }
    #prompt { width: 1fr; height: 4; border: none; background: transparent; color: #f0f2f5; padding: 0; }
    #prompt:focus { background: transparent; }
    #composer-hint { width: 100%; height: 1; color: #707b87; text-align: right; }
    #statusbar { width: 100%; height: 2; padding: 1 1 0 1; color: #8993a2; }
    TextPanel, SelectionPanel, ValuePanel { align: center middle; background: rgba(5, 7, 10, 0.72); }
    #modal-card, #selection-card { width: 110; max-width: 94%; height: 82%; padding: 1 2; background: #191d24; border: round #46536a; }
    #value-card { width: 76; max-width: 94%; height: 9; padding: 1 2; background: #191d24; border: round #46536a; }
    .modal-title { height: 2; text-style: bold; color: #dce3ed; }
    .modal-hint { dock: bottom; height: 1; color: #707b8c; text-align: right; }
    #modal-scroll, #selection-list { height: 1fr; }
    #panel-body { width: 100%; }
    """

    def __init__(self, controller: InteractiveCLI | None = None) -> None:
        super().__init__()
        self.controller = controller or InteractiveCLI(write=lambda value: None)
        self.controller.write = lambda value: self.post_message(
            ControllerOutput(str(value))
        )
        self.cancel_event = threading.Event()
        self.agent_worker: Any = None
        self.tool_views: dict[str, ToolCallView] = {}
        self.last_error_details = ""
        self.pending_request: str | None = None
        self.pending_mode: str | None = None
        self.pending_api_provider: str | None = None
        self.pending_model_provider: str | None = None
        self.active_agent_turn: Vertical | None = None
        self.active_agent_body: Vertical | None = None
        self.command_matches: list[str] = []
        self.command_suggestion_index = 0

    def compose(self) -> ComposeResult:
        with Container(id="viewport"):
            with Vertical(id="content-column"):
                with VerticalScroll(id="conversation"):
                    yield Static(id="welcome")
                    yield Vertical(id="feed")
                with VerticalScroll(id="command-suggestions"):
                    yield Static(id="command-suggestions-body")
                with Container(id="composer"):
                    with Horizontal(id="composer-row"):
                        yield Static(">", id="prompt-glyph")
                        yield ComposerInput(
                            placeholder="描述你想构建的内容…",
                            soft_wrap=True,
                            show_line_numbers=False,
                            id="prompt",
                        )
                    yield Static(
                        "Enter 发送 · Shift+Enter / Ctrl+J 换行",
                        id="composer-hint",
                    )
                yield Static(id="statusbar")

    def on_mount(self) -> None:
        for logger_name in ("LiteLLM", "litellm", "httpx", "httpcore"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        self.query_one("#command-suggestions", VerticalScroll).display = False
        self._resize_conversation(self.size.height)
        self.refresh_chrome()
        self.query_one("#prompt", ComposerInput).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._resize_conversation(event.size.height)

    def _resize_conversation(self, terminal_height: int) -> None:
        suggestion_rows = (
            min(5, len(self.command_matches)) + 2 if self.command_matches else 0
        )
        self.query_one("#conversation", VerticalScroll).styles.max_height = max(
            10, terminal_height - 13 - suggestion_rows
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "prompt":
            self._update_command_suggestions(event.text_area.text)

    @property
    def command_suggestions_visible(self) -> bool:
        return bool(self.command_matches)

    def _update_command_suggestions(self, value: str) -> None:
        if value.startswith("/") and " " not in value and "\n" not in value:
            prefix = value.lower()
            self.command_matches = [
                command for command in SLASH_COMMANDS if command.startswith(prefix)
            ]
        else:
            self.command_matches = []
        self.command_suggestion_index = min(
            self.command_suggestion_index, max(0, len(self.command_matches) - 1)
        )
        self._render_command_suggestions()

    def _render_command_suggestions(self) -> None:
        panel = self.query_one("#command-suggestions", VerticalScroll)
        body = self.query_one("#command-suggestions-body", Static)
        panel.display = bool(self.command_matches)
        if self.command_matches:
            content = Text()
            for index, command in enumerate(self.command_matches):
                selected = index == self.command_suggestion_index
                glyph = "›" if selected else " "
                line = f"{glyph} {command:<12} {COMMAND_DESCRIPTIONS[command]}"
                content.append(line, style="bold reverse" if selected else "")
                if index < len(self.command_matches) - 1:
                    content.append("\n")
            body.update(content)
            panel.call_after_refresh(
                panel.scroll_to,
                y=max(0, self.command_suggestion_index - 2),
                animate=False,
            )
        else:
            body.update("")
        self._resize_conversation(self.size.height)

    def move_command_suggestion(self, direction: int) -> None:
        if self.command_matches:
            self.command_suggestion_index = (
                self.command_suggestion_index + direction
            ) % len(self.command_matches)
            self._render_command_suggestions()

    def selected_command_suggestion(self) -> str | None:
        if not self.command_matches:
            return None
        return self.command_matches[self.command_suggestion_index]

    def hide_command_suggestions(self) -> None:
        self.command_matches = []
        self.command_suggestion_index = 0
        self._render_command_suggestions()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        for command, help_text in COMMAND_DESCRIPTIONS.items():
            yield SystemCommand(
                command, help_text, lambda value=command: self.submit_text(value)
            )

    def on_composer_submitted(self, event: ComposerSubmitted) -> None:
        value = event.value.strip()
        if not value:
            return
        prompt = self.query_one("#prompt", ComposerInput)
        prompt.remember(value)
        prompt.text = ""
        self.submit_text(value)

    def submit_text(self, value: str) -> None:
        if value.startswith("/"):
            command, _, argument = value[1:].partition(" ")
            self.handle_command(command.lower(), argument.strip())
        else:
            self.start_request(value)

    def start_request(self, request: str) -> None:
        if self.agent_worker and not self.agent_worker.is_finished:
            self.add_system("智能体正在工作；Ctrl+C 可请求中断。", error=True)
            return
        if not self.controller.session:
            if is_git_repo(self.controller.cwd):
                if not self.controller._create_session(self.controller.cwd):
                    return
            else:
                self.pending_request = request
                self.push_screen(
                    ValuePanel("选择 Git 项目", "项目目录的绝对路径"),
                    self._project_selected,
                )
                return
        self._launch_agent(request)

    def _project_selected(self, value: str | None) -> None:
        request, self.pending_request = self.pending_request, None
        mode, self.pending_mode = self.pending_mode, None
        if not value:
            return
        if self.controller._create_session(Path(value).expanduser().resolve()):
            if mode:
                self._apply_mode(mode)
            if request:
                self.set_timer(0.2, lambda: self._launch_agent(request))

    def _launch_agent(self, request: str) -> None:
        self.add_message("user", request)
        self._start_agent_turn()
        self.cancel_event.clear()
        self.set_working(True)
        self.agent_worker = self.run_worker(
            lambda: self._agent_thread(request),
            thread=True,
            exit_on_error=False,
            name="agent-turn",
        )

    def _agent_thread(self, request: str) -> None:
        try:
            result = self.controller.chat(
                request,
                event_sink=lambda kind, payload: self.post_message(
                    AgentEvent(kind, payload)
                ),
                cancel_check=self.cancel_event.is_set,
                emit_output=False,
            )
            if result:
                self.call_from_thread(self._agent_result, result)
        except Exception as exc:  # noqa: BLE001 - keep worker failures out of the terminal
            self.last_error_details = traceback.format_exc()
            self.call_from_thread(self.add_system, f"智能体运行失败：{exc}", True)
        finally:
            self.call_from_thread(self.set_working, False)

    def _agent_result(self, result: Any) -> None:
        self.add_message("agent", result.summary)
        if result.evidence:
            self._mount_agent_widget(
                Static(f"依据：{result.evidence}", classes="system", markup=False)
            )
        if result.status.value != "completed":
            classes = "system"
            if result.status.value not in {"interrupted", "blocked"}:
                classes += " error"
            self._mount_agent_widget(
                Static(
                    f"已停止：{result.status.value}（{result.stop_reason}）",
                    classes=classes,
                    markup=False,
                )
            )
        meta = self._turn_meta(result)
        if meta:
            self._mount_agent_widget(Static(meta, classes="turn-meta", markup=False))
        self.active_agent_turn = None
        self.active_agent_body = None
        self.refresh_chrome()

    def on_agent_event(self, event: AgentEvent) -> None:
        kind, payload = event.event_type, event.payload
        if kind == "tool_started":
            title = self._tool_title(payload["name"], payload.get("arguments", {}))
            view = ToolCallView(payload["id"], title)
            self.tool_views[payload["id"]] = view
            self._mount_agent_widget(view)
        elif kind == "tool_finished":
            view = self.tool_views.get(payload["id"])
            if view:
                view.finish(
                    ok=bool(payload["ok"]),
                    summary=self._tool_result_summary(payload),
                    detail=self._truncate_detail(str(payload.get("output", ""))),
                )
        elif kind == "error":
            self.last_error_details = str(payload.get("details", ""))
        elif kind == "run_finished":
            self.refresh_chrome(payload.get("budget", {}).get("usage"))

    def on_controller_output(self, event: ControllerOutput) -> None:
        self.add_system(event.text)
        self.refresh_chrome()

    def handle_command(self, command: str, argument: str) -> None:
        if command == "help":
            self.push_screen(TextPanel("帮助", HELP_ZH))
        elif command in {"status", "context", "cost", "config"} and not argument:
            self._open_capture_panel(command)
        elif command == "diff":
            self._open_capture_panel("diff", diff=True)
        elif command == "sessions":
            self._open_sessions()
        elif command == "model" and not argument:
            self._open_model_panel()
        elif command == "thinking" and not argument:
            self._open_thinking_panel()
        elif command == "thinking":
            self.controller._thinking(argument)
            self.refresh_chrome()
        elif command == "api" and not argument:
            self._open_api_panel()
        elif command in {"test", "lint"} or (command == "build" and argument == "run"):
            kind = command if command != "build" else "build"
            labels = {"test": "测试工作流", "lint": "代码检查", "build": "构建工作流"}
            self._launch_controller_work(
                lambda: self.controller._workflow(kind), labels[kind]
            )
        elif command == "undo":
            self._launch_controller_work(self.controller._undo, "撤销")
        elif command in {"plan", "build"}:
            self._switch_mode(command, argument)
        elif command == "exit":
            self.action_save_exit()
        elif command == "details":
            self.push_screen(
                TextPanel("错误详情", self.last_error_details or "暂无错误详情。")
            )
        elif command == "new" and not argument and not is_git_repo(self.controller.cwd):
            self.push_screen(ValuePanel("新建会话", "Git 项目目录"), self._new_project)
        else:
            self.controller.handle(
                "/" + command + ((" " + argument) if argument else "")
            )
            self.refresh_chrome()

    def _switch_mode(self, mode: str, request: str) -> None:
        if not self.controller.session:
            if is_git_repo(self.controller.cwd):
                if not self.controller._create_session(self.controller.cwd):
                    return
            else:
                self.pending_mode = mode
                self.pending_request = request or None
                self.push_screen(
                    ValuePanel("选择 Git 项目", "项目目录的绝对路径"),
                    self._project_selected,
                )
                return
        self._apply_mode(mode)
        if request:
            self.start_request(request)

    def _apply_mode(self, mode: str) -> None:
        if not self.controller.session:
            return
        self.controller.session.mode = mode
        self.controller.sessions.save(self.controller.session)
        self.add_system(f"模式：{mode.upper()}")
        self.refresh_chrome()

    def _launch_controller_work(self, callback: Callable[[], None], label: str) -> None:
        if self.agent_worker and not self.agent_worker.is_finished:
            self.add_system("智能体正在工作；请先中断或等待完成。", error=True)
            return
        self.set_working(True, label)

        def run() -> None:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                self.last_error_details = traceback.format_exc()
                self.call_from_thread(self.add_system, f"{label} 失败：{exc}", True)
            finally:
                self.call_from_thread(self.set_working, False)
                self.call_from_thread(self.refresh_chrome)

        self.agent_worker = self.run_worker(
            run, thread=True, exit_on_error=False, name=label
        )

    def _launch_api_test(self, provider: str | None = None) -> None:
        self.add_system("正在验证身份认证、模型访问与工具调用…")

        def probe() -> None:
            try:
                current_provider, current_model = self.controller.current_model()
                selected = provider or current_provider
                if selected != current_provider or not current_model:
                    raise ValueError("请先通过 /model 为该服务商选择模型。")
                test_provider_api(
                    self.controller.config,
                    self.controller.credentials,
                    provider=selected,
                    model=current_model,
                    thinking_level=(
                        self.controller.session.thinking
                        if self.controller.session
                        else "auto"
                    ),
                    timeout_seconds=min(45, self.controller.config.timeout_seconds),
                )
                self.call_from_thread(
                    self.add_system,
                    f"API 测试通过 ✓ {PROVIDERS[selected]['label']} / {current_model} · 工具调用可用",
                    False,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error_details = getattr(
                    exc, "details", traceback.format_exc()
                )
                self.call_from_thread(
                    self.add_system, f"API 测试失败 ✗ {exc}（/details 查看详情）", True
                )
            finally:
                self.call_from_thread(self.set_working, False)

        self.set_working(True, "API 测试")
        self.agent_worker = self.run_worker(
            probe, thread=True, exit_on_error=False, name="api-test"
        )

    def _open_capture_panel(self, command: str, *, diff: bool = False) -> None:
        if not self.controller.session and command not in {"config"}:
            self.push_screen(
                TextPanel(
                    self._command_panel_title(command),
                    "当前没有会话。首次自然语言输入时会自动创建。",
                )
            )
            return
        output = self._capture(lambda: self.controller.handle(f"/{command}"))
        self.push_screen(
            TextPanel(
                self._command_panel_title(command), output or "无内容。", diff=diff
            )
        )

    @staticmethod
    def _command_panel_title(command: str) -> str:
        return {
            "status": "状态",
            "diff": "代码差异",
            "context": "上下文",
            "cost": "令牌与费用",
            "config": "配置",
        }.get(command, command)

    def _capture(self, callback: Callable[[], None]) -> str:
        lines: list[str] = []
        original = self.controller.write
        self.controller.write = lambda value: lines.append(str(value))
        try:
            callback()
        finally:
            self.controller.write = original
        return "\n".join(lines)

    def _open_sessions(self) -> None:
        sessions = self.controller.sessions.list()
        if not sessions:
            self.push_screen(TextPanel("会话", "暂无保存的会话。"))
            return
        options = [
            (
                item.id,
                f"{item.id[:8]}  {item.mode.upper():5}  {Path(item.repo).name}  {item.updated_at[:19]}",
            )
            for item in sessions
        ]
        self.push_screen(SelectionPanel("会话", options), self._resume_selected)

    def _resume_selected(self, session_id: str | None) -> None:
        if session_id:
            self.controller._resume(session_id)
            self.refresh_chrome()

    def _open_model_panel(self) -> None:
        providers = self.controller.usable_providers()
        if not providers:
            self.push_screen(
                TextPanel("模型", "当前没有配置完整且可用的服务商。请先使用 `/api`。")
            )
            return
        current, _ = self.controller.current_model()
        options = [
            (
                provider,
                f"{'● ' if provider == current else ''}{PROVIDERS[provider]['label']}",
            )
            for provider in providers
        ]
        self.push_screen(
            SelectionPanel("选择已配置的服务商", options),
            self._model_provider_selected,
        )

    def _model_provider_selected(self, provider: str | None) -> None:
        if not provider:
            return
        self.pending_model_provider = provider
        self._open_models_for_provider(provider)

    def _open_models_for_provider(self, provider: str) -> None:
        cached = self.controller.cached_models(provider)
        options = [(model, model) for model in cached]
        options.extend(
            [
                ("__refresh__", "↻ 刷新模型列表"),
                ("__manual__", "手动输入模型 ID…"),
            ]
        )
        suffix = f" · 已缓存 {len(cached)} 个" if cached else " · 暂无缓存"
        self.push_screen(
            SelectionPanel(f"{PROVIDERS[provider]['label']} 模型{suffix}", options),
            self._model_selected,
        )

    def _model_selected(self, model: str | None) -> None:
        provider = self.pending_model_provider
        if not provider:
            return
        if model == "__refresh__":
            self._launch_model_refresh(provider)
        elif model == "__manual__":
            self.push_screen(
                ValuePanel("模型 ID", "输入服务商返回的真实模型 ID"),
                self._custom_model,
            )
        elif model:
            self._select_model(provider, model)

    def _custom_model(self, model: str | None) -> None:
        if model and self.pending_model_provider:
            self._select_model(self.pending_model_provider, model)

    def _select_model(self, provider: str, model: str) -> None:
        try:
            self.controller.select_model(provider, model)
            self.refresh_chrome()
        except Exception as exc:  # noqa: BLE001 - concise TUI boundary
            self.last_error_details = getattr(exc, "details", traceback.format_exc())
            self.add_system(f"模型切换失败：{exc}", error=True)

    def _launch_model_refresh(self, provider: str) -> None:
        self.add_system(f"正在从 {PROVIDERS[provider]['label']} 获取模型…")

        def refresh() -> None:
            try:
                models = self.controller.refresh_models(provider)
                self.call_from_thread(
                    self.add_system,
                    f"模型列表刷新成功 ✓ 共 {len(models)} 个，已更新本地缓存。",
                    False,
                )
                self.call_from_thread(self._open_models_for_provider, provider)
            except Exception as exc:  # noqa: BLE001
                self.last_error_details = getattr(
                    exc, "details", traceback.format_exc()
                )
                cached = self.controller.cached_models(provider)
                self.call_from_thread(
                    self.add_system,
                    f"刷新模型列表失败：{exc}。{'继续使用旧缓存。' if cached else '当前没有旧缓存。'}",
                    True,
                )
                if cached:
                    self.call_from_thread(self._open_models_for_provider, provider)
            finally:
                self.call_from_thread(self.set_working, False)

        self.set_working(True, "刷新模型列表")
        self.agent_worker = self.run_worker(
            refresh, thread=True, exit_on_error=False, name="refresh-models"
        )

    def _open_thinking_panel(self) -> None:
        session = self.controller.session
        if not session:
            self.push_screen(
                TextPanel("推理强度", "推理强度属于具体会话；首次对话后再设置。")
            )
            return
        levels = self.controller.current_capability().thinking_levels
        if not levels:
            self.push_screen(TextPanel("推理强度", "当前模型的推理能力未知或不支持。"))
            return
        labels = {
            "auto": "自动",
            "low": "低",
            "medium": "中",
            "high": "高",
            "max": "最大",
        }
        options = [
            (level, f"{'● ' if level == session.thinking else ''}{labels[level]}")
            for level in levels
        ]
        self.push_screen(SelectionPanel("推理强度", options), self._thinking_selected)

    def _thinking_selected(self, level: str | None) -> None:
        if level:
            self.controller._thinking(level)
            self.refresh_chrome()

    def _open_api_panel(self) -> None:
        options = []
        for provider, metadata in PROVIDERS.items():
            configured = provider in self.controller.config.provider_configs
            try:
                configured = configured or bool(
                    self.controller.credentials.get_api_key(provider)
                )
            except ConfigError:
                pass
            options.append(
                (provider, f"{'● ' if configured else ''}{metadata['label']}")
            )
        self.push_screen(
            SelectionPanel("API · 服务商", options), self._api_provider_selected
        )

    def _api_provider_selected(self, provider: str | None) -> None:
        if not provider:
            return
        self.pending_api_provider = provider
        base = provider_api_base(self.controller.config, provider) or "未配置"
        options = [
            ("key", "设置或替换 API 密钥"),
            ("base", f"基础 URL · {base}"),
            ("test", "连接测试 · 认证 + 模型 + 工具调用"),
            ("delete", "删除服务商配置"),
        ]
        self.push_screen(
            SelectionPanel(f"API · {PROVIDERS[provider]['label']}", options),
            self._api_action_selected,
        )

    def _api_action_selected(self, action: str | None) -> None:
        provider = self.pending_api_provider
        if not action or not provider:
            return
        if action == "key":
            self.push_screen(
                ValuePanel(
                    f"{PROVIDERS[provider]['label']} API 密钥",
                    "安全保存；不会写入会话、上下文、轨迹或日志",
                    password=True,
                ),
                self._store_api_key,
            )
        elif action == "base":
            self.push_screen(
                ValuePanel("基础 URL", "服务商 API 地址"), self._store_api_base
            )
        elif action == "test":
            self._launch_api_test(provider)
        elif action == "delete":
            try:
                environment_remains = self.controller.delete_provider_config(provider)
                suffix = (
                    " 环境变量中的 API 密钥仍然有效。" if environment_remains else ""
                )
                self.add_system(
                    f"已删除 {PROVIDERS[provider]['label']} 服务商配置。{suffix}"
                )
                self.refresh_chrome()
            except ConfigError as exc:
                self.add_system(str(exc), error=True)

    def _store_api_key(self, value: str | None) -> None:
        if not value or not self.pending_api_provider:
            return
        try:
            self.controller.set_api_key(self.pending_api_provider, value)
            self.add_system("API 密钥已保存到操作系统凭据存储。")
        except ConfigError as exc:
            self.add_system(str(exc), error=True)

    def _store_api_base(self, value: str | None) -> None:
        if value is None or not self.pending_api_provider:
            return
        try:
            self.controller.set_api_base(self.pending_api_provider, value)
            self.add_system(f"基础 URL 已更新：{value}")
        except ConfigError as exc:
            self.add_system(str(exc), error=True)

    def _new_project(self, value: str | None) -> None:
        if value:
            self.controller._create_session(Path(value).expanduser().resolve())
            self.refresh_chrome()

    def add_message(self, role: str, content: str) -> None:
        self.query_one("#welcome", Static).display = False
        if role == "user":
            turn = Vertical(
                Static("你", classes="turn-label"),
                Markdown(content, classes="message-body"),
                classes="turn user-turn",
            )
            self._mount_conversation(turn)
            return
        if self.active_agent_turn is None:
            self._start_agent_turn()
        self._mount_agent_widget(Markdown(content, classes="message-body"))

    def _start_agent_turn(self) -> None:
        body = Vertical(classes="agent-body")
        turn = Vertical(
            Static("ForgeLoop", classes="turn-label"),
            body,
            classes="turn agent-turn",
        )
        self.active_agent_turn = turn
        self.active_agent_body = body
        self._mount_conversation(turn)

    def _mount_agent_widget(self, widget: Any) -> None:
        if self.active_agent_turn is None:
            self._start_agent_turn()
        assert self.active_agent_body is not None
        self.active_agent_body.mount(widget)
        self._scroll_feed()

    def add_system(self, content: str, error: bool = False) -> None:
        classes = "message system" + (" error" if error else "")
        self._mount_conversation(Static(content, classes=classes, markup=False))

    def _mount_conversation(self, widget: Any) -> None:
        feed = self.query_one("#feed", Vertical)
        feed.mount(widget)
        self._scroll_feed()

    def _scroll_feed(self) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.call_after_refresh(conversation.scroll_end, animate=False)

    def _turn_meta(self, result: Any) -> str:
        usage = (
            result.budget.get("usage", {}) if isinstance(result.budget, dict) else {}
        )
        parts: list[str] = []
        if self.controller.session:
            diff = str(
                self.controller.session.context_state.get("diff_summary", "")
            ).strip()
            if diff:
                summary = diff.splitlines()[-1].strip()
                if summary:
                    parts.append(summary)
        steps = int(usage.get("steps") or 0)
        if steps:
            parts.append(f"{steps} 步")
        cost = usage.get("cost_usd")
        if cost:
            parts.append(f"${float(cost):.4f}")
        return " · ".join(parts)

    def refresh_chrome(self, current_usage: dict[str, Any] | None = None) -> None:
        del current_usage
        session = self.controller.session
        mode = session.mode.upper() if session else "BUILD"
        _, model = self.controller.current_model()
        model_display = self._display_model(model) if model else "未选择模型"
        thinking = session.thinking if session else "auto"
        thinking_display = {
            "auto": "自动",
            "low": "低",
            "medium": "中",
            "high": "高",
            "max": "最大",
        }.get(thinking, thinking)
        welcome_model = model_display if model else "未配置"
        self.query_one("#welcome", Static).update(
            "[b]ForgeLoop[/b]\n"
            "[dim]描述你想构建的内容，或输入 /help[/dim]\n\n"
            f"[dim]模型：[/dim]    {welcome_model}\n"
            f"[dim]模式：[/dim]    {mode}\n"
            f"[dim]版本：[/dim]    {__version__}"
        )
        budget = self.controller.current_context_budget()
        if session and budget.usable_context:
            context = estimate_tokens(session.conversation)
            context_text = f"上下文：{context / budget.usable_context:.0%}"
        else:
            context_text = "上下文：-"
        if model:
            center = f"{model_display}   推理：{thinking_display}"
        else:
            center = "输入 /model 选择模型"
        self.query_one("#statusbar", Static).update(
            f"{mode}   {center}                         {context_text}"
        )

    @staticmethod
    def _display_model(model: str) -> str:
        words = model.replace("_", "-").split("-")
        aliases = {"deepseek": "DeepSeek", "gpt": "GPT", "claude": "Claude"}
        return " ".join(
            aliases.get(
                word.lower(),
                word.upper()
                if word.lower().startswith("v") and word[1:].isdigit()
                else word.title(),
            )
            for word in words
        )

    def set_working(self, working: bool, label: str = "智能体") -> None:
        prompt = self.query_one("#prompt", ComposerInput)
        prompt.placeholder = (
            f"{label}正在工作… Ctrl+C 中断" if working else "描述你想构建的内容…"
        )
        if not working:
            self.agent_worker = None
            prompt.focus()

    def action_interrupt(self) -> None:
        if self.agent_worker and not self.agent_worker.is_finished:
            self.cancel_event.set()
            self.add_system("已请求中断；当前 provider/tool 调用返回后将安全停止。")
        else:
            self.action_save_exit()

    def action_close_panel(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self.query_one("#prompt", ComposerInput).focus()

    def action_save_exit(self) -> None:
        self.cancel_event.set()
        if self.controller.session:
            self.controller.sessions.save(self.controller.session)
        self.exit()

    @staticmethod
    def _tool_title(name: str, arguments: dict[str, Any]) -> str:
        if name == "read_file":
            return f"正在读取 {arguments.get('path', '')}".rstrip()
        if name == "search_files":
            return f"正在搜索 {arguments.get('pattern', '')}".rstrip()
        if name == "list_files":
            return f"正在浏览 {arguments.get('path', '.')}"
        if name == "apply_patch":
            return f"正在编辑 {arguments.get('path', '')}"
        if name in {"git_diff", "git_inspect"}:
            return "正在检查修改"
        if name == "shell":
            command = str(arguments.get("command", ""))
            lowered = command.lower()
            if any(token in lowered for token in ("test", "pytest")):
                return "正在运行测试"
            if any(token in lowered for token in ("lint", "ruff", "eslint")):
                return "正在运行检查"
            if any(token in lowered for token in ("build", "compile")):
                return "正在构建"
            return f"正在运行 {command[:72]}"
        return name.replace("_", " ")

    @staticmethod
    def _tool_result_summary(payload: dict[str, Any]) -> str:
        name = str(payload.get("name", "tool"))
        ok = bool(payload.get("ok"))
        duration = float(payload.get("metadata", {}).get("duration_seconds", 0.0))
        state = {
            "read_file": "读取完成",
            "search_files": "搜索完成",
            "list_files": "浏览完成",
            "apply_patch": "编辑完成",
            "shell": "命令完成",
            "git_diff": "差异检查完成",
            "git_inspect": "Git 检查完成",
        }.get(name, name)
        output = str(payload.get("output", ""))
        key_error = ""
        if not ok:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            key_error = f" · {(lines[-1] if lines else '失败')[:120]}"
        return f"{state} · {duration:.1f}s{key_error}"

    @staticmethod
    def _truncate_detail(output: str, limit: int = 8_000) -> str:
        if len(output) <= limit:
            return output
        return output[:limit] + f"\n… 已折叠 {len(output) - limit:,} 个字符"


def run_tui(controller: InteractiveCLI | None = None) -> None:
    ForgeLoopTUI(controller).run()
