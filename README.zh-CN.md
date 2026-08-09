# ForgeLoop

[English](README.md) | 简体中文

ForgeLoop 是一个精简的 Coding Agent 平台，用于在可执行的软件工程任务上运行、
评测和持续改进 Agent。

核心能力：

- 与模型厂商无关的 `AgentLoop`、Tool Calling 和预算控制；
- 共用同一核心的交互式 TUI、Task 和 Goal 工作流；
- 本地与 Docker Runtime，以及明确的安全边界；
- Trajectory、Effect Event、Replay 和确定性 Explain；
- 面向长轨迹的确定性 Context 压缩与逐轮来源指标；
- 由 Verifier 驱动的内部评测与 DeepSWE 外部评测；
- 可追溯的数据集导出，为后续训练闭环提供数据。

## 快速开始

ForgeLoop 需要 Python 3.10+，推荐使用 [`uv`](https://docs.astral.sh/uv/)。

```powershell
uv sync --extra dev
uv run forgeloop
```

默认命令会打开对话式 TUI。使用 `/api` 配置 Provider，使用 `/model` 选择模型，
使用 `/help` 查看命令。

常用操作：

- `/plan` 和 `/build`：切换只读规划与代码修改；
- `/status`、`/diff`、`/test`、`/undo`：检查、验证或撤销修改；
- `/context`、`/compact`、`/cost`：查看上下文与用量；
- `Ctrl+C`：中断当前回合；`Ctrl+D`：保存并退出。

API Key 保存在操作系统凭据存储中，不会写入 Session、日志或 Trajectory。

## CLI 自动化

运行边界明确的任务：

```powershell
$env:OPENAI_API_KEY = "..."
uv run forgeloop task "修复失败的测试" --model openai/gpt-4.1
```

运行目标导向的任务：

```powershell
uv run forgeloop goal "让所有测试通过" --model anthropic/claude-sonnet-4-5
```

通过 `--api-base` 可连接 OpenAI-compatible endpoint；通过
`--policy-manifest` 可使用可复现的内置或自定义 Policy。

## Policy

当前 active 本地 open-weight policy 是 `qwen3.5-4b-local`，由 Ollama 在
`http://127.0.0.1:11434/v1` 提供服务：

```powershell
uv run forgeloop task "修复失败的测试" `
  --policy-manifest qwen3.5-4b-local
```

当前 V4-Flash Controller policy 是
`deepseek-v4-flash-controller-v1.3-simplified`。历史 Base、SFT 和 Controller
manifest 继续保留，用于 provenance 与结果对比。

参见[本地 Policy 配置](docs/open-weight-policy.md)、
[Controller v1.3 Simplified](docs/hybrid-controller-v1.3-simplified.md)和
[Agent Context Efficiency v1](docs/agent-context-efficiency-v1.md)。

## 评测

查看或运行内部 Verifier-backed smoke suite：

```powershell
uv run forgeloop eval --stage a
uv run forgeloop eval --stage a --live --repeats 1
```

检查并运行冻结的 DeepSWE Eval v2 20-task subset：

```powershell
uv sync --extra dev --extra deepswe
uv run forgeloop deepswe check
uv run forgeloop deepswe run --policy qwen3.5-4b-local
```

Live Eval 会调用付费或本地模型 endpoint；Dry Run 不会调用模型。参见
[内部评测](docs/eval.md)、[DeepSWE Eval v2](docs/deepswe-eval-v2.md)和
[Environment Foundry](docs/foundry.md)。

## Trajectory 与数据集

每次运行都会记录模型用量、Tool Call、Observation、Effect、Git 状态、Verifier
结果和 Terminal Reason。

```powershell
uv run forgeloop trace replay <trajectory-id-or-path>
uv run forgeloop trace explain <trajectory-id-or-path>
uv run forgeloop dataset build
uv run forgeloop dataset inspect
uv run forgeloop dataset export
```

参见[可观测性](docs/observability.md)和[数据集格式](docs/dataset.md)。

## 安全说明

Local Runtime 会直接在宿主机执行命令，不是操作系统级沙箱。请只在可信仓库中
使用，并通过 `/diff` 审查修改；外部或需要隔离的任务应使用 Docker。文件 Tool
被限制在选定 Workspace 内，敏感路径会被阻止，Step、Token、Call、Cost 和
Timeout Budget 均会强制执行。

## 开发

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

架构与扩展边界见 [docs/design.md](docs/design.md)。ForgeLoop 使用 MIT License。
