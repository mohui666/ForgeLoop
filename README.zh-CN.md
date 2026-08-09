# ForgeLoop

[English](README.md) | 简体中文

ForgeLoop 是一套用于运行、评测和持续迭代 Coding Agent 的基础设施。它包含 CLI AgentLoop、与 Provider 无关的模型边界、Docker Runtime、trajectory 记录、确定性评测、小型真实仓库 SWE Environment Foundry，以及基于 Textual/Rich 的交互式 TUI。LiteLLM 提供可替换的多 Provider 适配层。

项目刻意保持小而清晰。目前已经在固定的 30 任务 smoke suite 和 8 任务真实仓库 suite 上完成验证。它是继续开展评测和训练数据工作的实用基础，而不是对 benchmark 领先地位或生产级沙箱隔离能力的宣称。

当前阶段明确不包含 Web UI、通用沙箱平台、容器池、多 Agent 编排、RAG、自行实现的推理服务器、大型 benchmark 基础设施、SFT/RL、Dashboard 或分布式执行。当前范围包括一套固定 smoke eval、按任务隔离的轻量 Docker Runtime、小型精选 Environment Foundry 路径，以及用于外部自托管 open-weight base policy 的 adapter。

## 快速开始

推荐使用 Python 3.10+ 和 `uv`：

```powershell
uv sync --extra dev
uv run forgeloop
```

不带参数运行会打开对话式 TUI。启动页不会显示仓库列表、最近 Session Dashboard 或文件树。直接输入自然语言即可与 Coding Agent 对话，输入 `/help` 查看 ForgeLoop 控制命令。首次对话时，如果当前目录是 Git 仓库，ForgeLoop 会创建 Session 并绑定当前目录；否则会要求选择项目路径。

界面只保留欢迎信息、连续对话、折叠 Tool Call、输入框和一行轻状态；没有 Dashboard、侧栏、文件树或常驻详情面板。`Ctrl+P` 打开命令面板，`Ctrl+C` 请求安全中断当前 Agent，`Esc` 关闭临时面板，`Ctrl+D` 保存退出。输入框支持上下键历史和 Tab 命令补全。

在 TUI 中按以下流程配置模型：

```text
/api
/model
/thinking
```

`/api` 统一管理 Provider、API Key、Base URL、Connection Test 和配置删除。API Key 保存在操作系统凭据存储中，不会写入配置、Session、context、日志或 trajectory。LiteLLM 常用的 Provider 环境变量仍然受支持。

`/model` 只显示配置完整且可用的 Provider。进入 Provider 后，会优先显示按 Provider + Base URL 隔离的本地模型缓存；Refresh Models 会调用真实 Provider 模型接口。刷新失败时旧缓存仍可用，也可以手动输入真实 Model ID。ForgeLoop 内部生成 LiteLLM canonical route，但普通用户只需要处理 Provider 与 Model ID。Preflight 会在进入 AgentLoop 前阻止缺失或不匹配的 route、credential 和 endpoint。Connection Test 会强制执行一次真实 tool call，同时验证认证、模型访问和 tool-calling 支持。

`/thinking` 根据当前模型 capability 只显示真实支持的 Auto / Low / Medium / High / Max 子集。切换模型后会重新解析能力；不兼容的 Session thinking 设置会回退到 Auto。Capability 按 Provider metadata、本地缓存、ForgeLoop 已验证 registry、unknown 的顺序解析，未知限制不会被猜测或伪造。

`/context` 显示当前模型的 context limit、usable context、当前估算用量、reserved output、thinking/tool reserve、自动压缩阈值和压缩次数。阈值会随模型动态变化；切换到更小的上下文模型前会先压缩，压缩后仍无法安全容纳时会阻止切换。

常用交互命令包括 `/plan`、`/build`、`/status`、`/diff`、`/undo`、`/test`、`/lint`、`/build run`、`/context`、`/compact`、`/cost`、`/sessions`、`/resume` 和 `/new`。Plan Mode 为只读模式。Build Mode 会在每个 Agent turn 前创建 Git object checkpoint，使 `/undo` 能同时恢复原有 worktree 和 index 状态。

`/sessions`、`/diff`、`/status`、`/context`、`/cost`、`/api`、`/model` 和 `/help` 使用临时 TUI 面板。较长的 Tool stdout/stderr 默认折叠；失败的 Tool 会保留简短错误摘要。`/details` 可查看最近一次已经脱敏的 Provider 诊断信息，不会把 LiteLLM traceback 直接倾倒到对话中。

全局非敏感 Provider/API 默认值和 budget 保存在 `~/.forgeloop/config.json`，模型 metadata 保存在 `~/.forgeloop/model_cache.json`。仓库路径、对话、diff/checkpoint 状态、context、选中的 Provider/Model、thinking、mode、runtime、usage 和 trajectory 引用按 Session 保存。模型缓存不包含 secret，并按 Provider + Base URL + Model 隔离。

项目级 `FORGELOOP.md` 和 `AGENTS.md` 会加载到每个 Agent turn。文件 Tool 被限制在选定仓库内，敏感凭据文件会被阻止访问，宿主机 Shell 会拒绝已知的破坏性命令和凭据暴露模式。Token、cost、step、tool-call、model-call 和 timeout budget 均由现有 `AgentLoop` 边界执行。

原有自动化入口保持不变：

```powershell
$env:OPENAI_API_KEY = "..."
uv run forgeloop task "修复 AuthService 中的空指针错误" --model openai/gpt-4.1
```

Goal Mode 接收最终目标，并允许 Agent 自行分解任务：

```powershell
uv run forgeloop goal "让这个项目的所有测试通过" --model anthropic/claude-sonnet-4-5
```

Task Mode 用于边界明确的软件工程任务：

```powershell
uv run forgeloop task "修复 AuthService 中的空指针错误" --model openai/gpt-4.1
```

Provider credential 遵循 LiteLLM 标准环境变量。`--api-base` 支持 OpenAI-compatible endpoint，`--model` 或 `FORGELOOP_MODEL` 用于选择 Provider/Model route。

## Open-weight Base Policy

ForgeLoop 当前唯一 active open-weight policy 是 `qwen3.5-4b-local`：通过 Ollama 在 `http://127.0.0.1:11434/v1` 运行 `qwen3.5:4b`，并复用现有 LiteLLM/OpenAI-compatible Provider 边界。内置 policy manifest 固定 Ollama digest、tokenizer artifact、inference backend、serving profile、generation profile 和 model capability：

```powershell
uv run forgeloop task "修复失败的测试" `
  --policy-manifest qwen3.5-4b-local
```

此前的 9B/vLLM manifest 只保留历史 provenance 兼容，不再是 active policy。当前阶段只接入并追踪 Base Policy，不执行训练。Ollama 安装、LocalRuntime rollout、identity schema 和 live validation gate 见 [docs/open-weight-policy.md](docs/open-weight-policy.md)。

## Budget 与记录

每次运行都会执行 step、model-call、tool-call、wall-clock 和 reported-token 限制。可以通过 `--max-cost-usd` 设置可选 cost limit。运行 `uv run forgeloop goal --help` 可查看全部选项。

自动化运行产生的 append-only JSONL trajectory 默认写入 `.forgeloop/runs/`。交互式 trajectory 按 Session 存放在 `~/.forgeloop/trajectories/`。其中包含标准化 model request/response、tool call、observation、budget snapshot 和 terminal result。API Key 不会写入运行配置或 trajectory event。

带 Verifier 结果的 eval trajectory 可以转换为可追溯、可分类的训练记录和与训练框架无关的 SFT conversation JSONL：

```powershell
uv run forgeloop dataset build
uv run forgeloop dataset inspect
uv run forgeloop dataset export
```

默认 SFT export 只包含已经验证且执行效率正常的 candidate，并排除全部 infrastructure failure。Schema、分类、provenance、过滤和 sanitization 保证见 [docs/dataset.md](docs/dataset.md)。

文件、Shell、Git、测试和安全相关的 Tool 副作用会作为结构化 trajectory evidence 记录。Replay 和确定性 Explain 完全离线，不会重新执行原始副作用：

```powershell
uv run forgeloop trace replay <trajectory-id-or-path>
uv run forgeloop trace explain <trajectory-id-or-path>
```

Effect Event schema、TraceSeal 设计来源、evidence 安全、分析规则和兼容性见 [docs/observability.md](docs/observability.md)。

## 本地执行警告

Local Runtime 的 Shell 命令会在独立 PowerShell 或 `/bin/sh` 进程中直接运行于宿主机。文件 Tool 无法逃逸选定 workspace。ForgeLoop 会从子进程中移除 credential-like 环境变量，并阻止已知危险命令形态，但这些措施只是 guardrail，不是操作系统级沙箱。只应在可信 workspace 中使用 ForgeLoop，并通过 `/diff` 审查变更。

Agent 和 workflow 在 Textual thread worker 中执行，因此终端能够保持响应。`Ctrl+C` 为协作式中断：它会立即记录中断请求，并在当前同步 Provider 或 Tool call 返回后结束 turn。已有 Session、checkpoint、trajectory 和代码修改仍会保留。

## 开发

```powershell
uv sync --extra dev
uv run pytest
uv run forgeloop --help
```

架构评估、参考项目和扩展边界见 [docs/design.md](docs/design.md)。

## 固定 Smoke Eval

内置 `python-smoke-v1` suite 包含 30 个由 Verifier 驱动的任务，覆盖 easy、medium 和 hard 三个层级。Eval 默认为 dry-run，并通过 staged execution 避免意外模型开销。Live run 默认对每个任务执行三次独立 attempt；使用 `--repeats 2` 可改为两次：

```powershell
uv run forgeloop eval --stage a

# 仅在准备好执行一次真实 canary task 时使用：
$env:AGENT_TEMP_KEY = [Environment]::GetEnvironmentVariable("agent_temp_key", "Machine")
uv run forgeloop eval --stage a --live --repeats 3
Remove-Item Env:AGENT_TEMP_KEY
```

在每个任务独占的 disposable Docker container 中运行相同 canary：

```powershell
$env:AGENT_TEMP_KEY = [Environment]::GetEnvironmentVariable("agent_temp_key", "Machine")
uv run forgeloop eval --stage a --live --runtime docker --repeats 3
Remove-Item Env:AGENT_TEMP_KEY
```

Stage `a` 选择一个 canary，`b` 选择另外三种任务类型，`c` 运行完整 suite。任务定义、可复现性保证、结果字段和 DeepSeek V4 Flash 配置见 [docs/eval.md](docs/eval.md)。

## 真实仓库 Foundry

从固定的公开 Git commit 构建精选的 8 任务 Stage B suite，并要求每个任务通过两次独立 Docker FAIL-to-PASS 验证：

```powershell
uv run forgeloop foundry build
uv run forgeloop eval --suite real-swe --runtime docker
uv run forgeloop eval --suite real-swe --runtime docker --live --repeats 2
```

如果无法完整构建 suite，Foundry 会拒绝发布部分结果。生成的 fixture 和隐藏 gold patch 位于 `.forgeloop/foundry/real-swe`；只有 fixture 内容会复制到 Agent workspace。默认命令不会调用模型，只有明确加入 `--live` 才会执行被评测模型。Task provenance、screening record、trust boundary 和扩展标准见 [docs/foundry.md](docs/foundry.md)。
