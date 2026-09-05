# ForgeLoop Project Charter

## Astra 协作约定

- 以用户当前目标和本轮明确约束为准。任务要求实施时，完成实际修改与必要验证，不停在计划、建议或“是否继续”。普通实现选择自行决定；只有缺失信息会实质改变结果或操作超出授权时才询问，并先完成不依赖答案的工作。
- 用户指令优先于本地 skills 的工作流建议。只读取与当前任务直接相关的文件和技能；不因关键词命中就串联整套技能、生成流程工件或增加审批。
- 保留既有业务规则、数据所有权、用户改动和明确的工具限制。只改当前目标需要的内容，不顺手重构、升级依赖、搬目录或扩展产品范围。

## 拒绝过度防御性编程

- 直接使用已有输入、文件、依赖和运行环境，不重复做环境、权限、目录或文件存在性预检查。
- 不为假想故障添加重复参数验证、大量极端输入分支、宽泛 `try/catch`、默认值兜底、静默失败或伪造成功。契约不满足时暴露具体错误。
- 不主动新增重试、退避、熔断、降级、备用实现、兼容层、自动备份、回滚、迁移或恢复机制。
- 不主动添加 SHA、MD5、签名、文件哈希、完整性校验、CI/CD、发布门禁、安全扫描、许可证审计、复杂日志、监控、遥测或诊断框架。
- 不为未来需求预建插件系统、通用框架或抽象层，不为小改动铺设大量单元测试、回归测试、故障注入或性能基准。
- 只在缺少检查会立即阻止核心功能、造成明显数据损坏或掩盖真实错误时保留最小必要检查。现有鉴权、真实业务校验和数据保护功能继续遵守其契约；本规则不授权删除这些功能。
- 例外必须来自用户明确要求，或与本次改动直接相关的既有产品契约。旧文档中泛化的“每次全量检查”“必须先审批”“自动完善”不构成额外任务。

## 验证与交付

- 选择能证明本次行为的最小验证：文档或提示词改动检查内容和 diff；代码改动运行相关构建、现有定向测试或核心流程冒烟。低影响、可逆改动不新增仅复述实现的测试。
- 必要检查通过即交付；只有新改动、失败或具体未解决疑点才扩大或重复验证。不要为了收尾重跑无关全量测试、打包、实机流程或基准。
- 错误如实报告。区分实际运行通过、静态检查、未运行与真实环境验证；历史测试数量不能当作本次证据。
- 仅在任务需要时使用子代理；不强制委派、切换模型或修改推理档位，遵守当前会话设置与工具权限。
- 按当前授权和项目约定执行 Git 操作，只提交本任务文件；不要为清空工作区而夹带其他改动，不强推或丢弃用户内容。没有远端时报告，不擅自创建远端。
- 用简明中文交代实际修改、验证结果和已知问题。只有需求、接口或已验证事实改变时同步相关文档，不追加与交付无关的报告。

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
