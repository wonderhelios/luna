# Luna 架构设计与开发路线图 v2.1 (TPAR)

> Local-first Agent Runtime with Deterministic Code Intelligence

---

## 1. 核心理念

### 1.1 一句话定位

Luna 是一个 **Local-first 的 Agent Runtime**：从入口接收任务、理解意图、制定计划、执行行动、反思结果，并持续优化。

### 1.2 为什么采用 TPAR 而非 ReAct？

| 架构 | 流程 | 特点 | 适用场景 |
|------|------|------|----------|
| **ReAct** | Thought → Action → Observation | 单步循环，即时反馈 | 简单任务、快速响应 |
| **TPAR** | Task → Plan → Act → Review | 先规划后执行，全局优化 | 复杂任务、多步骤、需要反思 |

**TPAR 优势**：
1. **全局规划**：先制定完整计划，避免局部最优
2. **可回溯**：Review 阶段可以反思、修正、学习
3. **RL 友好**：Plan 和 Review 为策略网络提供明确的学习信号
4. **复杂任务**：适合代码重构、多文件修改等需要规划的任务

### 1.3 核心差异化

| 特性 | Claude Code | Codex | Cursor | **Luna** |
|------|-------------|-------|--------|----------|
| 架构 | ReAct | ReAct | ReAct | **TPAR** |
| 代码理解 | LLM推断 | LLM推断 | RAG | **ScopeGraph精确解析** |
| 确定性 | 低 | 低 | 中 | **高（AST级）** |
| 可解释性 | 中 | 中 | 中 | **高（完整trace）** |
| 隐私 | 数据上传 | 本地 | 本地 | **本地** |

**Luna 的核心差异化**：
- **TPAR 架构**：先规划后执行，全局优化
- **ScopeGraph**：确定性代码理解，不是猜测
- **模块化 Runtime**：可插拔、可扩展

---

## 2. 系统架构

### 2.1 TPAR Agent Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        TPAR Agent Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│   │  Task   │───►│  Plan   │───►│   Act   │───► Review  │   │
│   │ (Intent)│    │(Planner)│    │ (Tools) │    │(Reflect)│   │
│   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘   │
│        │              │              │              │        │
│        │              │              │              │        │
│        ▼              ▼              ▼              ▼        │
│   ┌──────────────────────────────────────────────────────┐  │
│   │              Trajectory Recorder                     │  │
│   │  记录: Task → Plan → [Action, Observation]* → Review │  │
│   └──────────────────────────────────────────────────────┘  │
│                              │                                 │
│        ┌─────────────────────┼─────────────────────┐          │
│        │                     │                     │          │
│        ▼                     ▼                     ▼          │
│   ┌──────────┐        ┌──────────┐         ┌──────────┐      │
│   │  Safety  │        │ Context  │         │  Refill  │      │
│   │  Guard   │        │ Pipeline │◄───────►│ Pipeline │      │
│   │          │        │          │         │          │      │
│   │ - Plan   │        │ - Index  │         │ - Index  │      │
│   │ - Action │        │   Chunk  │         │   → Ref  │      │
│   │ - Review │        │ - Context│         │ - Dynamic│      │
│   └──────────┘        └──────────┘         └──────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 完整系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Ingress Layer                        │
│                      (luna-cli)                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼ RunRequest
┌─────────────────────────────────────────────────────────┐
│                    LUNA RUNTIME                         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              TPAR Agent Loop                    │   │
│  │                                                 │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐     │   │
│  │  │  Task   │───►│  Plan   │───►│   Act   │     │   │
│  │  │ (Intent)│    │(Planner)│    │ (Tools) │     │   │
│  │  └─────────┘    └────┬────┘    └────┬────┘     │   │
│  │                      │              │          │   │
│  │                      ▼              ▼          │   │
│  │              Review/Reflect    Trajectory      │   │
│  │                                                 │   │
│  │  ┌──────────────────────────────────────────┐  │   │
│  │  │ SafetyGuard + EventStream                │  │   │
│  │  └──────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Context   │  │   Session   │  │   Vector    │    │
│  │  Pipeline   │  │    Store    │  │   Index     │    │
│  │             │  │             │  │             │    │
│  │ ┌─────────┐ │  │             │  │             │    │
│  │ │ Index   │ │  │             │  │             │    │
│  │ │ Chunk   │ │  │             │  │             │    │
│  │ │(Retrieval)│ │  │             │  │             │    │
│  │ └────┬────┘ │  │             │  │             │    │
│  │      │Refill│ │  │             │  │             │    │
│  │      ▼      │ │  │             │  │             │    │
│  │ ┌─────────┐ │  │             │  │             │    │
│  │ │ Context │ │  │             │  │             │    │
│  │ │ Chunk   │ │  │             │  │             │    │
│  │ │(for LLM)│ │  │             │  │             │    │
│  │ └─────────┘ │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐                     │
│  │ AgentConfig │  │   Project   │                     │
│  │  (Skills)   │  │   Memory    │                     │
│  └─────────────┘  └─────────────┘                     │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  luna-   │     │  luna-   │     │  luna-   │
  │  tools   │     │intelligence│    │  llm     │
  └──────────┘     └──────────┘     └──────────┘
```

---

## 3. TPAR 各阶段详解

### 3.1 Task（任务理解）

```rust
/// 任务理解：将用户输入解析为结构化任务
pub struct Task {
    /// 任务类型
    pub task_type: TaskType,
    /// 原始输入
    pub raw_input: String,
    /// 提取的意图
    pub intent: Intent,
    /// 涉及的代码实体（从输入中提取的符号）
    pub entities: Vec<CodeEntity>,
}

pub enum TaskType {
    /// 查询类："X在哪里定义"
    Query,
    /// 编辑类："修改这个函数"
    Edit,
    /// 重构类："重构认证模块"
    Refactor,
    /// 修复类："修复这个bug"
    Fix,
    /// 解释类："解释这段代码"
    Explain,
}
```

### 3.2 Plan（计划制定）

```rust
/// 计划：一系列有序的行动步骤
pub struct Plan {
    /// 计划步骤
    pub steps: Vec<PlanStep>,
    /// 步骤间的依赖关系
    pub dependencies: DependencyGraph,
    /// 可并行执行的步骤组
    pub parallel_groups: Vec<Vec<StepId>>,
    /// 预估的 token 消耗
    pub estimated_tokens: usize,
}

pub struct PlanStep {
    pub id: StepId,
    /// 步骤描述（给 LLM 看）
    pub description: String,
    /// 要调用的工具
    pub tool: String,
    /// 工具参数
    pub args: Value,
    /// 预期结果
    pub expected_outcome: String,
}
```

### 3.3 Act（行动执行）

执行 Plan 中的步骤，支持并行执行。

### 3.4 Review（反思评估）

检查执行结果，决定：
- Success：任务完成
- NeedsRevision：需要修改计划并重试
- NeedsRollback：需要回滚到某一步

### 3.5 Context Pipeline（上下文管线）

大代码库场景下的高效上下文管理系统，实现**检索阶段与生成阶段解耦**。

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │   Index     │ ───► │   Refill    │ ───► │   Context   │ │
│  │   Chunk     │      │   Pipeline  │      │   Chunk     │ │
│  │             │      │             │      │             │ │
│  │ • 粗糙粒度   │      │ • 符号注入   │      │ • 精炼内容   │ │
│  │ • 快速召回   │      │ • 去重合并   │      │ • Token优化 │ │
│  │ • 向量索引   │      │ • 动态补充   │      │ • LLM就绪   │ │
│  └─────────────┘      └─────────────┘      └─────────────┘ │
│                                                             │
│  检索阶段                转换阶段              生成阶段       │
│  (IndexChunk)         (Refill)            (ContextChunk)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```rust
/// 检索阶段的原始代码块
///
/// 用于快速召回候选内容，可以粗糙但覆盖率高
pub struct IndexChunk {
    pub id: ChunkId,
    /// 原始内容（可能较长）
    pub content: String,
    /// 来源位置
    pub source: SourceLocation,
    /// 向量嵌入（可选，用于语义检索）
    pub embedding: Option<Vec<f32>>,
    /// 关联的符号（ScopeGraph 提取）
    pub symbols: Vec<SymbolId>,
    /// 语言类型
    pub language: LanguageId,
    /// 最后更新时间
    pub modified_at: TimestampMs,
}

/// 生成阶段送给 LLM 的精炼上下文
///
/// 经过 Refill 管线处理，Token 效率最优
pub struct ContextChunk {
    pub id: ChunkId,
    /// 精炼后的内容（可能截断或摘要）
    pub content: String,
    /// 来源位置
    pub source: SourceLocation,
    /// 相关性分数（综合精确匹配 + 语义相似度）
    pub relevance_score: f32,
    /// 实际 Token 数量
    pub token_count: usize,
    /// 关联符号的签名信息（已注入）
    pub symbol_signatures: Vec<String>,
}

/// Refill 管线：IndexChunk → ContextChunk 的转换引擎
///
/// 核心职责：
/// 1. 初始检索：基于 ScopeGraph + 向量索引快速召回
/// 2. 精炼转换：去重、排序、截断、注入符号签名
/// 3. 动态补充：根据 LLM 反馈补充缺失上下文
pub struct RefillPipeline {
    /// 向量索引存储
    index_store: Arc<dyn IndexStore>,
    /// ScopeGraph 符号解析
    scope_graph: Arc<ScopeGraph>,
    /// Token 预算管理
    budget: TokenBudget,
    /// 会话级上下文缓存（支持增量 Refill）
    cache: ContextCache,
}

impl RefillPipeline {
    /// 初始检索：混合精确匹配 + 向量语义检索
    pub fn retrieve(&self, query: &Query, top_k: usize) -> Vec<IndexChunk>;

    /// 精炼为 ContextChunk（核心转换逻辑）
    ///
    /// 处理流程：
    /// - 基于 ScopeGraph 去重（同一符号的多处引用合并）
    /// - 按 relevance_score 排序
    /// - Token 预算截断（优先保留高相关 chunk）
    /// - 注入符号签名（函数签名、类型定义等）
    pub fn refine(&self, chunks: &[IndexChunk]) -> Vec<ContextChunk>;

    /// 动态补充：根据执行反馈补充上下文
    ///
    /// 触发条件：
    /// - LLM 请求 "查看更多代码"
    /// - 工具执行发现新的相关符号
    /// - Review 阶段发现上下文不足
    pub fn refill(&self,
        current: &[ContextChunk],
        missing_symbols: &[SymbolId]
    ) -> Vec<ContextChunk>;

    /// 构建最终 LLM 上下文
    pub fn build_prompt(&self, chunks: &[ContextChunk]) -> String;
}

/// 查询类型
pub enum Query {
    /// 精确符号查询（走 ScopeGraph）
    Symbol { name: String },
    /// 位置查询（文件:行号）
    Position { path: PathBuf, line: usize },
    /// 语义概念查询（走向量检索）
    Concept { description: String },
    /// 混合查询
    Hybrid { symbol: Option<String>, concept: String },
}
```

**Context Pipeline 与 TPAR 的集成**:

```rust
// 在 Plan 阶段调用 Context Pipeline
impl RuleBasedPlanner {
    fn plan(&self, task: &Task, ctx: &PlanContext) -> Plan {
        // 1. 通过 Context Pipeline 获取相关上下文
        let context_chunks = ctx.refill_pipeline.retrieve_and_refine(
            &Query::from_task(task),
            ctx.budget.max_context_tokens,
        );

        // 2. 基于上下文制定计划
        let steps = match task.task_type {
            TaskType::Query => vec![
                PlanStep::Intelligence {
                    context: context_chunks,
                    query: task.raw_input.clone(),
                }
            ],
            TaskType::Edit => {
                // 编辑任务需要更精确的上下文
                let refined = ctx.refill_pipeline.refine_with_symbols(
                    &context_chunks,
                    &task.entities,
                );
                vec![
                    PlanStep::ReadFile { context: refined },
                    PlanStep::EditFile { ... },
                ]
            }
            // ...
        };

        Plan { steps, estimated_tokens, ... }
    }
}
```

**核心优势**:

| 特性 | 传统 RAG | Luna Context Pipeline |
|------|----------|----------------------|
| 检索粒度 | 固定 chunk | IndexChunk 可粗糙，灵活召回 |
| 生成粒度 | 原始 chunk | ContextChunk 精炼，Token 高效 |
| 符号理解 | 文本匹配 | ScopeGraph 精确解析 |
| 动态补充 | 无 | Refill 按需补充 |
| 大代码库 | 线性扫描 | 分层索引 + 智能缓存 |

---

## 4. 开发路线图（TPAR 版）

### Phase 0: TPAR Runtime 框架（Week 1-2）

**目标**：建立 TPAR 主链，实现基本的 Task → Plan → Act → Review 流程

**任务**：
- [x] 组装最小 TPAR 主链（`src/runtime/src/tpar.rs` + `src/runtime/src/runtime.rs`）
- [x] Task（意图识别 + 实体抽取）MVP（`src/runtime/src/intent.rs`）
- [x] Plan（路由级规划）MVP：将任务映射到 Router/安全检查/兜底（`src/runtime/src/tpar.rs`）
- [x] Act（执行）MVP：symbol navigation 走 `RuntimeRouter`（`src/runtime/src/router.rs`）
- [x] Review/Reflect（最小）MVP：记录 TPAR 事件（`src/runtime/src/response.rs` 的 `RuntimeEvent::Tpar*`）
- [x] 定义结构化 `Task/Plan/PlanStep`（可序列化，用于 Trajectory 的 step/action）
- [x] ActExecutor：将“计划步骤”落到真正的工具执行（ToolCall/ToolRegistry）
- [x] Reviewer：支持 `NeedsRevision/NeedsRollback`（文件编辑失败时自动回滚）

**验收标准**：

> 注：下面的输出示例描述的是“目标形态”的可观测性。当前实现已具备 TPAR 主链与事件，但 Task/Plan/Act 的详细逐步输出与回放仍在后续里程碑中补齐。

```bash
# 测试 1: 简单查询任务
> find_main_function 在哪里？

[Task] type: Query, intent: FindDefinition, entities: ["find_main_function"]
[Plan] 
  Step 1: search_symbol(name="find_main_function")
  Step 2: read_file(path=<result from step 1>)
[Act]
  Step 1: ✅ Found in src/tools/src/lib.rs:45
  Step 2: ✅ Read file content
[Review] ✅ Success

结果：
find_main_function 定义在 src/tools/src/lib.rs:45
函数签名：pub fn find_main_function(query: &str) -> Vec<SymbolLocation>
```

```bash
# 测试 2: 编辑任务
> 修改 src/main.rs 第 10 行 为 fn hello(name: &str) {

[Task] type: Edit, intent: ModifyFunction, entities: ["hello", "src/main.rs"]
[Plan]
  Step 1: read_file(path="src/main.rs")
  Step 2: edit_file(path="src/main.rs", line_1=10, new_line="fn hello(name: &str) {")
  Step 3: verify (read file again)
[Act]
  Step 1: ✅ Read file
  Step 2: ✅ Edit file
  Step 3: ✅ Verification passed
[Review] ✅ Success

结果：
✅ 修改完成
```

```bash
# 测试 3: 失败回滚
> 修改不存在的函数

[Task] type: Edit, intent: ModifyFunction, entities: ["nonexistent_func"]
[Plan]
  Step 1: search_symbol(name="nonexistent_func")
[Act]
  Step 1: ❌ Symbol not found
[Review] ❌ Failed - Symbol "nonexistent_func" not found in codebase

结果：
❌ 任务失败：找不到函数 nonexistent_func
```

---

### Phase 1: Intelligence 集成（Week 3-4）

**目标**：让 Plan 阶段能利用 ScopeGraph 进行确定性代码理解

**任务**：
- [x] 符号导航核心：`goto_definition/get_symbol_context/find_references`（`src/intelligence/navigation.rs`）
- [x] Runtime 路由：对符号查询优先走 Intelligence（`src/runtime/src/router.rs`）
- [x] 上下文输出：签名 + snippet + refs 渲染（`src/runtime/src/render.rs`）
- [x] 将 Intelligence 从“符号查询专用”提升为通用 Plan 能力（在 PlanStep 中引入 `Intelligence` step）
- [x] 对比测试：ScopeGraph vs 纯文本搜索（`runtime::tpar` 单测覆盖）
- [x] 基准测试骨架：提供 `#[ignore]` 的 latency smoke（`runtime::tpar`）

**验收标准**：

```bash
# 测试 1: 精确符号定位（对比 grep）
> find_main_function 在哪里定义？

# Luna (ScopeGraph)
[Plan] search_symbol(name="find_main_function")
[Act] ✅ Found definition at src/tools/src/lib.rs:45
      Signature: pub fn find_main_function(query: &str) -> Vec<SymbolLocation>
      Callers: [src/runtime/src/runtime.rs:257, src/cli/src/main.rs:8]

结果：100% 准确，包含签名和调用关系

# 对比：纯 grep 可能返回 10+ 个匹配（调用处+定义处），需要人工判断
```

```bash
# 测试 2: 同名函数区分
> UserService 在哪里定义？

[Plan] search_symbol(name="UserService")
[Act] ✅ Found 3 definitions:
      1. src/services/user.rs:23 - struct UserService (pub)
      2. src/tests/mock.rs:45 - struct UserService (test mock)
      3. src/old/legacy.rs:78 - struct UserService (deprecated)

根据上下文，最可能的是：src/services/user.rs:23
```

```bash
# 测试 3: 调用链分析
> 谁调用了 authenticate 函数？

[Plan] search_symbol(name="authenticate") + get_callers
[Act] ✅ authenticate defined at src/auth.rs:56
      Callers (5):
      - src/api/login.rs:45
      - src/api/oauth.rs:78
      - src/middleware/auth.rs:123
      - tests/auth_test.rs:34
      - examples/demo.rs:89

结果：完整的调用关系图
```

**性能指标**：
- 目标：符号查询延迟 < 10ms（1000 文件项目）
- 目标：准确率 100%（对比 grep 的 ~70%）

> 注：目前仓库尚未提供对应的基准/对比测试来自动验证上述指标，建议在 Phase 1 补齐。

> 注：目前已提供对比测试与基准测试骨架；如需在 CI 中强约束性能，需要引入更稳定的 bench 基线与数据采集。

---

### Phase 2: Session + Safety + Trace（Week 5-6）

**目标**：会话持久化、安全防护、完整轨迹记录

**任务**：
- [x] Session 持久化（JSONL）：`~/.luna/sessions/<session_id>.jsonl`（`src/session/src/jsonl_store.rs`）
- [x] 会话管理命令：`/sessions`、`/switch`（`src/runtime/src/command.rs` + `src/runtime/src/runtime.rs`）
- [x] SafetyGuard：Action 危险命令拦截（pattern deny）（`src/runtime/src/safety.rs`）
- [x] SafetyGuard：重复编辑检测（MVP，基于近期 action digest）（`src/runtime/src/safety.rs`）
- [x] TrajectoryRecorder（JSONL）：记录 run/event/step（`src/runtime/src/recorder.rs` + `src/runtime/src/recorder_jsonl.rs`）
- [x] SafetyGuard：Plan 合理性检查（step 级 safety check：执行前检查，warn/deny 直接终止）
- [x] Token 预算（MVP）：限制输入长度/单步 IO bytes/最大 step 数（`src/runtime/src/config.rs`）
- [x] 编辑执行升级为真实 `edit_file` 工具调用（需要显式提供替换内容："修改 <path> 第 <line> 行 为 <new_line>"）

**验收标准**：

```bash
# 测试 1: 会话持久化
$ luna
Session: local:a1b2c3d4 (resumed, 5 messages)

> 你好
[对话内容]

> exit
$ luna
Session: local:a1b2c3d4 (resumed, 6 messages)  # 历史保留
```

```bash
# 测试 2: 危险命令拦截
> 删除根目录

[Task] type: Edit, intent: DeleteFiles
[Plan] run_terminal(command="rm -rf /")
[SafetyGuard] ⚠️ BLOCKED: Dangerous command detected
              Command "rm -rf /" would delete system files
[Review] ❌ Blocked by SafetyGuard

结果：❌ 危险命令被拦截
```

```bash
# 测试 3: 重复编辑检测
> 修改 src/main.rs 第 10 行 为 // change-1
[Edit] ✅ Success

> 修改 src/main.rs 第 10 行 为 // change-2  # 再次修改同一位置
⚠️ Warning: 重复编辑，请检查逻辑
（本次执行中止，需要用户重新确认/调整计划）
```

```bash
# 测试 4: 轨迹记录验证
$ cat ~/.luna/trajectories/<session_id>.jsonl

# 当前实现会记录 SessionCreated/MessageAppended 等事件，以及 step 级的 TrajectoryStep（每个 PlanStep 一条）
{"type":"event","session_id":"...","event":"SessionCreated"}
{"type":"event","session_id":"...","event":{"MessageAppended":{"role":"User","bytes":123}}}
{"ts_ms":...,"session_id":"...","request_id":"...","state":...,"action":...,"reward":0.2,"outcome":...}
```

```bash
# 测试 5: 会话管理
> /sessions
Active sessions:
  local:a1b2c3d4 - my-project (12 messages) [current]
  local:e5f6g7h8 - another-project (5 messages)

> /switch local:e5f6g7h8
Switched to session local:e5f6g7h8
```

---

### Phase 3A: 流式输出（EventStream + TUI Streaming）（Week 7-8）

**目标**：把“过程”变成一等公民：Task/Plan/Act/Review 的事件可以实时推送到 CLI/TUI

**任务**：
- [x] Runtime EventStream：将 `RuntimeEvent` 从”返回 Vec”升级为可订阅流（channel/stream）
- [x] 让 TUI 逐事件渲染（不必等 turn 完成；支持滚动/取消）
- [x] 统一关键节点事件：TaskClassified / PlanBuilt / StepStarted / StepCompleted / Reviewed
- [x] Trajectory step 与 EventStream 对齐（同一 request_id 的事件可回放）

**验收标准**：

```bash
# 测试 1: 流式事件
> /sessions

[Task] ...
[Plan] ...
[Act] ...
[Review] ...
（TUI 按事件实时追加输出，而非等待命令完成后一次性输出）
```

```bash
# 测试 2: 长耗时任务不中断 UI
> 重构认证模块

[Plan - streaming]
...
[Act - streaming]
...
（可以滚动查看历史输出，且输入框仍可响应）
```

---

### Phase 3B: LlmBasedPlanner（Week 8-9）

**目标**：让 Plan 由“路由级”升级为“可执行步骤序列”，并引入 LLM 提升计划质量

**任务**：
- [x] 定义可序列化的 `Plan/PlanStep`（与 Phase 0 的目标形态对齐）
- [x] `RuleBasedPlanner` 输出结构化 Plan（作为基线）
- [x] `LLMBasedPlanner` 输出结构化 Plan（JSON schema），并做校验/降级
- [x] Plan 与 SafetyGuard/Trajectory 深度集成（每个 Step 都有 action/outcome）
- [x] 实现 `OpenAIClient` 支持真实 LLM API（OpenAI/OpenRouter/SiliconFlow）
- [ ] 对比评测：RuleBased vs LlmBased（成功率、步骤数、耗时、回滚率）

**验收标准**：

```bash
# 测试 1: 结构化计划输出（需要配置 LLM）
export LUNA_LLM_API_KEY="your-api-key"
export LUNA_LLM_MODEL="gpt-4o-mini"  # 或其他模型
# 可选: export LUNA_LLM_BASE_URL="https://api.openai.com/v1"
# 可选: export LUNA_PLANNER="llm"  # 启用 LLM planner（默认 rule）

> 修复构建错误

[Plan - LLM Generated]
  Step 1: run_terminal(cargo build)
  Step 2: search_symbol(...)
  Step 3: edit_file(...)
  Step 4: run_tests(...)
```

**环境变量配置**：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LUNA_LLM_API_KEY` | LLM API 密钥（必需） | - |
| `LUNA_LLM_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `LUNA_LLM_MODEL` | 模型名称 | `gpt-4o-mini` |
| `LUNA_LLM_TIMEOUT_SECS` | 请求超时（秒） | `60` |
| `LUNA_PLANNER` | Planner 类型 (`rule`/`llm`) | `rule` |

**支持的提供商**：
- OpenAI: `https://api.openai.com/v1`
- OpenRouter: `https://openrouter.ai/api/v1`
- SiliconFlow: `https://api.siliconflow.cn/v1`
- 任意 OpenAI 兼容 API

---

### Phase 4: Context Pipeline + Hybrid Search（Week 9-10）

**目标**：大代码库的高效上下文管理，检索与生成阶段解耦

**核心设计**：`IndexChunk` → `Refill` → `ContextChunk`

**任务**：
- [ ] **Context Pipeline 基础架构**
  - 定义 `IndexChunk` / `ContextChunk` / `RefillPipeline` 结构
  - 实现 `luna-context` crate，作为 Plan 阶段的子系统
  - Token 预算在 Chunk 级别的精细化管理

- [ ] **分层检索系统**
  - L1: ScopeGraph 精确符号检索（<10ms，准确率 100%）
  - L2: 向量语义检索（本地 embedding，概念查询）
  - L3: 文件级倒排索引（快速文件过滤）

- [ ] **Refill 管线实现**
  - `retrieve()`: 混合检索，召回候选 IndexChunk
  - `refine()`: 去重、排序、截断、符号签名注入
  - `refill()`: 根据 LLM 反馈动态补充上下文

- [ ] **Hybrid Search 策略**
  - Identifier 优先：明确符号查询走 ScopeGraph
  - 语义兜底：概念查询走向量检索
  - 结果融合：综合相关度排序

- [ ] **LLM 增强 Review**（与 Context Pipeline 配合）
  - Review 阶段可触发 Refill 补充上下文
  - 失败分析后自动修正计划

**验收标准**：

```bash
# 测试 1: Context Pipeline - 大代码库符号定位
> 在 src/runtime/src/router.rs:45 附近找相关代码

[Context Pipeline]
  [Retrieve] ScopeGraph: goto_definition(router.rs:45) → 3 symbols
             Vector: semantic_similarity("router", "runtime") → 5 chunks
             Total IndexChunks: 8

  [Refine]   Deduplicate: 8 → 5 unique symbols
             Sort by relevance: [0.95, 0.92, 0.88, 0.75, 0.70]
             Token budget: 2000 tokens
             Select top 3: src/runtime/src/router.rs, src/runtime/src/tpar.rs
             Inject signatures: ✓

  [Context]  Generated 3 ContextChunks (1850 tokens)
             1. router.rs:45 - RuntimeRouter struct + goto_definition
             2. tpar.rs:120 - Plan execution context
             3. tpar.rs:200 - ActExecutor implementation

[Plan]
  Step 1: read_file_with_context(router.rs, context=chunk_1)
  Step 2: search_references("RuntimeRouter")

[Act] ✅ 完整上下文已加载，无需额外文件读取
```

```bash
# 测试 2: Refill 动态补充
> 解释 handle_request 函数的实现

[Context Pipeline - Initial]
  Retrieved 3 ContextChunks (1500 tokens)
  - handler.rs:45 - handle_request function
  - middleware.rs:23 - auth middleware

[LLM Response]
  "我看到 handle_request 调用了 authenticate_user，
   但没有看到 authenticate_user 的实现..."

[Context Pipeline - Refill]
  Detected missing symbol: authenticate_user
  Refill triggered → retrieve("authenticate_user")
  New ContextChunk: auth.rs:78 - authenticate_user implementation

[LLM Response - After Refill]
  "handle_request 调用 authenticate_user 进行权限校验，
   authenticate_user 使用 bcrypt 对比哈希..."
```

```bash
# 测试 3: Hybrid Search - 概念查询
> 找出所有处理权限的代码

[Query Analysis]
  No clear identifier found → Hybrid Search

[Hybrid Search]
  [L1: ScopeGraph] Search symbols with "permission|auth|role" → 5 hits
  [L2: Vector]     Semantic search "权限处理 authorization" → 8 chunks
  [Merge]          Deduplicate → 10 unique IndexChunks

[Refine]
  10 IndexChunks → 6 ContextChunks (token-optimized)
  - auth/permission.rs:23   - check_permission (signature injected)
  - middleware/auth.rs:45   - AuthMiddleware (signature injected)
  - rbac/roles.rs:78        - Role enum (signature injected)
  ...

结果：
1. src/auth/permission.rs:23 - check_permission 函数
   签名：pub fn check_permission(user: &User, resource: &str) -> bool

2. src/middleware/auth.rs:45 - AuthMiddleware 结构
   实现权限校验中间件

3. src/rbac/roles.rs:78 - Role 枚举
   定义用户角色权限
```

```bash
# 测试 4: 大代码库性能（10万+ 文件）
> 重构认证模块

[Context Pipeline]
  [Retrieve]
    - ScopeGraph: 符号解析 8ms
    - Vector: 语义检索 45ms (ANN index)
    - File filter: 语言过滤 2ms
    Total: 55ms → 15 IndexChunks

  [Refine]
    - Deduplicate by symbol: 15 → 8
    - Sort by relevance score
    - Token budget: 4000 tokens
    - Select top 6: 6 ContextChunks (3800 tokens)

  [Refill - during execution]
    - Step 3: Missing symbol detected "OAuthConfig"
    - Refill: 12ms → 1 new ContextChunk
    - Continue execution

性能指标：
  初始检索: 55ms
  Refine: 8ms
  单次 Refill: 12ms
  总 Context 加载: <100ms（vs 线性扫描 5s+）
```

```bash
# 测试 5: 对比传统 RAG vs Context Pipeline
> 分析数据库连接池实现

# 传统 RAG（固定 chunk）
结果：20 chunks，15000 tokens
问题：
  - 包含无关代码（import 语句、注释）
  - 函数被截断，需要多次读取
  - 符号关系缺失

# Luna Context Pipeline
结果：5 ContextChunks，2800 tokens
优势：
  - 基于 ScopeGraph 精确提取相关符号
  - 函数签名完整注入
  - 调用关系图已加载
  - 无需额外文件读取
```

---

### Phase 5: Project Memory + AgentConfig（Week 11-12）

**目标**：个性化、不同行为模式

**任务**：
- [ ] Project Memory 自动学习（build_cmd, test_cmd, style_prefs）
- [ ] AgentConfig：不同 system_prompt + 可用工具组合
- [ ] LUNA.md 支持：显式项目说明
- [ ] 模式切换命令：`/mode chat`, `/mode fix`, `/mode refactor`

**验收标准**：

```bash
# 测试 1: Project Memory 自动学习
> cargo build  # 用户执行构建

[Project Memory] Learned: build_cmd = "cargo build"

> cargo test  # 用户执行测试

[Project Memory] Learned: test_cmd = "cargo test"

> /memory
Project Memory:
  build_cmd: cargo build
  test_cmd: cargo test
  package_manager: cargo
  preferred_tools: ["cargo", "rustc"]
  style_prefs: ["4-space indent", "snake_case"]
```

```bash
# 测试 2: AgentConfig 模式切换
> /mode fix
Switched to fix mode

> 修复构建错误
[Fix Mode - Plan]
  1. run_terminal(cargo build)  # 自动使用 learned build_cmd
  2. analyze_error
  3. search_references
  4. edit_file
  5. run_tests  # 自动使用 learned test_cmd

> /mode explain
Switched to explain mode

> 解释这段代码
[Explain Mode - Plan]
  1. read_file
  2. analyze_code_structure
  3. generate_explanation
  # 不会执行编辑操作
```

```bash
# 测试 3: LUNA.md 支持
$ cat LUNA.md
# My Project

## Build
- `cargo build` - Build the project
- `cargo test` - Run tests

## Architecture
- Core: luna-runtime
- Tools: luna-tools

$ luna
🌙 Luna (loaded LUNA.md)

> 怎么构建项目？
[Context] LUNA.md found, using project-specific knowledge

根据 LUNA.md，你可以使用 `cargo build` 构建项目。
```

---

### Phase 6: RlBasedPlanner（Week 13+）

**目标**：学习最优任务分解策略

**任务**：
- [ ] Policy Network 设计（小模型，<10MB）
- [ ] Value Network 设计（评估计划质量）
- [ ] 从 Trajectory 数据中提取训练样本
- [ ] PPO/GRPO 训练
- [ ] A/B 测试：RL vs LLM vs Rule

**验收标准**：

```bash
# 测试 1: RL 计划 vs 其他计划对比
> 重构认证模块

# RuleBasedPlanner
[Plan] 3 steps, 5 min, 80% success rate (historical)

# LLMBasedPlanner  
[Plan] 5 steps, 8 min, 90% success rate (historical)

# RlBasedPlanner
[RL Policy Network analyzing...]
[Plan] 4 steps, 6 min, estimated 95% success rate
       Optimized based on similar tasks in this project

[执行结果]
✅ Success in 5 min (better than estimated)
```

```bash
# 测试 2: 自适应项目特征
> 在小型项目（100文件）中重构
[RL] Plan: 3 steps, simple refactoring

> 在大型项目（10000文件）中重构  
[RL] Plan: 8 steps, including:
       - Pre-analysis phase
       - Parallel search for references
       - Staged rollout (dev → test → prod)
       - Full test suite validation

结论：RL Planner 根据项目规模自动调整策略
```

```bash
# 测试 3: 持续学习
# 初始：RL Planner 成功率 70%
# 训练 100 个任务后：成功率 85%
# 训练 500 个任务后：成功率 93%

$ luna --stats
RL Planner Performance:
  Total tasks: 523
  Success rate: 93%
  Avg plan steps: 4.2
  Avg execution time: 3.5 min
  Improvement over RuleBased: +23%
```

---

## 5. 关键技术决策

| 决策 | 方案 | 理由 |
|------|------|------|
| **架构** | TPAR | 全局规划、可回溯、RL友好 |
| **Planner 演进** | Rule → LLM → RL | 渐进式，逐步智能化 |
| **Review 机制** | 基础 + LLM 增强 | 先保证正确性，再提升智能 |
| **并行执行** | Plan 阶段识别 | 提升效率 |
| **上下文管理** | Context Pipeline (Index→Refill→Context) | 检索与生成解耦，大代码库友好 |
| **向量检索** | 简化版（本地 embedding） | 解决概念查询，不做复杂系统 |
| **Trace 记录** | 从 Phase 0 开始 | 为 RL 准备数据 |

---

## 6. 成功标准总结

| Phase | 核心交付 | 关键指标 |
|-------|----------|----------|
| Phase 0 | TPAR 主链 | Task → Plan → Act → Review 完整流程 |
| Phase 1 | Intelligence | 符号查询 <10ms, 准确率 100% |
| Phase 2 | Safety + Trace | 危险命令拦截, 轨迹完整记录 |
| Phase 3 | 流式 + LLM Plan | 实时反馈, LLM 计划质量提升 |
| Phase 4 | Context Pipeline + Hybrid Search | Index→Refill→Context, 大代码库 <100ms |
| Phase 5 | Memory + Config | 个性化, 模式切换 |
| Phase 6 | RL Plan | 成功率 >90%, 持续学习 |

---

## 7. 附录：测试命令速查

```bash
# 基础功能测试
cargo test --package luna-runtime -- test_tpar_loop
cargo test --package luna-intelligence -- test_scope_graph

# 集成测试
cargo test --test integration -- test_full_workflow

# 性能测试
cargo test --release --bench benchmark_symbol_query

# 运行 Luna
export OPENAI_API_KEY="your-key"
cargo run --release
```
