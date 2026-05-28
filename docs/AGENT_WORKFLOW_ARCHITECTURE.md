# AgentScope Java 本体驱动 Agent 工作时序与代码架构

> **文档目的**：描述 AgentScope Java 本体驱动框架（`OntologyDrivenAgent`）的完整工作时序流程、核心组件职责、模块间数据流，以及底层代码架构。
>
> **版本**: v1.0.0
> **日期**: 2026-05-19
> **基于代码版本**: agentscope-java OD-agent 示例

---

## 目录

1. [概述](#1-概述)
2. [架构分层总览](#2-架构分层总览)
3. [模块详解](#3-模块详解)
   - [3.1 入口层：OntologyAgentBuilder 与 OntologyDrivenAgent](#31-入口层ontologyagentbuilder-与-ontologydrivenagent)
   - [3.2 意图识别层：IntentRouter 与 IntentClassification](#32-意图识别层intentrouter-与-intentclassification)
   - [3.3 策略绑定层：IntentBindings 与 IntentBinding](#33-策略绑定层intentbindings-与-intentbinding)
   - [3.4 任务规划层：Planner 与 TaskPlan](#34-任务规划层planner-与-taskplan)
   - [3.5 技能管理层：SkillRegistry 与 SkillManifest](#35-技能管理层skillregistry-与-skillmanifest)
   - [3.6 策略执行层：PolicyEngine](#36-策略执行层policyengine)
   - [3.7 验证层：TaskValidator](#37-验证层taskvalidator)
   - [3.8 审计层：AuditSink](#38-审计层auditsink)
   - [3.9 执行层：ReActAgent（底层引擎）](#39-执行层reactagent底层引擎)
   - [3.10 连接器层：Connector](#310-连接器层connector)
4. [完整工作时序流程](#4-完整工作时序流程)
5. [配置三阶段（M1/M2/M3）](#5-配置三阶段m1m2m3)
6. [关键设计模式](#6-关键设计模式)
7. [数据流图](#7-数据流图)
8. [文件索引](#8-文件索引)

---

## 1. 概述

`OntologyDrivenAgent` 是一个**本体驱动的企业级 Agent 框架**，位于 `agentscope-java/agentscope-examples/OD-agent` 模块。它通过组合（而非继承）的方式，包装了底层的 `ReActAgent` 运行时引擎，并在其外围构建了五层控制平面：

1. **意图识别**（Intent Classification）：将用户输入分类为结构化意图
2. **策略绑定**（Intent Binding）：将意图映射为五种执行策略之一
3. **任务规划**（Task Planning）：将策略展开为 DAG 任务列表或白名单
4. **本体验证**（Ontology Validation）：在工具调用前后校验本体一致性
5. **策略执行**（Policy Enforcement）：RBAC + HITL 门控

这套架构的核心设计原则是：**确定性逻辑优先（规则引擎），LLM 能力兜底**。所有路由决策在到达 ReAct 执行层之前都已完成，ReAct 只需要执行明确的任务 DAG 或白名单工具集。

---

## 2. 架构分层总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OntologyDrivenAgent                                 │
│                           (本体驱动 Agent — 控制平面)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ask(userInput) 入口                               │   │
│  │                                                                      │   │
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐  │   │
│  │  │ Intent   │───▶│ IntentBinding │───▶│ Planner  │───▶│ Decision│  │   │
│  │  │ Router   │    │   策略绑定     │    │ 任务规划  │    │ 路由分发 │  │   │
│  │  └──────────┘    └──────────────┘    └──────────┘    └────┬────┘  │   │
│  │                                                              │        │   │
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────┐         │        │   │
│  │  │ Policy   │    │ AuditSink   │    │ Task     │         │        │   │
│  │  │ Engine   │    │ 审计日志     │    │ Validator│         │        │   │
│  │  └──────────┘    └──────────────┘    └──────────┘         │        │   │
│  └──────────────────────────────────────────────────────────────│────────┘   │
│                                                                     │         │
└─────────────────────────────────────────────────────────────────────│─────────┘
                                                                      │
                           ┌──────────────────────────────────────────┐
                           │         ReActAgent（底层运行时引擎）        │
                           │         agentscope-core 模块              │
                           │                                          │
                           │  Memory ── Toolkit ── Model ── Hooks    │
                           │                                          │
                           │  ReAct Loop:                              │
                           │  user_input → reasoning → tool_call →   │
                           │  tool_result → reasoning → response      │
                           └──────────────────────────────────────────┘
```

### 各层职责对比

| 层级 | 组件 | 职责 | 是否 LLM 调用 |
|------|------|------|-------------|
| L1 | IntentRouter | 关键词/正则分类 → IntentClassification | 可选（LLM_INFER fallback） |
| L2 | IntentBinding | 意图 → 五种策略路由（REJECT/HITL/FIXED_SKILL/SKILL_WHITELIST/LLM_FREE） | 无 |
| L3 | Planner | 策略展开 → TaskPlan（DAG 或白名单） | 无 |
| L4 | PolicyEngine | RBAC 检查 + HITL 门控 | 无 |
| L5 | TaskValidator | 本体一致性硬闸（preflight + validate） | 无 |
| L6 | ReActAgent | 底层 ReAct 执行循环 | 是 |

---

## 3. 模块详解

### 3.1 入口层：OntologyAgentBuilder 与 OntologyDrivenAgent

**文件路径**：

- `OntologyAgentBuilder.java`
- `OntologyDrivenAgent.java`

#### 3.1.1 OntologyAgentBuilder（Fluent Builder）

`OntologyAgentBuilder` 是一个**流式构建器**，遵循渐进式配置原则。它支持三种配置模式（M1/M2/M3），在 build() 时自动应用自动生成逻辑。

**关键字段**：

```java
OntologySpec ontology;           // 本体定义（必需）
Model model;                     // LLM 模型（必需）
IntentConfig intents;            // 意图配置（可选）
SkillConfig skills;              // 技能配置（可选）
RagConfig rag;                   // RAG 配置（可选）
PolicyConfig policies;           // 策略配置（可选）
IntentBindings bindings;        // 意图绑定配置（可选）
Planner planner;                 // 任务规划器（可选）
QueryParserConfig queryParserConfig; // 查询解析配置（可选）
Memory memory;                   // 内存实现（可选）
AuditSink auditSink;            // 审计日志（可选）
Toolkit toolkit;                // 工具箱（可选）
```

**三种配置模式**：

| 模式 | 说明 | 自动生成内容 |
|------|------|------------|
| `ONTOLOGY_ONLY` | 仅本体，LLM 全权处理意图路由（M1） | 无 |
| `WITH_INTENTS` | 本体 + 自动生成意图规则（M2） | IntentConfig + IntentBindings |
| `FULL` | 完整配置（M3） | IntentConfig + IntentBindings + SkillConfig |

**使用示例**：

```java
// M1：最简配置
var agent = new OntologyAgentBuilder()
        .ontology("classpath:ontology/dtp_procurement.yaml")
        .model(OpenAIChatModel.builder().apiKey(key).build())
        .build();

// M3：完整配置
var agent = new OntologyAgentBuilder()
        .ontology("classpath:ontology/dtp_procurement.yaml")
        .model(model)
        .configurationMode(ConfigurationMode.FULL)
        .policies(policyConfig)
        .auditSink(kafkaAuditSink)
        .build();
```

#### 3.1.2 OntologyDrivenAgent（核心类）

`OntologyDrivenAgent` 通过**组合**而非继承的方式包装了 `ReActAgent`。它的构造函数完成两阶段初始化：

**Phase 1（必需）**：本体加载与一致性校验

```java
this.ontology = b.ontology;
this.ontology.validateInternalConsistency();
```

**Phase 2（可选渐进）**：各模块初始化，未配置时使用 no-op 单例

```java
this.intents     = b.intents    != null ? b.intents    : IntentConfig.EMPTY;
this.skills      = b.skills     != null ? b.skills     : SkillConfig.EMPTY;
this.policies    = b.policies   != null ? b.policies   : PolicyConfig.DEFAULT;
this.router      = intents.isEmpty() ? IntentRouter.LLM_INFER
                                     : new RuleBasedIntentRouter(intents);
this.skillRegistry = new SkillRegistry(skills, ontology);  // 启动时一致性校验
this.policyEngine  = new PolicyEngine(policies);
this.validator     = new TaskValidator(ontology);
this.auditSink     = b.auditSink != null ? b.auditSink  : AuditSink.SLF4J;
this.planner       = b.planner   != null ? b.planner    : new RuleBasedPlanner();
```

**公共 API**：

```java
public Mono<Msg> ask(String userInput)   // 主入口：classify → plan → delegate
public Agent delegate()                  // 暴露底层 ReActAgent，用于 AG-UI 注册
public Model model()                      // 暴露模型，供外部编排器使用
public Toolkit toolkit()                  // 暴露工具箱，供外部编排器使用
```

---

### 3.2 意图识别层：IntentRouter 与 IntentClassification

**文件路径**：

- `intent/IntentRouter.java`（SPI 接口）
- `intent/RuleBasedIntentRouter.java`（默认实现）
- `intent/ConstrainedLlmIntentInferer.java`（LLM Fallback）
- `intent/IntentClassification.java`（结果记录）

#### 3.2.1 IntentRouter（SPI 接口）

```java
@FunctionalInterface
public interface IntentRouter {
    IntentClassification classify(String userInput);
    IntentRouter LLM_INFER = input -> IntentClassification.UNKNOWN;
}
```

当 `IntentConfig` 为空时，使用 `LLM_INFER` no-op 实现，将意图推断委托给 ReActAgent 层的 LLM。

#### 3.2.2 RuleBasedIntentRouter（默认实现）

**评分算法**：

```
score = Σ(keyword_weight if substring matched) + 0.3 * (any regex matched ? 1 : 0)
```

- 关键词匹配：输入文本包含关键词 → 加权重
- 正则匹配：正则捕获组 → 提取实体槽位（entitySlots）
- 最高分规则胜出，置信度 = `min(1.0, score)`

**返回结构** `IntentClassification`：

```java
record IntentClassification(
    String type,             // 意图类型标签（如 "query_po"）
    double confidence,      // 置信度 [0, 1]
    Map<String, String> entities,  // 提取的实体（如 {orderId: "M-001"}）
    List<Candidate> candidates     // 排名前 3 的候选意图
)
```

#### 3.2.3 ConstrainedLlmIntentInferer（LLM Fallback）

当 `IntentConfig.fallback() == "LLM_CONSTRAINED"` 且规则路由返回 UNKNOWN 时，调用 LLM 在已知意图类型白名单内进行受限推断：

```java
public static Mono<IntentClassification> infer(
        Model model, IntentConfig config, String userInput) {
    // 构建 prompt：列举所有已配置的意图类型和关键词
    // LLM 从白名单中选择，不允许发明新意图
}
```

---

### 3.3 策略绑定层：IntentBindings 与 IntentBinding

**文件路径**：

- `planner/IntentBindings.java`
- `planner/IntentBinding.java`

#### 3.3.1 IntentBindings（绑定表）

`IntentBindings` 是一个**查询表**，将意图类型映射到 `IntentBinding` 配置行。默认 fallback 为 **REJECT**（白名单模型）。

```java
public final class IntentBindings {
    private final Map<String, IntentBinding> byType;  // intentType → binding
    private final IntentBinding fallback;               // 未匹配时的默认行为

    public IntentBinding resolve(String intentType) {
        if (intentType == null || "UNKNOWN".equals(intentType)) return fallback;
        return byType.getOrDefault(intentType, fallback);
    }
}
```

#### 3.3.2 IntentBinding（五种策略）

```java
public record IntentBinding(
    String intentType,           // 匹配的意图类型
    Strategy strategy,           // 五种策略之一
    String skillId,              // FIXED_SKILL 专用：单技能 ID
    List<String> skillWhitelist, // SKILL_WHITELIST 专用：技能白名单
    List<String> requiredSlots,  // 必需槽位，缺失则升级为 HITL
    String dataSensitivity,      // PUBLIC / INTERNAL / PII
    String rejectMessage,        // REJECT 时返回给用户的文本
    String hitlPrompt,           // HITL_CONFIRM 时确认问题模板
    String rationaleTemplate,    // 任务节点理由模板（支持 {slot} 占位符）
    double confidenceFloor       // 置信度门槛，低于此值升级为 HITL_CONFIRM
) {
    public enum Strategy {
        REJECT,           // 直接拒绝
        HITL_CONFIRM,     // 暂停，等待人工确认
        FIXED_SKILL,      // 确定性执行单技能 DAG
        SKILL_WHITELIST,  // LLM 在技能白名单内自主编排
        LLM_FREE          // 完全自由，无约束
    }
}
```

---

### 3.4 任务规划层：Planner 与 TaskPlan

**文件路径**：

- `planner/Planner.java`（SPI 接口）
- `planner/RuleBasedPlanner.java`（默认实现）
- `planner/TaskPlan.java`（规划结果）

#### 3.4.1 Planner（SPI 接口）

```java
@FunctionalInterface
public interface Planner {
    TaskPlan plan(
            IntentClassification intent,
            OntologySpec ontology,
            SkillRegistry skills,
            IntentBindings bindings);

    Planner DELEGATE_ALL = ...;  // no-op：所有输入直接委托给 LLM
}
```

#### 3.4.2 RuleBasedPlanner（默认实现）

`RuleBasedPlanner` 接收 `IntentClassification`，通过 `IntentBindings` 确定策略，然后**零 LLM** 地生成 `TaskPlan`。

**规划决策流程**：

```
输入: IntentClassification (type, confidence, entities)
      IntentBinding (strategy, skillId, confidenceFloor, requiredSlots, ...)

Step 1: 歧义检测
  if isAmbiguous(gap=0.15, minConf=0.35) → CLARIFY（返回澄清问句）

Step 2: 置信度门槛检查（L1→L3 升级）
  if (strategy != REJECT && confidence < confidenceFloor) → HITL_CONFIRM

Step 3: 策略路由
  REJECT        → 返回拒绝理由
  HITL_CONFIRM → 返回确认问题 + 缺失槽位列表
  FIXED_SKILL  → materializeFixedSkill() 展开 DAG
  SKILL_WHITELIST → materializeWhitelist() 生成工具白名单
  LLM_FREE     → DELEGATE_LLM（直接进入 ReAct）
```

#### 3.4.3 TaskPlan（规划结果）

```java
record TaskPlan(
    IntentClassification intent,     // 原始分类结果
    String boundIntentType,           // 绑定后的意图类型
    Decision decision,                // 门控决策
    List<TaskNode> tasks,             // FIXED_SKILL 的 DAG 节点列表
    List<String> allowedTools,       // 允许的工具/动作 ID 列表
    String rejectedReason,            // REJECT 时的理由
    String hitlPrompt,               // HITL 确认问题
    List<String> warnings            // 警告信息
) {
    public enum Decision {
        REJECT,        // 拒绝执行
        CLARIFY,       // 需要澄清（歧义）
        HITL_CONFIRM,  // 需要人工确认
        EXECUTE,       // 执行 DAG 或白名单
        DELEGATE_LLM,  // 完全委托给 LLM
        SLOT_MISSING   // 必需槽位缺失（FIXED_SKILL 中）
    }
}
```

#### 3.4.4 TaskNode（任务节点）

每个 SkillManifest 的 flow 步骤被展开为一个 `TaskNode`：

```java
record TaskNode(
    String id,            // 节点 ID（如 "step_1"）
    String actionId,      // 调用的本体动作 ID（如 "purchase_orders/list"）
    Map<String, Object> params,   // 参数（{slot} 占位符已替换为具体值）
    List<String> dependsOn,        // 依赖节点 ID（DAG 拓扑排序用）
    String onFailure,    // 失败策略（skip / abort）
    String when,         // 条件执行表达式
    String rationale,    // 执行理由（用于 ReAct 的 tool call 描述）
    boolean hitl,        // 是否需要人工确认
    List<String> allowedTools,    // 该步骤允许的工具列表
    String riskTag       // 风险标签（HIGH / LOW）
)
```

#### 3.4.5 materializeFixedSkill：技能展开

FIXED_SKILL 策略将 SkillManifest 的 flow 展开为验证后的 DAG：

```java
// 输入：SkillManifest { id="po_query", flow=[
  { "action": "purchase_orders/list", "params": {material_id: "{material_id}"} },
  { "action": "purchase_orders/get_by_id", "depends_on": ["step_1"] }
]}
// 输出：TaskPlan { decision=EXECUTE, tasks=[step_1, step_2], allowedTools=[...] }
```

展开过程：
1. 遍历 flow 步骤，提取 action、params、depends_on
2. `{slot}` 占位符替换为 IntentClassification 中提取的实体值
3. 继承 HITL 标志（`requires_hitl` / `actSpec.requiresHitl()` / `isWrite()` / PII）
4. 继承风险标签（write action → HIGH）
5. DAG 拓扑排序验证（循环检测）
6. 生成 allowedTools 列表

---

### 3.5 技能管理层：SkillRegistry 与 SkillManifest

**文件路径**：

- `skill/SkillRegistry.java`
- `skill/SkillManifest.java`

#### 3.5.1 SkillManifest（技能定义）

```java
record SkillManifest(
    String id,                      // 技能唯一 ID
    String version,                  // 版本号
    String title,                   // 人类可读标题
    List<String> triggerIntents,    // 触发意图列表
    List<String> utterances,        // 示例话语
    List<Map<String, Object>> flow, // DAG 步骤定义
    boolean needsHitl               // 全局 HITL 标志
)
```

#### 3.5.2 SkillRegistry（启动时一致性校验）

**核心职责**：

1. **启动时校验**：所有 SkillManifest 引用的 action 必须存在于 OntologySpec
2. **意图索引**：按 triggerIntents 建立倒排索引
3. **系统提示注入**：生成 Skill 目录片段供 buildSystemPrompt 使用

```java
public SkillRegistry(SkillConfig skills, OntologySpec ontology) {
    for (SkillManifest m : skills.skills()) {
        // 校验 1：每个 action 必须存在于本体
        for (String actionId : m.referencedActions()) {
            if (ontology.actionById(actionId) == null) {
                throw new IllegalStateException(
                    "Skill '" + m.id() + "' references undeclared action '" + actionId + "'");
            }
        }
        // 索引
        byId.put(m.id(), m);
        for (String intent : m.triggerIntents()) {
            byIntent.computeIfAbsent(intent, k -> new ArrayList<>()).add(m);
        }
    }
}
```

---

### 3.6 策略执行层：PolicyEngine

**文件路径**：

- `policy/PolicyEngine.java`

#### 3.6.1 PolicyEngine

PolicyEngine 解释 `PolicyConfig`，提供两个核心判断：

```java
public class PolicyEngine {
    // RBAC 检查：用户角色是否允许调用该意图
    public boolean canInvoke(String intent, List<String> roles) { ... }

    // HITL 判断：是否需要人工确认
    public boolean requiresHitl(ActionSpec action, double confidence) {
        if (action == null) return false;
        if (cfg.alwaysConfirm().contains(action.id())) return true;
        if (action.isWrite()) return true;         // 所有写操作默认 HITL
        return confidence < cfg.confidenceGate();   // 低置信度升级
    }
}
```

---

### 3.7 验证层：TaskValidator

**文件路径**：

- `validator/TaskValidator.java`

#### 3.7.1 TaskValidator（本体硬闸）

TaskValidator 是**常开**的本体一致性门控，不依赖于任何可选配置。它在工具调用前进行 preflight 检查：

```java
public class TaskValidator {
    private final OntologySpec ontology;

    public Verdict preflight(String actionId, Map<String, Object> args) {
        ActionSpec action = ontology.actionById(actionId);
        if (action == null) {
            return Verdict.fail("Action '" + actionId + "' is not declared in ontology");
        }
        if (args != null) {
            for (String k : args.keySet()) {
                if (!action.inputs().containsKey(k)) {
                    return Verdict.fail(
                        "Parameter '" + k + "' is not declared on action '" + actionId + "'");
                }
            }
        }
        return Verdict.OK;
    }

    public record Verdict(boolean ok, String reason) {
        public static final Verdict OK = new Verdict(true, null);
        public static Verdict fail(String reason) { return new Verdict(false, reason); }
    }
}
```

---

### 3.8 审计层：AuditSink

**文件路径**：

- `audit/AuditSink.java`

#### 3.8.1 AuditSink（SPI 接口）

```java
@FunctionalInterface
public interface AuditSink {
    void write(String agentId, String eventType, String detail);

    AuditSink SLF4J = ...;  // 默认实现：写入 SLF4J 日志
}
```

M3 阶段可替换为 Kafka / 文件滚动审计日志。当前架构中记录的关键事件：

- `AGENT_STARTED`：Agent 启动
- `REQUEST`：每次 ask() 请求（意图类型 + 置信度 + 规划决策 + RAG 命中数）
- `QUERY_PARSING`：链式查询解析结果

---

### 3.9 执行层：ReActAgent（底层引擎）

**文件路径**：

- `agentscope-core/src/main/java/io/agentscope/core/ReActAgent.java`

#### 3.9.1 ReActAgent 定位

`ReActAgent` 是 AgentScope Java 的**底层 ReAct 运行时引擎**，由 `OntologyDrivenAgent` 通过组合（composition）方式使用。它负责：

- **ReAct 循环**：reasoning → tool_call → tool_result → reasoning → response
- **记忆管理**：短期记忆（InMemoryMemory / AutoContextMemory）+ 长期记忆（Mem0）
- **工具执行**：通过 Toolkit 调用 AgentTool / PrimitiveTool
- **Hook 系统**：PreReasoning / PostReasoning / PreActing / PostActing 等生命周期钩子
- **结构化输出**：StructuredOutputCapableAgent 提供类型安全的输出生成
- **HITL 中断**：PostReasoningEvent / PostActingEvent 中调用 stopAgent() 暂停
- **流式响应**：支持 Flux<Event> 流式推送中间结果

#### 3.9.2 构造 ReActAgent

`OntologyDrivenAgent` 通过 `OntologyAgentBuilder` 配置 ReActAgent：

```java
ReActAgent.Builder rb = ReActAgent.builder()
        .name(agentName)
        .sysPrompt(buildSystemPrompt(b.userSysPromptPrefix))  // 本体 + Skill 目录
        .model(this.model)
        .toolkit(this.toolkit)
        .memory(memory)
        .maxIters(b.maxIters);

if (memory instanceof AutoContextMemory) {
    rb.hook(new AutoContextHook());
}
if (ltmHook != null) {
    rb.hook(ltmHook);
}
this.delegate = rb.build();
```

#### 3.9.3 系统提示构建

`buildSystemPrompt()` 将三部分内容注入系统提示：

```java
private String buildSystemPrompt(String userPrefix) {
    StringBuilder sb = new StringBuilder();
    // 1. Agent 角色定义
    sb.append("你是一个基于本体驱动的 ").append(ontology.domain()).append(" 域智能体。\n");
    sb.append("请严格遵守本体定义；不得引用未声明的实体、字段或动作。\n\n");
    // 2. 本体片段（实体、动作、关系定义）
    sb.append(ontology.toPromptFragment());
    // 3. Skill 目录（优先显示与当前意图匹配的技能）
    if (!skillRegistry.isEmpty()) {
        sb.append('\n').append(skillRegistry.fragmentFor(null));
    }
    // 4. 用户自定义前缀
    if (userPrefix != null && !userPrefix.isBlank()) {
        sb.append("\n## 智能体系统提示词\n").append(userPrefix).append('\n');
    }
    return sb.toString();
}
```

---

### 3.10 连接器层：Connector

**文件路径**：

- `connector/Connector.java`

#### 3.10.1 Connector（SPI 接口）

Connector 是 Agent 与后端业务服务之间的**桥接层**：

```java
public interface Connector {
    String getId();                      // 连接器 ID（如 "procurement"）
    String getDescription();             // 人类可读描述
    List<ToolSpec> getToolSpecs();       // 暴露的 MCP 工具 schema
    Object execute(String actionId, Map<String, Object> params);  // 执行动作
    List<String> getSupportedActions(); // 支持的本体动作 ID 列表
}
```

---

## 4. 完整工作时序流程

### 4.1 ask() 主流程时序图

```
用户: "帮我查一下 M-001 物料的采购订单"
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 OntologyDrivenAgent.ask()                            │
│                                                                      │
│  [1] askWithChainParser / classifyWithFallback                       │
│      │                                                               │
│      ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ RuleBasedIntentRouter.classify(userInput)                       │  │
│  │                                                               │  │
│  │  - 遍历 IntentConfig.intents[]                                │  │
│  │  - 关键词权重累加（score += keyword_weight）                    │  │
│  │  - 正则匹配 + 实体提取（entities）                             │  │
│  │  - 返回 IntentClassification(type="query_mo", conf=0.85,      │  │
│  │    entities={material_id: "M-001"})                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      │                                                               │
│      ▼ LLM_CONSTRAINED fallback（如需要）                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ ConstrainedLlmIntentInferer.infer()                            │  │
│  │  - 构建意图白名单 prompt                                       │  │
│  │  - 调用 LLM 在已知意图内选择                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      │                                                               │
│      ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ RuleBasedPlanner.plan(classification, ontology, ...)          │  │
│  │                                                               │  │
│  │  Step 1: 歧义检测                                              │  │
│  │    if isAmbiguous(0.15, 0.35) → CLARIFY                       │  │
│  │                                                               │  │
│  │  Step 2: 置信度门槛检查                                        │  │
│  │    if conf < binding.confidenceFloor → HITL_CONFIRM           │  │
│  │                                                               │  │
│  │  Step 3: 策略路由                                              │  │
│  │    IntentBinding.resolve("query_mo")                          │  │
│  │    → Strategy.FIXED_SKILL, skillId="mo_query_skill"           │  │
│  │                                                               │  │
│  │  Step 4: materializeFixedSkill                                │  │
│  │    - 获取 SkillManifest("mo_query_skill")                     │  │
│  │    - 展开 flow DAG: [list_mo, get_by_id]                       │  │
│  │    - {material_id} 占位符替换为 "M-001"                        │  │
│  │    - 继承 HITL 标志（write action → hitl=true）               │  │
│  │    - 拓扑排序验证                                              │  │
│  │    → TaskPlan { decision=EXECUTE, tasks=[step_1, step_2] }   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      │                                                               │
│      ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ AuditSink.write("REQUEST", intent=query_mo, conf=0.85, ...)   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      │                                                               │
│      ▼                                                               │
│  [2] Decision 路由分发                                               │
│      │                                                               │
│      ├── REJECT        → return canned("⛔ " + reason)              │
│      ├── CLARIFY       → return canned("❓ " + prompt)              │
│      ├── HITL_CONFIRM  → return canned("🙋 请确认...")             │
│      │                                                               │
│      └── EXECUTE / DELEGATE_LLM → 委托 ReActAgent                   │
│          │                                                           │
│          ▼                                                           │
│      ┌──────────────────────────────────────────────────────────┐  │
│      │ TimeContextPrefixer.prefix(userInput)                     │  │
│      │  - 添加时间上下文前缀（当前日期/星期等）                    │  │
│      └──────────────────────────────────────────────────────────┘  │
│          │                                                           │
│          ▼                                                           │
│      ┌──────────────────────────────────────────────────────────┐  │
│      │ ReActAgent.call(Msg)                                       │  │
│      │                                                            │  │
│      │  ReAct Loop:                                              │  │
│      │  ┌──────────────────────────────────────────────────────┐ │  │
│      │  │ iter 1:                                              │ │  │
│      │  │   PreReasoning → LLM(reasoning) → PostReasoning      │ │  │
│      │  │   PreActing → ToolCall(purchase_orders/list)         │ │  │
│      │  │          ↓ TaskValidator.preflight() 检查本体一致性  │ │  │
│      │  │   ToolResult → PostActing                           │ │  │
│      │  ├──────────────────────────────────────────────────────┤ │  │
│      │  │ iter 2:                                              │ │  │
│      │  │   PreReasoning → LLM(reasoning with tool result)     │ │  │
│      │  │   PreActing → ToolCall(purchase_orders/get_by_id)    │ │  │
│      │  │          ↓ TaskValidator.preflight()                │ │  │
│      │  │   ToolResult → PostActing                           │ │  │
│      │  ├──────────────────────────────────────────────────────┤ │  │
│      │  │ iter 3:                                              │ │  │
│      │  │   PreReasoning → LLM(reasoning) → PostReasoning      │ │  │
│      │  │   [no more tools] → PostSummary → response           │ │  │
│      │  └──────────────────────────────────────────────────────┘  │
│      │                                                            │  │
│      │  Memory: [user_msg, assistant_reasoning, tool_call,        │  │
│      │           tool_result, ..., final_response]                │  │
│      │                                                            │  │
│      │  Hooks: AutoContextHook（自动上下文压缩）                  │  │
│      │         OdLongTermMemoryHook（Mem0 长期记忆）              │  │
│      └──────────────────────────────────────────────────────────┘  │
│          │                                                           │
│          ▼                                                           │
│      return Msg (assistant response)                                │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
用户收到助手回复
```

### 4.2 各策略执行路径

| 决策 | ask() 行为 | ReActAgent 参与 |
|------|-----------|---------------|
| **REJECT** | 直接返回拒绝消息（canned） | 不参与 |
| **CLARIFY** | 返回歧义消解问句 | 不参与 |
| **HITL_CONFIRM** | 返回确认问句（可带缺失槽位列表） | 不参与 |
| **SLOT_MISSING** | 返回槽位缺失警告 | 不参与 |
| **EXECUTE** | 构建 Msg，附加时间上下文，委托 delegate.call() | 全程参与 |
| **DELEGATE_LLM** | 同 EXECUTE，但无工具白名单约束 | 全程参与 |

---

## 5. 配置三阶段（M1/M2/M3）

| 阶段 | 说明 | 关键差异 |
|------|------|---------|
| **M1** | 仅本体，LLM 全权 | IntentConfig = EMPTY，Planner = DELEGATE_ALL，SkillRegistry = 空 |
| **M2** | 本体 + 意图规则 | RuleBasedIntentRouter + RuleBasedPlanner，支持 FIXED_SKILL/HITL/REJECT |
| **M3** | 完整配置 | M2 + SkillConfig + PolicyEngine + AuditSink + LongTermMemory |

**M2 的关键能力**：

- 意图识别从 LLM 黑盒变为规则白盒
- 写操作（HITL）自动升级为需确认
- 置信度门槛控制低匹配度时的行为
- DAG 验证在执行前完成（fail-fast）

**自动配置流程**（OntologyAgentBuilder.build()）：

```java
private void applyAutoConfiguration() {
    switch (configurationMode) {
        case ONTOLOGY_ONLY -> { /* no-op */ }
        case WITH_INTENTS -> {
            if (this.intents == null) {
                var result = intentAutoGenerator.generateWithBindings(
                    ontology, GenerationOptions.DEFAULT);
                this.intents = result.config();
                if (this.bindings == null) this.bindings = result.bindings();
            }
        }
        case FULL -> {
            // M2 内容
            if (this.intents == null) { ... }
            // 额外生成技能配置
            if (this.skills == null) {
                var skills = skillAutoGenerator.generate(
                    ontology, GenerationOptions.DEFAULT);
                this.skills = new SkillConfig(skills);
            }
        }
    }
}
```

---

## 6. 关键设计模式

### 6.1 渐进式配置（Progressive Enhancement）

所有可选模块均有 no-op 默认实现：

```java
this.intents   = b.intents   != null ? b.intents   : IntentConfig.EMPTY;
this.policies  = b.policies  != null ? b.policies  : PolicyConfig.DEFAULT;
this.router    = intents.isEmpty() ? IntentRouter.LLM_INFER
                                   : new RuleBasedIntentRouter(intents);
this.auditSink = b.auditSink != null ? b.auditSink : AuditSink.SLF4J;
```

### 6.2 白名单安全模型

默认 fallback = REJECT。任何未在 `IntentBindings` 中显式声明的意图都会被拒绝。这确保了 Agent 只能执行明确授权的操作。

### 6.3 组合而非继承

`OntologyDrivenAgent` 不继承 `ReActAgent`，而是通过 `delegate()` 方法暴露底层 agent。这保持了 `ReActAgent` 的包内可见性（构造函数是 package-private），同时允许框架层添加控制平面。

### 6.4 SPI + 默认实现

每个核心接口都有默认实现（RuleBasedXxx），允许替换为自定义实现：

```java
IntentRouter  → RuleBasedIntentRouter / ConstrainedLlmIntentInferer
Planner       → RuleBasedPlanner / (future: HybridPlanner)
AuditSink     → SLF4J / Kafka / FileRolling
```

### 6.5 本体即宪法

`OntologySpec` 在启动时通过 `TaskValidator` 强制执行：
- 所有 action 必须在本体内声明
- 所有参数必须匹配 action 的 input schema
- 无本体即无工具

---

## 7. 数据流图

### 7.1 请求级数据流

```
用户输入 "查 M-001 的采购订单"
     │
     ▼
IntentClassification(type="query_po", conf=0.78, entities={material_id:"M-001"})
     │
     ▼
IntentBinding(strategy=FIXED_SKILL, skillId="po_query_skill", confidenceFloor=0.35)
     │
     ▼
TaskPlan(decision=EXECUTE,
         tasks=[
           TaskNode(id="step_1", action="purchase_orders/list",
                    params={material_id="M-001"}, dependsOn=[]),
           TaskNode(id="step_2", action="purchase_orders/get_by_id",
                    params={order_id="${step_1.order_id}"}, dependsOn=["step_1"])
         ],
         allowedTools=["purchase_orders/list", "purchase_orders/get_by_id"])
     │
     ▼
RagRetriever.retrieve(userInput, topK=5) → List<Knowledge>
     │
     ▼
AuditSink.write(agentId, "REQUEST", "intent=query_po conf=0.78 decision=EXECUTE ...")
     │
     ▼
TimeContextPrefixer.prefix(userInput) → "当前日期: 2026-05-19, 用户输入: ..."
     │
     ▼
Msg(name="user", role=USER, content="当前日期: ...")
     │
     ▼
ReActAgent.call(Msg)
     │
     ├─▶ Memory.append(user_msg)
     │
     ├─▶ Loop (maxIters):
     │   │
     │   ├─▶ PreReasoning Hook
     │   │
     │   ├─▶ LLM.generate(reasoning, system_prompt)
     │   │   system_prompt = "你是一个基于本体驱动的 procurement 域智能体。\n"
     │   │                   + ontology.toPromptFragment()
     │   │                   + Skill目录 + RAG hits
     │   │
     │   ├─▶ PostReasoning Hook (HITL stopAgent 可能在此触发)
     │   │
     │   ├─▶ ToolCall(actionId, params)
     │   │   │
     │   │   └─▶ TaskValidator.preflight(actionId, params)
     │   │       │
     │   │       ├─▶ ontology.actionById(actionId) != null ?
     │   │       └─▶ params.keySet() ⊆ action.inputs().keySet() ?
     │   │
     │   ├─▶ Connector.execute(actionId, params)
     │   │   │
     │   │   └─▶ 后端服务调用（DB/API）
     │   │
     │   ├─▶ Memory.append(tool_call_msg, tool_result_msg)
     │   │
     │   └─▶ PostActing Hook
     │
     ├─▶ Memory.append(assistant_final_response)
     │
     └─▶ return Msg(role=ASSISTANT, content=final_response)
```

### 7.2 系统提示注入时机

系统提示在 `OntologyDrivenAgent` 构造时**一次性构建**（静态），包含：

```
┌────────────────────────────────────────────────────────────────┐
│  你是一个基于本体驱动的 procurement 域智能体。                    │
│  请严格遵守本体定义；不得引用未声明的实体、字段或动作。          │
│                                                                │
│  ## 本体定义（ontology.toPromptFragment()）                     │
│  ### 实体                                                     │
│  - Material: { id, name, unit, category, supplier_id }        │
│  - PurchaseOrder: { id, material_id, quantity, status, ... }  │
│  ### 动作                                                     │
│  - purchase_orders/list: 查询采购订单列表                       │
│    inputs: { material_id: string?, date_from: string?, ... }   │
│  - purchase_orders/get_by_id: 按 ID 查询                       │
│    inputs: { order_id: string }                               │
│                                                                │
│  ## Skill 目录                                                 │
│  ★ po_query_skill — 采购订单查询（intents: query_po）          │
│    po_create_skill — 创建采购订单（intents: create_po）        │
│                                                                │
│  [用户自定义 sysPromptPrefix]                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. 文件索引

### 8.1 OD-agent 框架核心（`agentscope-examples/OD-agent`）

| 文件 | 职责 |
|------|------|
| `ontology/OntologyDrivenAgent.java` | 核心控制平面主类，ask() 入口 |
| `ontology/OntologyAgentBuilder.java` | 流式构建器，支持 M1/M2/M3 配置 |
| `intent/IntentRouter.java` | 意图路由器 SPI 接口 |
| `intent/RuleBasedIntentRouter.java` | 关键词/正则意图分类实现 |
| `intent/ConstrainedLlmIntentInferer.java` | LLM 受限意图推断（fallback） |
| `intent/IntentClassification.java` | 意图分类结果记录 |
| `planner/Planner.java` | 任务规划器 SPI 接口 |
| `planner/RuleBasedPlanner.java` | 零 LLM 任务规划实现 |
| `planner/TaskPlan.java` | 规划结果：DAG + 决策 + 白名单 |
| `planner/IntentBindings.java` | 意图 → 策略绑定表 |
| `planner/IntentBinding.java` | 单条绑定记录（五种策略） |
| `skill/SkillRegistry.java` | 技能注册与启动时一致性校验 |
| `skill/SkillManifest.java` | 技能定义记录 |
| `policy/PolicyEngine.java` | RBAC + HITL 策略执行引擎 |
| `validator/TaskValidator.java` | 本体硬闸（preflight check） |
| `audit/AuditSink.java` | 审计日志 SPI 接口 |
| `connector/Connector.java` | 后端业务服务连接器 SPI |
| `config/IntentConfig.java` | 意图配置（规则 + 推理预算） |
| `config/SkillConfig.java` | 技能配置 |
| `config/PolicyConfig.java` | 策略配置（RBAC + alwaysConfirm） |
| `config/MemoryConfig.java` | 记忆配置（AutoContext + Mem0） |

### 8.2 底层引擎（`agentscope-core`）

| 文件 | 职责 |
|------|------|
| `ReActAgent.java` | 底层 ReAct 运行时引擎（~1700 行） |
| `StructuredOutputCapableAgent.java` | 结构化输出支持 |
| `memory/InMemoryMemory.java` | 内存消息存储 |
| `memory/autocontext/AutoContextMemory.java` | 自动上下文压缩记忆 |
| `memory/mem0/Mem0LongTermMemory.java` | Mem0 长期记忆集成 |
| `tool/Toolkit.java` | 工具注册与调用分发 |
| `model/Model.java` | LLM 模型抽象接口 |
| `hook/Hook.java` | 生命周期钩子系统 |
| `hook/PreReasoningEvent.java` | 推理前钩子事件 |
| `hook/PostActingEvent.java` | 工具执行后钩子事件 |

---

## 附录：核心类型速查

### A. IntentClassification 完整结构

```java
record IntentClassification(
    String type,                // 意图类型（如 "query_po"）
    double confidence,          // 置信度 [0.0, 1.0]
    Map<String, String> entities,  // 提取的实体 {slot: value}
    List<Candidate> candidates  // Top-3 候选（用于歧义检测）
) {
    record Candidate(String type, double confidence, Map<String, String> entities) {}
    static IntentClassification UNKNOWN = ...;
    boolean isAmbiguous(double gap, double minConfidence);
    boolean isUnknown();
}
```

### B. TaskPlan 完整结构

```java
record TaskPlan(
    IntentClassification intent,
    String boundIntentType,
    Decision decision,          // REJECT/CLARIFY/HITL_CONFIRM/EXECUTE/DELEGATE_LLM/SLOT_MISSING
    List<TaskNode> tasks,      // FIXED_SKILL DAG
    List<String> allowedTools,  // 工具白名单（FIXED_SKILL / SKILL_WHITELIST）
    String rejectedReason,
    String hitlPrompt,
    List<String> warnings
) {
    List<TaskNode> topologicalOrder();  // DAG 拓扑排序（含循环检测）
}
```

### C. Decision → ask() 行为对照

| Decision | delegate.call() | 返回值 |
|---------|-----------------|--------|
| REJECT | 不调用 | `Msg("⛔ " + reason)` |
| CLARIFY | 不调用 | `Msg("❓ " + prompt)` |
| HITL_CONFIRM | 不调用 | `Msg("🙋 " + prompt + "（请回复确认/取消）")` |
| SLOT_MISSING | 不调用 | `Msg("⚠️ " + reason)` |
| EXECUTE | 调用 | ReAct 执行结果 |
| DELEGATE_LLM | 调用 | ReAct 执行结果 |

---

> **文档维护**：每次 OntologyDrivenAgent 或其依赖模块有重大变更时，同步更新本文档对应章节。
