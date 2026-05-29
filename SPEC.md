# SDD 技术文档 - OD Agent

> 本文档记录项目的技术架构、模块拓扑和核心实现。
> 每次任务完成后请检查并更新此文档，保持与项目架构一致。

---

## 0. 语义原生 AI 后端概览

> 参见 `docs/2026-05-28_本体-GraphQL-AI-语义原生后端概念.md`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AI Agent (OD Assistant)                          │
│                    (Claude / DeepSeek / GPT via AgentScope)                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ MCP 协议 (Model Context Protocol)
┌──────────────────────────────▼───────────────────────────────────────────────┐
│                        MCP Server Layer                                        │
│              SemanticSkill → GraphQL → AI-callable Tools                      │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ GraphQL Queries / Mutations
┌──────────────────────────────▼───────────────────────────────────────────────┐
│                        GraphQL API Layer                                       │
│                    (Apollo / Cosmo compatible SDL)                             │
│              Schema = Business Object API Contract                            │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ OSI semantic_model.yaml → GraphQL SDL
┌──────────────────────────────▼───────────────────────────────────────────────┐
│                        OSI / Ontology Layer                                    │
│  Datasets / Relationships / Metrics / Actions / Rules / ai_context           │
│                    src/semantic/osi_model.py                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Pipeline (Step 0 → Step 5):
  Step 0: 业务域 YAML 定义 → OSIModel (DataSet, Relationship, Metric, Action, Rule, AIContext)
  Step 1: OSIModel → GraphQL SDL (GraphQLGenerator)
  Step 2: GraphQL Operations → MCP Tools (build_mcp_tools_from_osi)
  Step 3: ai_context → 向量语义索引 (SemanticIndexer + jieba + MMH3)
  Step 4: AI Agent 查询 → 语义检索 → GraphQL Operation → MCP Tool 调用
  Step 5: 结构化响应返回 Agent → 最终回复用户
```

---

## 1. 系统架构拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OD Agent                                       │
│                   (AgentScope ReAct 智能客服 Agent)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               │                               │
     ▼                               ▼                               ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│  Admin UI    │            │   FastAPI    │            │  AgentScope  │
│  (Embedded    │            │   HTTP API   │            │  ReAct Agent │
│   HTML/JS)   │            │  (src/app.py)│            │  (src/agent) │
└──────────────┘            └──────────────┘            └──────────────┘
                                                              │
                              ┌───────────────────────────────┤
                              │                               │
                              ▼                               ▼
                     ┌──────────────┐              ┌──────────────────────┐
                     │ Skill Manager│              │  DeepSeek Chat API   │
                     │ (src/skills/)│              │  (OpenAI Compatible) │
                     └──────────────┘              └──────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │MathTeacher   │    │ BaseSkill    │    │ Intent Router│
  │  Skill       │    │ (Abstract)   │    │ (Keyword)    │
  └──────────────┘    └──────────────┘    └──────────────┘
         │
         ▼
  ┌──────────────┐
  │ MathEngine   │
  │ (Calculations│
  │  & Solutions)│
  └──────────────┘
```

**核心特性：** 支持运行时配置化热发布的 Agent 后端服务，无需重启即可更新 Agent 行为。

---

## 2. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **框架** | AgentScope 1.0+ | 多智能体对话框架，核心为 ReAct Agent |
| **语言** | Python 3.11+ | 项目主力语言 |
| **HTTP 服务** | FastAPI 0.115+ | ASGI HTTP API 框架 |
| **ASGI 服务器** | Uvicorn 0.32+ | 生产级 ASGI 服务器 |
| **模板引擎** | Jinja2 3.1+ | 服务端 HTML 模板渲染（Admin UI） |
| **前端交互** | HTMX 1.9 + Alpine.js 3.13 | 局部刷新 + 响应式状态管理 |
| **AI 模型** | DeepSeek (deepseek-chat) | 通过 OpenAI 兼容接口调用 |
| **环境管理** | python-dotenv | 环境变量加载 |
| **测试** | pytest + pytest-asyncio + httpx | 异步测试支持 |
| **语义后端** | PyYAML + jieba + mmh3 | OSI YAML 解析 + 中文分词 + 向量哈希 |
| **部署** | Docker, Railway | 容器化与云平台部署 |

---

## 3. 模块拓扑

### 3.1 应用入口

| 文件 | 说明 |
|------|------|
| `src/app.py` | FastAPI 应用工厂 `create_app()`，包含所有 HTTP 路由；Admin UI 通过 Jinja2 模板渲染 |
| `src/agent.py` | AgentScope `ReActAgent` 封装，`CustomerServiceAgent` 类管理 Agent 生命周期和对话逻辑；`_setup_intent_pipeline()` 初始化四阶段流水线 |
| `src/models.py` | `ModelConfig` 数据类，封装 DeepSeek 模型配置 |
| `src/sse_stream.py` | SSE 流式事件定义与构建工具，符合 `docs/SSE流式响应规范.md` |
| `src/__init__.py` | 包初始化 |
| `src/temporal/__init__.py` | temporal 模块初始化 |
| `src/temporal/temporal_parser.py` | **Phase 0**：时间线预解析引擎 |
| `src/intent/intent_classifier.py` | **Phase 1**：意图分类器（关键词评分 + 实体提取 + 多意图） |
| `src/intent/intent_binding.py` | 意图绑定配置（5种策略 + 技能依赖） |
| `src/planner/planner.py` | **Phase 2**：任务规划器（依赖感知 DAG） |
| `src/skills/task_executor.py` | **Phase 3**：任务执行器（拓扑排序 + 跨技能注入） |
| `src/skills/manager.py` | 流水线编排器（`run_pipeline()` 串联 Phase 0-3） |

### 3.2 四阶段流水线

`CustomerServiceAgent` 在初始化时通过 `_setup_intent_pipeline()` 构建并串联四阶段流水线：

```
用户消息
    │
    ▼ Phase 0: TemporalParser.parse()
    TemporalContext { dates, timeline, journey_result, yesterday_note }
    │
    ▼ Phase 1: IntentClassifier.classify_with_context()
    List[IntentClassification] (多意图并行)
    │
    ▼ Phase 2: RuleBasedPlanner.plan() — 依赖感知 DAG 构建
    TaskPlan { decision, tasks[], allowed_tools }
    │
    ▼ Phase 3: TaskExecutor.execute() — 拓扑排序执行
    List[TaskResult] → aggregate_responses() → 最终响应
```

关键意图类型与技能映射：

| Intent Type | Skill | 策略 | 依赖技能 |
|-------------|-------|------|---------|
| `date_range_diff` | Time Converter | FIXED_SKILL | — |
| `day_of_week` | Time Converter | FIXED_SKILL | — |
| `timezone` | Time Converter | FIXED_SKILL | — |
| `math` | Math Teacher | FIXED_SKILL | — |
| `journey_avg` | Math Teacher | FIXED_SKILL | Time Converter |
| `semantic_query` | Semantic Query | FIXED_SKILL | — |

**语义查询场景**（`查一下供应商的采购情况`）：

**语义查询场景**（`查一下供应商的采购情况`）：
1. `IntentClassifier` 识别 `semantic_query` 意图
2. `SemanticSkill.execute()` 调用 `SemanticIndexer.search("查一下供应商的采购情况")`
3. 语义索引检索：jieba 分词 + MMH3 哈希向量 → 找到 `Supplier.ai_context.synonyms` 中 "供应商" 匹配，`PurchaseOrder` 关系检索
4. 返回 top result：score=0.55, dataset=Supplier, tool=query_suppliers, GraphQL fragment
5. `TaskExecutor` 聚合结果 → 用户收到 GraphQL 操作建议

**行程场景处理示例**（`上周五出发...平均每天多远`）：
1. `TemporalParser` 解析 `上周五` → `2026-05-22`，`昨天` → 叙事推断为 `周一+1天`
2. `IntentClassifier` 识别 `journey_avg` 意图，提取 `total_km=2080`, `remaining_km=1000`
3. `RuleBasedPlanner` 构建 DAG：`[Time Converter (dep)]` → `[Math Teacher (journey_avg)]`
4. `TaskExecutor` 先执行 Time Converter 获取天数，再执行 Math Teacher 计算 2080/4=520 km/天

### 3.3 路由层

所有路由在 `src/app.py` 中通过 `create_app()` 动态注册：

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 欢迎页，返回服务信息 |
| `/health` | GET | 健康检查 |
| `/readiness` | GET | 就绪检查 |
| `/liveness` | GET | 存活检查 |
| `/chat` | POST | 对话入口，`stream=false` 返回完整响应，`stream=true` 返回 SSE 流式 chunks |
| `/chat/stream` | POST | 向后兼容别名，等价于 `POST /chat` + `stream=true` |
| `/process` | POST | AgentApp 兼容格式，处理对话请求 |
| `/config` | GET | 获取当前 Agent 配置 |
| `/config` | PUT | 更新 Agent 配置（支持热更新） |
| `/config/reload` | POST | 热重载 Agent 实例 |
| `/config/reset` | POST | 重置对话历史 |
| `/models` | GET | 获取 LLM 厂商列表及模型目录（供前端下拉框） |
| `/intents` | GET/POST | 获取/添加意图路由规则 |
| `/intents/{index}` | GET/PUT/DELETE | 单条意图的查改删 |
| `/intents/detect` | POST | 检测消息匹配的意图 |
| `/skills` | GET | 获取所有已注册技能 |
| `/skills/{skill_name}` | GET | 获取指定技能详情 |
| `/skills/detect` | POST | 检测消息应触发哪个技能 |
| `/semantic/config` | GET/PUT | 获取/更新语义后端配置 |
| `/semantic/schema` | GET | 获取当前 GraphQL SDL |
| `/semantic/tools` | GET | 获取 MCP tools 列表 |
| `/semantic/search` | POST | 语义搜索测试 |
| `/semantic/reload` | POST | 热重载语义后端 |
| `/admin` | GET | Admin 配置管理页面（Jinja2 模板渲染） |
| `/admin/intents/list` | GET | HTMX 片段：意图列表 |
| `/admin/intents/add` | POST | HTMX 片段：添加意图后刷新列表 |
| `/admin/intents/{index}` | DELETE | HTMX 片段：删除意图后刷新列表 |

### 3.4 数据模型层

| 文件 | 说明 |
|------|------|
| `src/app.py` | `ChatRequest`, `ChatResponse`, `ProcessRequest`, `AgentConfigUpdate`, `MessageInput`, `IntentFormData`, `SemanticConfigUpdate` (Pydantic BaseModel) |
| `src/models.py` | `ModelConfig` — 多厂商 LLM 配置封装，支持 thinking 模式热切换 |
| `src/llm_providers.py` | `LLMProvider` — LLM 厂商定义（DeepSeek / Kimi / Minimax / Custom），含模型列表和 thinking 参数模板 |
| `src/skills/base.py` | `BaseSkill` — 技能抽象基类 |
| `src/skills/math_engine.py` | `MathResult` dataclass — 数学解题结果 |
| `config/agent_config.json` | Agent 名称、系统提示词、模型配置的持久化存储 |
| `config/intent_routing.json` | 意图路由规则列表（关键词 + 优先级 + 处理器） |

### 3.4 服务层

| 文件 | 说明 |
|------|------|
| `src/agent.py` | `CustomerServiceAgent` — AgentScope ReActAgent 封装，单例管理；`_load_intent_rules()` 从 JSON 加载路由规则；`chat()` 方法执行意图检测与技能调度 |
| `src/skills/manager.py` | `SkillManager` — 技能注册、意图检测（优先级排序）、`detect_and_execute()` 协调技能执行 |
| `src/skills/base.py` | `BaseSkill` — 技能抽象接口，`execute()`, `match()`, `get_system_prompt()`, `to_dict()` |
| `src/skills/math_teacher.py` | `MathTeacherSkill` — 数学教师技能，50+ 中文关键词，正则模式匹配数学表达式 |
| `src/skills/math_engine.py` | `MathEngine` — 数学解题引擎，支持方程、百分比、数列、导数、积分、几何、统计等题型 |
| `src/skills/time_converter.py` | `TimeConverterSkill` — 时间转换技能，68 个关键词，支持相对/绝对日期、时长、时区、星期几 |

### 3.5 Admin UI

Admin 配置页面采用 Jinja2 模板引擎渲染，配合 HTMX 局部刷新和 Alpine.js 响应式状态管理。

| 文件 | 说明 |
|------|------|
| `templates/admin/base.html` | Admin 页面主模板，含 Alpine.js 全局状态和 deep-chat 组件 |
| `templates/admin/config.html` | 配置 Tab 片段（Agent 名称、System Prompt、Model 配置） |
| `templates/admin/intents.html` | Intent 路由 Tab 片段（含 HTMX 驱动的 CRUD 表单） |
| `templates/admin/skills.html` | Skills Tab 片段（技能列表 + 技能检测测试） |
| `templates/admin/partials/intent_list.html` | HTMX 片段：意图列表渲染 |
| `static/css/admin.css` | Admin 页面样式（从内联 CSS 提取） |

**交互架构**：Alpine.js 驱动表单状态和 Tab 切换；HTMX 处理 Intent 列表的 CRUD 局部刷新（添加/删除后不刷新整页）；deep-chat 组件负责对话功能。

### 3.6 语义原生 AI 后端层

|| 文件 | 说明 |
||------|------|
|| `src/semantic/__init__.py` | 语义模块入口，导出 OSIModel 等核心类型 |
|| `src/semantic/osi_model.py` | OSI 数据模型：DataSet, Relationship, Metric, Action, Rule, AIContext, FieldDefinition, FieldType |
|| `src/semantic/graphql_generator.py` | `GraphQLGenerator` — 将 OSIModel 转换为 GraphQL SDL（含 Query/Mutation/Enum/Input/Subscription 类型） |
|| `src/semantic/mcp_server.py` | `MCPServer` — MCP 工具注册与调用；`MCPTool` — 单个 MCP 工具；`build_mcp_tools_from_osi()` — 从 OSI model 构建 MCP tools |
|| `src/semantic/semantic_indexer.py` | `SemanticIndexer` — ai_context 向量化索引（jieba 分词 + MMH3 哈希）；`SimpleEmbeddingFunction` — 轻量嵌入函数 |
|| `src/semantic/semantic_backend.py` | `SemanticBackend` — 完整语义后端组装器（OSI → GraphQL → MCP → Index）；`load_osi_model()` — YAML 加载；`demo_supplier_model()` — 演示模型 |
|| `src/skills/semantic_skill.py` | `SemanticSkill` — 集成到 SkillManager 的语义查询技能 |

---

## 4. API 接口文档

### 4.1 对话接口

```
POST /chat

Request (blocking, default):
{
  "message": "你好，我想咨询产品",
  "session_id": "可选的会话ID",
  "stream": false
}

Response:
{
  "response": "你好！很高兴为你服务...",
  "session_id": null
}

Request (streaming SSE):
{
  "message": "你好",
  "stream": true
}

Response: Server-Sent Events stream per docs/SSE流式响应规范.md

event: think
data: {"content": "用户问好，我应该礼貌回应。\n"}

event: think_done
data: {"status": "done"}

event: content
data: {"content": "你好！很高兴为你服务。\n"}

event: content
data: {"content": "请问有什么可以帮助你的吗？"}

event: done
data: [DONE]
```

**工具调用示例**（当 Agent 调用 Skill / MCP / RAG 时）：

```
event: tool_call
data: {"type": "skill", "name": "Time Converter", "input": {"start": "2026-05-22", "end": "2026-05-26"}, "id": "call_001"}

event: tool_result
data: {"id": "call_001", "name": "Time Converter", "status": "success", "output": {"days": 5}, "error": null}

event: content
data: {"content": "从上周五到周三一共5天。\n"}

event: done
data: [DONE]
```

deep-chat 组件发送 `{"messages": [{"role": "user", "content": "..."}]}` 格式，
`/chat` 接口自动适配两种格式。Skill pipeline 结果通过 TaskExecutor.execute_stream() 流式输出
（tool_call / tool_result / content 事件），LLM 响应检测 thinking block 并分别输出
think / think_done / content 事件。

### 4.2 配置热更新接口

```
GET /config

Response:  (merged with provider catalog for dropdowns)
{
  "agent_name": "Agent名称",
  "system_prompt": "系统提示词",
  "provider_id": "deepseek",
  "model_name": "deepseek-v4-pro",
  "base_url": "https://api.deepseek.com/v1",
  "temperature": 0.7,
  "max_tokens": 4000,
  "thinking": false,
  "thinking_budget": 4000,
  "providers": [
    {
      "id": "deepseek",
      "name": "DeepSeek",
      "description": "...",
      "models": [...],
      "supports_thinking": true,
      "thinking_param": {"extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}},
      "thinking_disabled_params": ["temperature", "top_p", ...]
    },
    ...
  ],
  "supports_thinking": true
}

PUT /config
Request:
{
  "agent_name": "新名称",
  "system_prompt": "新提示词",
  "llm_config": {
    "provider_id": "deepseek",
    "model_name": "deepseek-v4-pro",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 4000,
    "thinking": true,
    "thinking_budget": 6000
  }
}

POST /config/reload
Response:
{
  "status": "success",
  "message": "Agent reloaded",
  "agent_name": "新名称"
}
```

### 4.3 意图路由接口

```
GET /intents

Response:
[
  {
    "name": "Math Teacher",
    "handler": "math_teacher",
    "keywords": ["计算", "导数", ...],
    "priority": 50,
    "description": "..."
  }
]

POST /intents/detect
Request:
{
  "message": "帮我计算 2x + 3 = 7"
}

Response:
{
  "detected": true,
  "intent": "Math Teacher",
  "handler": "math_teacher",
  "matched_keyword": "计算"
}
```

### 4.4 技能管理接口

```
GET /skills

Response:
{
  "skills": [
    {
      "name": "Math Teacher",
      "description": "A specialized tutor...",
      "keywords_count": 53,
      "priority": 50
    }
  ]
}

POST /skills/detect
Request:
{
  "message": "解方程 x^2 + 5x + 6 = 0"
}

Response:
{
  "matched": true,
  "intent": "Math Teacher",
  "handler": "math_teacher",
  "matched_keyword": "方程"
}
```

---

## 5. 关键流程

### 5.1 Agent 对话处理流程（四阶段流水线）

```
用户消息
    │
    ▼ Phase 0: TemporalParser.parse()
    ├─ 解析所有日期锚点（上周五、昨天、周一等）
    ├─ 推断叙事相对日期（昨天 = 周一的后一天）
    └─ 预计算行程（如有）
    │
    ▼ Phase 1: RuleBasedIntentClassifier.classify_with_context()
    ├─ 多意图并行识别
    └─ 返回 List[IntentClassification]
    │
    ▼ Phase 2: RuleBasedPlanner.plan()
    ├─ 歧义检测（多意图时跳过）
    ├─ 置信度门槛检查
    ├─ 策略路由（FIXED_SKILL → 依赖感知 DAG 构建）
    └─ 返回 TaskPlan { decision, tasks[], allowed_tools }
    │
    ▼ Phase 3: TaskExecutor.execute()
    ├─ 拓扑排序
    ├─ 跨技能参数注入 {SkillName.field}
    └─ aggregate_responses() → 最终响应
    │
    └─ 非 EXECUTE 决策 → 直接返回或委托 LLM
```

### 5.2 配置热更新流程

```
PUT /config (修改 JSON 文件)
    │
    ▼
POST /config/reload
    │
    ├── _agent = None
    ├── reload_skill_manager() → 新建 SkillManager 实例
    └── get_agent() → 重新加载配置，创建新 Agent 实例
```

### 5.3 数学解题流程

```
用户输入数学问题
    │
    ▼
MathTeacherSkill.match() → 正则 + 关键词双重匹配
    │
    ▼
MathEngine.solve() → 模式识别
    │
    ├─ quadratic    → 一元二次方程求根公式
    ├─ linear       → 移项合并同类项
    ├─ percentage   → 百分比公式
    ├─ arithmetic   → 等差数列通项与求和
    ├─ geometric    → 等比数列通项与求和
    ├─ derivative   → 幂函数求导法则
    ├─ integral     → 不定积分/定积分
    ├─ geometry     → 圆/三角形面积周长
    ├─ statistics   → 平均数/中位数/方差/标准差
    └─ general      → 基础算术表达式
    │
    ▼
MathResult (answer + steps + formula + 举一反三)
    │
    ▼
格式化响应 → 用户
```

---

## 6. 配置说明

### 环境变量

```bash
# DeepSeek API
DEEPSEEK_API_KEY=sk-xxxx          # 必填，DeepSeek API Key
DEEPSEEK_MODEL=deepseek-chat      # 可选，默认 deepseek-chat
DEEPSEEK_TEMPERATURE=0.7           # 可选，默认 0.7
DEEPSEEK_MAX_TOKENS=2000           # 可选，默认 2000

# 应用配置
APP_NAME=CustomerServiceAgent      # 可选
APP_DESCRIPTION=AI-powered...       # 可选
HOST=0.0.0.0                       # 可选，默认 0.0.0.0
PORT=8000                          # 可选，默认 8000
LOG_LEVEL=INFO                      # 可选
```

### 热更新配置文件

| 文件 | 路径 | 说明 |
|------|------|------|
| Agent 配置 | `config/agent_config.json` | agent_name, system_prompt, model_config |
| 意图路由 | `config/intent_routing.json` | 意图规则列表（name, handler, keywords, priority） |

---

## 7. 目录结构

```
od-agent/
├── src/
│   ├── __init__.py
│   ├── app.py              # FastAPI 应用（含路由，Admin UI 由 Jinja2 模板渲染）
│   ├── agent.py            # AgentScope ReAct Agent 封装
│   ├── models.py           # DeepSeek 模型配置
│   ├── semantic/           # 语义原生 AI 后端（OSI / GraphQL / MCP / Semantic Index）
│   │   ├── __init__.py
│   │   ├── osi_model.py        # OSI 数据模型
│   │   ├── graphql_generator.py # OSI → GraphQL SDL
│   │   ├── mcp_server.py       # MCP Server + Tools
│   │   ├── semantic_indexer.py  # 语义索引（RAG）
│   │   └── semantic_backend.py  # 完整后端组装器 + YAML 加载器
│   └── skills/
│       ├── __init__.py
│       ├── base.py         # BaseSkill 抽象类
│       ├── manager.py      # SkillManager 技能编排器
│       ├── math_teacher.py # 数学教师技能
│       ├── math_engine.py  # 数学解题引擎
│       ├── semantic_skill.py # 语义查询技能（SemanticSkill）
│       └── time_converter.py # 时间转换技能
├── templates/
│   └── admin/
│       ├── base.html       # Admin 主模板
│       ├── config.html     # 配置 Tab
│       ├── intents.html    # Intent 路由 Tab
│       ├── skills.html     # Skills Tab
│       └── partials/
│           └── intent_list.html  # HTMX 意图列表片段
├── static/
│   └── css/
│       └── admin.css       # Admin 页面样式
├── config/
│   ├── agent_config.json   # Agent 配置（热更新）
│   ├── intent_routing.json # 意图路由配置（热更新）
│   └── semantic_config.json # 语义后端配置（yaml_path、endpoint、use_demo）
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_app.py
├── .env                    # 环境变量（git 忽略）
├── .env.example            # 环境变量示例
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 部署配置
├── railway.json           # Railway 平台配置
├── README.md              # 项目说明
├── SPEC.md                # 本文档
├── plan.md                # 迭代计划
└── .specify/
    └── memory/
        └── constitution.md # SDD 规范文件
```

---

## 8. 开发规范

### 8.1 接口完整性检查（防止半完成改动）

在实现涉及多个模块/方法协作的功能时（如参数注入、流水线级联调用），
必须逐层验证参数传递链路。以下规则强制要求每条链路的两端都被正确连接：

**规则一：调用方传参 vs 被调用方用参，必须同时验证。**
当修改一个函数的签名（新增/删除参数）时，必须同时检查：
1. 所有调用该函数的地方是否传入了新参数（调用方完整性）
2. 该函数内部是否实际使用了新参数（实现完整性）

例如：`TemporalParser.__init__(today_date)` + `TemporalParser.parse(today_date)` + `run_pipeline()` 传入，三者缺一不可。
任何一端未连接，参数就形同虚设。

**规则二：同一信息源只能有一个权威来源。**
如果某数据已由某个模块/对象产生（如 `TimeConverterSkill.today`），其他所有需要该数据的模块必须引用同一个来源，
禁止在下游直接调用裸 `date.today()` 替代品。

**规则三：每个参数都有明确的"生产者"和"消费者"。**
新增跨模块参数时，在改动说明中必须明确写出：
- 谁产生这个值（Producer）
- 谁消费这个值（Consumer）
- 传递路径（Producer → 中间层 → Consumer）

如果说不清楚这三点，说明设计尚未完成，不要提交。

**规则四：改完后必须做端到端验证。**
在声称"改动完成"之前，必须实际运行调用链路，验证：
1. 参数确实被传递到了末端
2. 末端确实使用了传入的值（而非默认值/硬编码值）

对于 `date.today()` 这类隐式依赖，尤其要验证不同日期下行为是否一致。

### 8.2 代码风格

- 方法注释：只写不显而易见的设计意图、权衡取舍、约束条件，禁止写"本方法做X"这类废话注释
- 数据类属性：按 `Attributes:` 格式列出所有字段及其含义
- 异步方法：必须使用 `async/await`，禁止在异步函数内使用阻塞调用
- 错误处理：捕获异常时必须包含原始错误信息 `str(e)`，禁止空 `except`

---

## 9. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-05-29 | **语义原生 AI 后端**（feature/semantic-ontology-graphql-backend 分支）：新增 `src/semantic/` 模块（OSI 数据模型、GraphQL SDL 生成器、MCP Server、语义索引器、完整后端组装器）；新增 `SemanticSkill` 并集成到 SkillManager 和四阶段流水线；新增 `semantic_query` 意图路由；GraphQL 生成器支持 Enum/Query/Mutation/Subscription/Filter/OrderBy 类型；语义索引器使用 jieba 中文分词 + MMH3 哈希向量实现 RAG 检索 |
| 2026-05-29 | **Admin Semantic Backend 配置界面**（ID-003）：新增 Semantic Tab（4个卡片：模型配置/Schema查看/MCP工具列表/语义搜索测试）；新增 6 个后端 API 端点（`/semantic/config`、`/semantic/schema`、`/semantic/tools`、`/semantic/search`、`/semantic/reload`）；新增 `config/semantic_config.json` 配置文件；新增 `SemanticConfigUpdate` Pydantic 模型；Skills Tab 展示 SemanticSkill MCP tool 数量 |
| 2026-05-29 | 新增开发规范（8.1 接口完整性检查），防止参数传递链路断裂导致半完成改动 |
| 2026-05-20 | 初始化 SPEC.md，完成技术架构文档编写 |
| 2026-05-28 | Admin UI 重构为 LLM 厂商+模型+推理参数三级配置（DeepSeek / Kimi / Minimax / Custom）；新增 `src/llm_providers.py` 厂商定义模块；Thinking Mode 不再受模型约束，支持所有支持推理的厂商；新增 `build_temporal_context()` 多锚点时间线推理和行程预计算 |
| 2026-05-28 | Agent Chat 支持 SSE 流式响应：模型输出逐 token 流式显示在 deep-chat 组件中；`POST /chat` 升级为统一端点，`stream=true` 返回 SSE，`stream=false` 返回完整响应（向后兼容）；`OpenAIChatModel` 改为 `stream=True`；新增 `chat_stream()` / `_llm_chat_stream()` 方法返回异步生成器；`POST /chat/stream` 保留作为向后兼容别名；deep-chat 组件配置 `stream="true"` |
| 2026-05-28 | **四阶段流水线升级**：新增 `src/temporal/temporal_parser.py`（Phase 0：日期锚点解析 + 叙事相对日期推断 + 行程预计算）；扩展 `IntentBinding` 支持 `depends_on_skills` 和 `composite_meta_intent`（技能间依赖）；改造 `RuleBasedPlanner` 支持依赖感知 DAG 构建；新增 `MathTeacherSkill._execute_journey_avg()` 操作类型；重构 `SkillManager.run_pipeline()` 串联四阶段；修复 `MathTeacherSkill.execute()` 中 unreachable code bug |
