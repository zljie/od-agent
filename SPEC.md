# SDD 技术文档 - OD Agent

> 本文档记录项目的技术架构、模块拓扑和核心实现。
> 每次任务完成后请检查并更新此文档，保持与项目架构一致。

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
| **AI 模型** | DeepSeek (deepseek-chat) | 通过 OpenAI 兼容接口调用 |
| **环境管理** | python-dotenv | 环境变量加载 |
| **测试** | pytest + pytest-asyncio + httpx | 异步测试支持 |
| **部署** | Docker, Railway | 容器化与云平台部署 |

---

## 3. 模块拓扑

### 3.1 应用入口

| 文件 | 说明 |
|------|------|
| `src/app.py` | FastAPI 应用工厂 `create_app()`，包含所有 HTTP 路由、内嵌 Admin UI 页面 (HTML/JS/CSS) |
| `src/agent.py` | AgentScope `ReActAgent` 封装，`CustomerServiceAgent` 类管理 Agent 生命周期和对话逻辑 |
| `src/models.py` | `ModelConfig` 数据类，封装 DeepSeek 模型配置 |
| `src/__init__.py` | 包初始化 |

### 3.2 路由层

所有路由在 `src/app.py` 中通过 `create_app()` 动态注册：

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 欢迎页，返回服务信息 |
| `/health` | GET | 健康检查 |
| `/readiness` | GET | 就绪检查 |
| `/liveness` | GET | 存活检查 |
| `/chat` | POST | 对话入口，返回 `{response, session_id}` |
| `/chat/stream` | POST | 兼容 deep-chat 组件格式，返回 `{text}` |
| `/process` | POST | AgentApp 兼容格式，处理对话请求 |
| `/config` | GET | 获取当前 Agent 配置 |
| `/config` | PUT | 更新 Agent 配置（支持热更新） |
| `/config/reload` | POST | 热重载 Agent 实例 |
| `/config/reset` | POST | 重置对话历史 |
| `/intents` | GET/POST | 获取/添加意图路由规则 |
| `/intents/{index}` | GET/PUT/DELETE | 单条意图的查改删 |
| `/intents/detect` | POST | 检测消息匹配的意图 |
| `/skills` | GET | 获取所有已注册技能 |
| `/skills/{skill_name}` | GET | 获取指定技能详情 |
| `/skills/detect` | POST | 检测消息应触发哪个技能 |
| `/admin` | GET | Admin 配置管理页面（内嵌 HTML） |

### 3.3 数据模型层

| 文件 | 说明 |
|------|------|
| `src/app.py` | `ChatRequest`, `ChatResponse`, `ProcessRequest`, `AgentConfigUpdate`, `MessageInput` (Pydantic BaseModel) |
| `src/models.py` | `ModelConfig` — DeepSeek 模型参数封装 |
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

---

## 4. API 接口文档

### 4.1 对话接口

```
POST /chat

Request:
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
```

### 4.2 配置热更新接口

```
GET /config

Response:
{
  "agent_name": "和尚",
  "system_prompt": "你现在是个和尚...",
  "model_config": {
    "model_name": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 2000
  }
}

PUT /config
Request:
{
  "agent_name": "新名称",
  "system_prompt": "新提示词",
  "llm_config": { ... }
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

### 5.1 Agent 对话处理流程

```
用户消息
    │
    ▼
SkillManager.detect_and_execute()
    │
    ├─ 遍历 intent_routing.json 中的规则（按 priority 降序）
    │       │
    │       └─ 命中 → 执行对应 Skill
    │
    ├─ 未命中 → 遍历内置 Skill.match()（按 priority 降序）
    │       │
    │       └─ 命中 → 执行 Skill.execute()
    │
    └─ 仍未命中 → 调用 AgentScope ReActAgent.chat()
                      │
                      ▼
                 DeepSeek API (OpenAI Compatible)
                      │
                      ▼
                 返回响应文本
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
│   ├── app.py              # FastAPI 应用（含 Admin UI HTML）
│   ├── agent.py            # AgentScope ReAct Agent 封装
│   ├── models.py           # DeepSeek 模型配置
│   └── skills/
│       ├── __init__.py
│       ├── base.py         # BaseSkill 抽象类
│       ├── manager.py      # SkillManager 技能编排器
│       ├── math_teacher.py # 数学教师技能
│       └── math_engine.py  # 数学解题引擎
├── config/
│   ├── agent_config.json   # Agent 配置（热更新）
│   └── intent_routing.json # 意图路由配置（热更新）
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_app.py
├── SDK/versions/           # SDK 版本目录（预留）
├── .env                    # 环境变量（git 忽略）
├── .env.example            # 环境变量示例
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 部署配置
├── railway.json           # Railway 平台配置
├── README.md              # 项目说明
├── SPEC.md                # 本文档
├── plan.md                # 迭代计划
├── constitution.md        # 团队规范章程
└── .specify/
    └── memory/
        └── constitution.md # SDD 规范文件
```

---

## 8. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-05-20 | 初始化 SPEC.md，完成技术架构文档编写 |
