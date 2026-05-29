# SSE 流式响应规范

> 用途：定义 AI 聊天后端与前端之间的 SSE（Server-Sent Events）流式响应格式。
> 版本：v1.0
> 日期：2026-05-29

---

## 1. 概览

整体沿用现有的 `event: <type>` + `data: <payload>` + `\n\n` 格式。每个维度（thinking、content、tool）独立成一条事件流，前端各自拼接。

### 事件类型总览

| event 类型 | data 格式 | 说明 |
|---|---|---|
| `think` | `{"content": "..."}` | 思考过程逐字输出 |
| `think_done` | `{"status": "done"}` | think 流结束 |
| `content` | `{"content": "..."}` | 正文逐字/逐句输出 |
| `done` | `[DONE]` | 全文流结束 |
| `tool_call` | ToolCallPayload | 工具调用开始 |
| `tool_result` | ToolResultPayload | 工具调用结果 |

---

## 2. Think / 思考过程

thinking 阶段，大模型在"想"，还没开始回答。**非必选** —— 有就发，没有则跳过，直接进入 content。

```
event: think
data: {"content": "用户问了一个行程计算问题，"}

event: think
data: {"content": "我需要先理清时间线：\n"}

event: think
data: {"content": "- 上周五出发\n"}

event: think
data: {"content": "- 周一剩1000km\n- 昨天到拉萨\n"}

event: think
data: {"content": "总行程2080km，需要算出每天的距离。\n"}

event: think_done
data: {"status": "done"}
```

**前端行为：** 累积 `thinkContent`，在消息气泡中展开 "Thinking Process" 区域实时展示。

---

## 3. Content / 正文

正文开始后，不再发 think。content 可以分多次发，每次一小段（可以是句子也可以是 token，由后端控制粒度）。

```
event: content
data: {"content": "根据你的描述：\n\n"}

event: content
data: {"content": "从上周五到昨天（周三），一共**5天**，总行程**2080km**。\n"}

event: content
data: {"content": "**每天平均**：2080 ÷ 5 = **416km**\n\n"}

event: content
data: {"content": "**哪一天走得最远**：\n- 上周五到周一（共4天）：走了 2080 - 1000 = 1080km\n- 周二（昨天）：走了 1000km\n\n答案是**昨天（周二）**，走了整整 1000km。"}
```

**前端行为：** 累积 `content`，MarkdownBubble 实时渲染 markdown。

---

## 4. Tool Call / 工具调用

当大模型决定调用 Skill / MCP / RAG 时，在正文流中间插入。一次调用拆成三个阶段：**开始 → 参数 → 结果**。

### 4.1 tool_call

```
event: tool_call
data: {
  "type": "skill",
  "name": "math_calculator",
  "description": "执行数学计算",
  "input": {"expression": "(2080-1000)/4"},
  "id": "call_001"
}
```

**type 枚举：** `skill` | `mcp` | `rag`

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 工具类型 |
| `name` | string | 是 | 工具名称 |
| `description` | string | 否 | 工具用途说明 |
| `input` | object | 是 | 工具入参，格式自由，由后端定义 |
| `id` | string | 是 | 唯一调用 ID，用于匹配 result |

### 4.2 tool_result

```
event: tool_result
data: {
  "id": "call_001",
  "name": "math_calculator",
  "status": "success",
  "output": "270",
  "error": null
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 与 tool_call 中的 id 对应 |
| `name` | string | 是 | 工具名称 |
| `status` | string | 是 | `success` \| `error` \| `pending` |
| `output` | any | 否 | 工具返回，格式自由 |
| `error` | string | 否 | 错误信息，失败时有 |

### 4.3 完整调用示例

```
event: content
data: {"content": "让我查一下成都到拉萨的距离……"}

event: tool_call
data: {"type": "skill", "name": "math_calculator", "input": {"expression": "(2080-1000)/4"}, "id": "c1"}

event: tool_result
data: {"id": "c1", "name": "math_calculator", "status": "success", "output": "270", "error": null}

event: content
data: {"content": "算出来了，前面的平均速度是 270km/天。"}
```

---

## 5. 完成信号

正文流结束后，发 `[DONE]`，之后不再发任何事件，连接可关闭。

```
event: done
data: [DONE]
```

---

## 6. 完整示例（包含所有维度）

```
event: think
data: {"content": "用户问的是一个行程数学题。\n"}

event: think
data: {"content": "我需要：\n1. 理清时间线\n2. 计算每日距离\n3. 找最大值\n"}

event: think
data: {"content": "先调用数学计算工具。\n"}

event: think_done
data: {"status": "done"}

event: tool_call
data: {"type": "skill", "name": "math_calculator", "input": {"expr": "2080/5"}, "id": "c1"}

event: tool_result
data: {"id": "c1", "name": "math_calculator", "status": "success", "output": "416", "error": null}

event: tool_call
data: {"type": "mcp", "name": "map_distance_api", "input": {"from": "成都", "to": "拉萨"}, "id": "c2"}

event: tool_result
data: {"id": "c2", "name": "map_distance_api", "status": "success", "output": {"distance": "2080km", "route": "G318"}, "error": null}

event: tool_call
data: {"type": "rag", "name": "knowledge_base", "input": {"query": "成都拉萨行程 G318", "top_k": 3}, "id": "c3"}

event: tool_result
data: {"id": "c3", "name": "knowledge_base", "status": "success", "output": {"chunks": ["川藏线G318全长2080km...", "沿途经过康定、雅江..."]}, "error": null}

event: content
data: {"content": "根据G318川藏线的实测数据：\n\n"}

event: content
data: {"content": "**总行程**：2080km\n**总天数**：5天\n**日均**：416km\n\n"}

event: content
data: {"content": "**最后一天走得最远**：1000km（其余4天平均270km/天）。\n\n"}

event: content
data: {"content": "> 以上数据参考了川藏线G318历史行程数据库。"}

event: done
data: [DONE]
```

---

## 7. 前端数据结构（供后端参考）

前端会将每个 assistant message 组装成如下结构（**后端无需感知，前端自行维护**）：

```typescript
interface AssistantMessage {
  id: string
  role: 'assistant'
  thinkContent: string       // 所有 think 事件拼接
  thinkDone: boolean
  content: string            // 所有 content 事件拼接
  done: boolean
  toolCalls: ToolCall[]      // 所有 tool_call + tool_result 合并
}

interface ToolCall {
  id: string
  type: 'skill' | 'mcp' | 'rag'
  name: string
  description?: string
  input: any
  status: 'pending' | 'success' | 'error'
  output?: any
  error?: string
}
```

---

## 8. 注意事项

- **事件顺序不强制**：理论上 think / tool_call / content 可以交错出现，前端按到达顺序处理
- **think 为非必选**：如果后端不输出 thinking，忽略 `think` 和 `think_done` 事件即可
- **tool_call / tool_result 必须配对**：每个 `tool_call` 都应有一个对应的 `tool_result`
- **tool_result 可以异步返回**：如果结果返回较慢，前端会显示 pending 状态
- **content 中可穿插 tool_call**：正文和工具调用可以交错，前端会按顺序渲染
