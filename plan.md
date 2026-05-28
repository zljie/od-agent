# 项目迭代面板 (plan.md)

我们用这个文件做轻量迭代看板：每次开始一个新任务就新增一条卡片；实现完成并通过验收后，将卡片移动到 Done，并勾选验收标准。

---

## Backlog (待办)

### [ID-001] Admin UI Jinja2 + HTMX + Alpine.js 迁移

用户故事：作为开发者，我希望将内联 HTML 从 `src/app.py` 中迁移到 Jinja2 模板，以便于维护和扩展。

验收标准：
- [x] 添加 `jinja2>=3.1.0` 到 `requirements.txt`
- [x] 创建 `templates/admin/` 和 `static/css/` 目录结构
- [x] 提取内联 CSS 到 `static/css/admin.css`
- [x] `src/app.py` 中挂载 `StaticFiles` 和 `Jinja2Templates`
- [x] 新增 3 个 HTMX 片段路由 (`/admin/intents/list`、`/admin/intents/add`、`/admin/intents/{index} DELETE`)
- [x] `templates/admin/base.html` 主模板（含 Alpine.js 全局状态）
- [x] `templates/admin/config.html` — Config Tab
- [x] `templates/admin/intents.html` — Intents Tab
- [x] `templates/admin/skills.html` — Skills Tab
- [x] `templates/admin/partials/intent_list.html` — HTMX 意图列表片段
- [x] 删除 `src/app.py` 中的 `ADMIN_HTML_PAGE` 字符串
- [x] 更新 SPEC.md（技术栈、模块拓扑、目录结构）
- [ ] `/admin` 页面功能验收（见下方）
- [ ] pytest 测试通过

关联：第 381-1252 行 `src/app.py`（已删除）、`templates/`、`static/`

记录：创建日期 2026-05-28；完成日期 --

---

## In Progress (进行中)

---

## Done (已完成)

- Admin UI 迁移至 Jinja2 + HTMX + Alpine.js（templates/、static/、HTMX 路由、文档更新）
- 新增 TimeConverter Skill（68 个关键词，支持相对日期、绝对日期、时区、时长、星期几等）

---

## Backlog (待办)

### [ID-002] TimeConverter Skill

用户故事：作为系统，我希望通过代码直接解析用户输入中的时间表达式，减少模型的无谓消耗。

验收标准：
- [x] 创建 `src/skills/time_converter.py`
- [x] 支持相对日期（今天、明天、下周、上个月等）
- [x] 支持绝对日期（2024-01-15、2024年1月15日）
- [x] 支持时长计算（3天后、2周前、5个月后）
- [x] 支持星期几查询（今天是星期几、下周一）
- [x] 支持时区转换（北京时间、UTC、东京时间）
- [x] 集成到 `SkillManager` 自动注册
- [x] pytest 测试通过
- [x] 更新 SPEC.md

关联：`src/skills/time_converter.py`、`src/skills/manager.py`、`src/skills/__init__.py`

记录：创建日期 2026-05-28；完成日期 2026-05-28

---

## 需求卡片模板

```
[ID] 标题

用户故事：作为[角色]，我希望[能力]，以便[价值]

验收标准：
[ ] 条件 1
[ ] 条件 2
[ ] 条件 3

关联（可选）：页面/接口/关键文件
记录（可选）：创建日期 YYYY-MM-DD；完成日期 YYYY-MM-DD；备注
```

---

## 标签说明

- `[ ]` 未完成
- `[x]` 已完成
- `*italic*` 进行中
