"""Step 5: Response Generation for the 5-step pipeline."""

import json
from typing import List, Dict, Any, Optional
from .models import ResponseResult, SuggestedAction


class Step5ResponseGenerator:
    """Step 5: Response Generation.

    Generates the user-facing response based on execution results.
    """

    def generate(
        self,
        exec_result,
        plan_result,
        intent_result=None,
        ontology_result=None
    ) -> ResponseResult:
        """Generate the final response.

        Args:
            exec_result: Result from Step 4
            plan_result: Result from Step 3
            intent_result: Result from Step 1 (optional)
            ontology_result: Result from Step 2 (optional)

        Returns:
            ResponseResult with text and suggestions
        """
        # Aggregate statistics
        statistics = self._aggregate_statistics(exec_result)

        # Generate result summary
        summary = self._generate_summary(exec_result, plan_result)

        # Generate result definition
        definition = self._generate_definition(plan_result)

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            exec_result, plan_result, intent_result
        )

        # Detect ontology improvements
        ontology_improvements = self._detect_ontology_gaps(exec_result, ontology_result)

        # Build final text
        text = self._build_response_text(summary, statistics, suggested_actions)

        return ResponseResult(
            text=text,
            result_summary=summary,
            statistics=statistics,
            result_definition=definition,
            next_actions=suggested_actions,
            ontology_improvements=ontology_improvements,
        )

    def _aggregate_statistics(self, exec_result) -> List[Dict[str, Any]]:
        """Aggregate result statistics."""
        stats = []

        for record in exec_result.executions:
            if record.status == "success" and record.result_count is not None:
                stats.append({
                    "label": record.action_id.split("/")[-1],
                    "value": record.result_count,
                    "unit": "条",
                })

        return stats

    def _generate_summary(self, exec_result, plan_result) -> str:
        """Generate a natural language summary - uses LLM if available."""
        if not exec_result.executions:
            return "未执行任何操作"

        total = len(exec_result.executions)
        success = sum(1 for e in exec_result.executions if e.status == "success")

        if success == total:
            # Try LLM-generated summary first
            llm_summary = self._llm_generate_summary(exec_result, plan_result)
            if llm_summary:
                return llm_summary

            # Fallback to template-based summary
            for record in exec_result.executions:
                if record.result_count is not None and record.result_count > 0:
                    return f"已查询到 {record.result_count} 条符合条件的记录"
            return "执行成功"
        else:
            failed = total - success
            return f"执行完成：{success} 成功，{failed} 失败"

    def _llm_generate_summary(self, exec_result, plan_result) -> Optional[str]:
        """Use LLM to generate a natural language summary from execution results."""
        try:
            from ..models import get_default_model
            model = get_default_model()

            # Build context from execution results
            executions_info = []
            for record in exec_result.executions:
                exec_info = {
                    "action_id": record.action_id,
                    "status": record.status,
                    "result_count": record.result_count,
                    "response_summary": record.response_summary or "",
                }
                if record.data:
                    # Extract key info from data
                    if isinstance(record.data, dict):
                        if "items" in record.data and record.data["items"]:
                            first_item = record.data["items"][0]
                            exec_info["sample_data"] = {k: first_item.get(k) for k in list(first_item.keys())[:5]}
                        if "total" in record.data:
                            exec_info["total"] = record.data["total"]
                executions_info.append(exec_info)

            intent_label = plan_result.plan_summary if plan_result else ""
            query_conditions = []
            if plan_result and plan_result.query_conditions:
                for cond in plan_result.query_conditions[:3]:
                    query_conditions.append(f"{cond.label or cond.field}: {cond.operator} {cond.value}")

            prompt = f"""你是采购助手。请根据以下执行结果，用简洁的中文为用户生成一段回复。

## 用户意图
{intent_label or "采购查询"}

## 查询条件
{chr(10).join(query_conditions) if query_conditions else "无特定条件"}

## 执行结果
{json.dumps(executions_info, ensure_ascii=False, indent=2)}

## 要求
1. 用 1-2 句话总结查询结果，让用户清楚了解情况
2. 如果查询到数据，给出具体数字和关键信息
3. 如果执行失败，说明原因
4. 自然、友好、口语化，像在与用户对话
5. 不要列出技术细节

示例：
- "已为您查询到 12 条未执行的采购需求，其中研发部有 5 条，采购部有 7 条。"
- "采购订单 PO-20260528-001 当前状态正常，审批已通过，等待供应商发货。"
- "抱歉，查询失败：未找到匹配的采购需求记录。"

直接输出回复内容，不要有其他内容。
"""
            # Try sync call, handle async context gracefully
            try:
                response = model.generate(prompt, thinking_budget=300)
                if response and response.strip():
                    print(f"[Step5] LLM生成回复成功")
                    return response.strip()
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    print("[Step5] In async context, skipping LLM summary")
                else:
                    raise

        except Exception as e:
            print(f"[Step5] LLM回复生成失败: {e}")

        return None

    def _generate_definition(self, plan_result) -> str:
        """Generate the query definition."""
        conditions = []
        for cond in plan_result.query_conditions:
            conditions.append(f"{cond.label}：{cond.field} {cond.operator} {cond.value}")

        if conditions:
            return "查询口径说明：\n" + "\n".join(f"- {c}" for c in conditions)
        return ""

    def _generate_suggested_actions(
        self,
        exec_result,
        plan_result,
        intent_result=None
    ) -> List[SuggestedAction]:
        """Generate suggested next actions."""
        actions = []

        # Common actions
        actions.append(SuggestedAction(
            id="show_detail",
            label="展示明细",
            type="navigate"
        ))

        # Intent-specific actions
        if intent_result:
            if intent_result.intent == "procurement/query_open_pr":
                actions.extend([
                    SuggestedAction(id="group_by_dept", label="按部门汇总", type="execute"),
                    SuggestedAction(id="create_inquiry", label="生成询价单", type="execute"),
                    SuggestedAction(id="export", label="导出清单", type="export"),
                ])
            elif intent_result.intent == "procurement/query_order_status":
                actions.append(SuggestedAction(
                    id="track_receipt",
                    label="跟踪收货进度",
                    type="navigate"
                ))

        return actions

    def _detect_ontology_gaps(self, exec_result, ontology_result) -> List[Dict[str, Any]]:
        """Detect ontology gaps from execution results."""
        gaps = []

        if ontology_result and ontology_result.gaps:
            gaps.extend(ontology_result.gaps)

        # Check for data quality issues
        for record in exec_result.executions:
            if record.status == "success" and record.data:
                # Check if source_type is missing in results
                items = record.data.get("items", [])
                if items and "source_type" not in items[0]:
                    gaps.append({
                        "gap_type": "missing_attribute",
                        "description": "本体缺少 'source_type' 来源类型字段",
                        "suggestion": "补充后可以稳定统计需求来源分布（手工录入/批导/系统对接）"
                    })

        return gaps

    def _build_response_text(
        self,
        summary: str,
        statistics: List[Dict[str, Any]],
        suggested_actions: List[SuggestedAction]
    ) -> str:
        """Build the final response text."""
        parts = []

        parts.append(summary)

        # Add statistics
        if statistics:
            parts.append("\n统计信息：")
            for stat in statistics:
                unit = stat.get("unit", "")
                parts.append(f"- {stat['label']}：{stat['value']} {unit}")

        # Add next action suggestion
        if suggested_actions:
            action_labels = " / ".join(a.label for a in suggested_actions[:3])
            parts.append(f"\n您可以：{action_labels}")

        return "\n".join(parts)
