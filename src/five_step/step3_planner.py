"""Step 3: Task Planning for the 5-step pipeline."""

from typing import List, Dict, Any, Optional
from .models import TaskPlanResult, PlannedActionItem, QueryConditionItem


class Step3TaskPlanner:
    """Step 3: Task Planning.

    Generates the task execution plan based on intent and ontology results.
    Includes user-readable plan summary generation.
    """

    def __init__(self):
        self._plan_summary: Optional[str] = None

    def plan(self, intent_result, ontology_result) -> TaskPlanResult:
        """Generate a task execution plan.

        Args:
            intent_result: Result from Step 1
            ontology_result: Result from Step 2

        Returns:
            TaskPlanResult with planned actions and user-readable summary
        """
        # Generate action sequence
        actions = self._generate_action_sequence(intent_result, ontology_result)

        # Build query conditions
        query_conditions = self._build_query_conditions(intent_result, ontology_result)

        # Build aggregation rules
        aggregation_rules = self._build_aggregation_rules(intent_result, ontology_result)

        # Determine display fields
        display_fields = self._determine_display_fields(ontology_result)

        # Use risk level from intent
        risk_level = intent_result.risk_level

        # Use confirmation requirement from intent
        requires_confirmation = intent_result.requires_confirmation

        # Generate user-readable plan summary
        plan_summary = self._generate_user_summary(intent_result, ontology_result, actions, query_conditions)

        result = TaskPlanResult(
            planned_actions=actions,
            query_conditions=query_conditions,
            aggregation_rules=aggregation_rules,
            display_fields=display_fields,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            plan_summary=plan_summary,
        )
        
        # Store the summary for later retrieval
        self._plan_summary = plan_summary
        
        return result

    def _generate_user_summary(
        self, 
        intent_result, 
        ontology_result,
        actions: List[PlannedActionItem],
        query_conditions: List[QueryConditionItem]
    ) -> str:
        """Generate a user-readable summary of the execution plan.
        
        This summary helps users understand what the agent will do next,
        providing transparency and the opportunity to correct misunderstandings.
        """
        # Try LLM-powered generation first
        llm_summary = self._llm_generate_summary(intent_result, ontology_result, actions, query_conditions)
        if llm_summary:
            return llm_summary
        
        # Fallback to template-based summary
        return self._template_summary(intent_result, ontology_result, actions, query_conditions)

    def _llm_generate_summary(
        self,
        intent_result,
        ontology_result,
        actions: List[PlannedActionItem],
        query_conditions: List[QueryConditionItem]
    ) -> Optional[str]:
        """Use LLM to generate a user-friendly plan summary."""
        import asyncio

        try:
            from ..models import get_default_model

            # Build context
            action_list = "\n".join([
                f"- {a.action_label}: {a.description}"
                for a in actions
            ])

            condition_list = "\n".join([
                f"- {c.label or c.field} {c.operator} {c.value}"
                for c in query_conditions
            ]) if query_conditions else "（无过滤条件）"

            prompt = f"""将以下执行计划转化为用户可理解的自然语言描述。

## 用户意图
{intent_result.intent_label}

## 业务对象
{ontology_result.object_label}

## 将执行的操作
{action_list}

## 查询条件
{condition_list}

## 要求
1. 用简洁的中文描述整个计划，约 2-3 句话
2. 说明 agent 会查询什么数据
3. 可以给用户一个预期（如"约 2-3 秒"）
4. 不要使用技术术语，用业务语言
5. 让用户知道接下来会发生什么

示例输出：
"我将帮您查询所有未执行的采购需求，并按申请部门和物料汇总展示。整个查询预计需要 2-3 秒。"

"""
            model = get_default_model()

            # Try sync call, handle async context gracefully
            try:
                summary = model.generate(prompt, thinking_budget=300)
                return summary.strip() if summary else None
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    print("[Step3] In async context, skipping LLM summary")
                    return None
                raise

        except Exception as e:
            print(f"[Step3] LLM summary generation failed: {e}")
            return None

    def _template_summary(
        self,
        intent_result,
        ontology_result,
        actions: List[PlannedActionItem],
        query_conditions: List[QueryConditionItem]
    ) -> str:
        """Template-based summary fallback."""
        parts = []
        
        # What will be queried - use object_label from ontology_result or object_term from intent_result
        object_label = ontology_result.object_label if ontology_result.object_label != "未知对象" else getattr(intent_result, 'object_term', '业务对象')
        if object_label:
            parts.append(f"查询 {object_label} 数据")
        
        # Key conditions
        if query_conditions:
            condition_parts = []
            for cond in query_conditions[:3]:  # Limit to 3 conditions
                if cond.label:
                    condition_parts.append(cond.label)
                else:
                    condition_parts.append(f"{cond.field} {cond.operator} {cond.value}")
            
            if condition_parts:
                parts.append(f"，条件：{'、'.join(condition_parts)}")
        
        # Action description
        if actions:
            primary_action = actions[0]
            parts.append(f"，将执行「{primary_action.action_label}」")
        
        # Add time expectation
        parts.append("。预计耗时 1-3 秒。")
        
        return "".join(parts)

    def get_plan_summary(self) -> str:
        """Get the last generated plan summary."""
        return self._plan_summary or "执行计划已生成"

    def _generate_action_sequence(self, intent_result, ontology_result) -> List[PlannedActionItem]:
        """Generate the sequence of actions to execute."""
        actions = []

        # Map intent to action ID
        action_id = self._get_action_for_intent(intent_result, ontology_result)

        if action_id:
            actions.append(PlannedActionItem(
                sequence=1,
                action_id=action_id,
                action_label=self._get_action_label(action_id),
                connector="ProcurementConnector",
                description=self._get_action_description(intent_result, ontology_result, action_id),
            ))

        # Add additional actions based on intent type
        if intent_result.operation_type == "create":
            # Add pre-validation step
            actions.insert(0, PlannedActionItem(
                sequence=1,
                action_id="validate_before_create",
                action_label="规则校验",
                connector="PolicyEngine",
                description="执行创建前规则校验",
            ))
            # Renumber
            for i, action in enumerate(actions):
                action.sequence = i + 1

        return actions

    def _get_action_for_intent(self, intent_result, ontology_result) -> str:
        """Map intent to connector action ID, using ontology as reference."""
        # Try to find action from ontology based on entity
        if ontology_result.object_type and ontology_result.object_type != "unknown":
            entity_name = ontology_result.object_type
            operation = intent_result.operation_type
            
            # Map operation type to action suffix
            operation_suffix = {
                "query": "list",
                "create": "create",
                "update": "update",
                "delete": "delete",
            }.get(operation, "list")
            
            # Try to find matching action in ontology
            from ..procurement.ontology_loader import get_procurement_ontology
            ontology = get_procurement_ontology()
            
            for action in ontology.actions:
                if action.entity_name == entity_name and operation_suffix in action.id:
                    return action.id
        
        # Fallback to intent mapping - use the action from Intent definition
        if hasattr(intent_result, 'intent') and intent_result.intent:
            from ..procurement.intent_router import get_intent_router
            router = get_intent_router()
            intent_def = router.intents
            for i in intent_def:
                if i.id == intent_result.intent:
                    return i.action

        # Direct mapping for analytics intents
        intent_to_action = {
            "procurement/query_unexecuted_pr": "analytics/find_unexecuted_purchase_requests",
            "procurement/create_inquiry": "procurement/create_inquiry_from_pr",
            "analytics/generate_price_comparison": "analytics/generate_price_comparison",
            "analytics/get_purchase_order_execution_status": "analytics/get_purchase_order_execution_status",
            "procurement/submit_award_approval": "procurement/submit_award_approval",
            "procurement/create_purchase_order": "procurement/create_purchase_order_from_pr_and_quotation",
            "procurement/approve_pr": "purchase_request/approve",
            "rfq/collect_quotations": "rfq/collect_quotations",
            "procurement/query_inquiry": "purchase_inquiries/list",
            "procurement/query_quotation": "purchase_quotations/list",
            "procurement/query_purchase_order": "purchase_order_heads/list",
            # Legacy keys (keep for compatibility)
            "procurement/query_open_pr": "analytics/find_unexecuted_purchase_requests",
            "procurement/compare_quotations": "analytics/generate_price_comparison",
            "procurement/approve_quotation": "purchase_quotation/approve_award",
            "procurement/query_order_status": "analytics/get_purchase_order_execution_status",
        }
        return intent_to_action.get(intent_result.intent, f"{ontology_result.object_type}/list" if ontology_result.object_type != "unknown" else "purchase_requests/list")

    def _get_action_label(self, action_id: str) -> str:
        """Get display label for an action ID."""
        labels = {
            "purchase_requests/list": "查询采购需求",
            "purchase_requests/get_by_id": "获取采购需求详情",
            "purchase_requests/create": "创建采购需求",
            "purchase_inquiries/list": "查询询价单",
            "purchase_inquiries/create": "创建询价单",
            "purchase_quotations/list": "查询报价单",
            "purchase_quotations/compare": "比价分析",
            "purchase_order_heads/list": "查询采购订单",
            "purchase_order_items/list": "查询订单明细",
            "purchase_order_receipt_history/list": "查询收货历史",
            "analytics/find_unexecuted_purchase_requests": "查询未执行采购需求",
            "analytics/generate_price_comparison": "生成比价分析",
            "analytics/get_purchase_order_execution_status": "查询订单执行状态",
        }
        return labels.get(action_id, action_id.split("/")[-1] if "/" in action_id else action_id)

    def _get_action_description(self, intent_result, ontology_result, action_id: str) -> str:
        """Get detailed description for an action."""
        object_label = ontology_result.object_label if ontology_result.object_label != "未知对象" else getattr(intent_result, 'object_term', '业务对象')
        descriptions = {
            "purchase_requests/list": f"查询所有{object_label or '采购需求'}，并根据条件过滤",
            "purchase_inquiries/list": f"查询{object_label or '询价单'}列表",
            "purchase_quotations/list": f"查询{object_label or '报价单'}列表",
            "purchase_order_heads/list": f"查询{object_label or '采购订单'}抬头信息",
            "purchase_order_items/list": f"查询{object_label or '采购订单'}明细",
            "analytics/find_unexecuted_purchase_requests": "识别已审批但尚未生成采购订单的采购需求",
            "analytics/generate_price_comparison": "按物料维度对多家供应商报价进行比价分析",
            "analytics/get_purchase_order_execution_status": "查询采购订单的审批、发货、收货等执行状态",
        }
        return descriptions.get(action_id, f"执行 {self._get_action_label(action_id)}")

    def _build_query_conditions(self, intent_result, ontology_result) -> List[QueryConditionItem]:
        """Build query filter conditions.
        
        Enhanced to handle:
        - Time range from intent_result.temporal_context
        - Business conditions from intent_result.extracted_slots
        """
        conditions = []

        # Base condition: exclude deleted records
        conditions.append(QueryConditionItem(
            field="delete_flag",
            operator="=",
            value="0",
            label="排除已删除"
        ))

        # 1. Handle time range conditions
        if hasattr(intent_result, 'temporal_context') and intent_result.temporal_context:
            ctx = intent_result.temporal_context
            if ctx.get('date_from') and ctx.get('date_to'):
                conditions.append(QueryConditionItem(
                    field="delivery_date",
                    operator="BETWEEN",
                    value=f"['{ctx['date_from']}', '{ctx['date_to']}']",
                    label=f"需求日期：{ctx.get('label', '时间范围')}"
                ))
            elif ctx.get('resolved_date'):
                # Single date
                conditions.append(QueryConditionItem(
                    field="delivery_date",
                    operator="=",
                    value=ctx['resolved_date'],
                    label=f"需求日期：{ctx.get('label', '指定日期')}"
                ))

        # 2. Handle business condition slots
        if hasattr(intent_result, 'extracted_slots') and intent_result.extracted_slots:
            slots = intent_result.extracted_slots

            # Pr type filter
            if slots.get('pr_type'):
                conditions.append(QueryConditionItem(
                    field="pr_type",
                    operator="=",
                    value=slots['pr_type'],
                    label=f"采购类型：{slots['pr_type']}"
                ))

            # Material category filter (maps to material_d or a category field)
            if slots.get('material_category'):
                conditions.append(QueryConditionItem(
                    field="material_category",
                    operator="=",
                    value=slots['material_category'],
                    label=f"物料分类：{slots['material_category']}"
                ))

            # Department filter
            if slots.get('apply_dep') or slots.get('department'):
                dept_value = slots.get('apply_dep') or slots.get('department')
                conditions.append(QueryConditionItem(
                    field="apply_dep",
                    operator="=",
                    value=dept_value,
                    label=f"申请部门：{dept_value}"
                ))

            # Execution status filter
            if slots.get('execution_status'):
                status = slots['execution_status']
                # Map display status to flow_status values
                status_map = {
                    "未执行": None,  # Will be handled by connector logic
                    "已执行": "EXECUTED",
                    "待审批": "PENDING_APPROVAL",
                    "已审批": "APPROVED",
                    "已驳回": "REJECTED",
                }
                mapped_status = status_map.get(status, status)
                if mapped_status:
                    conditions.append(QueryConditionItem(
                        field="flow_status",
                        operator="=",
                        value=mapped_status,
                        label=f"审批状态：{status}"
                    ))

        # 3. Intent-specific base conditions
        if intent_result.intent == "procurement/query_open_pr":
            conditions.append(QueryConditionItem(
                field="flow_status",
                operator="IN",
                value="['S0','APPROVED']",
                label="已审批"
            ))

        return conditions

    def _build_aggregation_rules(self, intent_result, ontology_result) -> List[Dict[str, Any]]:
        """Build aggregation/statistics rules."""
        rules = []

        if intent_result.intent == "procurement/query_open_pr":
            rules.extend([
                {"type": "count", "fields": [], "label": "总数量"},
                {"type": "group_by", "fields": ["apply_dep"], "label": "按申请部门统计"},
                {"type": "group_by", "fields": ["material_d"], "label": "按物料统计"},
            ])

        return rules

    def _determine_display_fields(self, ontology_result) -> List[str]:
        """Determine which fields to display in results."""
        # Use attributes from ontology result if available
        if ontology_result.attributes:
            return [attr.get("name", "") for attr in ontology_result.attributes[:8]]
        
        return [
            "pr_id", "material_d", "quantity", "apply_dep",
            "applicant", "delivery_date"
        ]
