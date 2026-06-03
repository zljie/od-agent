"""Step 3: Task Planning for the 5-step pipeline.

PHASE 4: Action Planning (formerly "Task Planning")
================================================

Step 3's role has been upgraded from "Task Planning" to "Action Planning".

Before (Phase 3):
    Planner = Action Mapping (often re-guessed the action)

After (Phase 4):
    Planner = Parameter Completer (uses Semantic Contract's action)

Key changes:
1. The action from Semantic Contract is the SINGLE SOURCE OF TRUTH
2. Planner's job is to COMPLETE the parameters:
   - Resolve time ranges (最近3个月 → date_from, date_to)
   - Map display values to field codes (销售部 → department_code)
   - Fill in missing business logic parameters
3. Planner CANNOT change the action
4. Action Drift Detection: if planner wants to use a different action, flag it

This prevents "Semantic Collapse" where the high-confidence action from
Step 1 gets overwritten by planner's guesswork.
"""

from typing import List, Dict, Any, Optional
from .models import TaskPlanResult, PlannedActionItem, QueryConditionItem, SemanticContract


class Step3TaskPlanner:
    """Step 3: Action Planning.

    Generates the task execution plan based on Semantic Contract and ontology results.
    Includes user-readable plan summary generation.

    Phase 4 Changes:
    - Uses Semantic Contract action as the single source of truth
    - Planner completes parameters, not re-maps actions
    - Detects and reports action drift
    """

    def __init__(self):
        self._plan_summary: Optional[str] = None
        self._action_drift_detected: bool = False
        self._drift_reason: str = ""

    def plan(self, intent_result, ontology_result) -> TaskPlanResult:
        """Generate a task execution plan using Semantic Contract.

        Args:
            intent_result: Result from Step 1 (contains Semantic Contract)
            ontology_result: Result from Step 2 (contains ontology expansion)

        Returns:
            TaskPlanResult with planned actions and user-readable summary
        """
        # PHASE 4: Extract Semantic Contract
        semantic_contract = getattr(intent_result, 'semantic_contract', None)

        # Generate action sequence (uses Semantic Contract action)
        actions = self._generate_action_sequence(intent_result, ontology_result, semantic_contract)

        # Check for action drift
        self._check_action_drift(intent_result, ontology_result, semantic_contract, actions)

        # Build query conditions (uses Semantic Contract slots)
        query_conditions = self._build_query_conditions(intent_result, ontology_result, semantic_contract)

        # Build aggregation rules
        aggregation_rules = self._build_aggregation_rules(intent_result, ontology_result, semantic_contract)

        # Determine display fields
        display_fields = self._determine_display_fields(ontology_result)

        # Use risk level from intent
        risk_level = intent_result.risk_level

        # Use confirmation requirement from intent
        requires_confirmation = intent_result.requires_confirmation

        # Generate user-readable plan summary
        plan_summary = self._generate_user_summary(intent_result, ontology_result, actions, query_conditions, semantic_contract)

        result = TaskPlanResult(
            planned_actions=actions,
            query_conditions=query_conditions,
            aggregation_rules=aggregation_rules,
            display_fields=display_fields,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            plan_summary=plan_summary,
        )

        # Store Semantic Contract reference for diagnostics
        result.semantic_contract = semantic_contract
        result.action_drift_detected = self._action_drift_detected
        result.drift_reason = self._drift_reason

        # Store the summary for later retrieval
        self._plan_summary = plan_summary

        return result

    def _check_action_drift(
        self,
        intent_result,
        ontology_result,
        semantic_contract: Optional[SemanticContract],
        actions: List[PlannedActionItem]
    ):
        """Detect if the planned action differs from Semantic Contract's action."""
        self._action_drift_detected = False
        self._drift_reason = ""

        if not semantic_contract:
            return

        if not actions:
            return

        planned_action = actions[0].action_id

        # Check if planned action matches Semantic Contract action
        if planned_action != semantic_contract.action:
            # Allow generic list actions if Semantic Contract has analytics action
            if semantic_contract.action.startswith("analytics/") and "/list" in planned_action:
                self._action_drift_detected = True
                self._drift_reason = (
                    f"Semantic Contract action '{semantic_contract.action}' "
                    f"被替换为 '{planned_action}'"
                )
                print(f"[Step3][WARNING] Action Drift Detected: {self._drift_reason}")

    def _generate_user_summary(
        self, 
        intent_result, 
        ontology_result,
        actions: List[PlannedActionItem],
        query_conditions: List[QueryConditionItem],
        semantic_contract: Optional[SemanticContract] = None
    ) -> str:
        """Generate a user-readable summary of the execution plan.

        PHASE 4: Uses Semantic Contract info if available.
        """
        # Try LLM-powered generation first
        llm_summary = self._llm_generate_summary(intent_result, ontology_result, actions, query_conditions, semantic_contract)
        if llm_summary:
            return llm_summary

        # Fallback to template-based summary
        return self._template_summary(intent_result, ontology_result, actions, query_conditions, semantic_contract)

    def _llm_generate_summary(
        self,
        intent_result,
        ontology_result,
        actions: List[PlannedActionItem],
        query_conditions: List[QueryConditionItem],
        semantic_contract: Optional[SemanticContract] = None
    ) -> Optional[str]:
        """Use LLM to generate a user-friendly plan summary."""
        import asyncio

        try:
            from ..models import get_default_model

            # PHASE 4: Build enhanced context with Semantic Contract
            action_list = "\n".join([
                f"- {a.action_label}: {a.description}"
                for a in actions
            ])

            condition_list = "\n".join([
                f"- {c.label or c.field} {c.operator} {c.value}"
                for c in query_conditions
            ]) if query_conditions else "（无过滤条件）"

            # PHASE 4: Add Semantic Contract context
            semantic_context = ""
            if semantic_contract:
                semantic_context = f"""

## Semantic Contract (Single Source of Truth)
- Object: {semantic_contract.object} ({semantic_contract.object_label})
- Action: {semantic_contract.action} ({semantic_contract.action_label})
- Confidence: {semantic_contract.confidence:.2%}
- Slots: {', '.join(semantic_contract.slots.keys()) if semantic_contract.slots else 'None'}
"""
                if semantic_contract.alternatives:
                    semantic_context += "\n## Alternative Actions (not selected):\n"
                    for alt in semantic_contract.alternatives[:3]:
                        semantic_context += f"- {alt.get('intent_id')} (confidence: {alt.get('confidence', 0):.2%})\n"

            prompt = f"""将以下执行计划转化为用户可理解的自然语言描述。

## 用户意图
{intent_result.intent_label}

## 业务对象
{ontology_result.object_label}
{semantic_context}

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
        query_conditions: List[QueryConditionItem],
        semantic_contract: Optional[SemanticContract] = None
    ) -> str:
        """Template-based summary fallback."""
        parts = []

        # PHASE 4: Use Semantic Contract info if available
        if semantic_contract:
            # Use Semantic Contract's action label for more accurate description
            action_label = semantic_contract.action_label
            if actions:
                action_label = actions[0].action_label
            parts.append(f"查询 {semantic_contract.object_label or '业务数据'}")
            parts.append(f"，执行「{action_label}」")
        else:
            # Legacy template
            object_label = ontology_result.object_label if ontology_result.object_label != "未知对象" else getattr(intent_result, 'object_term', '业务对象')
            if object_label:
                parts.append(f"查询 {object_label} 数据")

        # Key conditions
        if query_conditions:
            condition_parts = []
            for cond in query_conditions[:3]:
                if cond.label:
                    condition_parts.append(cond.label)
                else:
                    condition_parts.append(f"{cond.field} {cond.operator} {cond.value}")

            if condition_parts:
                parts.append(f"，条件：{'、'.join(condition_parts)}")

        # Add time expectation
        parts.append("。预计耗时 1-3 秒。")

        return "".join(parts)

    def _build_aggregation_rules(
        self,
        intent_result,
        ontology_result,
        semantic_contract: Optional[SemanticContract] = None
    ) -> List[Dict[str, Any]]:
        """Build aggregation/statistics rules."""
        rules = []

        # PHASE 4: Check Semantic Contract for specific analytics actions
        if semantic_contract and semantic_contract.action:
            if semantic_contract.action == "analytics/find_unexecuted_purchase_requests":
                rules.extend([
                    {"type": "count", "fields": [], "label": "总数量"},
                    {"type": "group_by", "fields": ["apply_dep"], "label": "按申请部门统计"},
                    {"type": "group_by", "fields": ["material_d"], "label": "按物料统计"},
                ])
                return rules

        # Legacy path
        if intent_result.intent == "procurement/query_open_pr":
            rules.extend([
                {"type": "count", "fields": [], "label": "总数量"},
                {"type": "group_by", "fields": ["apply_dep"], "label": "按申请部门统计"},
                {"type": "group_by", "fields": ["material_d"], "label": "按物料统计"},
            ])

        return rules

    def get_plan_summary(self) -> str:
        """Get the last generated plan summary."""
        return self._plan_summary or "执行计划已生成"

    def _generate_action_sequence(
        self,
        intent_result,
        ontology_result,
        semantic_contract: Optional[SemanticContract] = None
    ) -> List[PlannedActionItem]:
        """Generate the sequence of actions to execute.

        PHASE 4: Uses Semantic Contract action as the single source of truth.
        """
        actions = []

        # PHASE 4: Use Semantic Contract action if available
        if semantic_contract and semantic_contract.action:
            action_id = semantic_contract.action
            action_label = semantic_contract.action_label or self._get_action_label(action_id)
            actions.append(PlannedActionItem(
                sequence=1,
                action_id=action_id,
                action_label=action_label,
                connector="ProcurementConnector",
                description=self._get_action_description(intent_result, ontology_result, action_id),
            ))
            return actions

        # Legacy path: no Semantic Contract
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
            actions.insert(0, PlannedActionItem(
                sequence=1,
                action_id="validate_before_create",
                action_label="规则校验",
                connector="PolicyEngine",
                description="执行创建前规则校验",
            ))
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

    def _build_query_conditions(
        self,
        intent_result,
        ontology_result,
        semantic_contract: Optional[SemanticContract] = None
    ) -> List[QueryConditionItem]:
        """Build query filter conditions.

        PHASE 4: Uses Semantic Contract slots as the primary source.
        Supplements with extracted_slots from intent_result.
        """
        conditions = []

        # PHASE 4: Use Semantic Contract slots if available
        if semantic_contract and semantic_contract.slots:
            conditions = self._build_conditions_from_semantic_contract(semantic_contract)
            return conditions

        # Legacy path: no Semantic Contract, use extracted_slots
        conditions = self._build_conditions_from_slots(intent_result)
        return conditions

    def _build_conditions_from_semantic_contract(
        self,
        semantic_contract: SemanticContract
    ) -> List[QueryConditionItem]:
        """Build query conditions from Semantic Contract slots.

        This is the Phase 4 primary path for condition building.
        """
        conditions = []

        # Base condition: exclude deleted records
        conditions.append(QueryConditionItem(
            field="delete_flag",
            operator="=",
            value="0",
            label="排除已删除"
        ))

        # Build conditions from Semantic Contract slots
        for slot_name, slot in semantic_contract.slots.items():
            condition = self._slot_to_condition(slot_name, slot)
            if condition:
                conditions.append(condition)

        # Handle special analytics actions
        if semantic_contract.action == "analytics/find_unexecuted_purchase_requests":
            # Add unexecuted filter (business logic)
            conditions.append(QueryConditionItem(
                field="flow_status",
                operator="IN",
                value="['S0', 'APPROVED']",
                label="已审批待执行"
            ))

        return conditions

    def _slot_to_condition(self, slot_name: str, slot) -> Optional[QueryConditionItem]:
        """Convert a SemanticContractSlot to a QueryConditionItem."""
        # Map slot names to field names and operators
        slot_mappings = {
            "department": ("apply_dep", "=", slot.display_value, f"申请部门：{slot.display_value}"),
            "apply_dep": ("apply_dep", "=", slot.display_value, f"申请部门：{slot.display_value}"),
            "date_range": None,  # Special handling for time ranges
            "time_label": None,  # Will be handled by time_date_from/time_date_to
            "time_date_from": None,  # Will be combined with time_date_to
            "time_date_to": None,  # Will be combined with time_date_from
            "execution_status": ("flow_status", "=", slot.display_value, f"状态：{slot.display_value}"),
            "pr_type": ("pr_type", "=", slot.display_value, f"采购类型：{slot.display_value}"),
            "material_category": ("material_category", "=", slot.display_value, f"物料分类：{slot.display_value}"),
        }

        mapping = slot_mappings.get(slot_name)
        if not mapping:
            return None

        field, operator, value, label = mapping
        return QueryConditionItem(
            field=field,
            operator=operator,
            value=value,
            label=label
        )

    def _build_conditions_from_slots(self, intent_result) -> List[QueryConditionItem]:
        """Build query conditions from extracted_slots (legacy path)."""
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

            # Material category filter
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

    def _build_aggregation_rules(
        self,
        intent_result,
        ontology_result,
        semantic_contract: Optional[SemanticContract] = None
    ) -> List[Dict[str, Any]]:
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
