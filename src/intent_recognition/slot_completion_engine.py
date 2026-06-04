"""
Slot Completion Engine
======================
Layer 2.5 of the BeBISO 7-layer intent recognition framework.

Responsible for the Action Readiness Check — the gate that prevents
"False Success" by ensuring all required slots are filled before
execution is permitted.

Position in pipeline
--------------------
Layer 2: Ontology Match
        ↓
Layer 2.5: Slot Completion Engine  ← inserted here
        ├─ missing_required → decision=clarify → Layer 6 HITL
        └─ slots齐备 → 进入 Layer 3 Fusion
        ↓
Layer 3: Confidence Fusion

Action Readiness Score
-----------------------
ReadinessScore = filled_required_slots / total_required_slots × 100

Decision rules:
  - 0-60%:  decision=clarify  (mandatory — block execution)
  - 60-90%: decision=suggest  (advisory — warn but allow override)
  - 90-100%: decision=execute (ready to proceed)

Priority rule: missing_required_slots > 0 → decision=clarify
This overrides ALL confidence scores.
"""

from typing import Any, Dict, List, Optional

from .models import LLMLightResult, OntologyMatchResult, SlotCheckResult
from .slot_definitions import get_slot_schema, IntentSlotSchema


class SlotCompletionEngine:
    """Checks whether an intent has all required slots filled.

    Acts as the execution gate between Layer 2 (Ontology) and Layer 3 (Fusion).
    When required slots are missing, it forces the pipeline into clarification
    mode regardless of confidence scores.
    """

    # Readiness thresholds
    CLARIFY_THRESHOLD: float = 60.0   # < 60% → mandatory clarification
    SUGGEST_THRESHOLD: float = 100.0  # 60–99% → advisory suggestion

    def check(
        self,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        intent_id: str = "",
    ) -> SlotCheckResult:
        """Check slot readiness for the identified intent.

        Parameters
        ----------
        llm_result:
            Layer-1 output containing primary intent and extracted slots.
        ontology_match:
            Layer-2 output containing ontology object match.
        intent_id:
            Optional explicit intent_id from deep reasoning candidates.
            Used for more precise schema lookup.

        Returns
        -------
        SlotCheckResult
            Contains readiness score, decision, and missing/filled slot lists.
        """
        # Resolve intent template and object
        intent_template = llm_result.primary_intent or "unknown"
        object_name = ""

        if llm_result.object_matches:
            object_name = llm_result.object_matches[0].object_name
        elif ontology_match.object_match:
            object_name = ontology_match.object_match.object_name

        # Try multiple lookup strategies to find the best schema
        schema = self._find_schema(intent_template, object_name, intent_id)

        # ========================================================================
        # RULE 2: schema_found=false → MUST block or clarify.
        # Never default readiness=100 and allow execute when we don't know
        # what slots are required. This prevents generic templates from bypassing
        # the slot completion gate.
        # ========================================================================
        if schema is None:
            return SlotCheckResult(
                intent_template=intent_template,
                object_name=object_name,
                action_readiness_score=0.0,  # BLOCKED: we don't know what's needed
                decision="clarify",           # Force clarification
                schema_found=False,
                metadata={
                    "reason": "no_schema_defined",
                    "blocking": True,
                    "message": f"No slot schema for '{intent_template}'. "
                               f"Generic templates cannot be executed directly."
                },
            )

        # Build extracted slots dict from LLM dimension matches
        extracted = self._build_extracted_slots(llm_result)

        # Compute readiness
        missing_required = schema.get_missing_required(extracted)
        missing_optional = self._get_missing_optional(schema, extracted)
        filled_required = [s.name for s in schema.required_slots if s.name not in missing_required]
        filled_optional = [s.name for s in schema.optional_slots if s.name not in missing_optional]

        readiness_score = schema.readiness_score(extracted)
        decision, _ = schema.decision(extracted)

        return SlotCheckResult(
            intent_template=intent_template,
            object_name=object_name,
            action_readiness_score=readiness_score,
            decision=decision,
            filled_required=filled_required,
            missing_required=missing_required,
            filled_optional=filled_optional,
            missing_optional=missing_optional,
            schema_found=True,
            metadata={
                "intent_id": intent_id,
                "schema": schema.intent_template,
            },
        )

    def _find_schema(
        self,
        intent_template: str,
        object_name: str,
        intent_id: str,
    ) -> Optional[IntentSlotSchema]:
        """Find the best-matching slot schema using multiple strategies."""
        # Strategy 1: intent_id from deep reasoning (most specific)
        if intent_id:
            schema = get_slot_schema(intent_template, object_name, intent_id)
            if schema:
                return schema

        # Strategy 2: intent_template + object
        schema = get_slot_schema(intent_template, object_name)
        if schema:
            return schema

        # Strategy 3: intent_template only
        schema = get_slot_schema(intent_template)
        if schema:
            return schema

        # Strategy 4: map intent_template to known schema keys
        mapped = self._map_intent_template(intent_template, object_name)
        if mapped:
            schema = get_slot_schema(mapped)
            if schema:
                return schema

        return None

    def _map_intent_template(
        self, intent_template: str, object_name: str
    ) -> Optional[str]:
        """Map composite intent templates to defined schema keys."""
        # create_object + purchase_requests → create_purchase_requests
        if intent_template == "create_object" and "purchase" in object_name.lower():
            return "create_purchase_requests"
        # submit_approval + approval_workflows → submit_approval
        if intent_template == "submit_approval":
            return "submit_approval"
        if intent_template == "approve_workflow":
            return "approve_workflow"
        if intent_template == "reject_workflow":
            return "reject_workflow"
        # generate_order on purchase_orders → create_purchase_order
        if intent_template == "generate_order" and object_name == "purchase_orders":
            return "create_purchase_order"
        if intent_template == "create_purchase_order":
            return "create_purchase_order"
        return None

    def _build_extracted_slots(
        self, llm_result: LLMLightResult
    ) -> Dict[str, Any]:
        """Build a flat dict of slot_name → value from LLM dimension matches."""
        extracted: Dict[str, Any] = {}

        for dim in llm_result.dimension_matches:
            name = dim.dimension_name
            value = dim.value

            # Normalize slot names to match schema keys
            normalized = self._normalize_slot_name(name)
            extracted[normalized] = value
            # Also keep original for debugging
            extracted[name] = value

        # Check for common aliases
        if "material" not in extracted:
            for dim in llm_result.dimension_matches:
                name_lower = dim.dimension_name.lower()
                if "物料" in dim.value or "material" in name_lower:
                    extracted["material"] = dim.value
                    break

        return extracted

    def _normalize_slot_name(self, name: str) -> str:
        """Normalize a slot name to canonical form."""
        # Remove common prefixes/suffixes
        name = name.strip()
        # Map Chinese terms to canonical slot names
        alias_map = {
            "物料": "material",
            "物料信息": "material",
            "物料名称": "material",
            "物料编码": "material",
            "数量": "quantity",
            "需求数量": "quantity",
            "采购数量": "quantity",
            "需求日期": "delivery_date",
            "期望日期": "delivery_date",
            "到货日期": "delivery_date",
            "申请部门": "apply_dep",
            "部门": "apply_dep",
            "采购部门": "apply_dep",
        }
        return alias_map.get(name, name)

    def _get_missing_optional(
        self, schema: IntentSlotSchema, extracted: Dict[str, Any]
    ) -> List[str]:
        """Return list of optional slot names that are not filled."""
        return [
            s.name for s in schema.optional_slots
            if extracted.get(s.name) is None or extracted.get(s.name) == ""
        ]

    def generate_clarification_question(
        self, slot_check: SlotCheckResult, object_name: str = ""
    ) -> str:
        """Generate a human-readable clarification question for missing slots.

        Parameters
        ----------
        slot_check:
            Result from check() containing missing_required slots.
        object_name:
            The object being acted upon (e.g., "采购需求").

        Returns
        -------
        str
            A natural-language question asking the user to fill in missing info.
        """
        if not slot_check.missing_required:
            return ""

        object_label = self._get_object_label(object_name)
        missing = slot_check.missing_required

        # Build slot descriptions for the question
        slot_descs = self._get_slot_descriptions(missing)

        # Format the question
        lines = [
            f"好的，我来帮您创建{object_label}。",
            f"还需要补充以下信息：",
        ]
        for i, (slot, desc) in enumerate(slot_descs, 1):
            lines.append(f"{i}. {desc}")
        lines.append("")
        lines.append("您可以直接回复，例如：")
        lines.append(self._get_example_reply(missing, object_label))

        return "\n".join(lines)

    def _get_object_label(self, object_name: str) -> str:
        """Map object name to human-readable label."""
        labels = {
            "purchase_requests": "采购需求",
            "purchase_inquiries": "询价单",
            "purchase_quotations": "报价单",
            "purchase_orders": "采购订单",
            "approval_workflows": "审批",
            "inquiries": "询价单",
            "quotations": "报价单",
            "purchase_order_heads": "采购订单",
        }
        return labels.get(object_name, object_name)

    def _get_slot_descriptions(
        self, missing_slots: List[str]
    ) -> List[tuple[str, str]]:
        """Get descriptions for missing slots."""
        from .slot_definitions import INTENT_SLOT_SCHEMAS

        descriptions = {}
        for schema in INTENT_SLOT_SCHEMAS.values():
            for slot in schema.required_slots + schema.optional_slots:
                descriptions[slot.name] = slot.description

        return [
            (slot, descriptions.get(slot, f"请提供{slot}"))
            for slot in missing_slots
        ]

    def to_slot_form_fields(
        self,
        missing_slots: List[str],
        object_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Convert missing slots to frontend form field definitions.

        Parameters
        ----------
        missing_slots:
            List of missing slot names (e.g., ["material", "quantity", "delivery_date"]).
        object_name:
            The business object name (e.g., "purchase_requests").

        Returns
        -------
        List[Dict[str, Any]]
            List of slot field definitions for the frontend form renderer.
            Each dict contains: id, label, type, required, placeholder, options.

        Example output:
            [
                {"id": "material", "label": "物料信息", "type": "text",
                 "required": True, "placeholder": "请输入物料名称或编码"},
                {"id": "quantity", "label": "采购数量", "type": "number",
                 "required": True, "placeholder": "请输入数量"},
                {"id": "delivery_date", "label": "需求日期", "type": "date",
                 "required": True, "placeholder": "请选择日期"},
            ]
        """
        # Build descriptions from slot definitions
        from .slot_definitions import INTENT_SLOT_SCHEMAS, get_slot_schema

        # Try to get schema for this object
        schema = None
        for schema_key, s in INTENT_SLOT_SCHEMAS.items():
            if s.object == object_name:
                schema = s
                break

        fields = []
        for slot_name in missing_slots:
            field_def: Dict[str, Any] = {
                "id": slot_name,
                "label": self._get_slot_label(slot_name),
                "type": "text",
                "required": True,
                "placeholder": self._get_placeholder(slot_name),
            }

            # Enrich from schema if available
            if schema:
                for slot_def in schema.required_slots + schema.optional_slots:
                    if slot_def.name == slot_name:
                        field_def["label"] = slot_def.description or self._get_slot_label(slot_name)
                        field_def["type"] = self._map_slot_type(slot_def.slot_type)
                        field_def["required"] = slot_def.required
                        if slot_def.enum_values:
                            field_def["options"] = [
                                {"value": v, "label": v} for v in slot_def.enum_values
                            ]
                        if slot_def.examples:
                            field_def["placeholder"] = f"例如：{slot_def.examples[0]}"
                        break

            fields.append(field_def)

        return fields

    def _get_slot_label(self, slot_name: str) -> str:
        """Get human-readable label for a slot name."""
        labels = {
            "material": "物料信息",
            "quantity": "采购数量",
            "delivery_date": "需求日期",
            "apply_dep": "申请部门",
            "pr_id": "采购需求编号",
            "pr_type": "采购类型",
            "material_id": "物料编码",
            "material_d": "物料描述",
            "factory_id": "工厂",
            "company_id": "公司",
            "unit_id": "单位",
            "source_type": "来源类型",
            "material_category": "物料分类",
            "appro_whe": "是否需要审批",
            "document_id": "单据编号",
            "document_type": "单据类型",
            "supplier_scope": "询价供应商范围",
            "deadline": "报价截止日期",
            "inquiry_type": "询价类型",
            "quotation_ids": "报价单编号",
            "approver": "指定审批人",
        }
        return labels.get(slot_name, slot_name)

    def _get_placeholder(self, slot_name: str) -> str:
        """Get placeholder text for a slot."""
        placeholders = {
            "material": "请输入物料名称或编码",
            "quantity": "请输入采购数量",
            "delivery_date": "请选择需求日期",
            "apply_dep": "请输入申请部门",
            "pr_id": "请输入采购需求编号",
            "pr_type": "请选择采购类型",
            "material_id": "请输入物料编码",
            "material_d": "请输入物料描述",
            "factory_id": "请输入工厂",
            "company_id": "请输入公司",
            "unit_id": "请输入单位",
            "document_id": "请输入单据编号",
            "document_type": "请选择单据类型",
            "supplier_scope": "请输入供应商名称",
            "deadline": "请选择截止日期",
            "approver": "请输入审批人",
        }
        return placeholders.get(slot_name, f"请输入{self._get_slot_label(slot_name)}")

    def _map_slot_type(self, slot_type: str) -> str:
        """Map schema slot_type to frontend form field type."""
        mapping = {
            "string": "text",
            "number": "number",
            "date": "date",
            "enum": "select",
            "boolean": "select",
            "array": "text",
        }
        return mapping.get(slot_type, "text")

    def _get_example_reply(
        self, missing_slots: List[str], object_label: str
    ) -> str:
        """Generate an example reply showing how to fill all slots at once."""
        examples = {
            "material": "联想笔记本电脑",
            "quantity": "10台",
            "delivery_date": "下周五",
            "apply_dep": "销售部",
            "pr_id": "PR-2026-001",
            "supplier_scope": "供应商A、B、C",
            "deadline": "本周五",
        }
        parts = [examples.get(s, f"[{s}]") for s in missing_slots]
        return " ".join(parts)
