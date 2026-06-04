"""
Intent-to-Slot Definitions
===========================
Defines required and optional slots for each intent template.

This is the authoritative source of truth for action readiness checks.
When the pipeline identifies an intent (e.g., "create_object" on
"purchase_requests"), this module provides the slot schema that determines
whether the action is executable or needs clarification.

Slot Policy Rules
-----------------
- Required slots (required_slots): MUST be filled before execution.
  If any required slot is missing → decision=clarify (mandatory).
- Optional slots (optional_slots): Can be filled or use defaults.
  Missing optional slots → decision=suggest (advisory) or proceed.
- Readiness Score = (filled_required / total_required) × 100
  - 0-60%:  decision=clarify  (block execution, ask user)
  - 60-90%: decision=suggest  (warn user, allow override)
  - 90-100%: decision=execute (ready to proceed)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlotDefinition:
    """Definition of a single slot."""

    name: str
    description: str = ""
    slot_type: str = "string"  # string, number, date, enum, boolean
    required: bool = False
    enum_values: List[str] = field(default_factory=list)
    default: Optional[Any] = None
    examples: List[str] = field(default_factory=list)


@dataclass
class IntentSlotSchema:
    """Slot schema for an intent template."""

    intent_template: str
    object: str = ""  # e.g., "purchase_requests"
    required_slots: List[SlotDefinition] = field(default_factory=list)
    optional_slots: List[SlotDefinition] = field(default_factory=list)
    all_fields: List[SlotDefinition] = field(default_factory=list)

    def get_slot(self, name: str) -> Optional[SlotDefinition]:
        """Find a slot definition by name."""
        for s in self.all_fields:
            if s.name == name:
                return s
        return None

    def get_missing_required(self, extracted_slots: Dict[str, Any]) -> List[str]:
        """Return list of required slot names that are not filled."""
        missing = []
        for slot in self.required_slots:
            value = extracted_slots.get(slot.name)
            if value is None or value == "" or value == []:
                missing.append(slot.name)
        return missing

    def readiness_score(self, extracted_slots: Dict[str, Any]) -> float:
        """Calculate action readiness score [0.0, 100.0]."""
        if not self.required_slots:
            return 100.0
        filled = 0
        for slot in self.required_slots:
            value = extracted_slots.get(slot.name)
            if value is not None and value != "" and value != []:
                filled += 1
        return (filled / len(self.required_slots)) * 100.0

    def decision(
        self, extracted_slots: Dict[str, Any]
    ) -> tuple[str, float]:
        """Determine decision and score based on readiness.

        Returns (decision, readiness_score):
            - clarify: readiness < 60% — mandatory clarification
            - suggest: 60% <= readiness < 100% — advisory suggestion
            - execute: readiness == 100% — ready to proceed
        """
        score = self.readiness_score(extracted_slots)
        if score < 60:
            return "clarify", score
        elif score < 100:
            return "suggest", score
        else:
            return "execute", score


# ---------------------------------------------------------------------------
# Intent Slot Schemas
# ---------------------------------------------------------------------------

INTENT_SLOT_SCHEMAS: Dict[str, IntentSlotSchema] = {}

# ----- purchase_requests -----

INTENT_SLOT_SCHEMAS["create_purchase_requests"] = IntentSlotSchema(
    intent_template="create_purchase_requests",
    object="purchase_requests",
    required_slots=[
        SlotDefinition(
            name="material",
            description="要采购的物料名称或编码",
            slot_type="string",
            required=True,
            examples=["联想笔记本电脑", "物料编码 M001"],
        ),
        SlotDefinition(
            name="quantity",
            description="采购数量",
            slot_type="number",
            required=True,
            examples=["10台", "100个"],
        ),
        SlotDefinition(
            name="delivery_date",
            description="需求日期/期望到货日期",
            slot_type="date",
            required=True,
            examples=["下周五", "2026-06-15"],
        ),
        SlotDefinition(
            name="apply_dep",
            description="申请部门",
            slot_type="string",
            required=True,
            examples=["销售部", "采购部"],
        ),
    ],
    optional_slots=[
        SlotDefinition(
            name="factory_id",
            description="工厂",
            slot_type="string",
            required=False,
        ),
        SlotDefinition(
            name="company_id",
            description="公司",
            slot_type="string",
            required=False,
        ),
        SlotDefinition(
            name="pr_type",
            description="采购类型（标准/紧急/生产）",
            slot_type="enum",
            required=False,
            enum_values=["标准", "紧急", "生产"],
        ),
        SlotDefinition(
            name="material_category",
            description="物料分类",
            slot_type="string",
            required=False,
            examples=["办公用品", "IT设备", "生产物料"],
        ),
        SlotDefinition(
            name="unit_id",
            description="单位",
            slot_type="string",
            required=False,
            examples=["台", "个", "箱"],
        ),
    ],
)

# Fallback for generic create_object intent
INTENT_SLOT_SCHEMAS["create_object"] = IntentSlotSchema(
    intent_template="create_object",
    object="",
    required_slots=[
        SlotDefinition(
            name="material",
            description="要创建的物料/对象名称",
            slot_type="string",
            required=True,
        ),
        SlotDefinition(
            name="quantity",
            description="数量",
            slot_type="number",
            required=True,
        ),
        SlotDefinition(
            name="delivery_date",
            description="期望日期",
            slot_type="date",
            required=True,
        ),
        SlotDefinition(
            name="apply_dep",
            description="申请部门",
            slot_type="string",
            required=True,
        ),
    ],
    optional_slots=[],
)

# ----- purchase_inquiries -----

INTENT_SLOT_SCHEMAS["create_inquiry"] = IntentSlotSchema(
    intent_template="create_inquiry",
    object="purchase_inquiries",
    required_slots=[
        SlotDefinition(
            name="pr_id",
            description="关联的采购需求编号",
            slot_type="string",
            required=True,
            examples=["PR-2026-001"],
        ),
        SlotDefinition(
            name="supplier_scope",
            description="询价供应商范围",
            slot_type="string",
            required=True,
            examples=["供应商A、B、C", "IT类供应商"],
        ),
        SlotDefinition(
            name="deadline",
            description="报价截止日期",
            slot_type="date",
            required=True,
            examples=["本周五", "2026-06-10"],
        ),
    ],
    optional_slots=[
        SlotDefinition(
            name="inquiry_type",
            description="询价类型",
            slot_type="enum",
            required=False,
            enum_values=["公开询价", "定向询价"],
        ),
    ],
)

# ----- purchase_quotations -----

INTENT_SLOT_SCHEMAS["compare_quotations"] = IntentSlotSchema(
    intent_template="compare_quotations",
    object="purchase_quotations",
    required_slots=[
        SlotDefinition(
            name="pr_id",
            description="采购需求编号",
            slot_type="string",
            required=True,
        ),
        SlotDefinition(
            name="quotation_ids",
            description="报价单编号列表（至少3家）",
            slot_type="array",
            required=True,
        ),
    ],
    optional_slots=[],
)

# ----- approval -----

INTENT_SLOT_SCHEMAS["submit_approval"] = IntentSlotSchema(
    intent_template="submit_approval",
    object="approval_workflows",
    required_slots=[
        SlotDefinition(
            name="document_id",
            description="单据编号",
            slot_type="string",
            required=True,
        ),
        SlotDefinition(
            name="document_type",
            description="单据类型",
            slot_type="enum",
            required=True,
            enum_values=["purchase_request", "purchase_order", "quotation"],
        ),
    ],
    optional_slots=[
        SlotDefinition(
            name="approver",
            description="指定审批人",
            slot_type="string",
            required=False,
        ),
    ],
)

INTENT_SLOT_SCHEMAS["approve_workflow"] = INTENT_SLOT_SCHEMAS["submit_approval"]

INTENT_SLOT_SCHEMAS["reject_workflow"] = INTENT_SLOT_SCHEMAS["submit_approval"]


# ----- create_purchase_order -----

INTENT_SLOT_SCHEMAS["create_purchase_order"] = IntentSlotSchema(
    intent_template="create_purchase_order",
    object="purchase_orders",
    required_slots=[
        SlotDefinition(
            name="pr_id",
            description="采购需求编号",
            slot_type="string",
            required=True,
            examples=["PR-2026-001"],
        ),
        SlotDefinition(
            name="supplier_id",
            description="供应商编号",
            slot_type="string",
            required=True,
            examples=["SUP-001", "供应商A"],
        ),
        SlotDefinition(
            name="material",
            description="物料明细",
            slot_type="string",
            required=True,
            examples=["联想笔记本电脑"],
        ),
        SlotDefinition(
            name="quantity",
            description="采购数量",
            slot_type="number",
            required=True,
            examples=["10台"],
        ),
        SlotDefinition(
            name="price",
            description="价格（单价或总价）",
            slot_type="number",
            required=True,
            examples=["5000元", "50000"],
        ),
        SlotDefinition(
            name="delivery_date",
            description="交货日期",
            slot_type="date",
            required=True,
            examples=["2026-06-15", "下周五"],
        ),
        SlotDefinition(
            name="payment_terms",
            description="付款条款",
            slot_type="string",
            required=False,
            examples=["预付30%", "货到付款"],
        ),
    ],
    optional_slots=[
        SlotDefinition(
            name="apply_dep",
            description="申请部门",
            slot_type="string",
            required=False,
            examples=["采购部", "销售部"],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_slot_schema(
    intent_template: str,
    object_name: str = "",
    intent_id: str = "",
) -> Optional[IntentSlotSchema]:
    """Find the best-matching slot schema for an intent.

    Priority:
    1. Match by full intent_id (e.g., "create_purchase_requests")
    2. Match by intent_template + object (e.g., "create_inquiry" + "purchase_inquiries")
    3. Match by intent_template only
    4. Return None (no schema defined — skip slot check)
    """
    # Try exact intent_id match
    if intent_id and intent_id in INTENT_SLOT_SCHEMAS:
        return INTENT_SLOT_SCHEMAS[intent_id]

    # Try template + object match
    key = f"{intent_template}_{object_name}"
    if key in INTENT_SLOT_SCHEMAS:
        return INTENT_SLOT_SCHEMAS[key]

    # Try template only
    if intent_template in INTENT_SLOT_SCHEMAS:
        return INTENT_SLOT_SCHEMAS[intent_template]

    return None


def build_all_fields(schema: IntentSlotSchema) -> IntentSlotSchema:
    """Populate the all_fields list from required + optional slots."""
    schema.all_fields = schema.required_slots + schema.optional_slots
    return schema


# Pre-build all_fields
for schema in INTENT_SLOT_SCHEMAS.values():
    build_all_fields(schema)
