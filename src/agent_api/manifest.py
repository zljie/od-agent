"""
Agent Tool Registry — Tool Manifest definitions and registry.

Per the API Manifest MVP protocol defined in
docs/prd/API_Manifest_后端服务方案.md Section 12.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool Manifest Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HitlPolicy:
    required: bool
    trigger: str = "before_invoke"
    confirm_message_template: str = ""
    risk_threshold: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "trigger": self.trigger,
            "confirm_message_template": self.confirm_message_template,
            "risk_threshold": self.risk_threshold,
        }


@dataclass
class PermissionPolicy:
    required_roles: List[str] = field(default_factory=list)
    scope: str = "department"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_roles": self.required_roles,
            "scope": self.scope,
        }


@dataclass
class AuditPolicy:
    enabled: bool = True
    level: str = "business_action"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "level": self.level,
        }


@dataclass
class RuntimeBinding:
    validate_endpoint: str
    dry_run_endpoint: str
    invoke_endpoint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validate_endpoint": self.validate_endpoint,
            "dry_run_endpoint": self.dry_run_endpoint,
            "invoke_endpoint": self.invoke_endpoint,
        }


@dataclass
class OntologyBinding:
    object: str
    action: str
    related_objects: List[str] = field(default_factory=list)
    field_mapping: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.object,
            "action": self.action,
            "related_objects": self.related_objects,
            "field_mapping": self.field_mapping,
        }


@dataclass
class InputSchemaField:
    name: str
    type: str
    title: str
    description: str = ""
    ontology_field: str = ""
    required: bool = False
    minimum: Optional[float] = None
    format_hint: Optional[str] = None
    enum_values: List[Any] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "ontology_field": self.ontology_field,
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.format_hint:
            result["format"] = self.format_hint
        if self.enum_values:
            result["enum"] = self.enum_values
        if self.examples:
            result["examples"] = self.examples
        return result


@dataclass
class InputSchema:
    required: List[str]
    properties: Dict[str, InputSchemaField]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "description": self.description,
            "required": self.required,
            "properties": {
                name: f.to_dict()
                for name, f in self.properties.items()
            },
        }


@dataclass
class OutputSchema:
    properties: Dict[str, Any]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class ToolManifest:
    name: str
    title: str
    description: str
    domain: str
    ontology: OntologyBinding
    intent_examples: List[str]
    input_schema: InputSchema
    output_schema: OutputSchema
    runtime: RuntimeBinding
    action_kind: str  # "query" | "command" | "analytics"
    hitl_policy: Optional[HitlPolicy] = None
    permission: Optional[PermissionPolicy] = None
    audit: Optional[AuditPolicy] = None
    tags: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Lightweight entry for /agent/manifest/tools listing."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "ontology_object": self.ontology.object,
            "action_type": self.action_kind,
            "risk_level": (
                self.hitl_policy.risk_threshold
                if self.hitl_policy and self.hitl_policy.required
                else "low"
            ),
            "requires_hitl": self.hitl_policy.required if self.hitl_policy else False,
            "tags": self.tags,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Full manifest for /agent/manifest/tools/{name}."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "ontology": self.ontology.to_dict(),
            "intent_examples": self.intent_examples,
            "input_schema_ref": f"/agent/manifest/tools/{self.name}/schema/input",
            "output_schema_ref": f"/agent/manifest/tools/{self.name}/schema/output",
            "runtime": self.runtime.to_dict(),
            "hitl_policy": self.hitl_policy.to_dict() if self.hitl_policy else None,
            "permission": self.permission.to_dict() if self.permission else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "action_kind": self.action_kind,
        }


# ---------------------------------------------------------------------------
# Full Tool Manifests (MVP — 3 core tools)
# ---------------------------------------------------------------------------

TOOL_MANIFESTS: Dict[str, ToolManifest] = {}


def _build_manifests() -> None:
    global TOOL_MANIFESTS

    # ── create_purchase_request ──────────────────────────────────────────────
    TOOL_MANIFESTS["create_purchase_request"] = ToolManifest(
        name="create_purchase_request",
        title="创建采购申请",
        description="根据部门、物料、数量和交付日期创建采购申请单。这是采购流程的起点。",
        domain="procurement",
        action_kind="command",
        ontology=OntologyBinding(
            object="PurchaseRequest",
            action="create",
            related_objects=["Department", "Material", "Employee"],
            field_mapping={
                "department": "PurchaseRequest.apply_dep",
                "item_name": "PurchaseRequest.item.name",
                "quantity": "PurchaseRequest.quantity",
                "delivery_date": "PurchaseRequest.delivery_date",
            },
        ),
        intent_examples=[
            "帮销售部门新增10箱A4打印纸采购申请",
            "为行政部申请5台显示器",
            "我要发起一条采购需求",
            "创建一个采购申请，物料是联想笔记本电脑，数量3台",
        ],
        input_schema=InputSchema(
            description="创建采购申请的输入参数",
            required=["department", "item_name", "quantity", "delivery_date"],
            properties={
                "department": InputSchemaField(
                    name="department",
                    type="string",
                    title="申请部门",
                    description="发起采购申请的业务部门",
                    ontology_field="PurchaseRequest.apply_dep",
                    required=True,
                    examples=["销售部", "行政部", "研发部"],
                ),
                "item_name": InputSchemaField(
                    name="item_name",
                    type="string",
                    title="物料名称",
                    description="需要采购的物料或商品名称",
                    ontology_field="PurchaseRequest.item.name",
                    required=True,
                    examples=["联想笔记本电脑", "A4打印纸", "Dell显示器"],
                ),
                "quantity": InputSchemaField(
                    name="quantity",
                    type="number",
                    title="采购数量",
                    description="采购物料数量，必须为正整数",
                    ontology_field="PurchaseRequest.quantity",
                    required=True,
                    minimum=1,
                    examples=["10", "5", "100"],
                ),
                "delivery_date": InputSchemaField(
                    name="delivery_date",
                    type="string",
                    title="期望交付日期",
                    description="用户期望收到物料的日期，ISO格式 YYYY-MM-DD",
                    ontology_field="PurchaseRequest.delivery_date",
                    required=True,
                    format_hint="date",
                    examples=["2026-06-15", "2026-07-01"],
                ),
                "material_category": InputSchemaField(
                    name="material_category",
                    type="string",
                    title="物料分类",
                    description="物料所属分类",
                    ontology_field="PurchaseRequest.material_category",
                    required=False,
                    enum_values=["办公用品", "IT设备", "生产物料", "工程物资"],
                    examples=["IT设备", "办公用品"],
                ),
                "pr_type": InputSchemaField(
                    name="pr_type",
                    type="string",
                    title="采购类型",
                    description="采购申请的类型",
                    ontology_field="PurchaseRequest.pr_type",
                    required=False,
                    enum_values=["NB", "标准采购", "紧急采购", "生产采购"],
                    examples=["标准采购", "紧急采购"],
                ),
            },
        ),
        output_schema=OutputSchema(
            description="创建采购申请的输出结果",
            properties={
                "pr_id": {
                    "type": "string",
                    "title": "采购申请编号",
                    "description": "系统生成的采购申请编号",
                },
                "status": {
                    "type": "string",
                    "title": "状态",
                    "description": "当前状态",
                },
                "created_at": {
                    "type": "string",
                    "title": "创建时间",
                    "format": "date-time",
                },
            },
        ),
        runtime=RuntimeBinding(
            validate_endpoint="/agent/tools/create_purchase_request/validate",
            dry_run_endpoint="/agent/tools/create_purchase_request/dry-run",
            invoke_endpoint="/agent/tools/create_purchase_request/invoke",
        ),
        hitl_policy=HitlPolicy(
            required=True,
            trigger="before_invoke",
            confirm_message_template="即将为 {{department}} 创建 {{quantity}} {{item_name}} 的采购申请，是否确认？",
            risk_threshold="medium",
        ),
        permission=PermissionPolicy(
            required_roles=["procurement_requester"],
            scope="department",
        ),
        audit=AuditPolicy(enabled=True, level="business_action"),
        tags=["procurement", "purchase_request", "create", "command"],
    )

    # ── query_purchase_requests ──────────────────────────────────────────────
    TOOL_MANIFESTS["query_purchase_requests"] = ToolManifest(
        name="query_purchase_requests",
        title="查询采购需求",
        description="分页查询采购需求列表，支持按部门、日期范围、状态等条件筛选。",
        domain="procurement",
        action_kind="query",
        ontology=OntologyBinding(
            object="PurchaseRequest",
            action="query",
            related_objects=["Department", "Material"],
        ),
        intent_examples=[
            "查询采购需求列表",
            "看看有哪些待审批的采购申请",
            "查询销售部门的采购需求",
            "列出本月所有采购需求",
        ],
        input_schema=InputSchema(
            description="查询采购需求的输入参数",
            required=[],
            properties={
                "page": InputSchemaField(
                    name="page",
                    type="integer",
                    title="页码",
                    description="分页页码，从1开始",
                    ontology_field="PurchaseRequest.page",
                    required=False,
                    examples=["1", "2"],
                ),
                "page_size": InputSchemaField(
                    name="page_size",
                    type="integer",
                    title="每页条数",
                    description="每页返回的记录数，默认50",
                    ontology_field="PurchaseRequest.page_size",
                    required=False,
                    minimum=1,
                    examples=["20", "50", "100"],
                ),
                "department": InputSchemaField(
                    name="department",
                    type="string",
                    title="申请部门",
                    description="按部门名称筛选",
                    ontology_field="PurchaseRequest.apply_dep",
                    required=False,
                    examples=["销售部", "采购部"],
                ),
                "date_from": InputSchemaField(
                    name="date_from",
                    type="string",
                    title="需求日期起",
                    description="需求日期范围起始，ISO格式",
                    ontology_field="PurchaseRequest.delivery_date",
                    required=False,
                    format_hint="date",
                    examples=["2026-01-01"],
                ),
                "date_to": InputSchemaField(
                    name="date_to",
                    type="string",
                    title="需求日期止",
                    description="需求日期范围结束，ISO格式",
                    ontology_field="PurchaseRequest.delivery_date",
                    required=False,
                    format_hint="date",
                    examples=["2026-12-31"],
                ),
                "status": InputSchemaField(
                    name="status",
                    type="string",
                    title="执行状态",
                    description="按执行状态筛选",
                    ontology_field="PurchaseRequest.flow_status",
                    required=False,
                    enum_values=["未执行", "已执行", "待审批", "已审批", "已驳回"],
                    examples=["未执行", "已审批"],
                ),
                "q": InputSchemaField(
                    name="q",
                    type="string",
                    title="关键词搜索",
                    description="按物料名称或编号搜索",
                    ontology_field="PurchaseRequest.material_d",
                    required=False,
                    examples=["笔记本电脑", "联想"],
                ),
            },
        ),
        output_schema=OutputSchema(
            description="查询采购需求的输出结果",
            properties={
                "items": {
                    "type": "array",
                    "title": "采购需求列表",
                    "description": "采购需求行项目数组",
                },
                "total": {
                    "type": "integer",
                    "title": "总记录数",
                },
                "page": {
                    "type": "integer",
                    "title": "当前页码",
                },
                "page_size": {
                    "type": "integer",
                    "title": "每页条数",
                },
                "total_pages": {
                    "type": "integer",
                    "title": "总页数",
                },
            },
        ),
        runtime=RuntimeBinding(
            validate_endpoint="/agent/tools/query_purchase_requests/validate",
            dry_run_endpoint="/agent/tools/query_purchase_requests/dry-run",
            invoke_endpoint="/agent/tools/query_purchase_requests/invoke",
        ),
        hitl_policy=HitlPolicy(
            required=False,
            trigger="on_risk",
            risk_threshold="low",
        ),
        permission=PermissionPolicy(
            required_roles=["procurement_requester", "procurement_viewer"],
            scope="department",
        ),
        audit=AuditPolicy(enabled=True, level="query"),
        tags=["procurement", "purchase_request", "query"],
    )

    # ── query_purchase_orders ────────────────────────────────────────────────
    TOOL_MANIFESTS["query_purchase_orders"] = ToolManifest(
        name="query_purchase_orders",
        title="查询采购订单执行情况",
        description="查询采购订单的审批状态、发货状态、收货状态和发票状态，以及执行进度。",
        domain="procurement",
        action_kind="query",
        ontology=OntologyBinding(
            object="PurchaseOrder",
            action="query",
            related_objects=["PurchaseOrderItem", "PurchaseOrderReceiptHistory", "Vendor"],
        ),
        intent_examples=[
            "查询采购订单 PO-20260604-001 的执行情况",
            "看看订单PO-20260603-005的状态",
            "采购订单到货了吗",
            "订单执行进度如何",
        ],
        input_schema=InputSchema(
            description="查询采购订单执行情况的输入参数",
            required=["po_id"],
            properties={
                "po_id": InputSchemaField(
                    name="po_id",
                    type="string",
                    title="采购订单编号",
                    description="需要查询的采购订单编号，如 PO-20260604-001",
                    ontology_field="PurchaseOrder.po_id",
                    required=True,
                    examples=["PO-20260604-001", "PO20260603005"],
                ),
            },
        ),
        output_schema=OutputSchema(
            description="查询采购订单执行情况的输出结果",
            properties={
                "po_id": {"type": "string", "title": "采购订单编号"},
                "approval_status": {"type": "string", "title": "审批状态"},
                "shipment_status": {"type": "string", "title": "发货状态"},
                "receipt_status": {"type": "string", "title": "收货状态"},
                "invoice_status": {"type": "string", "title": "发票状态"},
                "items": {"type": "array", "title": "订单行项目"},
                "receipts": {"type": "array", "title": "收货历史"},
            },
        ),
        runtime=RuntimeBinding(
            validate_endpoint="/agent/tools/query_purchase_orders/validate",
            dry_run_endpoint="/agent/tools/query_purchase_orders/dry-run",
            invoke_endpoint="/agent/tools/query_purchase_orders/invoke",
        ),
        hitl_policy=HitlPolicy(
            required=False,
            trigger="on_risk",
            risk_threshold="low",
        ),
        permission=PermissionPolicy(
            required_roles=["procurement_requester", "procurement_viewer"],
            scope="department",
        ),
        audit=AuditPolicy(enabled=True, level="query"),
        tags=["procurement", "purchase_order", "query"],
    )


_build_manifests()


# ---------------------------------------------------------------------------
# Agent Tool Registry
# ---------------------------------------------------------------------------

class AgentToolRegistry:
    """
    Central registry for Agent Tool Manifests.

    Provides read-only access to tool manifests (full and summary forms)
    and input schemas.
    """

    def __init__(self) -> None:
        self._manifests: Dict[str, ToolManifest] = dict(TOOL_MANIFESTS)

    # ── Read operations ──────────────────────────────────────────────────────

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return lightweight summary of all registered tools."""
        return [m.to_summary_dict() for m in self._manifests.values()]

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Return full manifest for a specific tool."""
        manifest = self._manifests.get(name)
        if manifest is None:
            return None
        return manifest.to_full_dict()

    def get_input_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Return input JSON schema for a specific tool."""
        manifest = self._manifests.get(name)
        if manifest is None:
            return None
        return manifest.input_schema.to_dict()

    def get_output_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Return output JSON schema for a specific tool."""
        manifest = self._manifests.get(name)
        if manifest is None:
            return None
        return manifest.output_schema.to_dict()

    def tool_exists(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._manifests

    def get_tool_manifest(self, name: str) -> Optional[ToolManifest]:
        """Return the raw ToolManifest object for a tool."""
        return self._manifests.get(name)

    def all_tool_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._manifests.keys())

    def filter_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Return tools filtered by domain."""
        return [
            m.to_summary_dict()
            for m in self._manifests.values()
            if m.domain == domain
        ]

    def filter_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        """Return tools filtered by action kind (query/command/analytics)."""
        return [
            m.to_summary_dict()
            for m in self._manifests.values()
            if m.action_kind == kind
        ]


# Singleton instance
_registry: Optional[AgentToolRegistry] = None


def get_tool_registry() -> AgentToolRegistry:
    """Get the global AgentToolRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = AgentToolRegistry()
    return _registry
