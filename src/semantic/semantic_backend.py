"""OSI YAML loader and SemanticBackend skill.

Loads semantic_model.yaml files and wires up the full semantic-native backend:
OSI YAML → OSIModel → GraphQL SDL → MCP Tools → Semantic Index → Agent Integration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .graphql_generator import GraphQLGenerator
from .mcp_server import MCPServer, MCPMode, build_mcp_tools_from_osi
from .osi_model import (
    AIContext,
    ActionParameter,
    DataSet,
    FieldDefinition,
    OSIModel,
    Relationship,
    Rule,
)
from .semantic_indexer import SemanticIndexer


def load_osi_model(path: str | Path) -> OSIModel:
    """Load an OSI semantic model from a YAML file.

    Supports two formats:
    1. Direct format: {datasets: [...], relationships: [...], metrics: [...], behavior: {...}}
    2. Nested semantic_model format: {semantic_model: [{name: ..., datasets: [...], ...}]}
       (the DTP procurement ontology uses this format)

    Args:
        path: Path to the YAML file.

    Returns:
        OSIModel instance with all datasets, relationships, metrics, actions, and rules loaded.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Normalise the YAML structure: handle the nested `semantic_model[]` wrapper.
    if "semantic_model" in raw:
        model_list = raw["semantic_model"]
        if isinstance(model_list, list) and len(model_list) > 0:
            data = model_list[0]
        elif isinstance(model_list, dict):
            data = model_list
        else:
            data = {}
    else:
        data = raw

    return _build_osi_model(data)


def _build_osi_model(data: Dict[str, Any]) -> OSIModel:
    """Build an OSIModel from a parsed YAML dict (no file I/O).

    Handles the DTP ontology field format where:
    - field entries use `type` instead of `gql_type`
    - extra fields like `expression`, `primary_key`, `source` are preserved as metadata
    - ai_context can be at the top level (for the model itself)
    - `operation_catalog` entries are loaded as actions when `behavior.actions` is absent
    """
    from .osi_model import Action, Metric, Rule

    # ── Load datasets ───────────────────────────────────────────────────────
    datasets = []
    for ds_raw in data.get("datasets", []):
        ai_ctx = None
        if "ai_context" in ds_raw:
            ai_ctx = AIContext(**ds_raw.pop("ai_context"))

        fields = []
        for f_raw in ds_raw.pop("fields", []):
            # Normalise `type` → `gql_type` (DTP format uses `type`)
            gql_type = f_raw.pop("type", None)
            if gql_type and "gql_type" not in f_raw:
                f_raw["gql_type"] = gql_type

            # Preserve SQL expression metadata for schema documentation
            expr = f_raw.pop("expression", None)
            if expr:
                f_raw["ai_hint"] = f"SQL: {expr}"

            # Strip non-OSI fields that DataSet/FieldDefinition don't know about
            for _field in list(f_raw.keys()):
                if _field not in (
                    "name", "gql_type", "description", "required", "is_list",
                    "enum_values", "enum_name", "relation_target", "ai_hint",
                ):
                    f_raw.pop(_field, None)

            fields.append(FieldDefinition(**f_raw))

        # Strip non-OSI top-level dataset fields
        for _f in list(ds_raw.keys()):
            if _f not in ("name", "description", "fields", "ai_context"):
                ds_raw.pop(_f, None)

        datasets.append(DataSet(ai_context=ai_ctx, fields=fields, **ds_raw))

    # ── Load relationships ──────────────────────────────────────────────────
    relationships = []
    for r in data.get("relationships", []):
        # Normalise `from`/`to` → `from_dataset`/`to_dataset` (DTP format)
        # Must happen before the strip loop below.
        if "from" in r and "from_dataset" not in r:
            r["from_dataset"] = r.pop("from")
        if "to" in r and "to_dataset" not in r:
            r["to_dataset"] = r.pop("to")
        # Strip non-OSI fields
        for _f in list(r.keys()):
            if _f not in (
                "from_dataset", "to_dataset", "relation_type", "description",
                "via_field", "ai_hint",
            ):
                r.pop(_f, None)
        relationships.append(Relationship(**r))

    # ── Load metrics ───────────────────────────────────────────────────────
    metrics = []
    for m in data.get("metrics", []):
        ai_ctx = None
        if "ai_context" in m:
            ai_ctx = AIContext(**m.pop("ai_context"))
        for _f in list(m.keys()):
            if _f not in (
                "name", "description", "unit", "dataset", "aggregation",
                "filter_fields", "ai_context",
            ):
                m.pop(_f, None)
        metrics.append(Metric(ai_context=ai_ctx, **m))

    # ── Load actions (from `behavior.actions` or `operation_catalog`) ───────
    actions = []
    behavior_section = data.get("behavior", {})
    action_sources = behavior_section.get("actions", []) or data.get("operation_catalog", [])

    for a in action_sources:
        ai_ctx = None
        if "ai_context" in a:
            ai_ctx = AIContext(**a.pop("ai_context"))

        # operation_catalog entries use `name` or `id`; prefer `name`
        action_name = a.get("name") or a.get("id", "unknown")
        # operation_catalog format: {kind, operation, entity_name, io_schema, ...}
        # Map to OSI Action fields
        params = []
        io_schema = a.get("io_schema", {})
        input_schema = io_schema.get("input_schema", {})
        if isinstance(input_schema, dict):
            required = input_schema.get("required", [])
            for pname, pdef in input_schema.get("properties", {}).items():
                ptype = _map_type(pdef.get("type", "string"))
                params.append(ActionParameter(
                    name=pname,
                    gql_type=ptype,
                    description=pdef.get("description", ""),
                    required=pname in required,
                ))

        for _extra in ["name", "id", "kind", "operation", "entity_name",
                       "io_schema", "labels", "applies_to"]:
            a.pop(_extra, None)
        # Now a only has OSI Action fields; use **a safely
        action_ds = a.pop("dataset", None) or ""
        actions.append(Action(
            ai_context=ai_ctx,
            name=action_name,
            dataset=action_ds,
            description=a.pop("description", ""),
            parameters=params,
            **a,
        ))

    # ── Load rules (from `behavior.rules`) ─────────────────────────────────
    rules = []
    for r in behavior_section.get("rules", []):
        ai_ctx = None
        if "ai_context" in r:
            ai_ctx = AIContext(**r.pop("ai_context"))
        for _f in list(r.keys()):
            if _f not in (
                "name", "description", "severity", "dataset", "condition",
                "enforcement", "ai_context",
            ):
                r.pop(_f, None)
        rules.append(Rule(ai_context=ai_ctx, **r))

    # ── Model-level ai_context ─────────────────────────────────────────────
    model_ai_ctx = None
    if "ai_context" in data:
        model_ai_ctx = AIContext(**data.pop("ai_context"))

    return OSIModel(
        version=data.get("version", "1.0"),
        domain=data.get("name", "") or data.get("domain", ""),
        description=data.get("description", ""),
        datasets=datasets,
        relationships=relationships,
        metrics=metrics,
        actions=actions,
        rules=rules,
    )


def _map_type(t: str) -> str:
    """Map YAML type strings to GraphQL type names."""
    mapping = {
        "string": "String",
        "integer": "Int",
        "number": "Float",
        "boolean": "Boolean",
        "date": "String",
        "datetime": "DateTime",
        "object": "JSON",
        "array": "String",
    }
    return mapping.get(t.lower(), "String")


def load_osi_model_from_dict(data: Dict[str, Any]) -> OSIModel:
    """Build an OSIModel from a parsed YAML dict (no file I/O)."""
    if "semantic_model" in data:
        model_list = data["semantic_model"]
        if isinstance(model_list, list) and len(model_list) > 0:
            data = model_list[0]
        elif isinstance(model_list, dict):
            data = model_list
    return _build_osi_model(data)


class ActionSpec:
    """Action specification loaded from YAML (used during model loading)."""

    def __init__(self, name: str, dataset: str = "", **kwargs):
        self.name = name
        self.dataset = dataset
        for k, v in kwargs.items():
            setattr(self, k, v)


class RuleSpec:
    """Rule specification loaded from YAML (used during model loading)."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            setattr(self, k, v)


# Re-export ActionSpec for compatibility
from .osi_model import Action, Metric, Rule


class SemanticBackend:
    """The complete semantic-native AI backend wired from OSI model.

    Combines:
    - OSI model (loaded from YAML)
    - GraphQL schema (generated from OSI)
    - MCP tools (derived from GraphQL operations)
    - Semantic index (RAG over ai_context)

    Usage:
        backend = SemanticBackend.from_yaml("semantic_model.yaml")
        backend.load()          # generate schema, build tools, build index
        schema = backend.schema  # GraphQL SDL
        tools = backend.mcp_server.list_tools()
        results = backend.search("查供应商的采购情况")
    """

    def __init__(
        self,
        model: Optional[OSIModel] = None,
        yaml_path: Optional[str] = None,
        graphql_endpoint: Optional[str] = None,
    ):
        self._model = model
        self._yaml_path = yaml_path
        self._graphql_endpoint = graphql_endpoint
        self._schema: str = ""
        self._mcp_server: Optional[MCPServer] = None
        self._indexer: Optional[SemanticIndexer] = None
        self._loaded: bool = False

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        graphql_endpoint: Optional[str] = None,
    ) -> "SemanticBackend":
        return cls(yaml_path=str(path), graphql_endpoint=graphql_endpoint)

    @classmethod
    def from_model(cls, model: OSIModel, graphql_endpoint: Optional[str] = None) -> "SemanticBackend":
        return cls(model=model, graphql_endpoint=graphql_endpoint)

    def load(self) -> "SemanticBackend":
        """Load the OSI model, generate GraphQL schema, build MCP tools, and index."""
        if self._loaded:
            return self

        if self._yaml_path:
            self._model = load_osi_model(self._yaml_path)

        if self._model is None:
            raise ValueError("No OSI model provided. Set yaml_path or pass model=...")

        gql_gen = GraphQLGenerator(self._model)
        self._schema = gql_gen.generate()

        tools = build_mcp_tools_from_osi(self._model)
        self._mcp_server = MCPServer(
            tools=tools,
            mode=MCPMode.DYNAMIC,
            endpoint=self._graphql_endpoint,
        )

        self._indexer = SemanticIndexer()
        self._indexer.index_model(self._model)

        self._loaded = True
        return self

    @property
    def schema(self) -> str:
        if not self._loaded:
            self.load()
        return self._schema

    @property
    def model(self) -> OSIModel:
        if not self._loaded:
            self.load()
        return self._model

    @property
    def mcp_server(self) -> MCPServer:
        if not self._loaded:
            self.load()
        return self._mcp_server

    @property
    def indexer(self) -> SemanticIndexer:
        if not self._loaded:
            self.load()
        return self._indexer

    def search(self, query: str) -> List[Any]:
        """Semantic search: natural language query → nearest GraphQL operation."""
        return self.indexer.search(query)

    def get_tool(self, name: str):
        """Get an MCP tool by name."""
        return self.mcp_server.get_tool(name)

    def manifest(self) -> Dict[str, Any]:
        """Get the MCP server manifest for AI agent tool registration."""
        return self.mcp_server.generate_mcp_manifest()


# ─── CLI / Dev helper ────────────────────────────────────────────────────────

def demo_supplier_model() -> OSIModel:
    """Build a minimal supplier-domain OSI model as a demonstration."""
    return OSIModel(
        version="1.0",
        domain="采购管理",
        description="供应商与采购订单领域模型",
        datasets=[
            DataSet(
                name="Supplier",
                description="外部供应商",
                ai_context=AIContext(
                    instructions="供应商仅指已准入的外部供应商，不含内部工厂。"
                    "状态枚举：Active(正常)/Blocked(冻结)/Suspended(暂停)。"
                    "Blocked 供应商在任何采购相关查询中应显示警告，不自动参与推荐。",
                    synonyms=["供应商", "供货商", "vendor", "vendor_code"],
                    examples=[
                        '查一下这个供应商的采购情况 → {supplierId} + PurchaseOrders',
                        '有哪些供应商是冻结状态 → {status: Blocked}',
                    ],
                ),
                fields=[
                    FieldDefinition(name="id", gql_type="ID", description="供应商编码"),
                    FieldDefinition(name="name", gql_type="String", description="供应商名称"),
                    FieldDefinition(
                        name="status",
                        gql_type="SupplierStatus",
                        enum_name="SupplierStatus",
                        description="供应商状态",
                        enum_values=["Active", "Blocked", "Suspended"],
                    ),
                    FieldDefinition(name="creditRating", gql_type="Int", description="信用等级（仅用于采购金额超过50万时的风险评估）"),
                    FieldDefinition(name="blockedReason", gql_type="String", description="冻结原因"),
                ],
            ),
            DataSet(
                name="PurchaseOrder",
                description="采购订单",
                ai_context=AIContext(
                    instructions="采购订单关联到具体供应商，状态包括 Pending/Approved/Closed。",
                    synonyms=["采购订单", "PO", "订单"],
                ),
                fields=[
                    FieldDefinition(name="orderId", gql_type="ID", description="订单号"),
                    FieldDefinition(name="supplierId", gql_type="ID", description="供应商编码"),
                    FieldDefinition(name="orderDate", gql_type="DateTime", description="下单日期"),
                    FieldDefinition(name="status", gql_type="POStatus", description="订单状态", enum_values=["Pending", "Approved", "Closed"]),
                    FieldDefinition(name="totalAmount", gql_type="Float", description="订单总金额"),
                ],
            ),
            DataSet(
                name="Material",
                description="物料",
                fields=[
                    FieldDefinition(name="id", gql_type="ID"),
                    FieldDefinition(name="name", gql_type="String"),
                    FieldDefinition(name="category", gql_type="String"),
                    FieldDefinition(name="unit", gql_type="String"),
                ],
            ),
        ],
        relationships=[
            Relationship(from_dataset="Supplier", to_dataset="PurchaseOrder", relation_type="ONE_TO_MANY", description="供应商 → 采购订单", ai_hint="查询某供应商的所有订单"),
            Relationship(from_dataset="PurchaseOrder", to_dataset="Material", relation_type="ONE_TO_MANY", description="订单 → 物料"),
        ],
        actions=[
            Action(
                name="blockSupplier",
                dataset="Supplier",
                description="冻结供应商",
                parameters=[],
            ),
            Action(
                name="releasePurchaseOrder",
                dataset="PurchaseOrder",
                description="释放采购订单",
                parameters=[],
            ),
        ],
        rules=[
            Rule(
                name="blocked_supplier_restriction",
                dataset="Supplier",
                description="Blocked 供应商不可下单",
                condition="status == 'Blocked'",
                severity="ERROR",
            ),
        ],
    )


def demo():
    """Demo: build OSI model → GraphQL → MCP tools → semantic search."""
    model = demo_supplier_model()
    backend = SemanticBackend.from_model(model)
    backend.load()

    print("=== GraphQL Schema ===")
    print(backend.schema[:2000])
    print("\n=== MCP Tools ===")
    for tool in backend.mcp_server.list_tools()[:3]:
        print(f"  - {tool.name}: {tool.description}")

    print("\n=== Semantic Search ===")
    results = backend.search("查一下供应商的采购情况")
    for r in results:
        print(f"  score={r.score:.3f}  {r.operation.operation_type} {r.operation.dataset}.{r.operation.field_name}")


if __name__ == "__main__":
    demo()
