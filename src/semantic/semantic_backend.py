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
    DataSet,
    FieldDefinition,
    OSIModel,
    Relationship,
    Rule,
)
from .semantic_indexer import SemanticIndexer


def load_osi_model(path: str | Path) -> OSIModel:
    """Load an OSI semantic model from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    datasets = []
    for ds_raw in data.get("datasets", []):
        ai_ctx = None
        if "ai_context" in ds_raw:
            ai_ctx = AIContext(**ds_raw.pop("ai_context"))

        fields = []
        for f_raw in ds_raw.pop("fields", []):
            fields.append(FieldDefinition(**f_raw))

        datasets.append(DataSet(ai_context=ai_ctx, fields=fields, **ds_raw))

    relationships = []
    for r in data.get("relationships", []):
        relationships.append(Relationship(**r))

    metrics = []
    for m in data.get("metrics", []):
        ai_ctx = None
        if "ai_context" in m:
            ai_ctx = AIContext(**m.pop("ai_context"))
        metrics.append(type(m).__class__.__name__)  # placeholder; use dict approach below

    behavior = data.get("behavior", {})
    actions = [ActionSpec(**a) for a in behavior.get("actions", [])]
    rules = [RuleSpec(**r) for r in behavior.get("rules", [])]

    return OSIModel(
        version=data.get("version", "1.0"),
        domain=data.get("domain", ""),
        description=data.get("description", ""),
        datasets=datasets,
        relationships=relationships,
        metrics=metrics,
        actions=actions,
        rules=rules,
    )


def load_osi_model_from_dict(data: Dict[str, Any]) -> OSIModel:
    """Build an OSIModel from a parsed YAML dict (no file I/O)."""
    datasets = []
    for ds_raw in data.get("datasets", []):
        ai_ctx = None
        if "ai_context" in ds_raw:
            ai_ctx = AIContext(**ds_raw.pop("ai_context"))
        fields = [FieldDefinition(**f) for f in ds_raw.pop("fields", [])]
        datasets.append(DataSet(ai_context=ai_ctx, fields=fields, **ds_raw))

    relationships = [Relationship(**r) for r in data.get("relationships", [])]

    from .osi_model import Action, Metric, Rule

    metrics = []
    for m in data.get("metrics", []):
        ai_ctx = None
        if "ai_context" in m:
            ai_ctx = AIContext(**m.pop("ai_context"))
        metrics.append(Metric(ai_context=ai_ctx, **m))

    actions = []
    for a in data.get("behavior", {}).get("actions", []):
        ai_ctx = None
        if "ai_context" in a:
            ai_ctx = AIContext(**a.pop("ai_context"))
        actions.append(Action(ai_context=ai_ctx, **a))

    rules = []
    for r in data.get("behavior", {}).get("rules", []):
        ai_ctx = None
        if "ai_context" in r:
            ai_ctx = AIContext(**r.pop("ai_context"))
        rules.append(Rule(ai_context=ai_ctx, **r))

    return OSIModel(
        version=data.get("version", "1.0"),
        domain=data.get("domain", ""),
        description=data.get("description", ""),
        datasets=datasets,
        relationships=relationships,
        metrics=metrics,
        actions=actions,
        rules=rules,
    )


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
            with open(self._yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._model = load_osi_model_from_dict(data)

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
