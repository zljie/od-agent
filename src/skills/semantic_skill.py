"""SemanticSkill — integrates the semantic-native backend into the OD Agent pipeline.

Bridges OSI models / GraphQL / MCP with the existing Intent → Plan → Execute pipeline:
1. Loads / rebuilds the semantic backend on demand
2. Handles semantic search intents (NL → GraphQL operation)
3. Registers MCP tools for downstream skills
4. Exposes the GraphQL schema for AI agent introspection
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..skills.base import BaseSkill
from ..semantic.semantic_backend import SemanticBackend, demo_supplier_model
from ..semantic.semantic_indexer import SemanticSearchResult

# Config file path — resolved relative to project root
_SEMANTIC_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "semantic_config.json"


def _load_semantic_config() -> Dict[str, Any]:
    """Load semantic config from file with fallback defaults."""
    defaults = {"yaml_path": "", "graphql_endpoint": "", "use_demo": True}
    if not _SEMANTIC_CONFIG_PATH.exists():
        return defaults
    try:
        with open(_SEMANTIC_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return {**defaults, **cfg}
    except Exception:
        return defaults


class SemanticSkill(BaseSkill):
    """Skill that exposes semantic-native backend capabilities to the AI agent.

    Responsibilities:
    - Maintain a SemanticBackend instance (lazy-loaded)
    - Handle semantic search: NL query → nearest GraphQL operation
    - Expose schema introspection and tool listing
    - Integrate with existing intent pipeline via `depends_on_skills`

    By default reads configuration from config/semantic_config.json:
    - yaml_path: path to the OSI YAML ontology file
    - graphql_endpoint: optional GraphQL backend URL
    - use_demo: if True, uses demo_supplier_model instead of the YAML file
    """

    name: str = "Semantic Query"
    description: str = "语义查询：通过自然语言在业务本体中查找数据，执行 GraphQL 操作"
    mcp_type: str = "rag"
    keywords: List[str] = [
        "查一下", "查询", "看看", "获取", "获取数据",
        "供应商", "采购", "订单", "物料", "客户",
        "情况", "状态", "金额", "数量",
        "最近", "三个月", "统计", "汇总",
    ]
    priority: int = 5

    def __init__(
        self,
        yaml_path: Optional[str] = None,
        graphql_endpoint: Optional[str] = None,
        use_demo_model: Optional[bool] = None,
    ):
        # Merge constructor args with config file values.
        # Explicit constructor args always take precedence.
        cfg = _load_semantic_config()
        self._yaml_path = yaml_path or os.environ.get("OSI_MODEL_PATH") or cfg.get("yaml_path", "")
        self._graphql_endpoint = graphql_endpoint or cfg.get("graphql_endpoint", "") or None
        self._use_demo_model = use_demo_model if use_demo_model is not None else cfg.get("use_demo", False)
        self._backend: Optional[SemanticBackend] = None
        self._initialized: bool = False

    def _ensure_loaded(self):
        if self._initialized:
            return
        if self._use_demo_model:
            model = demo_supplier_model()
            self._backend = SemanticBackend.from_model(model, self._graphql_endpoint)
        elif self._yaml_path and Path(self._yaml_path).exists():
            self._backend = SemanticBackend.from_yaml(self._yaml_path, self._graphql_endpoint)
        else:
            model = demo_supplier_model()
            self._backend = SemanticBackend.from_model(model, self._graphql_endpoint)
        self._backend.load()
        self._initialized = True

    @property
    def backend(self) -> SemanticBackend:
        self._ensure_loaded()
        return self._backend

    def match(self, message: str) -> bool:
        if super().match(message):
            return True
        semantic_triggers = [
            "语义查询", "semantic", "本体", "ontology",
            "供应商", "采购订单", "物料", "订单",
            "blocked", "active",
        ]
        msg_lower = message.lower()
        return any(t.lower() in msg_lower for t in semantic_triggers)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a semantic query.

        input_data expected keys:
            - message: str  — natural language user query
            - context: Optional[dict]  — optional temporal/intent context

        Returns:
            Dict with success, response (human-readable), metadata (tool/graphQL info).
        """
        message = input_data.get("message", "")
        if not message:
            return {"success": False, "response": "消息内容为空", "metadata": {}}

        self._ensure_loaded()

        search_results = self.backend.search(message)

        if not search_results or all(r.score < 0.1 for r in search_results):
            return {
                "success": False,
                "response": f"未找到匹配的语义操作：{message}",
                "metadata": {
                    "query": message,
                    "results_count": 0,
                },
            }

        top = search_results[0]
        tool = self.backend.get_tool(f"query_{top.operation.dataset.lower()}s")

        response_parts = [
            f"语义搜索结果（得分：{top.score:.2f}）",
            f"操作类型：{top.operation.operation_type}",
            f"数据集：{top.operation.dataset}",
            f"字段：{top.operation.field_name}",
        ]
        if top.operation.description:
            response_parts.append(f"说明：{top.operation.description}")
        if top.operation.examples:
            response_parts.append("使用示例：")
            for ex in top.operation.examples[:3]:
                response_parts.append(f"  - {ex}")

        if tool:
            response_parts.append(f"\n建议使用的 MCP 工具：{tool.name}")
            response_parts.append(f"GraphQL 操作：{top.operation.graphql_fragment}")

        return {
            "success": True,
            "response": "\n".join(response_parts),
            "metadata": {
                "query": message,
                "top_result": {
                    "operation_id": top.operation.operation_id,
                    "operation_type": top.operation.operation_type,
                    "dataset": top.operation.dataset,
                    "field_name": top.operation.field_name,
                    "score": top.score,
                    "graphql_fragment": top.operation.graphql_fragment,
                },
                "all_results": [
                    {
                        "dataset": r.operation.dataset,
                        "field_name": r.operation.field_name,
                        "score": r.score,
                        "type": r.operation.operation_type,
                    }
                    for r in search_results
                ],
                "mcp_tool": tool.name if tool else None,
                "mcp_manifest": self.backend.manifest() if tool else {},
            },
        }

    def get_system_prompt(self) -> str:
        self._ensure_loaded()
        manifest = self.backend.manifest()
        tool_count = len(manifest.get("tools", []))
        domain = self.backend.model.domain or "通用业务域"
        return (
            f"【Semantic Query 技能已激活】当前业务域：{domain}，"
            f"已注册 {tool_count} 个 MCP 工具。\n"
            "语义查询技能将尝试从业务本体中找到最匹配的操作。\n"
            "使用说明：用户提供自然语言查询 → 语义索引检索 → GraphQL 操作 → MCP 工具执行。"
        )

    def get_ontology_summary(self, max_datasets: int = 15) -> str:
        """Generate a formatted summary of the knowledge graph for LLM context injection.

        Produces a compact, human-readable description of all datasets, their fields,
        and key semantic hints. This is injected into the LLM context so the agent
        can reason about business semantics without needing a full schema dump.

        Args:
            max_datasets: Maximum number of datasets to include (prevents token overflow).

        Returns:
            A formatted string describing the ontology, suitable for LLM system prompt injection.
        """
        self._ensure_loaded()
        model = self.backend.model

        lines = [
            "## 业务本体知识图谱 (DTP Procurement Ontology)",
            f"版本: {model.version}",
            f"域: {model.domain}",
        ]

        if model.description:
            lines.append(f"说明: {model.description}")

        # Dataset summaries
        datasets_to_show = model.datasets[:max_datasets]
        lines.append(f"\n### 数据集 (共 {len(model.datasets)} 个，已显示 {len(datasets_to_show)} 个)")

        for ds in datasets_to_show:
            ai_ctx = ds.ai_context
            synonyms = ", ".join(ai_ctx.synonyms) if ai_ctx and ai_ctx.synonyms else ""
            examples = "\n  ".join(f"- {e}" for e in (ai_ctx.examples or [])[:2]) if ai_ctx and ai_ctx.examples else ""

            field_names = [f.name for f in ds.fields[:8]]
            if len(ds.fields) > 8:
                field_names.append(f"... (共 {len(ds.fields)} 个字段)")
            fields_str = ", ".join(field_names)

            lines.append(f"\n#### {ds.name}")
            lines.append(f"  说明: {ds.description or '（无描述）'}")
            if synonyms:
                lines.append(f"  同义词: {synonyms}")
            lines.append(f"  字段: {fields_str}")
            if examples:
                lines.append(f"  查询示例:\n  {examples}")

        if len(model.datasets) > max_datasets:
            lines.append(f"\n... (还有 {len(model.datasets) - max_datasets} 个数据集未显示)")

        # Key relationships
        if model.relationships:
            lines.append(f"\n### 关键关系 (共 {len(model.relationships)} 个)")
            for rel in model.relationships[:10]:
                lines.append(f"  - {rel.from_dataset} → {rel.to_dataset}: {rel.description or rel.relation_type}")

        # Key rules
        if model.rules:
            lines.append(f"\n### 业务规则 (共 {len(model.rules)} 个)")
            for rule in model.rules[:5]:
                lines.append(f"  - [{rule.severity}] {rule.name}: {rule.description or rule.condition}")

        return "\n".join(lines)

    def get_ontology_context_for_intent(self, user_message: str) -> Dict[str, Any]:
        """Get ontology context relevant to the current user intent.

        Performs a semantic search over the knowledge graph and returns
        relevant dataset/field information to help the LLM understand
        what the user is asking about.

        Args:
            user_message: The user's query string.

        Returns:
            Dict with relevant ontology elements keyed by relevance.
        """
        self._ensure_loaded()
        search_results = self.backend.search(user_message, top_k=5)

        if not search_results:
            return {"relevant_datasets": [], "top_operations": [], "search_query": user_message}

        # Group results by dataset
        datasets_seen = set()
        relevant_datasets = []
        for r in search_results:
            ds_name = r.operation.dataset
            if ds_name not in datasets_seen:
                datasets_seen.add(ds_name)
                ds = None
                for d in self.backend.model.datasets:
                    if d.name == ds_name:
                        ds = d
                        break
                if ds:
                    relevant_datasets.append({
                        "name": ds.name,
                        "description": ds.description,
                        "synonyms": ds.ai_context.synonyms if ds.ai_context else [],
                        "examples": ds.ai_context.examples if ds.ai_context else [],
                        "top_score": r.score,
                    })

        return {
            "relevant_datasets": relevant_datasets,
            "top_operations": [
                {
                    "dataset": r.operation.dataset,
                    "field": r.operation.field_name,
                    "description": r.operation.description,
                    "score": r.score,
                    "type": r.operation.operation_type,
                }
                for r in search_results
            ],
            "search_query": user_message,
        }

    def get_mcp_manifest(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return self.backend.manifest()

    def get_schema(self) -> str:
        self._ensure_loaded()
        return self.backend.schema

    def get_all_tools(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return [t.to_mcp_tool() for t in self.backend.mcp_server.list_tools()]
