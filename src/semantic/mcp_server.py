"""MCP server integration for GraphQL → AI tool exposure.

Wraps a GraphQLGenerator-produced schema and exposes GraphQL operations
as MCP tools that AI agents can call.

Supports three modes (per Apollo MCP Server patterns):
- STATIC: Pre-approved operations in .graphql files
- PERSISTED: Approved query registry (Apollo GraphOS / Cosmo)
- DYNAMIC: Full introspection-based exploration (dev/prototype)

The SemanticIndexer provides RAG over ai_context for NL2GraphQL enhancement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MCPMode(str, Enum):
    """MCP tool exposure mode."""

    STATIC = "static"
    PERSISTED = "persisted"
    DYNAMIC = "dynamic"


@dataclass
class MCPTool:
    """A single MCP tool, backed by a GraphQL operation.

    Exposed to AI agents via the MCP protocol.
    ai_context from the corresponding OSI dataset enriches tool descriptions.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    graphql_operation: str
    dataset: str = ""
    ai_context_examples: List[str] = field(default_factory=list)
    requires_auth: bool = False

    def to_mcp_tool(self) -> Dict[str, Any]:
        """Render this tool in MCP tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def to_llm_description(self) -> str:
        """Render a human-readable tool description for LLM context."""
        parts = [self.description]
        if self.ai_context_examples:
            parts.append("\n使用示例：")
            for ex in self.ai_context_examples:
                parts.append(f"  - {ex}")
        return "\n".join(parts)


class MCPServer:
    """MCP Server that exposes GraphQL operations as AI-callable tools.

    Wraps GraphQLGenerator output and provides:
    1. Tool discovery via schema introspection
    2. Tool invocation with GraphQL execution
    3. Semantic index for NL2GraphQL enhancement
    """

    def __init__(
        self,
        tools: Optional[List[MCPTool]] = None,
        mode: MCPMode = MCPMode.DYNAMIC,
        endpoint: Optional[str] = None,
    ):
        self._tools: Dict[str, MCPTool] = {}
        self._mode = mode
        self._endpoint = endpoint
        self._graphql_client: Any = None
        if tools:
            for t in tools:
                self._tools[t.name] = t
        self._init_graphql_client()

    def _init_graphql_client(self):
        if self._endpoint:
            try:
                from gql import Client, Transport
                self._graphql_client = Client(transport=Transport(url=self._endpoint))
            except ImportError:
                self._graphql_client = None

    def register_tool(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def list_tools(self) -> List[MCPTool]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a GraphQL operation via an MCP tool.

        Args:
            tool_name: Name of the registered MCP tool
            arguments: Tool arguments (mapped to GraphQL variables)

        Returns:
            GraphQL response as a dict, or error dict.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        if self._graphql_client is None:
            return {
                "error": "No GraphQL endpoint configured",
                "note": "Set endpoint in MCPServer constructor to enable execution",
            }

        try:
            from gql import gql
            operation = gql(tool.graphql_operation)
            result = self._graphql_client.execute(operation, variable_values=arguments)
            return {"data": result}
        except Exception as e:
            return {"error": str(e)}

    def generate_mcp_manifest(self) -> Dict[str, Any]:
        """Generate the MCP server manifest (tool registry).

        Returned dict conforms to the MCP tool discovery protocol,
        suitable for AI agents to browse available capabilities.
        """
        return {
            "name": "od-agent-semantic-backend",
            "version": "1.0.0",
            "mode": self._mode.value,
            "endpoint": self._endpoint,
            "tools": [t.to_mcp_tool() for t in self._tools.values()],
        }


def build_mcp_tools_from_osi(model: "OSIModel") -> List[MCPTool]:
    """Build MCP tools from an OSI model.

    For each dataset, creates a query tool.
    For each action, creates a mutation tool.
    ai_context.examples become tool usage examples.
    """
    from .osi_model import OSIModel

    tools: List[MCPTool] = []
    ds_map = model.dataset_map()

    for ds in model.datasets:
        query_tool = MCPTool(
            name=f"query_{ds.name.lower()}s",
            description=ds.description or f"查询 {ds.name} 数据",
            input_schema=_build_query_input_schema(ds),
            graphql_operation=_build_query_operation(ds),
            dataset=ds.name,
            ai_context_examples=ds.ai_context.examples if ds.ai_context else [],
        )
        tools.append(query_tool)

        get_by_id_tool = MCPTool(
            name=f"get_{ds.name.lower()}",
            description=f"根据 ID 获取单个 {ds.name} 详情",
            input_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            graphql_operation=f"query {{ {ds.name.lower()}(id: $id) {{ ...{ds.name}Fields }} }}",
            dataset=ds.name,
        )
        tools.append(get_by_id_tool)

    for action in model.actions:
        ds = ds_map.get(action.dataset)
        mutation_tool = MCPTool(
            name=f"mutate_{action.name.lower()}",
            description=action.description or f"执行 {action.name} 操作",
            input_schema=_build_action_input_schema(action),
            graphql_operation=_build_mutation_operation(action),
            dataset=action.dataset,
            ai_context_examples=action.ai_context.examples if action.ai_context else [],
        )
        tools.append(mutation_tool)

    return tools


def _build_query_input_schema(ds: "DataSet") -> Dict[str, Any]:
    props = {}
    required = []
    for f in ds.fields:
        t = _gql_to_json_schema_type(f.gql_type)
        props[f.name] = {"type": t, "description": f.description or f.ai_hint or ""}
        if f.required:
            required.append(f.name)
    return {"type": "object", "properties": props, "required": required}


def _build_action_input_schema(action: "Action") -> Dict[str, Any]:
    props = {}
    required = []
    for p in action.parameters:
        t = _gql_to_json_schema_type(p.gql_type)
        props[p.name] = {"type": t, "description": p.description}
        if p.required:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def _build_query_operation(ds: "DataSet") -> str:
    fields = " ".join(f.name for f in ds.fields if not f.relation_target)
    return f"query {{ {ds.name.lower()}s {{ {fields} }} }}"


def _build_mutation_operation(action: "Action") -> str:
    params = ", ".join(f"{p.name}: ${p.name}" for p in action.parameters)
    fields = "success message data { id }"
    return f"mutation {action.name}({params}) {{ {action.name}(input: {{ {params} }}) {{ {fields} }} }}"


def _gql_to_json_schema_type(gql_type: str) -> str:
    mapping = {
        "ID": "string",
        "String": "string",
        "Int": "integer",
        "Float": "number",
        "Boolean": "boolean",
        "DateTime": "string",
        "JSON": "object",
    }
    return mapping.get(gql_type, "string")
