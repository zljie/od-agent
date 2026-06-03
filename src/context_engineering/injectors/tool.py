"""
Tool Context Injector
====================
Injects available tools and connector capabilities for each pipeline step.
Step-specific filtering ensures the LLM only sees tools relevant to its current phase.
"""

from typing import Any, Dict, List, Optional

from src.context_engineering.models import RiskLevel, ToolDefinition


# ---------------------------------------------------------------------------
# Connector / tool registry access
# ---------------------------------------------------------------------------

_CONNECTOR_DEFINITIONS: List[Dict[str, Any]] = []  # Populated lazily


def _load_connector_definitions() -> List[Dict[str, Any]]:
    """Load connector definitions from the connector registry."""
    global _CONNECTOR_DEFINITIONS
    if _CONNECTOR_DEFINITIONS:
        return _CONNECTOR_DEFINITIONS

    try:
        # Lazy import to avoid circular dependency
        from src.procurement.connector_registry import get_all_connectors

        connectors = get_all_connectors()
        _CONNECTOR_DEFINITIONS = [
            {
                "id": c.get("id", ""),
                "name": c.get("name", ""),
                "description": c.get("description", ""),
                "actions": c.get("actions", []),
                "risk_level": c.get("risk_level", "low"),
                "requires_confirmation": c.get("requires_confirmation", False),
                "use_condition": c.get("use_condition", ""),
                "parameters": c.get("parameters", {}),
            }
            for c in connectors
        ]
    except ImportError:
        _CONNECTOR_DEFINITIONS = []

    return _CONNECTOR_DEFINITIONS


# ---------------------------------------------------------------------------
# ToolContextInjector
# ---------------------------------------------------------------------------


class ToolContextInjector:
    """Injects tool and connector capability context.

    Each step sees a different, scoped view of available tools:
    - Step 1 (intent recognition): lightweight query tools only
    - Step 2 (ontology grounding): ontology query tools
    - Step 3 (task planning): all tools with risk levels and conditions
    - Step 4 (execution): only the tools planned for execution
    - Step 5 (response generation): read-only summary tools
    """

    def __init__(self):
        self._tool_registry: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Public API — one method per pipeline step
    # ------------------------------------------------------------------

    def inject_for_step1(self, intent_result: Any) -> List[ToolDefinition]:
        """Tools available during intent recognition.

        Returns only lightweight query tools (no writes, no high-risk ops).
        """
        all_tools = self._get_tool_registry()
        query_only = [t for t in all_tools if t.get("kind") == "query"]
        return [self._to_tool_def(t) for t in query_only]

    def inject_for_step2(self, intent_result: Any) -> List[ToolDefinition]:
        """Tools available during ontology grounding.

        Returns ontology query tools that help resolve the object.
        """
        all_tools = self._get_tool_registry()
        ontology_tools = [
            t
            for t in all_tools
            if t.get("kind") in ("query", "ontology") or t.get("id", "").startswith("ontology_")
        ]
        return [self._to_tool_def(t) for t in ontology_tools]

    def inject_for_step3(self, ontology_result: Any) -> List[ToolDefinition]:
        """Tools available during task planning.

        Returns all available tools with risk levels and use conditions.
        """
        all_tools = self._get_tool_registry()
        return [self._to_tool_def(t, include_risk=True) for t in all_tools]

    def inject_for_step4(self, plan_result: Any) -> List[ToolDefinition]:
        """Tools available during execution.

        Returns only the tools planned for execution.
        Extracts tool IDs from the plan's planned_actions.
        """
        planned_ids = self._extract_planned_tool_ids(plan_result)
        all_tools = self._get_tool_registry()

        if planned_ids:
            filtered = [t for t in all_tools if t.get("id") in planned_ids]
        else:
            # Fall back to all low-risk tools if no plan yet
            filtered = [t for t in all_tools if t.get("risk_level") == "low"]

        return [self._to_tool_def(t, include_risk=True) for t in filtered]

    def inject_for_step5(self, exec_result: Any) -> List[ToolDefinition]:
        """Tools available during response generation.

        Returns read-only summary tools only.
        """
        all_tools = self._get_tool_registry()
        read_only = [t for t in all_tools if t.get("operation") in ("read", "query", "export")]
        return [self._to_tool_def(t) for t in read_only]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tool_registry(self) -> List[Dict[str, Any]]:
        """Get the tool registry from connector definitions."""
        if self._tool_registry is None:
            self._tool_registry = _load_connector_definitions()
        return self._tool_registry

    def _to_tool_def(
        self, tool: Dict[str, Any], include_risk: bool = False
    ) -> ToolDefinition:
        """Convert a raw connector definition to a ToolDefinition."""
        return ToolDefinition(
            name=tool.get("id", ""),
            type=tool.get("kind", "function"),
            risk_level=RiskLevel(tool.get("risk_level", "low")),
            description=tool.get("description", ""),
            confirmation_required=tool.get("requires_confirmation", False),
            use_condition=tool.get("use_condition", ""),
            parameters=tool.get("parameters", {}),
        )

    def _extract_planned_tool_ids(self, plan_result: Any) -> List[str]:
        """Extract tool/action IDs from a TaskPlanResult."""
        if plan_result is None:
            return []

        try:
            from src.five_step.models import TaskPlanResult

            if isinstance(plan_result, TaskPlanResult):
                return [action.action_id for action in plan_result.planned_actions]
        except ImportError:
            pass

        # Fallback: try duck-typing via attribute access
        ids = []
        for action in getattr(plan_result, "planned_actions", []):
            action_id = getattr(action, "action_id", None)
            if action_id:
                ids.append(action_id)
        return ids

    def _assess_tool_risk(self, tool: Dict[str, Any]) -> str:
        """Assess risk level of a tool based on its characteristics."""
        risk = tool.get("risk_level", "low")
        op = tool.get("operation", "")

        high_risk_ops = {"delete", "publish", "approve", "reject", "cancel"}
        if op in high_risk_ops:
            return "high"
        if tool.get("requires_confirmation"):
            return "medium"
        return risk

    def _get_tool_use_condition(self, tool: Dict[str, Any]) -> str:
        """Get the use condition/constraints for a tool."""
        condition = tool.get("use_condition", "")
        if condition:
            return condition

        op = tool.get("operation", "")
        conditions = {
            "create": "Only use when user explicitly requests to create a new record.",
            "update": "Only use when user explicitly confirms the update.",
            "delete": "Requires explicit user confirmation before executing.",
            "publish": "Requires supervisor approval before publishing.",
            "approve": "Verify user has approval authority before executing.",
        }
        return conditions.get(op, "Use only when necessary to fulfill the user's request.")
