"""Task executor: runs TaskPlan DAGs by invoking skills in topological order."""

import asyncio
import re as _re
from typing import Any, Dict, List, Optional

from .skill_registry import SkillRegistry
from ..planner.task_plan import TaskNode, TaskPlan


class TaskExecutor:
    """Executes a TaskPlan by running skills in topological order.

    Supports:
    - Parallel execution of independent tasks
    - Cross-skill result injection via {SkillName.field} placeholders
      (e.g. node.params["days"] = "{Time_Converter.days}" → resolved to computed value)
    - Graceful fallback to LLM when skill returns "需要更多信息"
    """

    def __init__(self, registry: Optional[SkillRegistry] = None):
        self._registry = registry or SkillRegistry()

    def set_registry(self, registry: SkillRegistry) -> None:
        self._registry = registry

    async def execute(self, plan: TaskPlan) -> List[Dict[str, Any]]:
        """Execute a TaskPlan and return a list of task results."""
        ordered = plan.topological_order()
        results: Dict[str, Dict[str, Any]] = {}
        pending: Dict[str, asyncio.Task] = {}

        async def run_node(node: TaskNode) -> Dict[str, Any]:
            # Wait for dependencies
            for dep_id in node.depends_on:
                if dep_id in pending:
                    await pending[dep_id]

            skill = self._registry.get(node.skill_id)
            if not skill:
                return self._node_result(node, None, False, f"Skill '{node.skill_id}' not found")

            # Resolve cross-skill placeholders (e.g. "{Time_Converter.days}")
            resolved_params = self._resolve_params(node.params, results)

            try:
                result = await skill.execute({"params": resolved_params, "message": ""})
                success = result.get("success", False)
                return self._node_result(node, result, success, None)
            except Exception as e:
                return self._node_result(node, None, False, str(e))

        # Launch all nodes
        for node in ordered:
            pending[node.node_id] = asyncio.create_task(run_node(node))

        # Gather results in order and build skill-name index (normalize spaces)
        results_list: List[Dict[str, Any]] = []
        for node in ordered:
            result = await pending[node.node_id]
            results_list.append(result)
            results[node.node_id] = result
            # Index by skill name, normalized (spaces → underscores)
            normalized = node.skill_id.replace(" ", "_")
            results["skill_" + normalized] = result  # also index by skill name

        return results_list

    def _node_result(
        self, node: TaskNode, result: Optional[Dict], success: bool, error: Optional[str]
    ) -> Dict[str, Any]:
        """Build a standardized task result dict."""
        metadata = {}
        if result:
            metadata = result.get("metadata", {})
        return {
            "node_id": node.node_id,
            "skill_id": node.skill_id,
            "success": success,
            "result": result or {},
            "error": error,
            "metadata": metadata,
        }

    def _resolve_params(
        self, params: Dict[str, Any], completed: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve {SkillName.field} placeholders from previously computed results.

        Example:
            params = {"expression": "2080 / {Time_Converter.days}"}
            completed = {"skill_Time_Converter": {"metadata": {"days": 6}}}
            → {"expression": "2080 / 6"}
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "{" in value:
                def replacer(m: Any) -> str:
                    ref = m.group(1)
                    parts = ref.split(".", 1)
                    normalized = parts[0].replace(" ", "_")
                    field = parts[1] if len(parts) > 1 else None
                    entry = completed.get("skill_" + normalized) or completed.get(parts[0])
                    if entry and entry.get("success"):
                        meta = entry.get("metadata", {})
                        if field and field in meta:
                            return str(meta[field])
                        if not field:
                            return str(meta)
                    return m.group(0)
                resolved[key] = _re.sub(r"\{([^}]+)\}", replacer, value)
            else:
                resolved[key] = value
        return resolved

    def aggregate_responses(self, results: List[Dict[str, Any]]) -> str:
        """Combine task results into a single user-facing response.

        Rules:
        - Successful results are always included in the output
        - Only delegate to LLM if EVERY task failed (no successful results)
        """
        parts: List[str] = []
        all_failed = True

        for r in results:
            if r.get("success"):
                all_failed = False
                response = r.get("result", {}).get("response", "")
                if response:
                    parts.append(response)
            else:
                err = r.get("error", "未知错误")
                parts.append(f"[{r['skill_id']}] {err}")

        if not parts:
            return "无法完成请求。"

        if all_failed:
            return f"__DELEGATE_LLM__\n" + "\n".join(parts)

        return "\n".join(parts)
