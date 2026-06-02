"""Task executor: runs TaskPlan DAGs by invoking skills in topological order."""

import asyncio
import re as _re
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from .skill_registry import SkillRegistry
from ..planner.task_plan import TaskNode, TaskPlan

if TYPE_CHECKING:
    from ..policy import PolicyEngine
    from ..semantic.task_validator import TaskValidator


# Write-action prefixes for automatic HITL escalation
_WRITE_ACTION_PREFIXES = frozenset([
    "create", "update", "delete", "block", "release",
    "approve", "reject", "add", "remove", "modify",
])


class TaskExecutor:
    """Executes a TaskPlan by running skills in topological order.

    Supports:
    - Parallel execution of independent tasks
    - Cross-skill result injection via {SkillName.field} placeholders
      (e.g. node.params["days"] = "{Time_Converter.days}" → resolved to computed value)
    - Graceful fallback to LLM when skill returns "需要更多信息"
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        task_validator: Optional["TaskValidator"] = None,
        policy_engine: Optional["PolicyEngine"] = None,
    ):
        self._registry = registry or SkillRegistry()
        self._validator = task_validator
        self._policy = policy_engine

    def set_registry(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def set_validator(self, validator: "TaskValidator") -> None:
        self._validator = validator

    def set_policy_engine(self, engine: "PolicyEngine") -> None:
        self._policy = engine

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

            # ── Ontology preflight: validate action against OSIModel ──────────────────
            # Extract the effective action_id from the semantic context if available
            action_id = self._resolve_action_id(node, skill, resolved_params)
            if self._validator is not None and action_id:
                verdict = self._validator.preflight(action_id, resolved_params)
                if not verdict.ok:
                    return self._node_result(
                        node, None, False,
                        f"[Ontology Preflight Failed] {verdict.reason}"
                    )

            # ── Policy check: RBAC + HITL gating ──────────────────────────────────
            if self._policy is not None and action_id:
                is_write = any(
                    action_id.lower().startswith(p) for p in _WRITE_ACTION_PREFIXES
                )
                if self._policy.requires_hitl(action_id, is_write_action=is_write):
                    return self._node_result(
                        node, None, False,
                        f"[HITL Required] {self._policy.get_hitl_prompt(action_id, resolved_params)}"
                    )

            try:
                result = await skill.execute({"params": resolved_params, "message": node.user_message or ""})
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

    def _build_plan_json(
        self, nodes: List[TaskNode], results: List[Dict[str, Any]], user_message: str = ""
    ) -> Dict[str, Any]:
        """Build the plan JSON payload for SSE emission and error reporting."""
        task_summaries = []
        for node, result in zip(nodes, results):
            task_summaries.append({
                "node_id": node.node_id,
                "skill_id": node.skill_id,
                "params": node.params,
                "rationale": node.rationale,
                "success": result.get("success", False),
                "error": result.get("error"),
            })

        return {
            "user_message": user_message,
            "task_count": len(nodes),
            "tasks": task_summaries,
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

    def _resolve_action_id(
        self,
        node: TaskNode,
        skill: Any,
        resolved_params: Dict[str, Any],
    ) -> Optional[str]:
        """Infer the effective action_id for a skill node for preflight validation.

        Tries: MCP tool name > semantic_skill backend action > skill_id itself.
        Returns None if no backend is configured (backward compat).
        """
        # Priority 1: MCP tool name from SemanticSkill
        if hasattr(skill, "backend") and skill.backend is not None:
            # Derive action_id from skill's intent_type mapping
            # SemanticQuery skill → ontology action mapping
            if hasattr(skill, "name") and skill.name == "Semantic Query":
                # For semantic queries, extract action from params or use default
                return resolved_params.get("action_id", "semanticQuery")

        # Priority 2: skill name as action_id (convention: skill names map to actions)
        skill_name = node.skill_id.replace(" ", "").replace("_", "")
        return skill_name

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
                raw = r.get("result", {})
                response = raw.get("response", "") if isinstance(raw, dict) else str(raw)
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

    def aggregate_responses_ex(
        self, results: List[Dict[str, Any]], user_message: str = ""
    ) -> Dict[str, Any]:
        """Extended aggregator: returns a dict with text, plan_json, and all_failed flag.

        Used by streaming to build the final content message that includes a
        "任务未完成" report when all skills failed with preflight/tool errors.
        """
        parts: List[str] = []
        all_failed = True
        failed_tasks: List[Dict[str, Any]] = []

        for r in results:
            if r.get("success"):
                all_failed = False
                raw = r.get("result", {})
                response = raw.get("response", "") if isinstance(raw, dict) else str(raw)
                if response:
                    parts.append(response)
            else:
                err = r.get("error", "未知错误")
                parts.append(f"[{r['skill_id']}] {err}")
                failed_tasks.append({
                    "skill_id": r.get("skill_id", ""),
                    "error": err,
                })

        if not parts:
            return {
                "text": "无法完成请求。",
                "plan_json": {"user_message": user_message, "tasks": []},
                "all_failed": True,
            }

        plan_json = self._build_plan_json_from_results(results, user_message)

        if all_failed:
            failure_report = self._build_failure_report(failed_tasks, user_message)
            return {
                "text": failure_report,
                "plan_json": plan_json,
                "all_failed": True,
            }

        return {
            "text": "\n".join(parts),
            "plan_json": plan_json,
            "all_failed": False,
        }

    def _build_plan_json_from_results(
        self, results: List[Dict[str, Any]], user_message: str = ""
    ) -> Dict[str, Any]:
        """Build plan JSON from results list (for aggregate_responses_ex)."""
        tasks = []
        for r in results:
            tasks.append({
                "skill_id": r.get("skill_id", ""),
                "success": r.get("success", False),
                "error": r.get("error"),
            })
        return {"user_message": user_message, "tasks": tasks}

    def _build_failure_report(
        self, failed_tasks: List[Dict[str, Any]], user_message: str = ""
    ) -> str:
        """Build a human-readable failure report when all tasks failed."""
        lines = []
        lines.append(f"未接收到相关信息，任务未完成。")

        if failed_tasks:
            tool_errors = [
                f"  - {t['skill_id']}: {t.get('error', '未知错误')}"
                for t in failed_tasks
                if t.get("error")
            ]
            if tool_errors:
                lines.append("")
                lines.append("技能调用失败详情：")
                lines.extend(tool_errors)

        return "\n".join(lines)

    async def execute_stream(
        self, plan: TaskPlan
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of execute: yields SSE-ready event dicts as each skill runs.

        Each yielded dict follows the SSE spec in docs/SSE流式响应规范.md.
        After all tool_call/tool_result pairs are yielded, final task results are
        aggregated and a single content event is yielded with the combined response.
        """
        ordered = plan.topological_order()
        results: Dict[str, Dict[str, Any]] = {}
        results_list: List[Dict[str, Any]] = []   # deduped for aggregate_responses
        pending: Dict[str, asyncio.Task] = {}
        tool_id_map: Dict[str, str] = {}   # skill_id → assigned tool_id

        async def run_node(node: TaskNode) -> Dict[str, Any]:
            for dep_id in node.depends_on:
                if dep_id in pending:
                    await pending[dep_id]

            skill = self._registry.get(node.skill_id)
            if not skill:
                return self._node_result(node, None, False, f"Skill '{node.skill_id}' not found")

            resolved_params = self._resolve_params(node.params, results)

            # ── Ontology preflight: validate action against OSIModel ──────────────────
            action_id = self._resolve_action_id(node, skill, resolved_params)
            if self._validator is not None and action_id:
                verdict = self._validator.preflight(action_id, resolved_params)
                if not verdict.ok:
                    return self._node_result(
                        node, None, False,
                        f"[Ontology Preflight Failed] {verdict.reason}"
                    )

            # ── Policy check: RBAC + HITL gating ──────────────────────────────────
            if self._policy is not None and action_id:
                is_write = any(
                    action_id.lower().startswith(p) for p in _WRITE_ACTION_PREFIXES
                )
                if self._policy.requires_hitl(action_id, is_write_action=is_write):
                    return self._node_result(
                        node, None, False,
                        f"[HITL Required] {self._policy.get_hitl_prompt(action_id, resolved_params)}"
                    )

            # Assign a spec-compliant tool_id
            tool_id = f"call_{len(results):03d}"
            tool_id_map[node.skill_id] = tool_id

            try:
                result = await skill.execute({"params": resolved_params, "message": node.user_message or ""})
                success = result.get("success", False)
                return self._node_result(node, result, success, None)
            except Exception as e:
                return self._node_result(node, None, False, str(e))

        for node in ordered:
            pending[node.node_id] = asyncio.create_task(run_node(node))

        for node in ordered:
            result = await pending[node.node_id]
            tool_id = tool_id_map.get(node.skill_id, f"call_{len(results):03d}")

            from ..sse_stream import tool_call, tool_result
            skill = self._registry.get(node.skill_id)
            skill_type = getattr(skill, "mcp_type", "skill") if skill else "skill"
            skill_name = node.skill_id
            resolved_params = self._resolve_params(node.params, results)

            yield tool_call(
                name=skill_name,
                input_data=resolved_params,
                type=skill_type,
                tool_id=tool_id,
                description=getattr(skill, "description", None) if skill else None,
            )

            if result.get("success"):
                yield tool_result(
                    tool_id=tool_id,
                    name=skill_name,
                    status="success",
                    output=result.get("result"),
                    error=None,
                )
            else:
                yield tool_result(
                    tool_id=tool_id,
                    name=skill_name,
                    status="error",
                    output=None,
                    error=result.get("error") or "Unknown error",
                )

            results[node.node_id] = result
            # Keep skill-name index for _resolve_params in subsequent nodes,
            # but track results_list separately so aggregate_responses sees no duplicates.
            results_list.append(result)
            normalized = node.skill_id.replace(" ", "_")
            results["skill_" + normalized] = result

        # Aggregate responses and emit the final content event
        response = self.aggregate_responses(results_list)
        from ..sse_stream import content
        if response:
            yield content(response)
