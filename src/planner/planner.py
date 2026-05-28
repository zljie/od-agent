"""Task planner: converts IntentClassification + IntentBinding → TaskPlan.

Supports both single-intent planning (one IntentClassification → one TaskPlan)
and multi-intent planning (List[IntentClassification] → TaskPlan with parallel DAG)
with skill dependency awareness for composite intents.

The planner operates in four steps:
    1. Ambiguity check: if top-2 candidates are too close → CLARIFY
    2. Confidence floor check: conf < binding.confidenceFloor → HITL_CONFIRM
    3. Strategy routing: REJECT / HITL_CONFIRM / FIXED_SKILL / SKILL_WHITELIST / LLM_FREE
    4. For FIXED_SKILL: materializeFixedSkill() → TaskPlan(EXECUTE)
       If the binding has depends_on_skills, builds a dependency-aware DAG.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..intent.intent_binding import IntentBinding, IntentBindingTable, Strategy
from ..intent.intent_classification import IntentClassification
from .task_plan import Decision, TaskNode, TaskPlan

if TYPE_CHECKING:
    from ..skills.manager import SkillRegistry
    from ..temporal import TemporalContext


class RuleBasedPlanner:
    """Planner that produces TaskPlan DAGs without LLM calls."""

    def __init__(self, binding_table: Optional[IntentBindingTable] = None):
        self._bindings = binding_table or IntentBindingTable()

    def set_binding_table(self, bindings: IntentBindingTable) -> None:
        self._bindings = bindings

    def plan(
        self,
        classification: Union[IntentClassification, List[IntentClassification]],
        skill_registry: Optional["SkillRegistry"] = None,
        temporal: Optional["TemporalContext"] = None,
    ) -> TaskPlan:
        """Plan execution for single or multiple intents.

        When given a list (multi-intent), produces a TaskPlan with parallel
        TaskNodes — one per intent — so all skills run concurrently.

        Args:
            classification: Single IntentClassification or list of them
            skill_registry: Optional registry for skill lookup
            temporal: Optional TemporalContext from Phase-0 (for composite intents)

        Returns:
            TaskPlan with decision and (if EXECUTE) DAG of TaskNodes
        """
        if isinstance(classification, list):
            return self._plan_multi(classification, skill_registry, temporal=temporal)
        return self._plan_single(classification, skill_registry, temporal=temporal)

    def _plan_single(
        self,
        classification: IntentClassification,
        skill_registry: Optional["SkillRegistry"],
        temporal: Optional["TemporalContext"] = None,
        skip_ambiguity_check: bool = False,
    ) -> TaskPlan:
        """Plan for a single IntentClassification."""
        if classification.is_unknown():
            return TaskPlan.delegate_llm(classification)

        if not skip_ambiguity_check and classification.is_ambiguous():
            return TaskPlan.clarify(
                classification,
                "我不太确定您的意思，请更具体地说明一下？",
            )

        binding = self._bindings.resolve(classification.intent_type)

        if (
            binding.strategy != Strategy.REJECT
            and classification.confidence < binding.confidence_floor
        ):
            return TaskPlan.hitl_confirm(
                classification,
                f"我对这个请求的把握不太高（置信度 {classification.confidence:.0%}），"
                f"请问您确认要继续吗？",
            )

        if binding.strategy == Strategy.REJECT:
            return TaskPlan.reject(classification, binding.reject_message)

        if binding.strategy == Strategy.HITL_CONFIRM:
            return TaskPlan.hitl_confirm(classification, binding.hitl_prompt)

        if binding.strategy == Strategy.FIXED_SKILL:
            return self._materialize_fixed_skill(
                classification, binding, skill_registry, temporal=temporal
            )

        return TaskPlan.delegate_llm(classification)

    def _plan_multi(
        self,
        classifications: List[IntentClassification],
        skill_registry: Optional["SkillRegistry"],
        temporal: Optional["TemporalContext"] = None,
    ) -> TaskPlan:
        """Plan for multiple intents (parallel DAG with optional dependency edges)."""
        if not classifications:
            return TaskPlan.delegate_llm(
                classifications[0] if classifications else IntentClassification.UNKNOWN
            )

        all_tasks: List[TaskNode] = []
        all_skills: List[str] = []
        warnings: List[str] = []

        for i, classification in enumerate(classifications):
            # In multi-intent mode, skip ambiguity check: if multiple distinct
            # intent types are detected, we handle them in parallel regardless of
            # whether their confidence scores are close. Ambiguity detection only
            # makes sense for single-intent queries where we must choose one.
            skip_ambig = len(classifications) > 1
            single_plan = self._plan_single(
                classification, skill_registry, temporal=temporal,
                skip_ambiguity_check=skip_ambig
            )
            if single_plan.decision == Decision.EXECUTE:
                for task in single_plan.tasks:
                    task.node_id = f"step_{i + 1}_{task.skill_id}"
                    all_tasks.append(task)
                    if task.skill_id not in all_skills:
                        all_skills.append(task.skill_id)
            elif single_plan.decision == Decision.SLOT_MISSING:
                warnings.extend(single_plan.warnings)

        if not all_tasks:
            return TaskPlan.delegate_llm(classifications[0])

        return TaskPlan(
            intent=classifications[0],
            bound_intent_type="multi",
            decision=Decision.EXECUTE,
            tasks=all_tasks,
            allowed_tools=all_skills,
            warnings=warnings,
        )

    def _materialize_fixed_skill(
        self,
        classification: IntentClassification,
        binding: IntentBinding,
        skill_registry: Optional["SkillRegistry"],
        temporal: Optional["TemporalContext"] = None,
    ) -> TaskPlan:
        """Expand FIXED_SKILL binding into a TaskPlan DAG.

        If the binding has depends_on_skills, creates bootstrap task nodes
        for each dependency and wires the main node's depends_on to them.
        The TaskExecutor resolves {SkillName.field} placeholders using
        the metadata returned by completed dependency tasks.
        """
        if not binding.skill_id:
            return TaskPlan.reject(classification, "FIXED_SKILL 缺少 skill_id")

        missing = binding.missing_slots(classification.entities)
        if missing:
            return TaskPlan.slot_missing(classification, missing)

        entities = dict(classification.entities)
        intent_op_map = {
            "date_range_diff": "range_diff",
            "day_of_week": "day_of_week",
            "timezone": "timezone",
            "math": "math",
            "journey_avg": "journey_avg",
        }
        if "operation" not in entities:
            entities.setdefault(
                "operation", intent_op_map.get(classification.intent_type, "range_diff")
            )

        # ── Build dependency chain ───────────────────────────────────────────
        tasks: List[TaskNode] = []
        node_counter = [0]  # Mutable counter for stable node IDs

        def next_id() -> str:
            node_counter[0] += 1
            return f"step_{node_counter[0]}"

        # For journey_avg, inject resolved dates from temporal context into entities
        if temporal and classification.intent_type == "journey_avg":
            entities = self._inject_temporal_dates(entities, temporal)

        if binding.depends_on_skills:
            # Bootstrap tasks: one per dependency, producing {skill_id: metadata}
            dep_ids: List[str] = []
            for dep_skill in binding.depends_on_skills:
                dep_id = next_id()
                dep_entities = self._build_dep_params(dep_skill, classification, temporal)
                tasks.append(
                    TaskNode(
                        node_id=dep_id,
                        skill_id=dep_skill,
                        params=dep_entities,
                        depends_on=[],
                        rationale=f"执行依赖技能 {dep_skill} 以获取中间结果",
                    )
                )
                dep_ids.append(dep_id)

            # Main task: depends on all bootstrap tasks
            # Replace unresolved placeholders with {SkillName.field} style references
            resolved_params = self._inject_dep_placeholders(
                entities, binding.depends_on_skills
            )
            tasks.append(
                TaskNode(
                    node_id=next_id(),
                    skill_id=binding.skill_id,
                    params=resolved_params,
                    depends_on=dep_ids,
                    rationale=f"执行 {binding.skill_id}（依赖 {binding.depends_on_skills}）",
                )
            )
        else:
            tasks.append(
                TaskNode(
                    node_id=next_id(),
                    skill_id=binding.skill_id,
                    params=entities,
                    depends_on=[],
                    rationale=f"执行 {binding.skill_id}，参数: {entities}",
                )
            )

        allowed = [binding.skill_id] + list(binding.depends_on_skills or [])

        return TaskPlan(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.EXECUTE,
            tasks=tasks,
            allowed_tools=allowed,
        )

    def _build_dep_params(
        self,
        dep_skill: str,
        classification: IntentClassification,
        temporal: Optional["TemporalContext"],
    ) -> Dict[str, Any]:
        """Build parameters for a dependency bootstrap task.

        Maps skill names to the parameters they need to produce the fields
        required by downstream tasks.
        """
        # Normalize skill name: "Time Converter" → "Time_Converter"
        norm = dep_skill.replace(" ", "_")

        if dep_skill in ("Time Converter", "Time_Converter"):
            # For journey scenarios, Time Converter needs to compute the date range
            entities = classification.entities or {}
            # Infer start/end from temporal context if available
            if temporal:
                start_label, end_label = self._infer_date_labels(temporal)
                return {
                    "operation": "range_diff",
                    "start": start_label,
                    "end": end_label,
                }
            # Fallback: try entities
            return {
                "operation": "range_diff",
                "start": entities.get("start", ""),
                "end": entities.get("end", ""),
            }

        return {}

    def _infer_date_labels(
        self, temporal: "TemporalContext"
    ) -> tuple:
        """Infer the start and end date labels from a TemporalContext.

        Returns (start_label, end_label) as strings suitable for Time Converter.
        For journey scenarios, the end date prefers narrative-relative "昨天"
        (resolved by TemporalParser as the day after the last weekday anchor).

        IMPORTANT: Always return ISO date strings for anchors that are already
        in temporal.dates, so that TimeConverterSkill receives unambiguous dates
        and does not re-resolve "昨天" against the current calendar date.
        """
        dates = temporal.dates
        if not dates:
            return ("", "")

        # Sort by date VALUE
        sorted_by_date = sorted(dates.items(), key=lambda x: x[1])

        # Return the *resolved ISO date* for each anchor, NOT the raw label.
        # This prevents TimeConverterSkill from re-interpreting "昨天" against
        # the current calendar date (which would give a wrong result).
        start_label = sorted_by_date[0][0]
        end_label = (
            dates["昨天"].isoformat()
            if "昨天" in dates
            else sorted_by_date[-1][0]
        )

        # If the start label is also in temporal.dates (i.e. it's a qualified
        # anchor like "上周五"), return its ISO string too.
        if start_label in dates:
            start_label = dates[start_label].isoformat()

        return (start_label, end_label)

    def _inject_temporal_dates(
        self, entities: Dict[str, Any], temporal: "TemporalContext"
    ) -> Dict[str, Any]:
        """Inject resolved date values from temporal context into task parameters.

        For journey_avg, we compute days from the narrative-relative journey
        dates (start → end of the actual trip), not from the raw min/max of
        all anchors. The journey end is "昨天" (resolved), not the latest anchor.
        """
        result = dict(entities)

        if "days" not in result or not str(result.get("days", "")).isdigit():
            # Compute days from the actual journey range, not raw anchor extremes.
            # Prefer "昨天" as end (narrative-relative) over the latest anchor.
            dates = temporal.dates
            if len(dates) >= 2:
                sorted_dates = sorted(dates.items(), key=lambda x: x[1])
                # Journey end: "昨天" if present, else latest anchor
                if "昨天" in dates:
                    end_date = dates["昨天"]
                else:
                    end_date = sorted_dates[-1][1]
                start_date = sorted_dates[0][1]
                days = (end_date - start_date).days
                if days > 0:
                    result["days"] = str(days)

        return result

    def _inject_dep_placeholders(
        self, entities: Dict[str, Any], dep_skills: List[str]
    ) -> Dict[str, Any]:
        """Replace bare numeric fields with {SkillName.field} placeholders.

        For each dep_skill in dep_skills, if a numeric field in entities
        is unresolved (empty string or placeholder), inject a {DepSkill.field}
        reference that TaskExecutor will resolve from completed dep results.

        Known field mappings:
            Time Converter → metadata.days
            Math Teacher → metadata.answer
        """
        result = {}
        for key, value in entities.items():
            if isinstance(value, str) and not value.strip():
                # Empty string: this field should come from a dependency
                # Map the key to the appropriate dependency field
                dep_field = self._dep_field_for(key, dep_skills)
                if dep_field:
                    # Use the first matching dependency
                    dep_skill = dep_skills[0].replace(" ", "_")
                    result[key] = f"{{{dep_skill}.{dep_field}}}"
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _dep_field_for(self, field_name: str, dep_skills: List[str]) -> Optional[str]:
        """Map a logical field name to a dependency skill metadata field."""
        field_map = {
            "days": "days",
            "weekday": "weekday",
            "expression": "answer",
            "avg_km_per_day": "answer",
            "start": "start",
            "end": "end",
        }
        return field_map.get(field_name)

    def _skill_params(
        self, skill_id: str, registry: Optional["SkillRegistry"]
    ) -> Dict[str, str]:
        """Query skill manifest parameters from registry."""
        if registry is None:
            return {}
        skill = registry.get(skill_id)
        if skill is None:
            return {}
        return getattr(skill, "intent_params", {})
