"""Skill Manager - orchestrates skill detection, planning, and execution.

Provides a unified interface that:
1. Manages SkillRegistry (skill registration/lookup)
2. Runs TemporalParser (Phase 0: date resolution + journey detection)
3. Runs IntentClassifier (Phase 1: intent classification with temporal context)
4. Runs RuleBasedPlanner (Phase 2: dependency-aware DAG planning)
5. Runs TaskExecutor (Phase 3: topological execution)

All components can be used independently or together.
"""

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseSkill
from .skill_registry import SkillRegistry
from .task_executor import TaskExecutor
from .math_teacher import MathTeacherSkill
from .time_converter import TimeConverterSkill
from .semantic_skill import SemanticSkill

if TYPE_CHECKING:
    from ..agent import CustomerServiceAgent


class SkillManager:
    """Manages skill registration, intent detection, planning, and execution.

    Pipeline: user_input → IntentClassifier → RuleBasedPlanner → TaskExecutor → response

    For backward compatibility, detect_and_execute() also supports the old
    keyword-matching mode when no IntentBinding is configured.
    """

    def __init__(self):
        self._registry = SkillRegistry()
        self._executor = TaskExecutor(self._registry)
        self._intent_config: List[Dict[str, Any]] = []
        self._binding_table: Any = None
        self._planner: Any = None
        self._classifier: Any = None
        self._temporal_parser: Any = None
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        """Register built-in skills."""
        self.register(MathTeacherSkill())
        self.register(TimeConverterSkill())
        self.register(SemanticSkill(use_demo_model=True))

    def _resolve_today_date(self):
        """Resolve the current local date via TimeConverterSkill.

        This ensures all date calculations throughout the pipeline use the same
        "today" value that TimeConverterSkill uses, rather than TemporalParser
        calling date.today() directly.
        """
        time_skill = self._registry.get("Time Converter")
        if time_skill is None:
            return None
        return getattr(time_skill, "today", None)

    # ─── SkillRegistry delegation ───────────────────────────────────────────────

    def register(self, skill: BaseSkill) -> None:
        """Register a skill."""
        self._registry.register(skill)

    def unregister(self, name: str) -> bool:
        """Unregister a skill by name."""
        return self._registry.unregister(name)

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name."""
        return self._registry.get(name)

    def get_all_skills(self) -> List[BaseSkill]:
        """Get all registered skills."""
        return self._registry.all()

    def get_skills_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all skills for UI display."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "keywords_count": len(getattr(s, "keywords", [])),
                "priority": getattr(s, "priority", 10),
            }
            for s in self._registry.all()
        ]

    # ─── Intent routing ────────────────────────────────────────────────────────

    def load_intent_rules(self, rules: List[Dict[str, Any]]) -> None:
        """Load intent routing rules from config (backward compat)."""
        self._intent_config = rules

    # ─── Full pipeline (Intent → Plan → Execute) ────────────────────────────────

    def setup_pipeline(
        self,
        classifier: Any,
        binding_table: Any,
        planner: Any,
        temporal_parser: Any = None,
    ) -> None:
        """Wire up the full Intent → Plan → Execute pipeline.

        Call this after constructing TemporalParser, IntentClassifier,
        IntentBindingTable, and RuleBasedPlanner.

        Args:
            classifier: RuleBasedIntentClassifier instance
            binding_table: IntentBindingTable instance
            planner: RuleBasedPlanner instance
            temporal_parser: TemporalParser instance (Phase 0)
        """
        self._classifier = classifier
        self._binding_table = binding_table
        self._planner = planner
        self._temporal_parser = temporal_parser

    async def run_pipeline(self, user_input: str) -> Dict[str, Any]:
        """Run the full four-phase pipeline: Temporal → Intent → Plan → Execute.

        Phase 0 (TemporalParser): Resolve all date anchors and detect journey patterns.
        Phase 1 (IntentClassifier): Classify intents with temporal context injected.
        Phase 2 (RuleBasedPlanner): Build dependency-aware DAG.
        Phase 3 (TaskExecutor): Execute DAG in topological order.

        Args:
            user_input: The raw user message

        Returns:
            Dict with keys:
                - decision: TaskPlan.Decision value
                - response: Final response string for the user
                - task_results: List of task execution results
                - classification: Primary IntentClassification result
                - classifications: List of all IntentClassification (multi-intent)
                - plan: TaskPlan result
                - temporal: TemporalContext from Phase 0
        """
        from ..intent.intent_classification import IntentClassification
        from ..planner import RuleBasedPlanner
        from ..temporal import TemporalParser

        # ── Phase 0 bootstrap: resolve "today" via TimeConverterSkill ─────────────
        # This ensures the entire date-resolution chain uses the same skill system,
        # rather than calling date.today() directly in TemporalParser.
        today_date = self._resolve_today_date()

        # Phase 0: Temporal parsing
        temporal = None
        if self._temporal_parser:
            temporal = self._temporal_parser.parse(user_input, today_date=today_date)
        else:
            temporal = TemporalParser(today_date=today_date).parse(user_input)

        # Phase 1: Intent classification
        if self._classifier:
            classifications = self._classifier.classify_with_context(user_input, temporal)
            primary = max(classifications, key=lambda c: c.confidence)
        else:
            primary = self._classify_fallback(user_input)
            classifications = [primary]

        # Phase 2: Task planning
        if self._planner:
            plan = self._planner.plan(classifications, self._registry, temporal=temporal)
        else:
            plan = self._plan_fallback(primary)

        # Inject user_message into every TaskNode so skills that need it (SemanticSkill)
        # receive the original user query for semantic search, not just entity dict keys.
        for task in plan.tasks:
            if not task.user_message:
                task.user_message = user_input

        decision = plan.decision.value

        # 3. Route on decision
        if decision == "reject":
            return {
                "decision": decision,
                "response": plan.rejected_reason or "无法处理该请求。",
                "task_results": [],
                "classification": primary,
                "classifications": classifications,
                "plan": plan,
                "temporal": temporal,
            }

        if decision == "clarify":
            return {
                "decision": decision,
                "response": plan.hitl_prompt,
                "task_results": [],
                "classification": primary,
                "classifications": classifications,
                "plan": plan,
                "temporal": temporal,
            }

        if decision == "hitl_confirm":
            return {
                "decision": decision,
                "response": plan.hitl_prompt,
                "task_results": [],
                "classification": primary,
                "classifications": classifications,
                "plan": plan,
                "temporal": temporal,
            }

        if decision == "slot_missing":
            return {
                "decision": decision,
                "response": "请提供必要的信息：" + ", ".join(plan.warnings),
                "task_results": [],
                "classification": primary,
                "classifications": classifications,
                "plan": plan,
                "temporal": temporal,
            }

        if decision == "delegate_llm":
            return {
                "decision": decision,
                "response": "__DELEGATE_LLM__",
                "task_results": [],
                "classification": primary,
                "classifications": classifications,
                "plan": plan,
                "temporal": temporal,
            }

        # Phase 3: Execute EXECUTE decision
        task_results = await self._executor.execute(plan)
        response = self._executor.aggregate_responses(task_results)

        return {
            "decision": decision,
            "response": response,
            "task_results": task_results,
            "classification": primary,
            "classifications": classifications,
            "plan": plan,
            "temporal": temporal,
        }

    async def run_pipeline_stream(self, user_input: str):
        """Streaming version of run_pipeline: yields SSE-ready event dicts.

        Phase 0-2 are the same as run_pipeline (fast, no I/O).
        Phase 3 uses TaskExecutor.execute_stream() which yields
        tool_call / tool_result events as each skill completes.

        Yields:
            Dicts with keys: event (str), data (str)
            matching docs/SSE流式响应规范.md

        Returns the final decision string as a second value via a special event
        ({"event": "_decision", "data": <decision>}) so the caller can route
        on it without re-running Phase 0-2.
        """
        from ..intent.intent_classification import IntentClassification
        from ..planner import RuleBasedPlanner
        from ..temporal import TemporalParser
        from ..sse_stream import (
            think,
            think_done,
            content,
            done,
        )

        today_date = self._resolve_today_date()

        temporal = None
        if self._temporal_parser:
            temporal = self._temporal_parser.parse(user_input, today_date=today_date)
        else:
            temporal = TemporalParser(today_date=today_date).parse(user_input)

        if self._classifier:
            classifications = self._classifier.classify_with_context(user_input, temporal)
            primary = max(classifications, key=lambda c: c.confidence)
        else:
            primary = self._classify_fallback(user_input)
            classifications = [primary]

        if self._planner:
            plan = self._planner.plan(classifications, self._registry, temporal=temporal)
        else:
            plan = self._plan_fallback(primary)

        for task in plan.tasks:
            if not task.user_message:
                task.user_message = user_input

        decision = plan.decision.value

        # Emit the decision as a private event so chat_stream can route on it
        yield {"event": "_decision", "data": decision}

        # Non-execution decisions: yield content and done
        if decision in ("reject", "clarify", "hitl_confirm", "slot_missing"):
            response = plan.rejected_reason or "无法处理该请求。"
            if decision == "clarify" or decision == "hitl_confirm":
                response = plan.hitl_prompt
            if decision == "slot_missing":
                response = "请提供必要的信息：" + ", ".join(plan.warnings)
            yield content(response)
            yield done()
            return

        if decision == "delegate_llm":
            # Yield a marker; the agent will call _llm_chat_stream directly
            yield {"event": "_delegate_llm", "data": "1"}
            return

        # EXECUTE decision: stream tool events from TaskExecutor
        async for event in self._executor.execute_stream(plan):
            yield event

        yield done()

    # ─── Backward-compatible API ───────────────────────────────────────────────

    async def detect_and_execute(self, message: str, agent: Any = None) -> Optional[Dict[str, Any]]:
        """Detect intent and execute skill (backward-compatible API).

        Priority:
        1. Full pipeline (if configured via setup_pipeline)
        2. Intent rules from config (keyword matching)
        3. Built-in skill matching (fallback)
        """
        # Try full pipeline if configured
        if self._classifier and self._planner:
            result = await self.run_pipeline(message)
            return result

        # Fallback: simple skill dispatch
        return await self._detect_and_execute_simple(message, agent)

    async def _detect_and_execute_simple(
        self, message: str, agent: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Simple skill dispatch (backward compatible)."""
        from ..intent.intent_classification import IntentClassification

        classification = self._classify_fallback(message)

        matched_skill = None
        matched_rule = None
        max_priority = -1

        # Check intent config rules
        message_lower = message.lower()
        for rule in self._intent_config:
            priority = rule.get("priority", 10)
            keywords = rule.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in message_lower and priority > max_priority:
                    max_priority = priority
                    matched_rule = rule
                    matched_skill = self._registry.get(rule.get("handler"))

        # Fallback: built-in skill matching
        if not matched_skill:
            for skill in sorted(self._registry.all(), key=lambda s: getattr(s, "priority", 10), reverse=True):
                if self._skill_matches(skill, message):
                    matched_skill = skill
                    break

        if matched_skill:
            try:
                result = await matched_skill.execute({"message": message})
                return {
                    "skill": matched_skill.name,
                    "executed": True,
                    "result": result,
                    "classification": classification,
                }
            except Exception as e:
                return {
                    "skill": matched_skill.name,
                    "executed": False,
                    "error": str(e),
                }

        return None

    def detect_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect which skill/intent should handle the message without executing."""
        for rule in sorted(self._intent_config, key=lambda r: r.get("priority", 10), reverse=True):
            keywords = rule.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in message.lower():
                    return {
                        "matched": True,
                        "intent": rule.get("name"),
                        "handler": rule.get("handler"),
                        "matched_keyword": keyword,
                    }

        for skill in sorted(self._registry.all(), key=lambda s: getattr(s, "priority", 10), reverse=True):
            if self._skill_matches(skill, message):
                return {
                    "matched": True,
                    "intent": skill.name,
                    "handler": skill.name,
                    "matched_keyword": "skill_keyword",
                }

        return {"matched": False}

    def build_temporal_context(self, message: str) -> Dict[str, Any]:
        """Extract and resolve all relative dates in the user message.

        All dates are resolved relative to the current local date (today), NOT
        relative to any narrative anchor in the text.

        Returns a dict with:
            - dates: {label: iso_date_string} mapping
            - timeline: human-readable ordered list
            - context_text: formatted string for LLM injection
            - has_multiple_anchors: bool
            - journey_result: always None — journey math is handled by the LLM
        """
        import re as _bare_re

        time_skill = self._registry.get("Time Converter")
        if not time_skill:
            return {"dates": {}, "timeline": [], "context_text": "", "has_multiple_anchors": False}

        # All relative date keywords
        qualified_candidates = [
            "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
            "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
            "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
            "今天", "昨天", "前天", "明天", "后天", "大后天", "大前天",
            "上周", "下周", "这周",
            "上个月", "下个月", "这个月",
            "去年", "今年", "明年",
        ]
        bare_weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日",
                         "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday_map = {
            "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
            "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
        }

        anchor_dates: Dict[str, date] = {}

        # Resolve qualified labels (上周五, 昨天, etc.) via TimeConverterSkill
        for cand in qualified_candidates:
            if cand in message:
                resolved = time_skill._resolve_date(cand)
                if resolved:
                    anchor_dates[cand] = resolved

        # ISO date literals
        import re as _re
        for m in _re.finditer(r"\d{4}-\d{2}-\d{2}", message):
            try:
                anchor_dates[m.group(0)] = date.fromisoformat(m.group(0))
            except ValueError:
                pass
        for m in _re.finditer(r"\d{4}年\d{1,2}月\d{1,2}日", message):
            cn = _re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", m.group(0))
            if cn:
                try:
                    anchor_dates[cn.group(0)] = date(int(cn.group(1)), int(cn.group(2)), int(cn.group(3)))
                except ValueError:
                    pass

        # Bare weekdays: resolve relative to TODAY's week (this week), NOT journey week.
        forbidden_from_qualified: set = set()
        qualified_week_labels = [
            "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
            "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
            "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
        ]
        for ql in qualified_week_labels:
            if ql in message:
                for bw in bare_weekdays:
                    if bw in ql:
                        forbidden_from_qualified.add(bw)

        this_monday = time_skill.today - timedelta(days=time_skill.today.weekday())
        for bw in bare_weekdays:
            if bw not in anchor_dates and bw not in forbidden_from_qualified:
                if _bare_re.search(_bare_re.escape(bw), message):
                    dow_idx = weekday_map[bw]
                    anchor_dates[bw] = this_monday + timedelta(days=dow_idx)

        if len(anchor_dates) < 2:
            return {"dates": {}, "timeline": [], "context_text": "", "has_multiple_anchors": False}

        # Build timeline
        resolved_entries = []
        for label, d in anchor_dates.items():
            wd = time_skill._get_chinese_weekday(d)
            resolved_entries.append(f"{label}({d.isoformat()},{wd})")

        dates_iso = {k: v.isoformat() for k, v in anchor_dates.items()}
        timeline_parts = sorted(resolved_entries, key=lambda x: x.split("(")[1].split(",")[0])

        context_parts = [
            f"[时间线分析 — 重要：所有相对日期均以今天({time_skill.today.isoformat()})为基准推算]",
            f"检测到 {len(anchor_dates)} 个日期锚点：{', '.join(sorted(anchor_dates.keys()))}",
            f"时间线：{' | '.join(timeline_parts)}",
            "请根据上述时间线理解'昨天'、'前天'等相对日期在叙事中的具体日期。",
        ]

        return {
            "dates": dates_iso,
            "timeline": timeline_parts,
            "context_text": "\n".join(context_parts),
            "has_multiple_anchors": True,
            "journey_result": None,
        }
    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _skill_matches(self, skill: BaseSkill, message: str) -> bool:
        """Check if a skill matches the message."""
        matcher = getattr(skill, "match", None)
        if matcher:
            return matcher(message)
        message_lower = message.lower()
        for kw in getattr(skill, "keywords", []):
            if kw.lower() in message_lower:
                return True
        return False

    def _classify_fallback(self, text: str):
        """Simple fallback classification when no IntentClassifier is configured."""
        from ..intent.intent_classification import IntentClassification, IntentCandidate

        message_lower = text.lower()
        best_score = 0.0
        best_type = "UNKNOWN"
        best_entities: Dict[str, str] = {}
        candidates: List[Any] = []

        for skill in self._registry.all():
            score = 0.0
            for kw in getattr(skill, "keywords", []):
                if kw.lower() in message_lower:
                    score += 1.0
            if score > 0:
                if score > best_score:
                    best_score = score
                    best_type = skill.name
                candidates.append(IntentCandidate(skill.name, score, {}))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        if best_score > 0:
            return IntentClassification(best_type, min(1.0, best_score), best_entities, candidates[:3])
        return IntentClassification.UNKNOWN

    def _plan_fallback(self, classification):
        """Simple fallback planning when no RuleBasedPlanner is configured."""
        from ..intent.intent_binding import Strategy
        from ..planner.task_plan import Decision, TaskNode, TaskPlan

        if classification.is_unknown():
            return TaskPlan.delegate_llm(classification)

        skill = self._registry.get(classification.intent_type)
        if skill:
            task = TaskNode(
                node_id="step_1",
                skill_id=skill.name,
                params={"message": ""},
                rationale=f"执行 {skill.name}",
            )
            return TaskPlan(
                intent=classification,
                bound_intent_type=classification.intent_type,
                decision=Decision.EXECUTE,
                tasks=[task],
            )

        return TaskPlan.delegate_llm(classification)


# ─── Global singleton ────────────────────────────────────────────────────────

_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get the global skill manager instance."""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager


def reload_skill_manager() -> SkillManager:
    """Reload and return the skill manager."""
    global _skill_manager
    _skill_manager = SkillManager()
    return _skill_manager
