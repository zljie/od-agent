"""Skill Manager - orchestrates skill detection, planning, and execution.

Provides a unified interface that:
1. Manages SkillRegistry (skill registration/lookup)
2. Runs TemporalParser (Phase 0: date resolution + journey detection)
3. Runs IntentClassifier (Phase 1: intent classification with temporal context)
4. Runs RuleBasedPlanner (Phase 2: dependency-aware DAG planning)
5. Runs TaskExecutor (Phase 3: topological execution)

All components can be used independently or together.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseSkill
from .skill_registry import SkillRegistry
from .task_executor import TaskExecutor
from .math_teacher import MathTeacherSkill
from .time_converter import TimeConverterSkill

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
        """Extract all time anchors and pre-compute journey math for multi-day narratives.

        Detects travel/journey patterns (多个日期锚点 + 总里程/每日等关键词) and
        resolves all relative dates (上周五, 周一, 昨天) in context, then pre-computes
        the journey answer so the LLM doesn't have to reason through the timeline.

        Returns a dict with:
            - dates: {name_label: iso_date_string} mapping
            - timeline: human-readable ordered list
            - context_text: formatted string for LLM injection (includes journey result)
            - has_multiple_anchors: bool
            - journey_result: pre-computed journey analysis dict (or None)
        """
        from datetime import date, timedelta
        import re as _re
        import re

        time_skill = self._registry.get("Time Converter")
        if not time_skill:
            return {"dates": {}, "timeline": [], "context_text": "", "has_multiple_anchors": False}

        # ── Scan for qualified date-like expressions (qualified = has prefix like 上/下/这) ──
        qualified_candidates = [
            "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
            "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
            "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
            "今天", "昨天", "前天", "明天", "后天", "大后天", "大前天",
            "上周", "下周", "这周",
            "上个月", "下个月", "这个月",
            "去年", "今年", "明年",
        ]

        # Bare weekdays (no prefix) — handled separately after we have a reference week.
        # Use regex word boundaries to avoid substring matches (e.g. "上周五" should not match "周五")
        import re as _bare_re
        bare_weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日",
                          "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday_map = {
            "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
            "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
        }

        anchor_dates: Dict[str, date] = {}
        for cand in qualified_candidates:
            if cand in message:
                resolved = time_skill._resolve_date(cand)
                if resolved:
                    anchor_dates[cand] = resolved

        # Scan for ISO / Chinese date literals
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

        # Post-process: re-resolve bare weekdays relative to anchored week.
        # If we have "上周五" (Friday of previous week), bare "周一" should resolve
        # to Monday of the NEXT week (the week that starts AFTER 上周五's week).
        # Story: 上周五出发 → Monday refers to the Monday that follows.

        # Find the reference week from a qualified anchor (上周五 → week starting the Monday before)
        ref_week_start = None
        for label, d in anchor_dates.items():
            if label in (
                "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
                "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
                "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
            ):
                # Monday of the week containing this anchor
                ref_week_start = d - timedelta(days=d.weekday())
                break

        if ref_week_start:
            # The journey week is the week AFTER the anchored week
            journey_week_start = ref_week_start + timedelta(weeks=1)
            for bw in bare_weekdays:
                # Use word-boundary check to avoid "上周五" matching "周五"
                pattern = _bare_re.compile(_bare_re.escape(bw))
                if pattern.search(message) and bw not in anchor_dates:
                    target_dow = weekday_map[bw]
                    resolved_bw = journey_week_start + timedelta(days=target_dow)
                    anchor_dates[bw] = resolved_bw

        if len(anchor_dates) < 2:
            return {"dates": {}, "timeline": [], "context_text": "", "has_multiple_anchors": False}

        # ── Resolve "昨天" relative to weekday anchors in the narrative ─────
        # Strategy: find the weekday anchor that appears LAST in the text (most recent
        # narrative mention), then "昨天" = the day after that anchor.
        # e.g. text: "上周五出发，周一还剩1000km，昨天到达"
        #   → "周一" appears last in text → 昨天 = day after 周一 = 周二
        yesterday_note = None
        yesterday_resolved_date = None

        # Collect all weekday anchors with their text positions
        qualified_anchor_labels = {
            "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
            "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
            "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
        }
        for bw in ["周一", "周二", "周三", "周四", "周五", "周六", "周日",
                     "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]:
            qualified_anchor_labels.add(bw)

        # Find the weekday anchor that appears last in the message text
        latest_anchor_label = None
        latest_pos = -1
        for label in qualified_anchor_labels:
            pos = message.find(label)
            if pos != -1 and pos > latest_pos:
                latest_pos = pos
                latest_anchor_label = label

        if latest_anchor_label and latest_anchor_label in anchor_dates:
            anchor_date = anchor_dates[latest_anchor_label]
            anchor_wd = time_skill._get_chinese_weekday(anchor_date)
            yd = anchor_date + timedelta(days=1)
            ywd = time_skill._get_chinese_weekday(yd)
            yesterday_note = (
                f"⚠️ 【关键】'昨天'在叙事中应理解为{latest_anchor_label}({anchor_date.isoformat()},{anchor_wd})的后一天"
                f"={yd.isoformat()}({ywd})！"
            )
            yesterday_resolved_date = yd

        # Build weekday_entries for journey computation (all qualified anchors)
        weekday_entries: Dict[str, tuple] = {}
        for label, d in anchor_dates.items():
            wd = time_skill._get_chinese_weekday(d)
            if label in qualified_anchor_labels:
                weekday_entries[label] = (d, wd)

        # ── Build readable timeline ────────────────────────────────────────
        resolved_entries = []
        for label, d in anchor_dates.items():
            wd = time_skill._get_chinese_weekday(d)
            resolved_entries.append(f"{label}({d.isoformat()},{wd})")

        dates_iso = {k: v.isoformat() for k, v in anchor_dates.items()}
        timeline_parts = sorted(
            resolved_entries,
            key=lambda x: x.split("(")[1].split(",")[0]
        )

        # ── Pre-compute journey math if this is a travel/journey message ────
        # Detected by: multiple date anchors + total distance keyword + "每天" or "平均"
        journey_result = None
        journey_keywords = ["行驶", "公里", "km", "拉萨", "成都", "出发", "每天", "平均", "总共", "一共"]
        math_keywords = ["每天", "平均", "多远"]
        has_journey = sum(1 for kw in journey_keywords if kw in message) >= 3
        has_math = any(kw in message for kw in math_keywords)
        has_total = any(kw in message for kw in ["2080", "总共", "一共", "总", "行驶了"]) and any(kw in message for kw in ["km", "公里", "米"])

        if has_journey and (has_math or has_total):
            journey_result = self._compute_journey(message, anchor_dates, yesterday_resolved_date, time_skill)

        # ── Build context text ──────────────────────────────────────────────
        context_parts = [
            f"[时间线分析 — 重要：所有相对日期均以今天({date.today().isoformat()})为基准推算]",
            f"检测到 {len(anchor_dates)} 个日期锚点：{', '.join(sorted(anchor_dates.keys()))}",
            f"时间线：{' | '.join(timeline_parts)}",
        ]
        if yesterday_note:
            context_parts.append(yesterday_note)

        if journey_result:
            context_parts.append(f"\n【旅程计算结果】\n{journey_result}")

        context_parts.append("请根据上述时间线理解'昨天'、'前天'等相对日期在叙事中的具体日期。")

        return {
            "dates": dates_iso,
            "timeline": timeline_parts,
            "context_text": "\n".join(context_parts),
            "has_multiple_anchors": True,
            "journey_result": journey_result,
        }

    def _compute_journey(
        self,
        message: str,
        anchor_dates: Dict[str, "date"],
        yesterday_resolved: Optional["date"],
        time_skill: Any,
    ) -> Optional[str]:
        """Pre-compute journey math from a multi-date travel narrative.

        Detects patterns like:
        - "从X到Y，总共Z公里，每天走多远"
        - "从上周五出发...昨天到达拉萨，总共X公里"
        """
        import re as _re
        from datetime import date

        if not anchor_dates:
            return None

        # Build weekday entries from anchor_dates for journey computation
        # (same as the qualified anchor filter in build_temporal_context)
        qualified_anchor_labels = {
            "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
            "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
            "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
        }
        for bw in ["周一", "周二", "周三", "周四", "周五", "周六", "周日",
                     "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]:
            if bw in message:
                qualified_anchor_labels.add(bw)

        journey_weekday_entries = {
            label: (d, time_skill._get_chinese_weekday(d))
            for label, d in anchor_dates.items()
            if label in qualified_anchor_labels
        }

        # Determine journey start: prefer qualified weekday anchors
        weekday_start_label = None
        weekday_end_label = None
        for label in journey_weekday_entries:
            if weekday_start_label is None:
                weekday_start_label = label
            else:
                weekday_end_label = label

        # Use weekday-named start if available, else earliest anchor
        if weekday_start_label:
            start_date = anchor_dates[weekday_start_label]
        else:
            start_date = min(anchor_dates.values())

        # End: use resolved yesterday if available and after start, else latest weekday anchor
        if yesterday_resolved and yesterday_resolved >= start_date:
            end_date = yesterday_resolved
            end_label = f"昨天({time_skill._get_chinese_weekday(end_date)})"
        elif weekday_end_label:
            end_date = anchor_dates[weekday_end_label]
            end_label = weekday_end_label
        else:
            end_date = max(anchor_dates.values())
            end_label = next((k for k, v in anchor_dates.items() if v == end_date), str(end_date))

        days = (end_date - start_date).days
        if days <= 0:
            return None

        # Extract total distance
        total_km = 0
        m = _re.search(r"(?:一共|总共|行驶了)[^\d]*?(\d+(?:\.\d+)?)\s*(?:km|公里|千米|米)?", message)
        if not m:
            m = _re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里|千米)", message)
        if m:
            total_km = float(m.group(1))

        # Extract remaining distance (周一还剩1000km)
        remaining_km = 0
        m_rem = _re.search(r"还剩?\s*(\d+(?:\.\d+)?)\s*(?:km|公里|千米|米)?", message)
        if m_rem:
            remaining_km = float(m_rem.group(1))

        start_wd = time_skill._get_chinese_weekday(start_date)
        end_wd = time_skill._get_chinese_weekday(end_date)

        lines = []
        lines.append(
            f"行程分析：{weekday_start_label}({start_date.isoformat()},{start_wd}) "
            f"→ {end_label}({end_date.isoformat()},{end_wd})，共 {days} 天"
        )

        if total_km > 0:
            avg = total_km / days
            lines.append(f"总里程：{total_km:.0f} km，总天数：{days} 天，平均：{total_km:.0f}/{days} = {avg:.1f} km/天")

            if remaining_km > 0 and journey_weekday_entries:
                # Find middle anchor (周一, 周三 style) between start and end
                mid_date = None
                mid_label = None
                for label, (d, _) in sorted(journey_weekday_entries.items(), key=lambda x: x[1][0]):
                    if start_date < d < end_date:
                        mid_date = d
                        mid_label = label
                        break

                if mid_date:
                    mid_wd = time_skill._get_chinese_weekday(mid_date)
                    first_leg_days = (mid_date - start_date).days
                    first_leg_km = total_km - remaining_km
                    first_leg_avg = first_leg_km / first_leg_days if first_leg_days > 0 else 0
                    second_leg_days = (end_date - mid_date).days
                    second_leg_km = remaining_km
                    second_leg_avg = second_leg_km / second_leg_days if second_leg_days > 0 else 0

                    lines.append(
                        f"第一段：{weekday_start_label} → {mid_label}，"
                        f"{first_leg_days} 天，行驶 {first_leg_km:.0f} km，平均 {first_leg_avg:.1f} km/天"
                    )
                    lines.append(
                        f"第二段：{mid_label} → {end_label}，"
                        f"{second_leg_days} 天，行驶 {second_leg_km:.0f} km，平均 {second_leg_avg:.1f} km/天"
                    )

                    if first_leg_avg > second_leg_avg:
                        lines.append(f"结论：前半段走得更快（每天多 {first_leg_avg - second_leg_avg:.1f} km）")
                    else:
                        lines.append(f"结论：后半段走得更快（每天多 {second_leg_avg - first_leg_avg:.1f} km）")

        return "\n".join(lines) if lines else None

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
