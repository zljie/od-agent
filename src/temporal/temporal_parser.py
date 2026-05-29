"""Temporal parser: Phase 0 of the pipeline — resolves all date anchors before intent classification.

Responsibilities:
- Extract all date expressions (relative, absolute, weekday-anchored)
- Resolve all relative dates (上周五, 昨天, 周一, etc.) against today's date
- Return a structured TemporalContext consumed by IntentClassifier and Planner

All relative dates are resolved relative to the current local date (today), not
to any narrative anchor in the text.
"""

import re as _re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional


# ─── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class TemporalContext:
    """Output of TemporalParser.parse().

    Consumed by IntentClassifier.classify_with_context() and RuleBasedPlanner.plan().

    Attributes:
        dates:       Mapping from date-label strings to resolved date objects.
                     e.g. {"上周五": date(2026,5,22), "周一": date(2026,5,25),
                           "昨天": date(2026,5,28)}
                     All dates are resolved relative to today (not narrative-relative).
        timeline:    Sorted list of human-readable date strings.
                     e.g. ["上周五(2026-05-22,星期五)", "周一(2026-05-25,星期一)",
                           "昨天(2026-05-28,星期四)"]
        context_text: Flat text for LLM injection.
        has_multiple_anchors: True when >= 2 date anchors detected.
        journey_result: Always None — journey math is handled by LLM after date resolution.
        yesterday_note: Always None — yesterday is resolved in-place by Phase-0.
    """

    dates: Dict[str, date] = field(default_factory=dict)
    timeline: List[str] = field(default_factory=list)
    context_text: str = ""
    has_multiple_anchors: bool = False
    journey_result: Optional[str] = None
    yesterday_note: Optional[str] = None

    def resolved_date(self, label: str) -> Optional[date]:
        """Look up a resolved date by its original label."""
        return self.dates.get(label)

    def get_days_between(self, start_label: str, end_label: str) -> Optional[int]:
        """Return the number of days between two resolved date labels."""
        d1 = self.dates.get(start_label)
        d2 = self.dates.get(end_label)
        if d1 and d2:
            return (d2 - d1).days
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dates": {k: v.isoformat() for k, v in self.dates.items()},
            "timeline": self.timeline,
            "context_text": self.context_text,
            "has_multiple_anchors": self.has_multiple_anchors,
            "journey_result": self.journey_result,
            "yesterday_note": self.yesterday_note,
        }


# ─── Core Parser ────────────────────────────────────────────────────────────────


class TemporalParser:
    """Phase-0 parser: resolves dates and detects journey patterns in user text.

    Usage:
        parser = TemporalParser()
        ctx = parser.parse("上周五从成都出发去拉萨，昨天到达拉萨，一共行驶了2080km")

    The `today_date` parameter (optional) allows the parser to receive the current
    date from an external source (e.g. TimeConverterSkill) rather than calling
    date.today() directly, ensuring the entire date-resolution chain uses the
    same skill system.
    """

    QUALIFIED_DATE_LABELS = [
        "上周五", "上周一", "上周二", "上周三", "上周四", "上周六", "上周日",
        "下周一", "下周二", "下周三", "下周四", "下周五", "下周六", "下周日",
        "这周一", "这周二", "这周三", "这周四", "这周五", "这周六", "这周日",
    ]

    QUALIFIED_BARE_WEEKDAYS = [
        "周一", "周二", "周三", "周四", "周五", "周六", "周日",
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日",
    ]

    WEEKDAY_LONG_NAMES = [
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"
    ]

    WEEKDAY_SHORT_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    WEEKDAY_MAP = {
        "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
        "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
    }

    def __init__(self, today_date: Optional[date] = None):
        self._today = today_date or date.today()

    @property
    def today(self) -> date:
        """Current date. Defaults to date.today() unless injected."""
        return self._today

    def parse(self, message: str, today_date: Optional[date] = None) -> TemporalContext:
        """Full Phase-0 parse: extract dates, resolve "昨天", detect journey.

        All relative dates are resolved relative to **today** (the current local date).
        "昨天" always means today - 1, never "the day after the last weekday anchor".
        """
        if today_date is not None:
            self._today = today_date
        anchor_dates = self._extract_all_dates(message)

        if len(anchor_dates) < 2:
            return TemporalContext()

        ctx = TemporalContext()
        ctx.dates = anchor_dates
        ctx.has_multiple_anchors = len(anchor_dates) >= 2

        # Note: "昨天" is already resolved in _extract_all_dates as today - 1.
        # No narrative-relative adjustment needed.
        ctx.yesterday_note = None

        # Build readable timeline
        ctx.timeline = self._build_timeline(anchor_dates)

        # No pre-computed journey: let the LLM handle the math after dates are resolved.
        # The agent's job is only to resolve ambiguous dates into clear absolute dates.
        ctx.journey_result = None

        # Backward-compat context text
        ctx.context_text = self._build_context_text(ctx)

        return ctx

    # ─── Date extraction ────────────────────────────────────────────────────────

    def _extract_all_dates(self, message: str) -> Dict[str, date]:
        """Extract all recognized date expressions from the message.

        All relative dates are resolved relative to **today**, not to any
        narrative anchor in the text. This means:
          - "上周五" = Friday of last week (today.weekday-based)
          - "昨天"   = today - 1 day (always)
          - bare "周一" = Monday of the current week (this week)
        """
        anchors: Dict[str, date] = {}

        # 1. Qualified labels with week prefix (上周五, 下周一, 这周三, etc.)
        for label in self.QUALIFIED_DATE_LABELS:
            if label in message:
                resolved = self._resolve_weekday_label(label)
                if resolved:
                    anchors[label] = resolved

        # 2. ISO numeric dates: 2026-05-22
        for m in _re.finditer(r"\d{4}-\d{2}-\d{2}", message):
            try:
                d = date.fromisoformat(m.group(0))
                anchors[m.group(0)] = d
            except ValueError:
                pass

        # 3. Chinese numeric dates: 2026年5月22日
        for m in _re.finditer(r"\d{4}年\d{1,2}月\d{1,2}日", message):
            cn = _re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", m.group(0))
            if cn:
                try:
                    anchors[cn.group(0)] = date(int(cn.group(1)), int(cn.group(2)), int(cn.group(3)))
                except ValueError:
                    pass

        # 4. Simple today/yesterday/tomorrow keywords — always relative to TODAY
        keyword_map = {
            "今天": self.today,
            "昨天": self.today - timedelta(days=1),
            "前天": self.today - timedelta(days=2),
            "明天": self.today + timedelta(days=1),
            "后天": self.today + timedelta(days=2),
        }
        for kw, d in keyword_map.items():
            if kw in message:
                anchors[kw] = d

        # 5. Bare weekdays — resolved relative to TODAY's week, NOT the journey week.
        # "周一" means Monday of THIS week (the week containing today).
        # Prevent substring collisions: if "上周五" was matched, "周五" inside it is NOT
        # a bare weekday anchor.
        forbidden_from_qualified: set = set()
        for ql in self.QUALIFIED_DATE_LABELS:
            for bw in self.QUALIFIED_BARE_WEEKDAYS:
                if ql in message and bw in ql:
                    forbidden_from_qualified.add(bw)

        # Monday of the current (today's) week
        this_monday = self.today - timedelta(days=self.today.weekday())
        for bw in self.QUALIFIED_BARE_WEEKDAYS:
            if bw not in anchors and bw not in forbidden_from_qualified:
                pattern = _re.compile(_re.escape(bw))
                if pattern.search(message):
                    dow_idx = self.WEEKDAY_MAP[bw]
                    resolved_bw = this_monday + timedelta(days=dow_idx)
                    anchors[bw] = resolved_bw

        return anchors

    def _resolve_weekday_label(self, label: str) -> Optional[date]:
        """Resolve a week-prefixed weekday label (上周五, 下周三, 这周一) to a date."""
        prefix_map = {"上": -1, "下": 1, "这": 0}
        for prefix, offset_weeks in prefix_map.items():
            for dow_name, dow_idx in self.WEEKDAY_MAP.items():
                if label == f"{prefix}{dow_name}":
                    this_monday = self.today - timedelta(days=self.today.weekday())
                    target_monday = this_monday + timedelta(weeks=offset_weeks)
                    return target_monday + timedelta(days=dow_idx)
        return None

    def _build_timeline(self, anchor_dates: Dict[str, date]) -> List[str]:
        """Build a sorted, human-readable timeline list."""
        entries = []
        for label, d in anchor_dates.items():
            wd = self._chinese_weekday(d)
            entries.append(f"{label}({d.isoformat()},{wd})")
        entries.sort(key=lambda x: x.split("(")[1].split(",")[0])
        return entries

    def _build_context_text(self, ctx: TemporalContext) -> str:
        """Build the flat context text for LLM injection."""
        parts = [
            f"[时间线分析 — 重要：所有相对日期均以今天({self.today.isoformat()})为基准推算]",
            f"检测到 {len(ctx.dates)} 个日期锚点：{', '.join(sorted(ctx.dates.keys()))}",
            f"时间线：{' | '.join(ctx.timeline)}",
            "请根据上述时间线理解'昨天'、'前天'等相对日期在叙事中的具体日期。",
        ]
        return "\n".join(parts)

    # ─── Public helpers ─────────────────────────────────────────────────────────

    def resolve_date(self, raw: str) -> Optional[date]:
        """Public entry point for resolving a single date expression."""
        if not raw or not raw.strip():
            return None
        return self._resolve_weekday_label(raw.strip())

    def chinese_weekday(self, d: date) -> str:
        """Return the Chinese weekday name for a date."""
        return self._chinese_weekday(d)

    def _chinese_weekday(self, d: date) -> str:
        return self.WEEKDAY_LONG_NAMES[d.weekday()]

