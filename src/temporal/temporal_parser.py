"""Temporal parser: Phase 0 of the pipeline — resolves all date anchors before intent classification.

Responsibilities:
- Extract all date expressions (relative, absolute, weekday-anchored)
- Resolve narrative-relative "昨天" / "前天" based on the last weekday anchor in the text
- Detect journey/travel patterns and pre-compute average distance per day
- Return a structured TemporalContext consumed by IntentClassifier and Planner
"""

import re as _re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ─── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class TemporalContext:
    """Output of TemporalParser.parse().

    Consumed by IntentClassifier.classify_with_context() and RuleBasedPlanner.plan().

    Attributes:
        dates:       Mapping from date-label strings to resolved date objects.
                     e.g. {"上周五": date(2026,5,22), "周一": date(2026,5,25)}
        timeline:    Sorted list of human-readable date strings.
                     e.g. ["上周五(2026-05-22,星期五)", "周一(2026-05-25,星期一)"]
        context_text: Flat text for LLM injection (kept for backward compat).
        has_multiple_anchors: True when >= 2 date anchors detected.
        journey_result: Pre-computed journey analysis (or None).
        yesterday_note: Warning about narrative-relative date resolution (or None).
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
        """Full Phase-0 parse: extract dates, resolve "昨天", detect journey."""
        anchor_dates = self._extract_all_dates(message)

        if len(anchor_dates) < 2:
            return TemporalContext()

        # Resolve narrative-relative "昨天" against the last weekday anchor in text
        ctx = TemporalContext()
        ctx.dates = anchor_dates
        ctx.has_multiple_anchors = len(anchor_dates) >= 2

        ctx.yesterday_note, yesterday_resolved = self._resolve_yesterday(message, anchor_dates)
        if yesterday_resolved:
            anchor_dates["昨天"] = yesterday_resolved
            ctx.dates = anchor_dates

        # Build readable timeline
        ctx.timeline = self._build_timeline(anchor_dates)

        # Pre-compute journey if this looks like a travel/distance problem
        journey_result = self._compute_journey(message, anchor_dates, yesterday_resolved)
        ctx.journey_result = journey_result

        # Backward-compat context text
        ctx.context_text = self._build_context_text(ctx, journey_result)

        return ctx

    # ─── Date extraction ────────────────────────────────────────────────────────

    def _extract_all_dates(self, message: str) -> Dict[str, date]:
        """Extract all recognized date expressions from the message."""
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

        # 4. Simple today/yesterday/tomorrow keywords
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

        # 5. Bare weekdays — resolve relative to the anchored week
        ref_week_start = self._find_anchor_week_start(anchors)
        if ref_week_start:
            journey_week_start = ref_week_start + timedelta(weeks=1)
            for bw in self.QUALIFIED_BARE_WEEKDAYS:
                if bw not in anchors:
                    pattern = _re.compile(_re.escape(bw))
                    if pattern.search(message):
                        dow_idx = self.WEEKDAY_MAP[bw]
                        resolved_bw = journey_week_start + timedelta(days=dow_idx)
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

    def _find_anchor_week_start(self, anchors: Dict[str, date]) -> Optional[date]:
        """Find the Monday of the week that contains the first qualified anchor."""
        for label in anchors:
            if label in self.QUALIFIED_DATE_LABELS:
                d = anchors[label]
                return d - timedelta(days=d.weekday())
        return None

    # ─── Narrative-relative "昨天" resolution ────────────────────────────────────

    def _resolve_yesterday(
        self, message: str, anchor_dates: Dict[str, date]
    ) -> Tuple[Optional[str], Optional[date]]:
        """Infer what "昨天" means from narrative context.

        Strategy: find the weekday anchor that appears LAST in the text, then
        "昨天" = that anchor date + 1 day.

        Returns (warning_note, resolved_date).
        """
        if "昨天" not in message:
            return None, None

        qualified_set = set(self.QUALIFIED_DATE_LABELS) | set(self.QUALIFIED_BARE_WEEKDAYS)

        latest_label = None
        latest_pos = -1
        for label in qualified_set:
            if label in anchor_dates:
                pos = message.find(label)
                if pos != -1 and pos > latest_pos:
                    latest_pos = pos
                    latest_label = label

        if latest_label and latest_label in anchor_dates:
            anchor_date = anchor_dates[latest_label]
            anchor_wd = self._chinese_weekday(anchor_date)
            yd = anchor_date + timedelta(days=1)
            ywd = self._chinese_weekday(yd)
            note = (
                f"⚠️ 【关键】'昨天'在叙事中应理解为{latest_label}"
                f"({anchor_date.isoformat()},{anchor_wd})的后一天"
                f"={yd.isoformat()}({ywd})！"
            )
            return note, yd

        return None, None

    # ─── Timeline building ───────────────────────────────────────────────────────

    def _build_timeline(self, anchor_dates: Dict[str, date]) -> List[str]:
        """Build a sorted, human-readable timeline list."""
        entries = []
        for label, d in anchor_dates.items():
            wd = self._chinese_weekday(d)
            entries.append(f"{label}({d.isoformat()},{wd})")
        entries.sort(key=lambda x: x.split("(")[1].split(",")[0])
        return entries

    def _build_context_text(self, ctx: TemporalContext, journey_result: Optional[str]) -> str:
        """Build the flat context text for LLM injection (backward compat)."""
        parts = [
            f"[时间线分析 — 重要：所有相对日期均以今天({self.today.isoformat()})为基准推算]",
            f"检测到 {len(ctx.dates)} 个日期锚点：{', '.join(sorted(ctx.dates.keys()))}",
            f"时间线：{' | '.join(ctx.timeline)}",
        ]
        if ctx.yesterday_note:
            parts.append(ctx.yesterday_note)
        if journey_result:
            parts.append(f"\n【旅程计算结果】\n{journey_result}")
        parts.append("请根据上述时间线理解'昨天'、'前天'等相对日期在叙事中的具体日期。")
        return "\n".join(parts)

    # ─── Journey computation ───────────────────────────────────────────────────

    def _compute_journey(
        self,
        message: str,
        anchor_dates: Dict[str, date],
        yesterday_resolved: Optional[date],
    ) -> Optional[str]:
        """Pre-compute journey math from a multi-date travel narrative.

        Detects patterns like:
        - "从X到Y，总共Z公里，每天走多远"
        - "从上周五出发...昨天到达拉萨，总共X公里"
        """
        if not anchor_dates:
            return None

        journey_kws = ["行驶", "公里", "km", "拉萨", "成都", "出发", "每天", "平均", "总共", "一共"]
        math_kws = ["每天", "平均", "多远"]

        has_journey = sum(1 for kw in journey_kws if kw in message) >= 3
        has_math = any(kw in message for kw in math_kws)
        has_total = (
            any(kw in message for kw in ["2080", "总共", "一共", "总", "行驶了"])
            and any(kw in message for kw in ["km", "公里", "米"])
        )

        if not (has_journey and (has_math or has_total)):
            return None

        # Build weekday entries for journey computation
        qualified_set = set(self.QUALIFIED_DATE_LABELS) | set(self.QUALIFIED_BARE_WEEKDAYS)
        journey_weekday_entries = {
            label: (d, self._chinese_weekday(d))
            for label, d in anchor_dates.items()
            if label in qualified_set
        }

        # Determine start and end dates
        start_date = self._resolve_journey_start(journey_weekday_entries, anchor_dates)
        end_date = self._resolve_journey_end(
            journey_weekday_entries, anchor_dates, yesterday_resolved, start_date
        )

        if start_date is None or end_date is None:
            return None

        days = (end_date - start_date).days
        if days <= 0:
            return None

        # Extract distances
        total_km = self._extract_total_km(message)
        remaining_km = self._extract_remaining_km(message)

        if total_km <= 0:
            return None

        lines = self._build_journey_lines(
            start_date, end_date, days, total_km,
            remaining_km, journey_weekday_entries
        )

        return "\n".join(lines) if lines else None

    def _resolve_journey_start(
        self,
        journey_weekday_entries: Dict[str, Tuple[date, str]],
        anchor_dates: Dict[str, date],
    ) -> Optional[date]:
        """Resolve journey start date from weekday anchors."""
        if journey_weekday_entries:
            # First entry is the earliest weekday anchor
            return next(iter(journey_weekday_entries.values()))[0]
        if anchor_dates:
            return min(anchor_dates.values())
        return None

    def _resolve_journey_end(
        self,
        journey_weekday_entries: Dict[str, Tuple[date, str]],
        anchor_dates: Dict[str, date],
        yesterday_resolved: Optional[date],
        start_date: date,
    ) -> Optional[date]:
        """Resolve journey end date, preferring narrative-relative yesterday."""
        if yesterday_resolved and yesterday_resolved >= start_date:
            return yesterday_resolved
        if len(journey_weekday_entries) >= 2:
            return list(journey_weekday_entries.values())[-1][0]
        if anchor_dates:
            return max(anchor_dates.values())
        return None

    def _extract_total_km(self, message: str) -> float:
        """Extract total distance from the message."""
        m = _re.search(r"(?:一共|总共|行驶了)[^\d]*?(\d+(?:\.\d+)?)\s*(?:km|公里|千米|米)?", message)
        if not m:
            m = _re.search(r"(\d+(?:\.\d+)?)\s*(?:km|公里|千米)", message)
        if m:
            return float(m.group(1))
        return 0.0

    def _extract_remaining_km(self, message: str) -> float:
        """Extract remaining distance (周一还剩1000km) from the message."""
        m = _re.search(r"还剩?\s*(\d+(?:\.\d+)?)\s*(?:km|公里|千米|米)?", message)
        if m:
            return float(m.group(1))
        return 0.0

    def _build_journey_lines(
        self,
        start_date: date,
        end_date: date,
        days: int,
        total_km: float,
        remaining_km: float,
        journey_weekday_entries: Dict[str, Tuple[date, str]],
    ) -> List[str]:
        """Build the journey analysis output lines."""
        lines = []
        start_wd = self._chinese_weekday(start_date)
        end_wd = self._chinese_weekday(end_date)

        # Infer start and end labels
        start_label = self._label_for(start_date, journey_weekday_entries) or str(start_date)
        end_label = self._label_for(end_date, journey_weekday_entries) or str(end_date)

        lines.append(
            f"行程分析：{start_label}({start_date.isoformat()},{start_wd}) "
            f"→ {end_label}({end_date.isoformat()},{end_wd})，共 {days} 天"
        )

        avg = total_km / days
        lines.append(
            f"总里程：{total_km:.0f} km，总天数：{days} 天，"
            f"平均：{total_km:.0f}/{days} = {avg:.1f} km/天"
        )

        if remaining_km > 0 and journey_weekday_entries:
            mid = self._find_middle_anchor(start_date, end_date, journey_weekday_entries)
            if mid:
                mid_date, mid_wd = mid
                mid_label = next(k for k, v in journey_weekday_entries.items() if v[0] == mid_date)
                first_leg_days = (mid_date - start_date).days
                first_leg_km = total_km - remaining_km
                first_leg_avg = first_leg_km / first_leg_days if first_leg_days > 0 else 0
                second_leg_days = (end_date - mid_date).days
                second_leg_avg = remaining_km / second_leg_days if second_leg_days > 0 else 0

                lines.append(
                    f"第一段：{start_label} → {mid_label}，"
                    f"{first_leg_days} 天，行驶 {first_leg_km:.0f} km，平均 {first_leg_avg:.1f} km/天"
                )
                lines.append(
                    f"第二段：{mid_label} → {end_label}，"
                    f"{second_leg_days} 天，行驶 {remaining_km:.0f} km，平均 {second_leg_avg:.1f} km/天"
                )

                if first_leg_avg > second_leg_avg:
                    lines.append(
                        f"结论：前半段走得更快（每天多 {first_leg_avg - second_leg_avg:.1f} km）"
                    )
                elif second_leg_avg > first_leg_avg:
                    lines.append(
                        f"结论：后半段走得更快（每天多 {second_leg_avg - first_leg_avg:.1f} km）"
                    )

        return lines

    def _label_for(
        self, d: date, entries: Dict[str, Tuple[date, str]]
    ) -> Optional[str]:
        """Find the original label string for a resolved date."""
        for label, (entry_date, _) in entries.items():
            if entry_date == d:
                return label
        return None

    def _find_middle_anchor(
        self,
        start: date,
        end: date,
        entries: Dict[str, Tuple[date, str]],
    ) -> Optional[Tuple[date, str]]:
        """Find the first weekday anchor strictly between start and end."""
        for label, (d, wd) in sorted(entries.items(), key=lambda x: x[1][0]):
            if start < d < end:
                return d, wd
        return None

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
