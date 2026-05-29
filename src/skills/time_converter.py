"""Time Converter Skill - parses and converts time expressions in user messages."""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from .base import BaseSkill


@dataclass
class TimeResult:
    """Result of time parsing and conversion."""

    original_input: str
    parsed_times: List[Dict[str, Any]]
    formatted_output: str
    summary: str


class TimeConverterSkill(BaseSkill):
    """Time Converter Skill that handles time/date parsing and conversion.

    Activated when user message contains time-related expressions:
    - Relative dates (今天, 明天, 下周, 去年)
    - Date patterns (2024-01-15, 2024年1月15日, 1月15日)
    - Time zones (北京时间, UTC, EST)
    - Relative calculations (三天后, 两周前, 下个月)
    - Day-of-week queries (今天是星期几, 下周一)
    """

    name = "Time Converter"
    description = "Parses and converts time/date expressions, calculates relative time differences, and formats dates in various styles."
    # Slots expected from IntentClassifier entities (Planner uses these for {slot} substitution)
    intent_params = {
        "operation": "range_diff",
        "start": "",
        "end": "",
    }
    keywords = [
        # Chinese relative dates
        "今天", "明天", "后天", "大后天",
        "昨天", "前天", "大前天",
        "上周", "这周", "下周", "下下周",
        "上个月", "这个月", "下个月", "明年的",
        "去年", "今年", "明年",
        # Day of week
        "星期几", "周几", "礼拜几", "礼拜天", "星期天",
        "周一", "周二", "周三", "周四", "周五", "周六", "周日",
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六",
        # Time phrases
        "几点", "什么时候", "哪一天", "日期",
        "多久", "几天", "几个小时", "几个月",
        "以前", "以后", "之前", "之后",
        # Time zone
        "北京时间", "东京时间", "纽约时间", "伦敦时间",
        "UTC", "GMT", "时区", "时差",
        # Duration
        "天前", "天后", "周前", "周后", "月前", "月后", "年前", "年后",
        "工作日", "周末", "节假日", "假期",
    ]
    priority = 60

    def __init__(self):
        self.today = date.today()
        self.now = datetime.now()
        self._chinese_weekday_names = [
            "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"
        ]
        self._chinese_weekday_short = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the time converter skill.

        Supports two input modes:
        - message-based (legacy): input_data["message"] contains raw user text
        - entity-based (Planner): input_data contains structured params
            e.g. {"start": "上周五", "end": "今天", "operation": "range_diff"}
        """
        params = input_data.get("params", {})
        raw_message = input_data.get("message", "")

        if params:
            # Planner-driven: receive resolved entity slots
            return await self.execute_with_entities(params)

        # Legacy: receive raw message
        if not raw_message:
            return {
                "success": False,
                "response": "请告诉我你想查询或转换什么时间？",
                "metadata": {},
            }

        result = self._parse_and_convert(raw_message)
        return {
            "success": True,
            "response": result.formatted_output,
            "metadata": {
                "original_input": result.original_input,
                "parsed_times": result.parsed_times,
                "summary": result.summary,
            },
        }

    async def execute_with_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time calculation from structured entity input (Planner-driven).

        Supported operations:
            range_diff: {"start": "...", "end": "...", "operation": "range_diff"}
            day_of_week: {"date": "...", "operation": "day_of_week"}
            duration: {"expression": "...", "operation": "duration"}
            timezone: {"tz": "...", "operation": "timezone"}

        Returns:
            Structured result with computed fields for the Planner/Agent to use.
        """
        operation = params.get("operation", "range_diff")
        result = self._compute(operation, params)

        if result is None:
            return {
                "success": False,
                "response": f"无法解析时间参数: {params}",
                "metadata": {"operation": operation, "params": params},
            }

        return {
            "success": True,
            "response": result["response"],
            "metadata": {
                "operation": operation,
                "start": result.get("start"),
                "end": result.get("end"),
                "days": result.get("days"),
                "weekday": result.get("weekday"),
                "original_params": params,
            },
        }

    def _compute(self, operation: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch to the appropriate time computation method."""
        if operation == "range_diff":
            return self._compute_range_diff(params.get("start", ""), params.get("end", ""))
        if operation == "day_of_week":
            return self._compute_day_of_week(params.get("date", "今天"))
        if operation == "duration":
            return self._compute_duration(params.get("expression", ""))
        if operation == "timezone":
            return self._compute_timezone(params.get("tz", "北京时间"))
        return None

    def _compute_range_diff(self, start_raw: str, end_raw: str) -> Optional[Dict[str, Any]]:
        """Compute the day difference between two date expressions."""
        start_date = self._resolve_date(start_raw)
        end_date = self._resolve_date(end_raw)
        if not start_date or not end_date:
            return None

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        delta = (end_date - start_date). days

        start_wd = self._get_chinese_weekday(start_date)
        end_wd = self._get_chinese_weekday(end_date)

        if delta == 0:
            response = f"起点和终点是同一天：{start_date.isoformat()}（{start_wd}）"
        else:
            response = f"从 {start_date.isoformat()}（{start_wd}）到 {end_date.isoformat()}（{end_wd}），一共 **{delta}** 天"

        return {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": delta,
            "weekday": end_wd,
            "response": response,
        }

    def _compute_day_of_week(self, date_raw: str) -> Optional[Dict[str, Any]]:
        """Compute the day of week for a date expression."""
        d = self._resolve_date(date_raw)
        if not d:
            return None
        wd = self._get_chinese_weekday(d)
        return {
            "start": d.isoformat(),
            "end": d.isoformat(),
            "days": 0,
            "weekday": wd,
            "response": f"{date_raw} 是 {d.isoformat()}（{wd}）",
        }

    def _compute_duration(self, expression: str) -> Optional[Dict[str, Any]]:
        """Compute a duration expression like '3天后'."""
        if not expression:
            return None
        results = self._extract_durations(expression)
        if not results:
            return None
        item = results[0]
        return {
            "start": item["iso"],
            "end": item["iso"],
            "days": 0,
            "weekday": item["weekday"],
            "response": f"{item['expression']} 是 {item['isoformat']}（{item['weekday']}）" if "isoformat" in item else item["display"],
        }

    def _compute_timezone(self, tz_raw: str) -> Optional[Dict[str, Any]]:
        """Compute timezone conversion."""
        results = self._extract_time_zone_info(tz_raw)
        if not results:
            return None
        item = results[0]
        return {
            "start": "",
            "end": "",
            "days": 0,
            "weekday": "",
            "response": item["display"],
        }

    def _resolve_date(self, raw: str) -> Optional["date"]:
        """Resolve a date expression string to a date object."""
        raw = raw.strip()
        if not raw:
            return None

        # Direct keyword
        keyword_map = {
            "今天": self.today,
            "昨天": self.today - timedelta(days=1),
            "明天": self.today + timedelta(days=1),
            "后天": self.today + timedelta(days=2),
            "前天": self.today - timedelta(days=2),
            "大前天": self.today - timedelta(days=3),
            "大后天": self.today + timedelta(days=3),
        }
        if raw in keyword_map:
            return keyword_map[raw]

        # ISO date
        import re as _re
        iso_m = _re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if iso_m:
            return date.fromisoformat(iso_m.group(0))

        # Chinese date
        cn_m = _re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw)
        if cn_m:
            return date(int(cn_m.group(1)), int(cn_m.group(2)), int(cn_m.group(3)))
        cn_m2 = _re.search(r"(\d{1,2})月(\d{1,2})日", raw)
        if cn_m2:
            return date(self.today.year, int(cn_m2.group(1)), int(cn_m2.group(2)))

        # Weekday-relative (上周五, 下周三)
        weekday_map = {
            "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
            "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
            "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
        }
        prefix_map = {"上": -1, "下": 1, "这": 0}
        for prefix, offset_weeks in prefix_map.items():
            for dow_name, dow_idx in weekday_map.items():
                if raw == f"{prefix}{dow_name}":
                    this_monday = self.today - timedelta(days=self.today.weekday())
                    target_monday = this_monday + timedelta(weeks=offset_weeks)
                    return target_monday + timedelta(days=dow_idx)
            if raw == f"{prefix}周":
                this_monday = self.today - timedelta(days=self.today.weekday())
                return this_monday + timedelta(weeks=offset_weeks)

        # Bare weekday (nearest future)
        for dow_name, dow_idx in weekday_map.items():
            if raw == dow_name:
                diff = (dow_idx - self.today.weekday()) % 7
                return self.today + timedelta(days=diff)

        return None

    def _parse_and_convert(self, message: str) -> TimeResult:
        """Parse time expressions and build formatted output."""
        parsed_times: List[Dict[str, Any]] = []
        lines: List[str] = []
        summaries: List[str] = []

        lines.append("**时间解析结果**\n")

        # 1. Absolute date patterns
        abs_dates = self._extract_absolute_dates(message)
        for item in abs_dates:
            parsed_times.append(item)
            lines.append(f"- **{item['display']}** = {item['iso']} ({item['weekday']})")
            summaries.append(f"{item['display']} 是 {item['iso']} {item['weekday']}")

        # 2. Relative date expressions
        rel_dates = self._extract_relative_dates(message)
        for item in rel_dates:
            parsed_times.append(item)
            lines.append(f"- {item['expression']} = **{item['iso']}** ({item['weekday']})")
            summaries.append(f"{item['expression']} 是 {item['iso']} {item['weekday']}")

        # 3. Duration calculations (e.g., "3天后")
        durations = self._extract_durations(message)
        for item in durations:
            parsed_times.append(item)
            lines.append(f"- {item['expression']} = **{item['iso']}** ({item['weekday']})")
            summaries.append(item['expression'])

        # 4. Day-of-week queries
        dow_result = self._extract_day_of_week_query(message)
        if dow_result:
            parsed_times.append(dow_result)
            lines.append(f"- **{dow_result['display']}**")
            summaries.append(dow_result['display'])

        # 5. Time zone conversions
        tz_results = self._extract_time_zone_info(message)
        for item in tz_results:
            parsed_times.append(item)
            lines.append(f"- {item['display']}")

        # 6. Range/diff calculation (MUST run after dates are extracted)
        range_result = self._extract_range_diff(message, rel_dates, abs_dates)
        if range_result:
            parsed_times.append(range_result)
            lines.append(f"- **{range_result['display']}**")
            summaries.append(range_result['summary'])

        # 7. Countdown / elapsed (single-direction "还有N天")
        countdown = self._extract_countdown(message)
        if countdown:
            parsed_times.append(countdown)
            lines.append(f"- {countdown['display']}")
            summaries.append(countdown['summary'])

        if not parsed_times:
            return TimeResult(
                original_input=message,
                parsed_times=[],
                formatted_output="未能在消息中识别出有效的时间表达式。",
                summary="",
            )

        formatted = "\n".join(lines)
        summary = "；".join(summaries)

        return TimeResult(
            original_input=message,
            parsed_times=parsed_times,
            formatted_output=formatted,
            summary=summary,
        )

    def _extract_absolute_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract absolute date patterns."""
        results: List[Dict[str, Any]] = []

        # Pattern: 2024-01-15 or 2024/01/15
        for m in re.finditer(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                d = date(year, month, day)
                results.append({
                    "type": "absolute",
                    "display": m.group(0),
                    "iso": d.isoformat(),
                    "weekday": self._get_chinese_weekday(d),
                    "timestamp": datetime.combine(d, datetime.min.time()).timestamp(),
                })
            except ValueError:
                pass

        # Pattern: 2024年1月15日 or 2024年01月15日
        for m in re.finditer(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", text):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                d = date(year, month, day)
                results.append({
                    "type": "absolute",
                    "display": m.group(0),
                    "iso": d.isoformat(),
                    "weekday": self._get_chinese_weekday(d),
                    "timestamp": datetime.combine(d, datetime.min.time()).timestamp(),
                })
            except ValueError:
                pass

        # Pattern: 1月15日 or 01月15日 (current year)
        for m in re.finditer(r"(\d{1,2})月(\d{1,2})日?", text):
            month, day = int(m.group(1)), int(m.group(2))
            try:
                d = date(self.today.year, month, day)
                results.append({
                    "type": "absolute",
                    "display": m.group(0),
                    "iso": d.isoformat(),
                    "weekday": self._get_chinese_weekday(d),
                    "timestamp": datetime.combine(d, datetime.min.time()).timestamp(),
                })
            except ValueError:
                pass

        return results

    def _extract_relative_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract Chinese relative date expressions."""
        results: List[Dict[str, Any]] = []
        today = self.today

        relatives = [
            ("今天", today),
            ("昨天", today - timedelta(days=1)),
            ("前天", today - timedelta(days=2)),
            ("大前天", today - timedelta(days=3)),
            ("明天", today + timedelta(days=1)),
            ("后天", today + timedelta(days=2)),
            ("大后天", today + timedelta(days=3)),
            ("上周", today - timedelta(weeks=1)),
            ("这周", today),
            ("下周", today + timedelta(weeks=1)),
            ("上个月", date(self.today.year, self.today.month - 1 if self.today.month > 1 else 12,
                            min(self.today.day, 28))),
            ("这个月", self.today),
            ("下个月", date(self.today.year if self.today.month < 12 else self.today.year + 1,
                            self.today.month + 1 if self.today.month < 12 else 1,
                            min(self.today.day, 28))),
            ("去年", date(self.today.year - 1, self.today.month, self.today.day)),
            ("今年", self.today),
            ("明年", date(self.today.year + 1, self.today.month, self.today.day)),
        ]

        for keyword, d in relatives:
            if keyword in text:
                results.append({
                    "type": "relative",
                    "expression": keyword,
                    "iso": d.isoformat(),
                    "weekday": self._get_chinese_weekday(d),
                    "display": f"{keyword} = {d.isoformat()} {self._get_chinese_weekday(d)}",
                })

        return results

    def _extract_durations(self, text: str) -> List[Dict[str, Any]]:
        """Extract duration expressions like 3天后, 2周前."""
        results: List[Dict[str, Any]] = []

        # N天后 / N天前
        for m in re.finditer(r"(\d+)\s*天\s*(后|前|以前|以后)", text):
            n = int(m.group(1))
            direction = m.group(2)
            d = self.today + timedelta(days=n if direction in ("后", "以后") else -n)
            results.append({
                "type": "duration",
                "expression": m.group(0),
                "iso": d.isoformat(),
                "weekday": self._get_chinese_weekday(d),
                "display": f"{m.group(0)} = {d.isoformat()} {self._get_chinese_weekday(d)}",
            })

        # N周后 / N周前
        for m in re.finditer(r"(\d+)\s*周\s*(后|前|以前|以后)", text):
            n = int(m.group(1))
            direction = m.group(2)
            d = self.today + timedelta(weeks=n if direction in ("后", "以后") else -n)
            results.append({
                "type": "duration",
                "expression": m.group(0),
                "iso": d.isoformat(),
                "weekday": self._get_chinese_weekday(d),
                "display": f"{m.group(0)} = {d.isoformat()} {self._get_chinese_weekday(d)}",
            })

        # N个月后 / N个月前
        for m in re.finditer(r"(\d+)\s*个月\s*(后|前|以前|以后)", text):
            n = int(m.group(1))
            direction = m.group(2)
            month = self.today.month + n if direction in ("后", "以后") else self.today.month - n
            year = self.today.year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            d = date(year, month, min(self.today.day, 28))
            results.append({
                "type": "duration",
                "expression": m.group(0),
                "iso": d.isoformat(),
                "weekday": self._get_chinese_weekday(d),
                "display": f"{m.group(0)} = {d.isoformat()} {self._get_chinese_weekday(d)}",
            })

        # N年前 / N年后
        for m in re.finditer(r"(\d+)\s*年\s*(后|前|以前|以后)", text):
            n = int(m.group(1))
            direction = m.group(2)
            d = date(self.today.year + n if direction in ("后", "以后") else self.today.year - n,
                     self.today.month, self.today.day)
            results.append({
                "type": "duration",
                "expression": m.group(0),
                "iso": d.isoformat(),
                "weekday": self._get_chinese_weekday(d),
                "display": f"{m.group(0)} = {d.isoformat()} {self._get_chinese_weekday(d)}",
            })

        return results

    def _extract_range_diff(self, message: str, rel_dates: List[Dict], abs_dates: List[Dict]) -> Optional[Dict[str, Any]]:
        """Detect range/diff patterns like '上周五到今天一共几天' and calculate difference.

        Strategy: split on known conjunction words, resolve each side, compute diff.
        Must be called AFTER extracting individual dates so partial names (e.g., '周五')
        can be resolved against already-parsed dates.
        """
        # Split on conjunctions that connect two date endpoints
        # Supported patterns:
        #   X到Y一共几天  /  X至Y  /  X和Y相差N天  /  从X到Y
        connectors = ["到", "至"]
        for conn in connectors:
            if conn in message:
                parts = message.split(conn)
                if len(parts) >= 2:
                    # Left: everything before conn
                    left_raw = parts[0].strip()
                    # Right: the text after conn (used for date scanning below)
                    right_text = parts[1].strip()
                    # Extract the FIRST date expression from right_text.
                    # This is more robust than naive stripping because patterns like
                    # "今天一共几天" would fail stripping-based approaches.
                    right_date = None
                    # Try relative keywords first
                    for rel in ["今天", "明天", "昨天", "后天", "前天", "大后天"]:
                        if right_text.startswith(rel):
                            right_date = self._resolve_date_endpoint(rel, rel_dates, abs_dates)
                            break
                    # Try ISO pattern
                    if not right_date:
                        iso_m = re.search(r"\d{4}-\d{2}-\d{2}", right_text)
                        if iso_m:
                            right_date = date.fromisoformat(iso_m.group(0))
                    # Try Chinese date pattern
                    if not right_date:
                        cn_m = re.search(r"\d{4}年\d{1,2}月\d{1,2}日", right_text)
                        if cn_m:
                            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", cn_m.group(0))
                            if m:
                                right_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    # Try N月N日 pattern
                    if not right_date:
                        cn_m2 = re.search(r"\d{1,2}月\d{1,2}日", right_text)
                        if cn_m2:
                            m = re.search(r"(\d{1,2})月(\d{1,2})日", cn_m2.group(0))
                            if m:
                                right_date = date(self.today.year, int(m.group(1)), int(m.group(2)))
                    # Fallback: resolve the full right_raw (e.g. "上周五")
                    if not right_date:
                        right_date = self._resolve_date_endpoint(right_text, rel_dates, abs_dates)

                    left_date = self._resolve_date_endpoint(left_raw, rel_dates, abs_dates)

                    if left_date and right_date:
                        today = self.today
                        wl = self._get_chinese_weekday(left_date)
                        wr = self._get_chinese_weekday(right_date)

                        # Normalize: ensure left <= right for range calculation
                        if left_date > right_date:
                            left_date, right_date = right_date, left_date
                            wl, wr = wr, wl

                        if left_date == right_date:
                            # Same date → count from today
                            diff = abs((left_date - today).days)
                            display = f"{left_date.isoformat()}（{wl}）距离今天还有 **{diff}** 天"
                            summary = f"{left_date.isoformat()} 距离今天还有 {diff} 天"
                            days_val = diff
                        elif right_date <= today:
                            # Both in past (or right is today) → completed range
                            delta = (right_date - left_date).days
                            display = (
                                f"从 {left_date.isoformat()}（{wl}）"
                                f" 到 {right_date.isoformat()}（{wr}）"
                                f" 一共 **{delta}** 天"
                            )
                            summary = f"从 {left_date.isoformat()} 到 {right_date.isoformat()} 一共 {delta} 天"
                            days_val = delta
                        else:
                            # Both in future → countdown to the nearer one
                            diff = (left_date - today).days
                            display = (
                                f"从 {left_date.isoformat()}（{wl}）"
                                f" 到 {right_date.isoformat()}（{wr}），"
                                f"今天（{self._get_chinese_weekday(today)}）距离还有 **{diff}** 天"
                            )
                            summary = f"从今天到 {left_date.isoformat()} 还有 {diff} 天"
                            days_val = diff

                        return {
                            "type": "range_diff",
                            "start": left_date.isoformat(),
                            "end": right_date.isoformat(),
                            "days": days_val,
                            "display": display,
                            "summary": summary,
                        }

        # Fallback: "X和Y相差/相距N天" pattern
        m = re.search(r"(.+?)和(.+?)\s*(?:相差|相距|隔了)\s*(\d+)\s*天", message)
        if m:
            left_date = self._resolve_date_endpoint(m.group(1).strip(), rel_dates, abs_dates)
            right_date = self._resolve_date_endpoint(m.group(2).strip(), rel_dates, abs_dates)
            if left_date and right_date:
                delta = abs((right_date - left_date).days)
                return {
                    "type": "range_diff",
                    "start": left_date.isoformat(),
                    "end": right_date.isoformat(),
                    "days": delta,
                    "display": f"{left_date.isoformat()} 和 {right_date.isoformat()} 相差 {delta} 天",
                    "summary": f"{left_date.isoformat()} 和 {right_date.isoformat()} 相差 {delta} 天",
                }

        return None

    def _resolve_date_endpoint(
        self,
        raw: str,
        rel_dates: List[Dict],
        abs_dates: List[Dict],
    ) -> Optional[date]:
        """Resolve a raw text endpoint (e.g., '上周五', '今天') to a date object.

        Resolution order:
        1. Direct keyword match in already-parsed relative dates
        2. Strip weekday prefix (上周/下周一) -> weekday-only name
        3. Numeric date pattern
        4. Short weekday name (周五, 周三)
        """
        raw = raw.strip()

        # 1. Match against already-parsed relative dates
        for item in rel_dates:
            if item.get("expression") == raw:
                return date.fromisoformat(item["iso"])

        # 2. Numeric ISO date
        iso_m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if iso_m:
            return date.fromisoformat(iso_m.group(0))

        # 3. Chinese date pattern
        cn_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw)
        if cn_m:
            return date(int(cn_m.group(1)), int(cn_m.group(2)), int(cn_m.group(3)))
        cn_m2 = re.search(r"(\d{1,2})月(\d{1,2})日", raw)
        if cn_m2:
            return date(self.today.year, int(cn_m2.group(1)), int(cn_m2.group(2)))

        # 4. 上周/下周/这周 and weekday-named ranges (上周五, 下周三)
        weekday_map = {
            "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
            "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
            "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
        }

        prefix_map = {"上": -1, "下": 1, "这": 0}
        for prefix, offset_weeks in prefix_map.items():
            # Handle "上周" / "下周" / "这周" (whole week reference)
            if raw == f"{prefix}周":
                this_monday = self.today - timedelta(days=self.today.weekday())
                target_monday = this_monday + timedelta(weeks=offset_weeks)
                return target_monday

            for dow_name, dow_idx in weekday_map.items():
                if raw == f"{prefix}{dow_name}":
                    # Monday of the target week
                    this_monday = self.today - timedelta(days=self.today.weekday())
                    target_monday = this_monday + timedelta(weeks=offset_weeks)
                    target_date = target_monday + timedelta(days=dow_idx)
                    return target_date

        # 5. Fallback: bare weekday name (nearest occurrence in future)
        for dow_name, dow_idx in weekday_map.items():
            if raw == dow_name:
                current_dow = self.today.weekday()
                diff = (dow_idx - current_dow) % 7
                return self.today + timedelta(days=diff)

        return None

    def _extract_day_of_week_query(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle day-of-week queries like '今天是星期几'."""
        if any(kw in text for kw in ["星期几", "周几", "礼拜几", "哪天", "今日星期几"]):
            dow = self._get_chinese_weekday(self.today)
            dow_short = self._chinese_weekday_short[self.today.weekday()]
            return {
                "type": "day_of_week",
                "display": f"今天是 {dow_short}（{dow}）",
                "iso": self.today.isoformat(),
                "weekday": dow,
                "weekday_index": self.today.weekday(),
            }

        # 下周一 pattern (always the Monday of next week, not nearest Monday)
        m = re.search(r"下个?(周一|周二|周三|周四|周五|周六|周日|一|二|三|四|五|六|日)", text)
        if m and "下" in text:
            target = m.group(1)
            days_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6,
                        "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6}
            target_idx = days_map.get(target, 0)
            this_monday = self.today - timedelta(days=self.today.weekday())
            next_monday = this_monday + timedelta(weeks=1)
            d = next_monday + timedelta(days=target_idx)
            return {
                "type": "day_of_week",
                "display": f"下{target} = {d.isoformat()} {self._get_chinese_weekday(d)}",
                "iso": d.isoformat(),
                "weekday": self._get_chinese_weekday(d),
                "weekday_index": d.weekday(),
            }

        return None

    def _extract_time_zone_info(self, text: str) -> List[Dict[str, Any]]:
        """Handle timezone conversion queries."""
        results: List[Dict[str, Any]] = []
        lines: List[str] = []

        tz_map = {
            "北京时间": ("Asia/Shanghai", 8),
            "东京时间": ("Asia/Tokyo", 9),
            "纽约时间": ("America/New_York", -5),
            "伦敦时间": ("Europe/London", 0),
            "UTC": ("UTC", 0),
            "GMT": ("GMT", 0),
        }

        for name, (tz_id, offset) in tz_map.items():
            if name in text:
                local_hour = (self.now.hour + offset - 8) % 24
                lines.append(
                    f"- **{name}**：{local_hour:02d}:{self.now.minute:02d}:{self.now.second:02d} "
                    f"（当前 {self.now.strftime('%H:%M:%S')} 北京时间，UTC{'+' if offset >= 0 else ''}{offset}）"
                )
                results.append({
                    "type": "timezone",
                    "display": lines[-1],
                    "tz_name": name,
                    "tz_offset": offset,
                    "local_time": f"{local_hour:02d}:{self.now.minute:02d}",
                })

        return results

    def _extract_countdown(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle countdown / elapsed time calculations."""
        # Pattern: 距离 X 还有多少天
        m = re.search(r"距离(.+?)还有?\s*(\d+)\s*天", text)
        if m:
            target = m.group(1)
            days_left = int(m.group(2))
            target_date = self.today + timedelta(days=days_left)
            return {
                "type": "countdown",
                "display": f"距离「{target}」还有 {days_left} 天 = {target_date.isoformat()} {self._get_chinese_weekday(target_date)}",
                "summary": f"距「{target}」还有 {days_left} 天",
                "target_date": target_date.isoformat(),
                "days_remaining": days_left,
            }

        # Pattern: X 和 Y 相差几天
        m = re.search(r"(.+?)\s*和\s*(.+?)\s*(相差|隔|距离|相隔)\s*(\d+)\s*天", text)
        if m:
            item1, item2, _, days = m.group(1), m.group(2), m.group(3), int(m.group(4))
            d1 = self.today - timedelta(days=days // 2)
            d2 = self.today + timedelta(days=days // 2)
            return {
                "type": "countdown",
                "display": f"{item1}（{d1.isoformat()}）和 {item2}（{d2.isoformat()}）相差 {days} 天",
                "summary": f"{item1} 和 {item2} 相差 {days} 天",
            }

        return None

    def _get_chinese_weekday(self, d: date) -> str:
        """Return Chinese weekday name."""
        return self._chinese_weekday_names[d.weekday()]

    def match(self, message: str) -> bool:
        """Check if message contains time-related expressions."""
        message_lower = message.lower()

        for keyword in self.keywords:
            if keyword in message_lower:
                return True

        time_patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{1,2}月\d{1,2}日",
            r"\d+\s*天\s*(后|前|以后|以前)",
            r"\d+\s*周\s*(后|前|以后|以前)",
            r"\d+\s*个月\s*(后|前|以后|以前)",
            r"\d+\s*年\s*(后|前|以后|以前)",
            r"星期[一二三四五六日]",
            r"周[一二三四五六日]",
            r"礼拜[一二三四五六日天]",
        ]

        for pattern in time_patterns:
            if re.search(pattern, message_lower):
                return True

        return False

    def get_system_prompt(self) -> str:
        """Return the system prompt when this skill is activated."""
        return """你是一个时间转换助手。

当用户提到时间相关的问题时，你应该：
1. 等待 Time Converter Skill 解析出时间信息
2. 基于解析结果，以友好、简洁的方式回复用户
3. 如果用户询问"还有多久到某个日期"，计算并告知剩余天数
4. 如果用户问"是星期几"，直接告知答案
5. 时间信息已由代码解析，不要重复计算

回复风格：
- 简洁明了，直接给出答案
- 可以补充一些有用信息（如是否是周末、距离远近）
- 遇到不确定的时间表达，询问用户确认"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "priority": self.priority,
        }
