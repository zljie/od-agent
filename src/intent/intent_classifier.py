"""Rule-based intent classifier with keyword scoring and entity extraction."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .intent_classification import IntentCandidate, IntentClassification


# ─── Date-expression helpers ───────────────────────────────────────────────────


_DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
    r"^\d{4}年\d{1,2}月\d{1,2}日?",
    r"^\d{1,2}月\d{1,2}日?",
    r"^[今昨明前后大][天日后早]",
    r"^[上下]周",
    r"^周[一二三四五六日天]",
    r"^星期[一二三四五六日天]",
    r"^礼拜[一二三四五六日天]",
    r"^本[年月周]",
    r"^当[天周]",
    r"^去年|^今年|^明年",
    r"^上个月|^下个月|^这个月",
]


def _is_valid_date_string(raw: str) -> bool:
    """Return True if raw is a valid standalone date string (no extraction)."""
    if not raw or raw.startswith("每天"):
        return False
    for pat in _DATE_PATTERNS:
        if re.match(pat, raw):
            return True
    # Fallback: need ≥2 date-component characters
    date_chars = frozenset("今明昨前后大上下这那周星期礼拜月日号年")
    return sum(1 for ch in date_chars if ch in raw) >= 2


def _extract_date_from_raw(raw: str) -> Optional[str]:
    """Extract the first date-like substring from raw (e.g. '今天一共' → '今天')."""
    patterns = [
        r"(下周一|下周二|下周三|下周四|下周五|下周六|下周日)",
        r"(上周[一二三四五六日]?)",
        r"(今天|昨天|前天|明天|后天|大后天|大前天)",
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"(\d{4}年\d{1,2}月\d{1,2}日?)",
        r"(\d{1,2}月\d{1,2}日?)",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    return None


def _looks_like_date(raw: str) -> bool:
    """Return True if raw is a date expression.

    Handles embedded dates like '今天一共' by extracting the date portion first.
    """
    raw = raw.strip()
    if not raw or raw.startswith("每天"):
        return False
    # If raw itself is a valid date string, accept it
    if _is_valid_date_string(raw):
        return True
    # Try to extract a date from within raw
    extracted = _extract_date_from_raw(raw)
    if extracted:
        return _is_valid_date_string(extracted)
    return False


# ─── Entity extractors ────────────────────────────────────────────────────────


def extract_range_diff_entities(text: str) -> Optional[Dict[str, str]]:
    """Extract start/end dates from a date-range query.

    Requires a date-range keyword (几天, 共几天, 相差几天, etc.) to be present.
    Uses rfind to find the last connector, so "做到了最终到了拉萨" is skipped.
    Extracts the date portion from noisy strings like '今天一共' and '1月10日相差'.

    Returns {"start": "...", "end": "..."} or None.
    """
    # Longer suffixes first so "相差几天" is tried before "几天"
    suffixes = sorted(
        ["相差几天", "相距几天", "相隔几天", "共几天",
         "隔几天", "有几周", "共几周", "几天"],
        key=len, reverse=True
    )

    for suffix in suffixes:
        kw_pos = text.rfind(suffix)       # Last occurrence
        if kw_pos == -1:
            continue

        before = text[:kw_pos].strip()    # Everything before "几天"

        # Pattern A: X到Y / X至Y
        for conn in ("到", "至"):
            cp = before.rfind(conn)      # Last connector
            if cp == -1:
                continue
            start = before[:cp].strip().lstrip("从").strip()
            raw_end = before[cp + len(conn):].strip()
            # Extract just the date from the noisy end string
            end = _extract_date_from_raw(raw_end)
            if not end:
                end = raw_end
            if _looks_like_date(start) and _looks_like_date(end):
                return {"start": start, "end": end}

        # Pattern B: X和Y相差几天
        pattern = r"(.+?)\s*和\s*(.+?)\s*" + re.escape(suffix)
        m = re.search(pattern, text)
        if m:
            start = m.group(1).strip()
            raw_end = m.group(2).strip()
            end = _extract_date_from_raw(raw_end) or raw_end
            if _looks_like_date(start) and _looks_like_date(end):
                return {"start": start, "end": end}

    return None


def extract_math_entities(text: str) -> Optional[Dict[str, str]]:
    """Extract math expression from text.

    Scores each sentence by: num_count * 2 + kw_count.
    Returns the best sentence that has both numbers and math keywords.

    Returns {"expression": "..."} or None.
    """
    math_keywords = [
        "平均", "每天", "多远", "除以", "除", "加", "减", "乘以", "乘",
        "等于", "多少", "+", "-", "*", "/", "×", "÷",
        "km", "公里", "元", "kg", "米", "个",
    ]
    num_pattern = r"(-?\d+(?:\.\d+)?)"

    best_seg, best_score = None, 0

    for seg in re.split(r"[。；\n]", text):
        seg = seg.strip()
        if len(seg) < 3:
            continue
        nums = re.findall(num_pattern, seg)
        if not nums:
            continue
        kw_count = sum(1 for kw in math_keywords if kw in seg)
        if kw_count == 0:
            continue
        score = len(nums) * 2 + kw_count
        if score > best_score:
            best_score = score
            best_seg = seg

    if best_seg:
        return {"expression": best_seg}

    # Fallback: full text
    if (re.search(num_pattern, text) and
            any(kw in text for kw in math_keywords)):
        return {"expression": text}

    return None


def extract_day_of_week_entities(text: str) -> Optional[Dict[str, str]]:
    """Extract the date from a day-of-week query.

    Looks for keywords like "今天是星期几", "今天周几", "今日星期几",
    "下周一", "下周三" etc. and returns {"date": "..."} if found.

    Returns None if no recognizable date expression is found.
    """
    # Normalize: strip common trailing/leading noise
    cleaned = text.strip().rstrip("？?。").strip()

    # Try full keyword patterns first
    day_kw_patterns = [
        "今日星期几", "今天是星期几", "今天是周几", "今天是礼拜几",
        "今天周几", "今天礼拜几",
    ]
    for kw in day_kw_patterns:
        if kw in cleaned:
            return {"date": "今天", "operation": "day_of_week"}

    # Try "X是星期几" / "X星期几" patterns (e.g. "明天是星期几", "后天星期几")
    relative_kw = ["今天", "明天", "后天", "昨天", "前天", "大后天", "大前天"]
    for rel in relative_kw:
        suffix_opts = ["是星期几", "是周几", "是礼拜几", "星期几", "周几", "礼拜几"]
        for suffix in suffix_opts:
            if cleaned == f"{rel}{suffix}":
                return {"date": rel, "operation": "day_of_week"}

    # "X是星期几" pattern: "明后天是星期几"
    for rel in relative_kw:
        if cleaned.startswith(rel):
            rest = cleaned[len(rel):]
            if re.match(r"^是?(星期|周|礼拜)[几天]?$", rest):
                return {"date": rel, "operation": "day_of_week"}

    # "下周X是几号" / "下周X星期几" — redirect to day_of_week
    # e.g. "下周三是几号" → date="下周三"
    next_week_pattern = re.compile(
        r"^(下[周]?)?周?([一二三四五六日天])是?(几号|星期几|周几|礼拜几)$"
    )
    m = next_week_pattern.match(cleaned)
    if m:
        prefix = m.group(1) or ""
        dow = m.group(2)
        suffix = m.group(3)
        date_str = (prefix + "周" + dow).replace("下周周", "下")
        return {"date": date_str, "operation": "day_of_week"}

    # "今天是几号" / "今天几号" — date query, not dow, but still time-related
    # Include in day_of_week context since it goes to Time Converter
    if re.match(r"^(今|明|昨|前|大前|大后)天(是)?(几号|哪一天|什么日期)$", cleaned):
        date_match = re.match(r"^((今|明|昨|前|大前|大后)天)", cleaned)
        if date_match:
            return {"date": date_match.group(1), "operation": "day_of_week"}

    # Bare weekday: "下周一", "下周三", "周五", "礼拜天"
    bare_pattern = re.compile(
        r"^(上|下|这)?(周|星期|礼拜)?([一二三四五六日天])$"
    )
    m = bare_pattern.match(cleaned)
    if m:
        return {"date": cleaned, "operation": "day_of_week"}

    return None


# ─── IntentRule ────────────────────────────────────────────────────────────────


@dataclass
class IntentRule:
    """A single intent classification rule."""

    intent_type: str
    keywords: List[str]
    keyword_weights: Dict[str, float] = field(default_factory=dict)
    entity_extractor: Optional[callable] = None
    description: str = ""

    def keyword_score(self, text: str) -> float:
        score = 0.0
        text_lower = text.lower()
        for kw in self.keywords:
            if kw.lower() in text_lower:
                score += self.keyword_weights.get(kw, 1.0)
        return score

    def extract_entities(self, text: str) -> Optional[Dict[str, str]]:
        if self.entity_extractor is None:
            return None
        try:
            return self.entity_extractor(text)
        except Exception:
            return None


@dataclass
class IntentClassifierConfig:
    """Container for all intent rules."""

    rules: List[IntentRule] = field(default_factory=list)

    def intent_by_type(self, intent_type: str) -> Optional[IntentRule]:
        for r in self.rules:
            if r.intent_type == intent_type:
                return r
        return None


class RuleBasedIntentClassifier:
    """Rule-based intent classifier.

    A rule fires only when BOTH keyword match AND entity extraction succeed.
    """

    def __init__(self, config: Optional[IntentClassifierConfig] = None):
        self._config = config or IntentClassifierConfig()

    def classify(self, text: str) -> IntentClassification:
        """Classify user input and extract entity slots.

        A rule fires when:
        - It has at least one keyword in the text, AND
        - Either it has no entity_extractor (keyword-only rule), OR
          its entity_extractor returns non-empty entities.
        """
        candidates: List[Tuple[str, float, Dict[str, str]]] = []

        for rule in self._config.rules:
            kw_score = rule.keyword_score(text)
            if kw_score <= 0:
                continue

            # Rules without an entity_extractor fire on keyword match alone
            if rule.entity_extractor is None:
                candidates.append((rule.intent_type, kw_score, {}))
                continue

            # Rules with an entity_extractor require successful extraction
            entities = rule.extract_entities(text)
            if not entities:
                continue

            candidates.append((rule.intent_type, kw_score, entities))

        if not candidates:
            return IntentClassification.UNKNOWN

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_type, top_score, top_entities = candidates[0]

        intent_candidates = [
            IntentCandidate(intent_type=t, confidence=min(1.0, s), entities=e)
            for t, s, e in candidates[:3]
        ]

        return IntentClassification(
            intent_type=top_type,
            confidence=min(1.0, top_score),
            entities=top_entities,
            candidates=intent_candidates,
        )

    def classify_multi(self, text: str) -> List[IntentClassification]:
        """Classify all intents in a single message.

        Two-pass approach:
          1. Full-text (so date ranges spanning punctuation work)
          2. Per-sentence (for multi-intent queries like "上周五...，数学...")

        Each intent type appears at most once.
        """
        results_by_type: Dict[str, Tuple[str, float, Dict[str, str]]] = {}

        # Pass 1: full text — rules without extractor fire on keyword alone
        for rule in self._config.rules:
            kw_score = rule.keyword_score(text)
            if kw_score <= 0:
                continue
            if rule.entity_extractor is None:
                results_by_type[rule.intent_type] = (rule.intent_type, kw_score, {})
            else:
                entities = rule.extract_entities(text)
                if entities:
                    results_by_type[rule.intent_type] = (rule.intent_type, kw_score, entities)

        # Pass 2: per-sentence for unmatched types
        for seg in re.split(r"[。；\n]", text):
            seg = seg.strip()
            if len(seg) < 3:
                continue
            for rule in self._config.rules:
                if rule.intent_type in results_by_type:
                    continue
                kw_score = rule.keyword_score(seg)
                if kw_score <= 0:
                    continue
                if rule.entity_extractor is None:
                    results_by_type[rule.intent_type] = (rule.intent_type, kw_score, {})
                else:
                    entities = rule.extract_entities(seg)
                    if entities:
                        results_by_type[rule.intent_type] = (rule.intent_type, kw_score, entities)

        if not results_by_type:
            return [IntentClassification.UNKNOWN]

        all_candidates = sorted(results_by_type.values(), key=lambda x: x[1], reverse=True)

        return [
            IntentClassification(
                intent_type=t,
                confidence=min(1.0, s),
                entities=e,
                candidates=[
                    IntentCandidate(intent_type=tc, confidence=min(1.0, sc), entities=ec)
                    for tc, sc, ec in all_candidates
                ],
            )
            for t, s, e in all_candidates
        ]

    def classify_with_context(
        self,
        text: str,
        temporal_context: Optional["TemporalContext"] = None,
    ) -> List[IntentClassification]:
        """Classify intents with temporal context injected.

        This is Phase 1 of the pipeline. When a TemporalContext is provided,
        the planner uses its resolved dates to build dependency parameters
        (e.g. injecting days count from the date range into the task plan).

        The returned list includes all matched intent types (multi-intent).

        Args:
            text: Raw user message
            temporal_context: Phase-0 TemporalContext (optional)

        Returns:
            List of IntentClassification, sorted by confidence descending.
        """
        # Pass original text to classify_multi — temporal dates are used
        # by the Planner phase, not here. Substituting ISO dates in the text
        # risks false-positive math classification (e.g. "2026-05-22" looks
        # like an arithmetic expression).
        return self.classify_multi(text)
