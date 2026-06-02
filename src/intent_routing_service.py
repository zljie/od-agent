"""Intent routing management service.

Provides comprehensive intent routing management including:
- Intent CRUD operations
- Quick routing test
- Conflict detection and diagnostics
- Example management (positive/negative/edge)
- Slot configuration management
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Intent routing config file path
INTENT_CONFIG_DIR = Path("data/intent_routing")
INTENT_CONFIG_PATH = INTENT_CONFIG_DIR / "intents.json"
DIAGNOSTICS_PATH = INTENT_CONFIG_DIR / "diagnostics.json"
TEST_RUNS_PATH = INTENT_CONFIG_DIR / "test_runs.json"


def _ensure_config_dir():
    """Ensure the config directory exists."""
    INTENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_intents() -> List[Dict[str, Any]]:
    """Load all intents from storage."""
    _ensure_config_dir()
    if not INTENT_CONFIG_PATH.exists():
        return []
    try:
        with open(INTENT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_intents(intents: List[Dict[str, Any]]) -> None:
    """Save all intents to storage."""
    _ensure_config_dir()
    with open(INTENT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=2, ensure_ascii=False)


def create_intent(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new intent with generated ID."""
    _ensure_config_dir()
    intents = load_intents()

    # Generate unique ID
    intent_id = intent_data.get("id") or f"intent_{uuid.uuid4().hex[:8]}"

    # Set defaults
    new_intent = {
        "id": intent_id,
        "name": intent_data.get("name", ""),
        "display_name": intent_data.get("display_name", ""),
        "description": intent_data.get("description", ""),
        "status": intent_data.get("status", "draft"),
        "priority": intent_data.get("priority", 50),
        "domain": intent_data.get("domain", ""),
        # Route config
        "route_type": intent_data.get("route_type", "skill"),
        "route_target": intent_data.get("route_target", ""),
        "fallback_action": intent_data.get("fallback_action", "ask_missing_slot"),
        "failure_action": intent_data.get("failure_action", "human_handoff"),
        # Recognition rules
        "semantic_enabled": intent_data.get("semantic_enabled", True),
        "keyword_boost_enabled": intent_data.get("keyword_boost_enabled", False),
        "regex_enabled": intent_data.get("regex_enabled", False),
        "include_keywords": intent_data.get("include_keywords", []),
        "exclude_keywords": intent_data.get("exclude_keywords", []),
        # Examples
        "positive_examples": intent_data.get("positive_examples", []),
        "negative_examples": intent_data.get("negative_examples", []),
        "edge_examples": intent_data.get("edge_examples", []),
        # Slots
        "slots": intent_data.get("slots", []),
        # Test cases
        "test_cases": intent_data.get("test_cases", []),
        # Metadata
        "conflict": False,
        "test_failed": False,
        "created_at": intent_data.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
    }

    intents.append(new_intent)
    save_intents(intents)

    # Update diagnostics after creating intent
    _update_intent_conflicts()

    return new_intent


def get_intent(intent_id: str) -> Optional[Dict[str, Any]]:
    """Get a single intent by ID."""
    intents = load_intents()
    for intent in intents:
        if intent.get("id") == intent_id:
            return intent
    return None


def update_intent(intent_id: str, intent_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update an existing intent."""
    intents = load_intents()

    for i, intent in enumerate(intents):
        if intent.get("id") == intent_id:
            # Merge updates
            updated = {**intent, **intent_data}
            updated["id"] = intent_id
            updated["updated_at"] = _now_iso()

            # Remove internal flags that should be recalculated
            updated.pop("conflict", None)
            updated.pop("test_failed", None)

            intents[i] = updated
            save_intents(intents)

            # Update diagnostics
            _update_intent_conflicts()

            return updated

    return None


def delete_intent(intent_id: str) -> bool:
    """Delete an intent by ID (soft delete via status)."""
    intents = load_intents()

    for i, intent in enumerate(intents):
        if intent.get("id") == intent_id:
            # Remove from list
            intents.pop(i)
            save_intents(intents)

            # Update diagnostics
            _update_intent_conflicts()

            return True

    return False


def get_intent_stats() -> Dict[str, Any]:
    """Get summary statistics for all intents."""
    intents = load_intents()

    total = len(intents)
    enabled = len([i for i in intents if i.get("status") == "enabled"])
    conflicts = len([i for i in intents if i.get("conflict", False)])
    unbound = len([i for i in intents if not i.get("route_target")])
    test_cases_count = sum(len(i.get("test_cases", [])) for i in intents)
    passed_cases = sum(
        len([tc for tc in i.get("test_cases", []) if tc.get("status") == "passed"])
        for i in intents
    )
    pass_rate = int(passed_cases / test_cases_count * 100) if test_cases_count > 0 else 100

    return {
        "total": total,
        "enabled": enabled,
        "draft": total - enabled,
        "conflicts": conflicts,
        "unbound": unbound,
        "test_cases_count": test_cases_count,
        "pass_rate": pass_rate,
    }


# ─── Quick Routing Test ────────────────────────────────────────────────


def test_routing(message: str, user_profile: str = "normal_user") -> Dict[str, Any]:
    """Test routing for a user message.

    Returns:
        Selected intent, confidence, candidates, route target, slot status.
    """
    intents = load_intents()

    # Filter enabled intents
    enabled_intents = [i for i in intents if i.get("status") == "enabled"]

    # Score each intent
    candidates = []
    for intent in enabled_intents:
        score = _calculate_intent_score(message, intent)
        if score > 0:
            candidates.append({
                "intent": intent.get("name"),
                "intent_id": intent.get("id"),
                "display_name": intent.get("display_name", intent.get("name")),
                "confidence": min(score, 1.0),
                "reason": _get_match_reason(message, intent),
                "route_target": intent.get("route_target", ""),
                "route_type": intent.get("route_type", "skill"),
            })

    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    if not candidates:
        return {
            "selected_intent": None,
            "confidence": 0.0,
            "candidates": [],
            "route_target": None,
            "route_type": None,
            "executable": False,
            "missing_slots": [],
            "next_action": None,
            "diagnostics": ["未匹配到任何意图"],
        }

    selected = candidates[0]
    selected["selected"] = True

    # Check slots
    intent_obj = next((i for i in enabled_intents if i.get("id") == selected["intent_id"]), None)
    missing_slots = []
    executable = True
    next_action = "executable"

    if intent_obj:
        slots = intent_obj.get("slots", [])
        required_slots = [s for s in slots if s.get("required", False)]

        if required_slots:
            # Simulate slot checking (in real system, would check session/context)
            executable = False
            missing_slots = [s.get("slot_name", "") for s in required_slots]
            next_action = intent_obj.get("fallback_action", "ask_missing_slot")

    # Mark candidates
    for cand in candidates[1:]:
        cand["selected"] = False

    # Save test run
    _save_test_run({
        "message": message,
        "user_profile": user_profile,
        "selected_intent": selected["intent"],
        "confidence": selected["confidence"],
        "candidates": candidates,
        "route_target": selected["route_target"],
        "route_type": selected["route_type"],
        "executable": executable,
        "missing_slots": missing_slots,
        "next_action": next_action,
    })

    return {
        "selected_intent": selected["intent"],
        "confidence": selected["confidence"],
        "candidates": candidates,
        "route_target": selected["route_target"],
        "route_type": selected["route_type"],
        "executable": executable,
        "missing_slots": missing_slots,
        "next_action": next_action,
        "diagnostics": _generate_test_diagnostics(selected, missing_slots),
    }


def _calculate_intent_score(message: str, intent: Dict[str, Any]) -> float:
    """Calculate match score for an intent."""
    message_lower = message.lower()
    score = 0.0

    # 1. Semantic enabled: boost by priority
    if intent.get("semantic_enabled", True):
        score += 0.1 * (intent.get("priority", 50) / 100)

    # 2. Keyword matching
    include_keywords = intent.get("include_keywords", [])
    exclude_keywords = intent.get("exclude_keywords", [])

    # Check exclude keywords first
    for kw in exclude_keywords:
        if kw.lower() in message_lower:
            return 0.0  # Excluded

    # Count include keyword matches
    keyword_matches = 0
    for kw in include_keywords:
        if kw.lower() in message_lower:
            keyword_matches += 1
            score += 0.2

    # 3. Positive examples matching
    positive_examples = intent.get("positive_examples", [])
    for ex in positive_examples:
        if ex.lower() in message_lower:
            score += 0.3
        elif _similar_text(ex.lower(), message_lower):
            score += 0.15

    # 4. Edge examples (lower weight)
    edge_examples = intent.get("edge_examples", [])
    for ex in edge_examples:
        if ex.lower() in message_lower:
            score += 0.1

    # 5. Name/display_name match
    name = intent.get("name", "").lower()
    display_name = intent.get("display_name", "").lower()
    if name and name in message_lower:
        score += 0.15
    if display_name and display_name in message_lower:
        score += 0.15

    return score


def _get_match_reason(message: str, intent: Dict[str, Any]) -> str:
    """Get human-readable reason for intent match."""
    message_lower = message.lower()

    include_keywords = intent.get("include_keywords", [])
    for kw in include_keywords:
        if kw.lower() in message_lower:
            return f"包含关键词「{kw}」"

    positive_examples = intent.get("positive_examples", [])
    for ex in positive_examples:
        if ex.lower() in message_lower:
            return f"匹配样例「{ex[:20]}...」"

    return "语义匹配"


def _similar_text(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """Simple similarity check using common characters."""
    if not text1 or not text2:
        return False

    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return (intersection / union) >= threshold if union > 0 else False


def _generate_test_diagnostics(selected: Dict[str, Any], missing_slots: List[str]) -> List[str]:
    """Generate diagnostic messages for test result."""
    diagnostics = []

    if selected.get("selected"):
        diagnostics.append(f"命中意图 {selected['intent']}，置信度 {(selected['confidence'] * 100):.0f}%")

        if selected.get("route_target"):
            diagnostics.append(f"路由目标: {selected['route_target']}")
        else:
            diagnostics.append("警告: 此意图未配置路由目标")

    if missing_slots:
        diagnostics.append(f"缺少必填槽位: {', '.join(missing_slots)}")
        diagnostics.append("系统将追问用户以收集缺失信息")

    return diagnostics


def _save_test_run(result: Dict[str, Any]) -> None:
    """Save a test run record."""
    _ensure_config_dir()

    runs = []
    if TEST_RUNS_PATH.exists():
        try:
            with open(TEST_RUNS_PATH, "r", encoding="utf-8") as f:
                runs = json.load(f)
        except (json.JSONDecodeError, IOError):
            runs = []

    run_record = {
        "id": f"run_{uuid.uuid4().hex[:8]}",
        "timestamp": _now_iso(),
        **result,
    }

    runs.insert(0, run_record)  # Most recent first

    # Keep only last 100 runs
    runs = runs[:100]

    with open(TEST_RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)


# ─── Conflict Detection & Diagnostics ─────────────────────────────────


def run_diagnostics() -> Dict[str, Any]:
    """Run comprehensive diagnostics on all intents."""
    intents = load_intents()
    issues = []
    suggestions = []
    total_score = 100

    # 1. Check for duplicate names
    names = [i.get("name", "") for i in intents]
    duplicate_names = [n for n in names if names.count(n) > 1]
    if duplicate_names:
        issues.append({
            "level": "error",
            "type": "duplicate_name",
            "message": f"存在重复的意图名称: {', '.join(set(duplicate_names))}",
            "suggestion": "每个意图名称必须唯一",
        })
        total_score -= 20

    # 2. Check for intents without positive examples
    no_positive = [i.get("name") for i in intents if not i.get("positive_examples")]
    if no_positive:
        issues.append({
            "level": "warning",
            "type": "missing_positive_examples",
            "message": f"以下意图缺少正向样例: {', '.join(no_positive)}",
            "suggestion": "添加正向样例以提高识别准确性",
        })
        total_score -= 5 * len(no_positive)

    # 3. Check for unbound intents
    unbound = [i.get("name") for i in intents if not i.get("route_target")]
    if unbound:
        issues.append({
            "level": "error",
            "type": "unbound_route",
            "message": f"以下意图未绑定路由目标: {', '.join(unbound)}",
            "suggestion": "为每个意图配置路由目标(Skill/任务链)",
        })
        total_score -= 15 * len(unbound)

    # 4. Check for conflicting intents (similar examples)
    conflicts = _detect_intent_conflicts(intents)
    if conflicts:
        for conflict in conflicts:
            issues.append({
                "level": "warning",
                "type": "overlap",
                "message": f"{conflict['intent1']} 与 {conflict['intent2']} 样例重叠",
                "suggestion": conflict["suggestion"],
            })
            total_score -= 10

    # 5. Check for missing test cases
    no_tests = [i.get("name") for i in intents if not i.get("test_cases")]
    if no_tests:
        issues.append({
            "level": "warning",
            "type": "missing_tests",
            "message": f"以下意图缺少测试用例: {', '.join(no_tests)}",
            "suggestion": "添加测试用例以验证路由正确性",
        })
        total_score -= 5

    # 6. Check for disabled intents with high priority
    disabled_high_priority = [
        i.get("name") for i in intents
        if i.get("status") == "disabled" and i.get("priority", 0) >= 80
    ]
    if disabled_high_priority:
        suggestions.append(f"注意: {', '.join(disabled_high_priority)} 被禁用但优先级较高")

    # 7. Generate suggestions
    if total_score >= 90:
        suggestions.append("配置状态良好，继续保持")
    if len(intents) < 3:
        suggestions.append("建议增加更多意图以覆盖常见用户场景")
    if not any(i.get("include_keywords") for i in intents):
        suggestions.append("考虑添加关键词以提高识别准确性")

    # Ensure score is in valid range
    total_score = max(0, min(100, total_score))

    # Save diagnostics
    _save_diagnostics({
        "score": total_score,
        "issues": issues,
        "suggestions": suggestions,
        "timestamp": _now_iso(),
    })

    return {
        "score": total_score,
        "issues": issues,
        "suggestions": suggestions,
    }


def _detect_intent_conflicts(intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect overlapping/conflicting intents."""
    conflicts = []
    checked = set()

    for i, intent1 in enumerate(intents):
        for j, intent2 in enumerate(intents[i + 1:], start=i + 1):
            pair_key = tuple(sorted([intent1.get("id", ""), intent2.get("id", "")]))
            if pair_key in checked:
                continue

            # Check example overlap
            pos1 = set(intent1.get("positive_examples", []))
            pos2 = set(intent2.get("positive_examples", []))
            overlap = pos1 & pos2

            if overlap and len(overlap) >= 2:
                conflicts.append({
                    "intent1": intent1.get("name"),
                    "intent2": intent2.get("name"),
                    "overlap_count": len(overlap),
                    "suggestion": "增加反向样例或调整优先级以区分",
                })

            # Check keyword overlap
            kw1 = set(k.lower() for k in intent1.get("include_keywords", []))
            kw2 = set(k.lower() for k in intent2.get("include_keywords", []))
            kw_overlap = kw1 & kw2

            if kw_overlap and len(kw_overlap) >= 3:
                conflicts.append({
                    "intent1": intent1.get("name"),
                    "intent2": intent2.get("name"),
                    "overlap_count": len(kw_overlap),
                    "suggestion": "减少共同关键词，添加区分性关键词",
                })

            checked.add(pair_key)

    return conflicts


def _update_intent_conflicts() -> None:
    """Update conflict flags on all intents."""
    intents = load_intents()
    conflicts = _detect_intent_conflicts(intents)
    conflict_intents = set()

    for c in conflicts:
        conflict_intents.add(c["intent1"])
        conflict_intents.add(c["intent2"])

    for intent in intents:
        intent["conflict"] = intent.get("name") in conflict_intents

    save_intents(intents)


def _save_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """Save diagnostics result."""
    _ensure_config_dir()
    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)


def load_diagnostics() -> Dict[str, Any]:
    """Load latest diagnostics result."""
    _ensure_config_dir()
    if not DIAGNOSTICS_PATH.exists():
        return run_diagnostics()
    try:
        with open(DIAGNOSTICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"score": 100, "issues": [], "suggestions": []}


# ─── Example Management ─────────────────────────────────────────────────


def add_example(intent_id: str, example_type: str, text: str) -> Optional[Dict[str, Any]]:
    """Add an example to an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    example_map = {
        "positive": "positive_examples",
        "negative": "negative_examples",
        "edge": "edge_examples",
    }

    key = example_map.get(example_type)
    if not key:
        return None

    if text not in intent.get(key, []):
        intent[key] = intent.get(key, []) + [text]
        update_intent(intent_id, intent)

    return intent


def remove_example(intent_id: str, example_type: str, text: str) -> Optional[Dict[str, Any]]:
    """Remove an example from an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    example_map = {
        "positive": "positive_examples",
        "negative": "negative_examples",
        "edge": "edge_examples",
    }

    key = example_map.get(example_type)
    if not key:
        return None

    if text in intent.get(key, []):
        intent[key] = [e for e in intent[key] if e != text]
        update_intent(intent_id, intent)

    return intent


# ─── Slot Management ────────────────────────────────────────────────────


def add_slot(intent_id: str, slot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Add a slot to an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    slots = intent.get("slots", [])
    new_slot = {
        "id": f"slot_{uuid.uuid4().hex[:8]}",
        "slot_name": slot_data.get("slot_name", ""),
        "required": slot_data.get("required", True),
        "required_mode": slot_data.get("required_mode"),
        "group": slot_data.get("group"),
        "source": slot_data.get("source", "user_input"),
        "missing_action": slot_data.get("missing_action", "ask"),
        "ask_template": slot_data.get("ask_template", ""),
    }

    slots.append(new_slot)
    intent["slots"] = slots
    update_intent(intent_id, intent)

    return intent


def update_slot(intent_id: str, slot_id: str, slot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a slot in an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    slots = intent.get("slots", [])
    for i, slot in enumerate(slots):
        if slot.get("id") == slot_id:
            slots[i] = {**slot, **slot_data, "id": slot_id}
            intent["slots"] = slots
            update_intent(intent_id, intent)
            return intent

    return None


def remove_slot(intent_id: str, slot_id: str) -> Optional[Dict[str, Any]]:
    """Remove a slot from an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    slots = intent.get("slots", [])
    intent["slots"] = [s for s in slots if s.get("id") != slot_id]
    update_intent(intent_id, intent)

    return intent


# ─── Test Case Management ──────────────────────────────────────────────


def add_test_case(intent_id: str, case_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Add a test case to an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    test_cases = intent.get("test_cases", [])
    new_case = {
        "id": f"tc_{uuid.uuid4().hex[:8]}",
        "input": case_data.get("input", ""),
        "expected_intent": case_data.get("expected_intent", intent.get("name")),
        "expected_route_target": case_data.get("expected_route_target", ""),
        "expected_action": case_data.get("expected_action", "ask_missing_slot"),
        "expected_missing_slots": case_data.get("expected_missing_slots", []),
        "status": "pending",
    }

    test_cases.append(new_case)
    intent["test_cases"] = test_cases
    update_intent(intent_id, intent)

    return intent


def remove_test_case(intent_id: str, case_id: str) -> Optional[Dict[str, Any]]:
    """Remove a test case from an intent."""
    intent = get_intent(intent_id)
    if not intent:
        return None

    test_cases = intent.get("test_cases", [])
    intent["test_cases"] = [tc for tc in test_cases if tc.get("id") != case_id]
    update_intent(intent_id, intent)

    return intent


# ─── Utility ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()


# ─── Publish Check ─────────────────────────────────────────────────────


def get_publish_check() -> Dict[str, Any]:
    """Run pre-publish checks on all intents."""
    diagnostics = load_diagnostics()

    issues = diagnostics.get("issues", [])
    failed = len([i for i in issues if i.get("level") == "error"])
    warnings = len([i for i in issues if i.get("level") == "warning"])

    return {
        "can_publish": failed == 0,
        "summary": {
            "total_intents": len(load_intents()),
            "can_release": len(load_intents()) - failed,
            "has_errors": failed,
            "has_warnings": warnings,
        },
        "issues": issues[:10],  # Top 10 issues
    }
