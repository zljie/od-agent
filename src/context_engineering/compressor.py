from __future__ import annotations

from collections import Counter
from dataclasses import is_dataclass
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime


def _safe_to_dict(obj: Any) -> Any:
    """Convert a dataclass or Pydantic model to dict recursively."""
    if isinstance(obj, dict):
        return {k: _safe_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_safe_to_dict(item) for item in obj)
    elif is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field_name in getattr(obj, '__dataclass_fields__', {}):
            value = getattr(obj, field_name, None)
            result[field_name] = _safe_to_dict(value)
        return result
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj


def _numeric_summary(values: List[Any]) -> Dict[str, Any]:
    """Return count/sum/avg/min/max for a list of numeric values."""
    numeric_values = []
    for v in values:
        try:
            numeric_values.append(float(v))
        except (TypeError, ValueError):
            continue

    if not numeric_values:
        return {}

    return {
        "count": len(numeric_values),
        "sum": round(sum(numeric_values), 2),
        "avg": round(sum(numeric_values) / len(numeric_values), 2),
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def _top_categorical(values: List[Any], top_n: int = 3) -> List[Dict[str, Any]]:
    """Return top-N most frequent values as [{"value": v, "count": c}]."""
    str_values = [str(v) for v in values]
    counter = Counter(str_values)
    return [{"value": v, "count": c} for v, c in counter.most_common(top_n)]


class ContextCompressor:
    """Compress context for LLM consumption."""

    def compress_dialog_history(
        self, dialog_turns: List[Dict], max_turns: int = 10
    ) -> Dict[str, Any]:
        """
        Compress dialog history to fit within max_turns.
        Strategy: keep first 3 (context), last 2 (recency), middle as summary.
        """
        total_turns = len(dialog_turns)

        if total_turns <= max_turns:
            return {
                "preserved_turns": dialog_turns,
                "summary": None,
                "total_turns": total_turns,
                "compression_ratio": 1.0,
            }

        first_turns = dialog_turns[:3]
        last_turns = dialog_turns[-2:]
        middle_turns = dialog_turns[3:-2]

        omitted_count = len(middle_turns)
        operation_types = []
        for turn in middle_turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if "function" in role or "tool" in role:
                operation_types.append("tool_call")
            elif role == "assistant":
                operation_types.append("assistant_response")
            elif role == "user":
                operation_types.append("user_query")

        summary_parts = [f"{omitted_count} turns omitted"]
        if operation_types:
            type_counts = Counter(operation_types)
            summary_parts.append(
                "Operations: " + ", ".join(
                    f"{k}({v})" for k, v in type_counts.items()
                )
            )

        summary = "; ".join(summary_parts)

        preserved_turns = first_turns + last_turns

        return {
            "preserved_turns": preserved_turns,
            "summary": summary,
            "total_turns": total_turns,
            "compression_ratio": round(len(preserved_turns) / total_turns, 2),
        }

    def compress_business_data(
        self, query_result: Dict, max_items: int = 20
    ) -> Dict[str, Any]:
        """
        Compress query results to fit within max_items.
        Returns statistical summary for truncated lists.
        """
        if not isinstance(query_result, dict):
            query_result = {"items": query_result} if query_result else {}

        items = query_result.get("items", query_result.get("data", query_result))

        if isinstance(items, dict):
            return {
                "items": items,
                "statistical_summary": None,
                "omitted_count": 0,
            }

        if not isinstance(items, list):
            items = [items]

        omitted_count = max(0, len(items) - max_items)

        if omitted_count == 0:
            return {
                "items": items,
                "statistical_summary": None,
                "omitted_count": 0,
            }

        truncated_items = items[:max_items]
        statistical_summary: Dict[str, Any] = {}

        numeric_fields: Dict[str, List[float]] = {}
        categorical_fields: Dict[str, List[Any]] = {}

        for item in truncated_items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_fields.setdefault(key, []).append(float(value))
                elif isinstance(value, str):
                    categorical_fields.setdefault(key, []).append(value)

        for field, values in numeric_fields.items():
            if values:
                statistical_summary[field] = _numeric_summary(values)

        for field, values in categorical_fields.items():
            if values:
                statistical_summary[f"{field}_top"] = _top_categorical(values)

        return {
            "items": truncated_items,
            "statistical_summary": statistical_summary if statistical_summary else None,
            "omitted_count": omitted_count,
        }

    def compress_ontology_graph(
        self, full_graph: Dict, relevant_types: List[str]
    ) -> Dict[str, Any]:
        """
        Filter ontology graph to only relevant_types and their connected edges.
        """
        relevant_set = set(relevant_types)

        objects = full_graph.get("objects", [])
        actions = full_graph.get("actions", [])
        attributes = full_graph.get("attributes", [])
        relationships = full_graph.get("relationships", [])

        filtered_objects = [
            obj for obj in objects
            if obj.get("type") in relevant_set or obj.get("object_type") in relevant_set
        ]
        filtered_actions = [
            act for act in actions
            if act.get("type") in relevant_set or act.get("action_type") in relevant_set
        ]

        filtered_object_ids = {
            obj.get("id") for obj in filtered_objects if obj.get("id")
        }
        filtered_action_ids = {
            act.get("id") for act in filtered_actions if act.get("id")
        }

        connected_ids = filtered_object_ids | filtered_action_ids

        def connects_to_filtered(attr: Dict) -> bool:
            source = attr.get("source") or attr.get("from") or attr.get("object_id", "")
            target = attr.get("target") or attr.get("to") or attr.get("value", "")
            return source in connected_ids or target in connected_ids

        def connects_to_filtered_rel(rel: Dict) -> bool:
            source = rel.get("source") or rel.get("from", "")
            target = rel.get("target") or rel.get("to", "")
            return source in connected_ids or target in connected_ids

        filtered_attributes = [
            attr for attr in attributes
            if connects_to_filtered(attr)
        ]
        filtered_relationships = [
            rel for rel in relationships
            if connects_to_filtered_rel(rel)
        ]

        original_count = (
            len(objects) + len(actions) + len(attributes) + len(relationships)
        )

        return {
            "objects": filtered_objects,
            "attributes": filtered_attributes,
            "actions": filtered_actions,
            "relationships": filtered_relationships,
            "original_count": original_count,
        }
