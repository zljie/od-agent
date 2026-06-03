"""Context Selector — selects relevant context per pipeline step."""

from typing import Any, Dict, List, Optional


class ContextSelector:
    """Selects relevant context from retrieved candidates based on step purpose."""

    # Keys to always include for each step
    STEP_INCLUDES: Dict[str, List[str]] = {
        "intent_recognition": [
            "user_raw_input",
            "normalized_input",
            "preprocessing_result",
            "ontology_summary",
            "dialog_context_summary",
        ],
        "ontology_grounding": [
            "detected_intent",
            "candidate_objects",
            "object_aliases",
            "attributes",
            "relationships",
            "actions",
            "rules",
            "dynamic_terms",
        ],
        "task_planning": [
            "ontology_grounding_result",
            "task_templates",
            "available_connectors",
            "required_rules",
            "risk_policy",
        ],
        "execution": [
            "task_plan",
            "current_step",
            "intermediate_results",
            "connector_error_info",
            "fallback_policy",
        ],
        "response_generation": [
            "execution_result",
            "business_explanation",
            "rule_results",
            "next_actions",
            "response_style",
        ],
    }

    # Keys to always exclude for each step
    STEP_EXCLUDES: Dict[str, List[str]] = {
        "intent_recognition": [
            "full_business_data",
            "connector_results",
            "unrelated_history",
        ],
        "ontology_grounding": [
            "history_execution_logs",
            "unrelated_business_data",
        ],
        "task_planning": [
            "long_user_chitchat",
            "unrelated_context",
        ],
        "execution": [
            "full_ontology",
            "unrelated_candidates",
        ],
        "response_generation": [
            "unrelated_candidate_paths",
            "internal_debug_info",
        ],
    }

    def select_for_intent_recognition(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Select context for Step 1.
        Include: user raw input, normalized input, preprocessing result,
                 ontology summary, dialog context summary
        Exclude: full business data, connector results, unrelated history
        """
        return self._select(candidates, "intent_recognition")

    def select_for_ontology_grounding(
        self, candidates: Dict[str, Any], intent_result: Any
    ) -> Dict[str, Any]:
        """Select context for Step 2.
        Include: detected intent, candidate objects, object aliases,
                 attributes, relationships, actions, rules, dynamic terms
        Exclude: history execution logs, unrelated business data
        """
        selected = self._select(candidates, "ontology_grounding")
        if intent_result is not None:
            selected["intent_result"] = self._safe_to_dict(intent_result)
        return selected

    def select_for_task_planning(
        self, candidates: Dict[str, Any], intent: Any, ontology: Any
    ) -> Dict[str, Any]:
        """Select context for Step 3.
        Include: ontology grounding result, task templates,
                 available connectors, required rules, risk policy
        Exclude: long user chitchat, unrelated context
        """
        selected = self._select(candidates, "task_planning")
        if intent is not None:
            selected["intent"] = self._safe_to_dict(intent)
        if ontology is not None:
            selected["ontology"] = self._safe_to_dict(ontology)
        return selected

    def select_for_execution(
        self, candidates: Dict[str, Any], plan: Any, current_step: int
    ) -> Dict[str, Any]:
        """Select context for Step 4.
        Include: task plan, current step, intermediate results,
                 connector error info, fallback policy
        Exclude: full ontology, unrelated candidates
        """
        selected = self._select(candidates, "execution")
        if plan is not None:
            selected["plan"] = self._safe_to_dict(plan)
        selected["current_step"] = current_step
        return selected

    def select_for_response_generation(
        self, candidates: Dict[str, Any], exec_result: Any
    ) -> Dict[str, Any]:
        """Select context for Step 5.
        Include: execution result, business explanation, rule results,
                 next actions, response style
        Exclude: unrelated candidate paths, internal debug info
        """
        selected = self._select(candidates, "response_generation")
        if exec_result is not None:
            selected["exec_result"] = self._safe_to_dict(exec_result)
        return selected

    def _select(self, candidates: Dict[str, Any], step: str) -> Dict[str, Any]:
        """Internal dispatch: build include/exclude sets and filter candidates."""
        includes = set(self.STEP_INCLUDES.get(step, []))
        excludes = set(self.STEP_EXCLUDES.get(step, []))
        selected: Dict[str, Any] = {}

        # Always include list takes precedence
        for key in includes:
            if key in candidates:
                selected[key] = candidates[key]

        # Remove explicitly excluded keys even if listed in includes
        for key in excludes:
            selected.pop(key, None)

        return selected

    def _apply_trust_filter(
        self, contexts: Dict[str, Any], min_trust: str
    ) -> Dict[str, Any]:
        """Filter contexts by minimum trust level.

        Args:
            contexts: Dict of context items, each may have a `_trust_level` key.
            min_trust: Minimum acceptable trust level ("high", "medium", "low").

        Returns:
            Filtered contexts dict keeping only items at or above min_trust.
        """
        trust_rank = {"high": 3, "medium": 2, "low": 1}
        min_rank = trust_rank.get(min_trust.lower(), 0)
        return {
            k: v
            for k, v in contexts.items()
            if trust_rank.get(v.get("_trust_level", "").lower(), 0) >= min_rank
        }

    def _apply_freshness_filter(
        self, contexts: Dict[str, Any], max_age_seconds: int
    ) -> Dict[str, Any]:
        """Filter out stale context items based on TTL.

        Args:
            contexts: Dict of context items, each may have a `_fetched_at` timestamp.
            max_age_seconds: Maximum acceptable age in seconds.

        Returns:
            Filtered contexts dict removing items older than max_age_seconds.
        """
        import time

        now = time.time()
        filtered: Dict[str, Any] = {}

        for k, v in contexts.items():
            fetched_at = v.get("_fetched_at")
            if fetched_at is None:
                filtered[k] = v
                continue

            if isinstance(fetched_at, (int, float)):
                age = now - fetched_at
            elif isinstance(fetched_at, str):
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                    age = now - dt.timestamp()
                except (ValueError, TypeError):
                    age = 0
            else:
                age = 0

            if age <= max_age_seconds:
                filtered[k] = v

        return filtered

    def _safe_to_dict(self, obj: Any) -> Any:
        """Convert dataclass / Pydantic objects to plain dict recursively."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for f in obj.__dataclass_fields__:
                val = getattr(obj, f)
                result[f] = self._safe_to_dict(val)
            return result
        if isinstance(obj, dict):
            return {k: self._safe_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._safe_to_dict(i) for i in obj]
        return obj
