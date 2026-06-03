"""Context Policy Engine — applies permission, freshness, trust, and security checks."""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class ContextPolicyEngine:
    """Applies policy checks: permissions, freshness, trust, security."""

    def __init__(self, user_context: Optional[Dict[str, Any]] = None):
        """Initialize with optional user context dict.

        Args:
            user_context: Dict containing at least:
                - user_id: str
                - roles: List[str]
                - permissions: List[str]
                - department: Optional[str]
        """
        self.user_context = user_context or {}

    # -------------------------------------------------------------------------
    # Trust level mapping
    # -------------------------------------------------------------------------
    HIGH_TRUST_SOURCES = {
        "connector",
        "ontology",
        "rule_engine",
        "user_input",
        "current_input",
    }
    MEDIUM_TRUST_SOURCES = {
        "history_summary",
        "llm_reasoning",
        "context_cache",
        "session",
    }
    LOW_TRUST_SOURCES = {
        "user_ambiguous",
        "fallback",
        "default",
    }

    # -------------------------------------------------------------------------
    # Permissions
    # -------------------------------------------------------------------------

    def check_permissions(self, required_actions: List[str]) -> Dict[str, bool]:
        """Check if user has permission for each required action.

        Args:
            required_actions: List of action strings to check.

        Returns:
            Dict mapping each action to True (allowed) or False (denied).
        """
        user_permissions: List[str] = self.user_context.get("permissions", [])
        user_roles: List[str] = self.user_context.get("roles", [])

        # Build a combined permission set from direct permissions and role grants
        granted: set = set(user_permissions)

        role_permissions = {
            "admin": {
                "read",
                "write",
                "delete",
                "execute",
                "approve",
                "export",
                "import",
                "manage_users",
                "manage_roles",
            },
            "manager": {
                "read",
                "write",
                "execute",
                "approve",
                "export",
            },
            "user": {
                "read",
                "write",
                "execute",
                "export",
            },
            "viewer": {
                "read",
            },
        }
        for role in user_roles:
            granted.update(role_permissions.get(role, set()))

        return {action: action in granted for action in required_actions}

    # -------------------------------------------------------------------------
    # Freshness
    # -------------------------------------------------------------------------

    def check_freshness(
        self, context_items: Dict[str, Any], default_ttl_seconds: int = 300
    ) -> Dict[str, Dict[str, Any]]:
        """Check if each context item is fresh.

        Args:
            context_items: Dict of item_key -> item_data.
            default_ttl_seconds: Default TTL to apply when item has no explicit TTL.

        Returns:
            Dict mapping item_key -> {
                "fresh": bool,
                "age_seconds": float,
                "reason": str,
            }
        """
        now = time.time()
        results: Dict[str, Dict[str, Any]] = {}

        for key, item in context_items.items():
            fetched_at = item.get("_fetched_at")
            ttl = item.get("_ttl_seconds", default_ttl_seconds)

            if fetched_at is None:
                results[key] = {
                    "fresh": True,
                    "age_seconds": 0.0,
                    "reason": "no_timestamp",
                }
                continue

            age = self._compute_age(fetched_at, now)
            results[key] = {
                "fresh": age <= ttl,
                "age_seconds": age,
                "reason": "ok" if age <= ttl else "expired",
            }

        return results

    def _compute_age(self, fetched_at: Any, now: float) -> float:
        """Return age in seconds for a fetched_at value."""
        if isinstance(fetched_at, (int, float)):
            return now - fetched_at
        if isinstance(fetched_at, str):
            try:
                dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
                return now - dt.timestamp()
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    # -------------------------------------------------------------------------
    # Trust
    # -------------------------------------------------------------------------

    def get_trust_level(self, context_item: Dict[str, Any]) -> str:
        """Determine trust level of a context item.

        Trust levels:
        - high: connector real-time return, ontology definition,
                rule engine, user current input
        - medium: history summary, LLM reasoning result
        - low: user ambiguous expression

        Args:
            context_item: A single context item dict.

        Returns:
            "high" | "medium" | "low"
        """
        source = (context_item.get("source") or "").lower()
        _type = (context_item.get("type") or "").lower()

        if source in self.HIGH_TRUST_SOURCES or _type in self.HIGH_TRUST_SOURCES:
            return "high"
        if source in self.MEDIUM_TRUST_SOURCES or _type in self.MEDIUM_TRUST_SOURCES:
            return "medium"
        if source in self.LOW_TRUST_SOURCES or _type in self.LOW_TRUST_SOURCES:
            return "low"

        # Heuristic from item shape
        if context_item.get("is_real_time") or context_item.get("connector_id"):
            return "high"
        if context_item.get("is_llm_generated"):
            return "medium"
        if context_item.get("is_fallback"):
            return "low"

        return "medium"

    # -------------------------------------------------------------------------
    # Security filter
    # -------------------------------------------------------------------------

    def apply_security_filter(
        self, context_items: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """Apply security/permission filters.

        Removes items that the user lacks permission to see
        and items flagged as restricted.

        Args:
            context_items: Dict of item_key -> item_data.

        Returns:
            (filtered_items, excluded_items) where excluded_items is a list of
            {"source": str, "reason": str} for each removed entry.
        """
        filtered: Dict[str, Any] = {}
        excluded: List[Dict[str, str]] = []

        user_roles: List[str] = self.user_context.get("roles", [])
        clearance: str = self.user_context.get("clearance", "default")

        clearance_rank = {"topsecret": 3, "secret": 2, "confidential": 1, "default": 0}
        user_clearance_rank = clearance_rank.get(clearance.lower(), 0)

        for key, item in context_items.items():
            # Skip items explicitly marked restricted
            if item.get("_restricted", False):
                excluded.append({"source": key, "reason": "restricted_flag"})
                continue

            # Check classification clearance
            required_clearance = clearance_rank.get(
                item.get("_classification", "default").lower(), 0
            )
            if required_clearance > user_clearance_rank:
                excluded.append({"source": key, "reason": "insufficient_clearance"})
                continue

            # Check department boundary
            item_dept = item.get("_department")
            user_dept = self.user_context.get("department")
            if item_dept and user_dept and item_dept != user_dept:
                if "cross_department" not in user_roles:
                    excluded.append({"source": key, "reason": "department_boundary"})
                    continue

            # Check PII flag — redact if user is not authorized
            if item.get("_contains_pii", False):
                if "pii_access" not in user_roles:
                    # Redact instead of exclude
                    item = {k: "***" for k in item}
                    item["_pii_redacted"] = True

            filtered[key] = item

        return filtered, excluded

    # -------------------------------------------------------------------------
    # Merge policy metadata into selected context
    # -------------------------------------------------------------------------

    def merge_policy_context(self, selected: Dict[str, Any]) -> Dict[str, Any]:
        """Merge policy results into selected context.

        Adds policy metadata keys to each context item:
            _policy_notes: List[str]
            _trust_level: str
            _freshness: Dict (fresh, age_seconds, reason)
            _security_passed: bool

        Args:
            selected: Context dict that has passed selection logic.

        Returns:
            The same dict with policy metadata merged in.
        """
        freshness_map = self.check_freshness(selected)
        filtered, _ = self.apply_security_filter(selected)

        for key, item in selected.items():
            if key.startswith("_"):
                continue

            # Trust level
            item["_trust_level"] = self.get_trust_level(item)

            # Freshness
            item["_freshness"] = freshness_map.get(key, {"fresh": True, "age_seconds": 0.0, "reason": "unknown"})

            # Security pass status
            item["_security_passed"] = key in filtered

            # Policy notes
            notes: List[str] = []
            if not item.get("_freshness", {}).get("fresh", True):
                notes.append("item_stale")
            if not item.get("_security_passed"):
                notes.append("security_filtered")
            item["_policy_notes"] = notes

        return selected

    # -------------------------------------------------------------------------
    # Audit / summary
    # -------------------------------------------------------------------------

    def get_policy_summary(self, contexts: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of policy decisions for audit logging.

        Args:
            contexts: Full candidate context dict.

        Returns:
            Dict with counts and details:
                total_items, trust_breakdown, freshness_breakdown,
                security_excluded, policy_pass_count
        """
        freshness = self.check_freshness(contexts)
        _, security_excluded = self.apply_security_filter(contexts)

        trust_breakdown: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        freshness_breakdown: Dict[str, int] = {"fresh": 0, "stale": 0}

        for key, item in contexts.items():
            level = self.get_trust_level(item)
            trust_breakdown[level] = trust_breakdown.get(level, 0) + 1

            fres = freshness.get(key, {})
            if fres.get("fresh"):
                freshness_breakdown["fresh"] += 1
            else:
                freshness_breakdown["stale"] += 1

        policy_pass = sum(
            1
            for item in contexts.values()
            if not item.get("_policy_notes")
        )

        return {
            "total_items": len(contexts),
            "trust_breakdown": trust_breakdown,
            "freshness_breakdown": freshness_breakdown,
            "security_excluded": security_excluded,
            "security_excluded_count": len(security_excluded),
            "policy_pass_count": policy_pass,
        }
