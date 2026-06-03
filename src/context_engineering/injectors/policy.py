"""
Policy Context Injector
======================
Injects business rules, safety constraints, and permission context.

Each step receives a different scope of policy context:
- Step 1 (intent recognition): safety constraints only
- Step 2 (ontology grounding): relevant rules + safety constraints
- Step 3 (task planning): full rules, permissions, safety, risk level
- Step 4 (execution): hard blocks, confirmation requirements, fallback policies
- Step 5 (response generation): safety constraints for output
"""

from typing import Any, Dict, List, Optional

from src.context_engineering.models import ContextPolicy, RuleDefinition, Severity


# ---------------------------------------------------------------------------
# Universal safety constraints (always injected)
# ---------------------------------------------------------------------------

_UNIVERSAL_SAFETY_CONSTRAINTS = [
    "Never expose internal error messages or stack traces to the user.",
    "Never reveal system file paths, internal IPs, or infrastructure details.",
    "Never perform operations that modify data without explicit user confirmation.",
    "Never execute destructive actions (delete, cancel, reject) without double confirmation.",
    "Do not hallucinate data; only report information retrieved from verified connectors.",
    "Respect user privacy: do not log or persist personal identifiers beyond session scope.",
]


# ---------------------------------------------------------------------------
# User permissions (loaded lazily)
# ---------------------------------------------------------------------------

_USER_PERMISSIONS: Dict[str, Any] = {}


def _load_user_permissions() -> Dict[str, Any]:
    """Load user permissions from the permissions store."""
    global _USER_PERMISSIONS
    if not _USER_PERMISSIONS:
        try:
            from src.procurement.permission_store import get_user_permissions

            _USER_PERMISSIONS = get_user_permissions()
        except ImportError:
            _USER_PERMISSIONS = {
                "can_create": True,
                "can_update": True,
                "can_delete": False,
                "can_export": True,
                "max_results": 100,
            }
    return _USER_PERMISSIONS


# ---------------------------------------------------------------------------
# PolicyContextInjector
# ---------------------------------------------------------------------------


class PolicyContextInjector:
    """Injects policy context: rules, safety, permissions.

    Scoped to the current pipeline step to balance completeness vs. token budget.
    """

    def __init__(self):
        self._rule_index: Optional[List[RuleDefinition]] = None
        self._user_permissions: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API — one method per pipeline step
    # ------------------------------------------------------------------

    def inject_for_step1(self, intent_result: Any = None) -> ContextPolicy:
        """Inject policy context for intent recognition.

        Minimal scope: just universal safety constraints.
        No rules or permissions are injected at this stage.
        """
        ctx = ContextPolicy()
        ctx.safety_constraints = list(_UNIVERSAL_SAFETY_CONSTRAINTS)
        ctx.permissions = {"can_query": True}
        return ctx

    def inject_for_step2(
        self, intent_result: Any, ontology_result: Any
    ) -> ContextPolicy:
        """Inject policy context for ontology grounding.

        Includes:
        - Relevant rules for the detected object/action
        - Safety constraints
        """
        ctx = ContextPolicy()
        ctx.safety_constraints = list(_UNIVERSAL_SAFETY_CONSTRAINTS)

        object_type = self._extract_object_type(intent_result, ontology_result)
        action = self._extract_action(intent_result, ontology_result)

        rules = self._get_relevant_rules(object_type or "", action or "")
        ctx.rules = rules

        return ctx

    def inject_for_step3(
        self, intent_result: Any, ontology_result: Any
    ) -> ContextPolicy:
        """Inject policy context for task planning.

        Includes:
        - All relevant rules with severity
        - User permissions
        - Safety constraints
        - Risk level derived from the intent
        """
        ctx = ContextPolicy()
        ctx.safety_constraints = list(_UNIVERSAL_SAFETY_CONSTRAINTS)

        object_type = self._extract_object_type(intent_result, ontology_result)
        action = self._extract_action(intent_result, ontology_result)

        rules = self._get_relevant_rules(object_type or "", action or "")
        ctx.rules = rules

        permissions = self._get_user_permissions()
        ctx.permissions = permissions

        # Risk level from intent
        risk = getattr(intent_result, "risk_level", "low") if intent_result else "low"
        ctx.permissions["effective_risk_level"] = risk

        return ctx

    def inject_for_step4(self, plan_result: Any) -> ContextPolicy:
        """Inject policy context for execution.

        Includes:
        - Hard blocks (mandatory rules that stop execution)
        - Confirmation requirements
        - Fallback policies
        """
        ctx = ContextPolicy()
        ctx.safety_constraints = list(_UNIVERSAL_SAFETY_CONSTRAINTS)

        # Get hard blocks from mandatory rules
        hard_blocks = self._get_hard_blocks(plan_result)
        ctx.permissions["hard_blocks"] = hard_blocks

        # Confirmation requirements from plan
        requires_confirmation = getattr(plan_result, "requires_confirmation", False)
        ctx.permissions["requires_confirmation"] = requires_confirmation

        # Fallback policies
        fallback_rules = self._get_fallback_rules(plan_result)
        ctx.rules.extend(fallback_rules)

        return ctx

    def inject_for_step5(self, exec_result: Any) -> ContextPolicy:
        """Inject policy context for response generation.

        Minimal: safety constraints for output.
        No rules or heavy permissions at this stage.
        """
        ctx = ContextPolicy()
        ctx.safety_constraints = list(_UNIVERSAL_SAFETY_CONSTRAINTS)

        # Add output-specific constraints
        ctx.safety_constraints.extend(
            [
                "Format response according to the user's language preference.",
                "Do not include connector names, internal IDs, or technical jargon in the response.",
                "Summarize execution results in plain language before presenting details.",
            ]
        )

        # Check if execution was fully successful
        all_success = getattr(exec_result, "all_success", True) if exec_result else True
        ctx.permissions["execution_successful"] = all_success

        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_relevant_rules(
        self, object_type: str, action: str
    ) -> List[RuleDefinition]:
        """Get rules relevant to the given object type and action."""
        rules: List[RuleDefinition] = []

        # Build rule index if needed
        if self._rule_index is None:
            self._rule_index = self._build_rule_index()

        for rule_def in self._rule_index:
            metadata = rule_def.metadata
            rule_object = metadata.get("entity", metadata.get("object", ""))
            rule_action = metadata.get("action", "")

            if rule_object and rule_object != object_type:
                continue
            if rule_action and rule_action != action:
                continue

            rules.append(rule_def)

        return rules

    def _build_rule_index(self) -> List[RuleDefinition]:
        """Build the rule index from the ontology."""
        try:
            from src.procurement.ontology_loader import get_procurement_ontology

            ontology = get_procurement_ontology()
            rule_defs = []
            for rule in ontology.rules:
                rule_defs.append(
                    RuleDefinition(
                        id=rule.id,
                        severity=Severity(rule.severity) if rule.severity else Severity.INFO,
                        description=rule.message,
                        enabled=True,
                        metadata={"entity": rule.when.get("entity", ""), "action": rule.when.get("action", "")},
                    )
                )
            return rule_defs
        except ImportError:
            return []

    def _check_permissions(self, actions: List[str]) -> Dict[str, bool]:
        """Check user permissions for a list of actions."""
        permissions = self._get_user_permissions()
        result: Dict[str, bool] = {}

        action_permission_map = {
            "create": "can_create",
            "update": "can_update",
            "delete": "can_delete",
            "export": "can_export",
            "query": "can_query",
            "read": "can_query",
        }

        for action in set(actions):
            perm_key = action_permission_map.get(action, "can_query")
            result[action] = permissions.get(perm_key, False)

        return result

    def _get_safety_constraints(self) -> List[str]:
        """Get universal safety constraints."""
        return list(_UNIVERSAL_SAFETY_CONSTRAINTS)

    def _get_user_permissions(self) -> Dict[str, Any]:
        """Get user permissions (lazy load)."""
        if self._user_permissions is None:
            self._user_permissions = _load_user_permissions()
        return self._user_permissions

    def _get_hard_blocks(self, plan_result: Any) -> List[str]:
        """Extract hard blocks from the plan (mandatory rule violations)."""
        hard_blocks: List[str] = []

        try:
            from src.procurement.ontology_loader import get_procurement_ontology

            ontology = get_procurement_ontology()
            for rule in ontology.rules:
                if rule.severity == "mandatory":
                    hard_blocks.append(f"{rule.id}: {rule.message}")

            # Check plan's confirmation requirement
            if plan_result:
                requires_conf = getattr(plan_result, "requires_confirmation", False)
                if requires_conf:
                    hard_blocks.append(
                        "CONFIRM_REQUIRED: This action requires user confirmation before execution."
                    )
        except ImportError:
            pass

        return hard_blocks

    def _get_fallback_rules(self, plan_result: Any) -> List[RuleDefinition]:
        """Get fallback policies when planned actions cannot be executed."""
        fallback_rules: List[RuleDefinition] = []

        # Generic fallback rule
        fallback_rules.append(
            RuleDefinition(
                id="fallback_query",
                severity=Severity.INFO,
                description="If the primary action fails, fall back to a read-only query to retrieve existing data.",
                enabled=True,
            )
        )

        # Timeout fallback
        fallback_rules.append(
            RuleDefinition(
                id="timeout_retry",
                severity=Severity.INFO,
                description="If execution times out, retry once after 5 seconds.",
                enabled=True,
            )
        )

        return fallback_rules

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_object_type(
        self, intent_result: Any, ontology_result: Any
    ) -> Optional[str]:
        """Extract the object type from intent or ontology result."""
        # From intent
        if intent_result:
            normalized = getattr(intent_result, "normalized_term", None)
            if normalized:
                return normalized
            object_term = getattr(intent_result, "object_term", None)
            if object_term:
                return object_term

        # From ontology
        if ontology_result:
            return getattr(ontology_result, "object_type", None)

        return None

    def _extract_action(
        self, intent_result: Any, ontology_result: Any
    ) -> Optional[str]:
        """Extract the action/intent from results."""
        if intent_result:
            return getattr(intent_result, "intent", None)
        if ontology_result:
            return getattr(ontology_result, "resolved_action", None)
        return None
