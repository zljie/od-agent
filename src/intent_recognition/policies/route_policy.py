"""
Route Policy
============
Manages action + object → runtime routing rules.

Based on Section 10.2 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Example:
    policy = RoutePolicyEngine()
    result = policy.route(action="Query", object="PurchaseRequirement")
    print(result.target)  # "workflow.query_purchase_requirement"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..semantic_frame import RuntimeTarget, RiskLevel


# ---------------------------------------------------------------------------
# Route Policy Model
# ---------------------------------------------------------------------------

@dataclass
class RoutePolicy:
    """Route policy definition.

    Maps action + object combinations to runtime targets.
    """
    action_intent: str                          # Action intent (e.g., "Query")
    ontology_object: str                         # Ontology object (e.g., "PurchaseRequirement")
    condition_pattern: Optional[str] = None      # Optional condition pattern
    target_runtime: RuntimeTarget = RuntimeTarget.WORKFLOW  # Target runtime
    target_id: str = ""                         # Target resource ID
    priority: int = 100                         # Priority (lower = higher priority)
    fallback_strategy: Optional[str] = None    # Fallback strategy
    need_hitl: bool = False                     # Whether HITL is required
    risk_level: RiskLevel = RiskLevel.LOW        # Risk level for this route

    def matches(self, action: str, object: str) -> bool:
        """Check if this policy matches the given action and object."""
        action_match = self.action_intent.lower() == action.lower()
        object_match = self.ontology_object.lower() == object.lower()
        return action_match and object_match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actionIntent": self.action_intent,
            "ontologyObject": self.ontology_object,
            "conditionPattern": self.condition_pattern,
            "targetRuntime": self.target_runtime.value,
            "targetId": self.target_id,
            "priority": self.priority,
            "fallbackStrategy": self.fallback_strategy,
            "needHitl": self.need_hitl,
            "riskLevel": self.risk_level.value,
        }


@dataclass
class RouteResult:
    """Result of route policy lookup."""
    target_runtime: RuntimeTarget
    target_id: str
    policy: Optional[RoutePolicy] = None
    is_fallback: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targetRuntime": self.target_runtime.value,
            "targetId": self.target_id,
            "isFallback": self.is_fallback,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Default Route Policies
# ---------------------------------------------------------------------------

DEFAULT_ROUTE_POLICIES: List[RoutePolicy] = [
    # PurchaseRequirement routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="PurchaseRequirement",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_purchase_requirement",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Create",
        ontology_object="PurchaseRequirement",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.create_purchase_requirement",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.HIGH,
    ),
    RoutePolicy(
        action_intent="Update",
        ontology_object="PurchaseRequirement",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.update_purchase_requirement",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.MEDIUM,
    ),
    RoutePolicy(
        action_intent="Approve",
        ontology_object="PurchaseRequirement",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.approve_purchase_requirement",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.MEDIUM,
    ),

    # Inquiry routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="Inquiry",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_inquiry",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Create",
        ontology_object="Inquiry",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.create_inquiry",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.HIGH,
    ),

    # Quotation routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="Quotation",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_quotation",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Compare",
        ontology_object="Quotation",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.compare_quotation",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Create",
        ontology_object="Quotation",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.create_quotation",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.HIGH,
    ),

    # PurchaseOrder routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="PurchaseOrder",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_purchase_order",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Create",
        ontology_object="PurchaseOrder",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.create_purchase_order",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.HIGH,
    ),
    RoutePolicy(
        action_intent="Update",
        ontology_object="PurchaseOrder",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.update_purchase_order",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.MEDIUM,
    ),

    # Contract routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="Contract",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_contract",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Create",
        ontology_object="Contract",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.create_contract",
        priority=10,
        need_hitl=True,
        risk_level=RiskLevel.CRITICAL,
    ),

    # Supplier routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="Supplier",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_supplier",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Compare",
        ontology_object="Supplier",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.compare_supplier",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),

    # Material routes
    RoutePolicy(
        action_intent="Query",
        ontology_object="Material",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.query_material",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Recommend",
        ontology_object="Material",
        target_runtime=RuntimeTarget.PLANNER,
        target_id="planner.material_recommendation",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),

    # Analyze routes
    RoutePolicy(
        action_intent="Analyze",
        ontology_object="PurchaseRequirement",
        target_runtime=RuntimeTarget.PLANNER,
        target_id="planner.analyze_purchase_requirement",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
    RoutePolicy(
        action_intent="Analyze",
        ontology_object="Supplier",
        target_runtime=RuntimeTarget.PLANNER,
        target_id="planner.analyze_supplier",
        priority=10,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),

    # Generic routes (fallback)
    RoutePolicy(
        action_intent="Query",
        ontology_object="*",
        target_runtime=RuntimeTarget.WORKFLOW,
        target_id="workflow.generic_query",
        priority=1000,
        need_hitl=False,
        risk_level=RiskLevel.LOW,
    ),
]


# ---------------------------------------------------------------------------
# Route Policy Engine
# ---------------------------------------------------------------------------

class RoutePolicyEngine:
    """Manages route policies and performs routing lookups.

    Example:
        engine = RoutePolicyEngine()
        result = engine.route(action="Query", object="PurchaseRequirement")
        print(result.target_id)  # "workflow.query_purchase_requirement"
    """

    def __init__(self, policies: Optional[List[RoutePolicy]] = None):
        """Initialize the route policy engine.

        Args:
            policies: Custom route policies. Uses default if not provided.
        """
        self.policies = sorted(policies or DEFAULT_ROUTE_POLICIES, key=lambda p: p.priority)

    def route(
        self,
        action: str,
        object: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RouteResult:
        """Find the appropriate route for action + object.

        Args:
            action: Action intent (e.g., "Query")
            object: Ontology object (e.g., "PurchaseRequirement")
            context: Additional context for routing

        Returns:
            RouteResult with target runtime and ID
        """
        context = context or {}

        # Find matching policies
        matching = [p for p in self.policies if p.matches(action, object)]

        if not matching:
            # Try wildcard match
            wildcard = [p for p in self.policies if p.matches(action, "*")]
            if wildcard:
                return RouteResult(
                    target_runtime=wildcard[0].target_runtime,
                    target_id=wildcard[0].target_id,
                    policy=wildcard[0],
                    is_fallback=True,
                    reason=f"通配符匹配: {action} + *",
                )

            # No match found
            return RouteResult(
                target_runtime=RuntimeTarget.REJECT,
                target_id="",
                reason=f"未找到匹配路由: {action} + {object}",
            )

        # Return highest priority match
        best = matching[0]
        return RouteResult(
            target_runtime=best.target_runtime,
            target_id=best.target_id,
            policy=best,
            is_fallback=False,
            reason=f"精确匹配: {action} + {object}",
        )

    def add_policy(self, policy: RoutePolicy) -> None:
        """Add a new route policy."""
        self.policies.append(policy)
        self.policies.sort(key=lambda p: p.priority)

    def remove_policy(self, action: str, object: str) -> bool:
        """Remove a route policy."""
        for i, p in enumerate(self.policies):
            if p.matches(action, object):
                self.policies.pop(i)
                return True
        return False

    def get_policies(self) -> List[RoutePolicy]:
        """Get all route policies."""
        return list(self.policies)

    def to_config(self) -> List[Dict[str, Any]]:
        """Export policies as config dict."""
        return [p.to_dict() for p in self.policies]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default_engine: Optional[RoutePolicyEngine] = None


def get_default_route_policy() -> RoutePolicyEngine:
    """Get the default route policy engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RoutePolicyEngine()
    return _default_engine
