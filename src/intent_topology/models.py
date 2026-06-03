"""
Intent Topology Data Models
===========================
Core domain types for the intent topology graph.

- PathType: Execution status of an intent path
- IntentPath: A single node in the topology (object × action × dimensions)
- IntentTopology: Aggregated topology with indexes for fast lookup
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PathType(Enum):
    """Execution status classification for an IntentPath.

    These categories drive downstream routing: which handler processes
    the intent, what confirmation / clarification UI to show, etc.
    """

    EXECUTABLE = "executable"
    """Intent is fully resolvable; execute directly."""

    RULE_BLOCKED = "rule_blocked"
    """Intent is blocked by one or more business rules."""

    CONFIRMATION_REQUIRED = "confirmation_required"
    """Intent is safe to execute but requires user confirmation first."""

    CLARIFICATION_REQUIRED = "clarification_required"
    """Intent is ambiguous and needs additional slots or context."""

    PARTIAL_EXECUTABLE = "partial_executable"
    """Intent has all required slots but some optional slots are missing."""

    ONTOLOGY_GAP = "ontology_gap"
    """No ontology path exists for this intent; human handoff required."""

    CONNECTOR_REQUIRED = "connector_required"
    """Intent requires a missing connector (API / tool) to execute."""

    DYNAMIC_LOOKUP_REQUIRED = "dynamic_lookup_required"
    """Intent requires a runtime dynamic lookup (e.g., vendor list)."""

    RECOMMENDATION_ONLY = "recommendation_only"
    """Intent returns a recommendation only; no write action performed."""


# ----------------------------------------------------------------------
# Intent Template Registry
# ----------------------------------------------------------------------
# Maps template keys to their semantic meaning and allowed action kinds.
# Used by IntentTopologyGenerator to classify (object, action) pairs.

INTENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "query_object": {
        "meaning": "查询对象",
        "action_kinds": ["read", "query"],
    },
    "process_object": {
        "meaning": "处理对象",
        "action_kinds": ["process", "update"],
    },
    "create_object": {
        "meaning": "创建对象",
        "action_kinds": ["create"],
    },
    "create_from_object": {
        "meaning": "基于对象创建新对象",
        "action_kinds": ["create"],
    },
    "delete_object": {
        "meaning": "删除对象",
        "action_kinds": ["delete"],
    },
    "revoke_object_status": {
        "meaning": "撤销状态",
        "action_kinds": ["update"],
    },
    "publish_object": {
        "meaning": "发布对象",
        "action_kinds": ["publish"],
    },
    "compare_objects": {
        "meaning": "比较对象",
        "action_kinds": ["query"],
    },
    "recommend_object": {
        "meaning": "推荐对象",
        "action_kinds": ["query"],
    },
    "submit_approval": {
        "meaning": "提交审批",
        "action_kinds": ["submit_approval"],
    },
    "generate_order": {
        "meaning": "生成订单",
        "action_kinds": ["create"],
    },
    "analyze_risk": {
        "meaning": "风险分析",
        "action_kinds": ["query"],
    },
    "analyze_performance": {
        "meaning": "表现分析",
        "action_kinds": ["query"],
    },
    "export_object": {
        "meaning": "导出对象",
        "action_kinds": ["query"],
    },
}


# ----------------------------------------------------------------------
# IntentPath
# ----------------------------------------------------------------------

@dataclass
class IntentPath:
    """A single intent path node in the topology graph.

    Attributes
    ----------
    id:
        Unique path identifier, formatted as ``{object}.{action}.{dimension_suffix}``.
        Example: ``purchase_requests.query.by_department``
    path_type:
        Execution status classification from ``PathType``.
    template:
        Intent template key (e.g. ``query_object``, ``create_from_object``).
    object:
        Ontology dataset name this path operates on.
        Example: ``purchase_requests``
    action:
        Ontology action ID this path invokes.
        Example: ``purchase_requests/list``
    dimensions:
        List of dimension / slot names that filter this path.
        Example: ``["apply_dep", "purchase_type"]``
    filters:
        Static filter conditions that always apply to this path.
        Example: ``{"flow_status": ["S0", "APPROVED"], "delete_flag": "0"}``
    required_rules:
        Rule IDs that must pass before this path is executable.
    blocked_by_rule:
        Rule IDs that will permanently block this path.
    candidate_next_goals:
        IDs of IntentPaths that are natural follow-up steps.
    example_utterances:
        Example user utterances that match this path.
    confidence:
        Base confidence score [0.0, 1.0] for this path on a match.
    description:
        Human-readable description of what this path does.
    """

    id: str
    path_type: PathType
    template: str
    object: str
    action: str
    dimensions: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    required_rules: List[str] = field(default_factory=list)
    blocked_by_rule: List[str] = field(default_factory=list)
    candidate_next_goals: List[str] = field(default_factory=list)
    example_utterances: List[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.confidence == 0.0:
            self.confidence = 0.85


# ----------------------------------------------------------------------
# IntentTopology
# ----------------------------------------------------------------------

@dataclass
class IntentTopology:
    """Complete intent topology built from the procurement ontology.

    Provides pre-built indexes for fast filtering and traversal.

    Attributes
    ----------
    version:
        Ontology version string used to generate this topology.
    paths:
        All ``IntentPath`` nodes in the topology.
    object_action_map:
        Mapping from object name → list of action IDs operating on it.
    rule_index:
        Mapping from rule ID → human-readable rule description.
    """

    version: str
    paths: List[IntentPath] = field(default_factory=list)
    object_action_map: Dict[str, List[str]] = field(default_factory=dict)
    rule_index: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def find_paths(
        self,
        object: Optional[str] = None,
        action: Optional[str] = None,
        template: Optional[str] = None,
        path_type: Optional[PathType] = None,
    ) -> List[IntentPath]:
        """Filter paths by any combination of attributes.

        All filter arguments are optional; omitted filters match everything.
        """
        results = self.paths
        if object is not None:
            results = [p for p in results if p.object == object]
        if action is not None:
            results = [p for p in results if p.action == action]
        if template is not None:
            results = [p for p in results if p.template == template]
        if path_type is not None:
            results = [p for p in results if p.path_type == path_type]
        return results

    def get_paths_for_object(self, object: str) -> List[IntentPath]:
        """Return all paths that operate on a specific ontology object."""
        return self.find_paths(object=object)

    def get_paths_for_action(self, action: str) -> List[IntentPath]:
        """Return all paths that invoke a specific action."""
        return self.find_paths(action=action)

    def get_executable_paths(self) -> List[IntentPath]:
        """Return all paths that are immediately executable (no blocks)."""
        return self.find_paths(
            path_type=PathType.EXECUTABLE,
        )

    def get_blocked_paths(self) -> List[IntentPath]:
        """Return all paths blocked by at least one business rule."""
        return self.find_paths(path_type=PathType.RULE_BLOCKED)

    def get_confirmation_paths(self) -> List[IntentPath]:
        """Return all paths requiring user confirmation."""
        return self.find_paths(path_type=PathType.CONFIRMATION_REQUIRED)

    def to_summary(self) -> Dict[str, Any]:
        """Return a serializable summary dict for debugging / logging."""
        return {
            "version": self.version,
            "total_paths": len(self.paths),
            "path_types": {
                pt.value: len([p for p in self.paths if p.path_type == pt])
                for pt in PathType
            },
            "objects": list(self.object_action_map.keys()),
            "rules": list(self.rule_index.keys()),
        }
