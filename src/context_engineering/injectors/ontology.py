"""
Ontology Context Injector
=========================
Injects a relevant ontology subgraph for each pipeline step.

The injector builds a scoped subgraph of the procurement ontology that is
proportional to the task at each step:
- Intent recognition  → lightweight summary (objects + actions list)
- Ontology grounding  → full subgraph for the detected intent
- Task planning       → task-relevant actions, connectors, rules
- Execution          → minimal rules/constraints only
"""

from typing import Any, Dict, List, Optional

from src.context_engineering.models import ContextOntology, SlotDefinition


# ---------------------------------------------------------------------------
# Ontology access
# ---------------------------------------------------------------------------

_ONTOLOGY_INSTANCE: Optional[Any] = None


def _get_ontology() -> Any:
    """Lazily load the procurement ontology."""
    global _ONTOLOGY_INSTANCE
    if _ONTOLOGY_INSTANCE is None:
        from src.procurement.ontology_loader import get_procurement_ontology

        _ONTOLOGY_INSTANCE = get_procurement_ontology()
    return _ONTOLOGY_INSTANCE


# ---------------------------------------------------------------------------
# OntologyContextInjector
# ---------------------------------------------------------------------------


class OntologyContextInjector:
    """Injects ontology context as a subgraph relevant to the current task.

    Each step receives a different granularity of ontology context, scoped
    to minimize token usage while ensuring correctness.
    """

    def __init__(self):
        self._ontology: Optional[Any] = None

    @property
    def ontology(self) -> Any:
        """Lazy ontology access."""
        if self._ontology is None:
            self._ontology = _get_ontology()
        return self._ontology

    # ------------------------------------------------------------------
    # Public API — one method per pipeline step
    # ------------------------------------------------------------------

    def inject_for_intent_recognition(self, user_input: str) -> ContextOntology:
        """Inject lightweight ontology summary for intent recognition.

        Only includes:
        - Object list with aliases (names, labels, synonyms)
        - Action list (IDs and names only)

        Returns a minimal ContextOntology that fits Step 1's tight token budget.
        """
        ctx = ContextOntology()

        # Object list: name, label, synonyms
        for name, ds in self.ontology.datasets.items():
            ctx.relationships.append(
                {
                    "from": "dataset",
                    "to": name,
                    "type": "object",
                    "label": ds.label,
                    "synonyms": ds.synonyms,
                    "keywords": ds.keywords,
                }
            )

        # Action list: id, name, entity_name
        for action in self.ontology.actions:
            ctx.relationships.append(
                {
                    "from": "action",
                    "to": action.id,
                    "type": "action",
                    "entity": action.entity_name,
                    "operation": action.operation,
                }
            )

        return ctx

    def inject_for_ontology_grounding(self, intent_result: Any) -> ContextOntology:
        """Inject full ontology subgraph for the detected intent.

        Includes:
        - Resolved object (dataset) with attributes
        - Relationships (incoming/outgoing)
        - Actions available for this entity
        - Applicable rules
        """
        ctx = ContextOntology()

        object_type = self._extract_object_type(intent_result)
        if not object_type:
            return ctx

        ctx.resolved_object = object_type

        # Resolve entity and its attributes
        entity = self.ontology.get_entity(object_type)
        if entity:
            ctx.resolved_slots = [
                SlotDefinition(
                    name=f.name,
                    value=f.type,
                    source="ontology",
                    confidence=1.0,
                    raw_phrase=f.description,
                )
                for f in entity.fields
            ]

        # Relationships
        related = self._get_related_relationships(object_type)
        ctx.relationships.extend(related)

        # Actions for this entity
        actions = self.ontology.get_actions_for_entity(object_type)
        ctx.resolved_action = "; ".join(a.id for a in actions)
        for action in actions:
            ctx.relationships.append(
                {
                    "from": object_type,
                    "to": action.id,
                    "type": "action",
                    "operation": action.operation,
                    "kind": action.kind,
                }
            )

        # Rules
        ctx.rules = self._get_applicable_rules(object_type)

        return ctx

    def inject_for_task_planning(self, ontology_result: Any) -> ContextOntology:
        """Inject task-relevant ontology for planning.

        Includes:
        - Actions, connectors, rules, relationships
        - Focused on the object resolved in Step 2
        """
        ctx = ContextOntology()

        object_type = getattr(ontology_result, "object_type", None)
        if not object_type:
            return ctx

        ctx.resolved_object = object_type

        # Actions with full detail
        actions = self.ontology.get_actions_for_entity(object_type)
        for action in actions:
            ctx.relationships.append(
                {
                    "from": object_type,
                    "to": action.id,
                    "type": "action",
                    "operation": action.operation,
                    "kind": action.kind,
                    "description": action.description,
                }
            )

        # Relationships (full depth)
        related = self._get_related_relationships(object_type, depth=2)
        ctx.relationships.extend(related)

        # Rules for planning
        ctx.rules = self._get_applicable_rules(object_type)

        # Connector info (from relationships)
        connectors = self._get_connectors_for_entity(object_type)
        for conn in connectors:
            ctx.relationships.append(
                {
                    "from": object_type,
                    "to": conn,
                    "type": "connector",
                }
            )

        return ctx

    def inject_for_execution(self, plan_result: Any) -> ContextOntology:
        """Inject minimal ontology for execution assist.

        Only includes relevant rules and constraints.
        """
        ctx = ContextOntology()

        # Extract object type from plan if available
        object_type = self._extract_object_type_from_plan(plan_result)

        ctx.rules = self._get_applicable_rules(object_type or "")

        # Only the most critical constraints
        critical_rules = [r for r in self.ontology.rules if r.severity in ("mandatory", "warning")]
        ctx.relationships = [
            {
                "from": "rule",
                "to": r.id,
                "type": "rule",
                "severity": r.severity,
                "message": r.message,
            }
            for r in critical_rules[:5]
        ]

        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ontology_subgraph(
        self, root_object: str, depth: int = 2
    ) -> Dict[str, Any]:
        """Build a subgraph of the ontology around a root object.

        Args:
            root_object: The primary entity to center the subgraph on.
            depth: Traversal depth.
                - Depth 1: object + direct relationships + actions
                - Depth 2: also include related objects

        Returns:
            A dict representing the subgraph.
        """
        subgraph: Dict[str, Any] = {
            "root": root_object,
            "entities": [],
            "relationships": [],
            "actions": [],
            "rules": [],
        }

        # Add root entity
        entity = self.ontology.get_entity(root_object)
        if entity:
            subgraph["entities"].append(
                {
                    "name": entity.name,
                    "label": entity.label,
                    "description": entity.description,
                }
            )

        # Add direct relationships
        direct_rels = self._get_related_relationships(root_object, depth=1)
        subgraph["relationships"].extend(direct_rels)

        # Depth 2: add related entities
        if depth >= 2:
            related_names = {r.get("to") for r in direct_rels}
            for rel_name in related_names:
                rel_entity = self.ontology.get_entity(rel_name)
                if rel_entity:
                    subgraph["entities"].append(
                        {
                            "name": rel_entity.name,
                            "label": rel_entity.label,
                            "description": rel_entity.description,
                        }
                    )

        # Add actions
        actions = self.ontology.get_actions_for_entity(root_object)
        for action in actions:
            subgraph["actions"].append(
                {
                    "id": action.id,
                    "name": action.name,
                    "operation": action.operation,
                    "kind": action.kind,
                }
            )

        # Add rules
        rules = self._get_applicable_rules(root_object)
        subgraph["rules"] = [{"id": r, "name": r} for r in rules]

        return subgraph

    def _extract_object_type(self, intent_result: Any) -> Optional[str]:
        """Extract the resolved object type from an intent recognition result."""
        if intent_result is None:
            return None

        # Try normalized_term first (the canonical object name)
        normalized = getattr(intent_result, "normalized_term", None)
        if normalized:
            return normalized

        # Try object_term
        object_term = getattr(intent_result, "object_term", None)
        if object_term:
            return object_term

        return None

    def _extract_object_type_from_plan(self, plan_result: Any) -> Optional[str]:
        """Extract the object type from a task plan result."""
        if plan_result is None:
            return None

        # Duck-typed access
        planned_actions = getattr(plan_result, "planned_actions", [])
        if not planned_actions:
            return None

        # Use the first action's connector/entity as hint
        first_action = planned_actions[0]
        connector = getattr(first_action, "connector", None)
        return connector

    def _get_related_relationships(
        self, entity_name: str, depth: int = 1
    ) -> List[Dict[str, str]]:
        """Get relationships for an entity."""
        related: List[Dict[str, str]] = []
        for rel in self.ontology.relationships:
            if rel.from_entity == entity_name:
                related.append(
                    {
                        "from": rel.from_entity,
                        "to": rel.to_entity,
                        "type": "outgoing",
                        "description": rel.description,
                    }
                )
            elif rel.to_entity == entity_name:
                related.append(
                    {
                        "from": rel.from_entity,
                        "to": rel.to_entity,
                        "type": "incoming",
                        "description": rel.description,
                    }
                )
        return related

    def _get_applicable_rules(self, object_type: str) -> List[str]:
        """Get rule IDs applicable to the given object type."""
        rule_ids = []
        for rule in self.ontology.rules:
            when = rule.when
            if isinstance(when, dict):
                if when.get("entity") == object_type or when.get("object") == object_type:
                    rule_ids.append(rule.id)
        return rule_ids

    def _get_connectors_for_entity(self, entity_name: str) -> List[str]:
        """Get connector names that operate on this entity."""
        connectors = []
        for action in self.ontology.actions:
            if action.entity_name == entity_name:
                connectors.append(action.id)
        return connectors
