"""
Intent Topology Generator
========================
Transforms a ProcurementOntology into an IntentTopology.

For each dataset + action pair the generator:
1. Determines the appropriate intent template from the action kind.
2. Extracts dimension fields from the dataset.
3. Builds static filter conditions from dataset enumerated values.
4. Checks business rules to classify the path type and collect blocking rules.
5. Wires cross-path next-goal candidates from relationship metadata.
6. Attaches example utterances derived from PROCUREMENT_INTENTS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.procurement.intents import PROCUREMENT_INTENTS
from src.procurement.ontology_loader import (
    Action,
    DataSet,
    ProcurementOntology,
    Rule,
    get_procurement_ontology,
)
from src.preprocessing.cleaner import BasicCleaner

from .models import (
    INTENT_TEMPLATES,
    IntentPath,
    IntentTopology,
    PathType,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a label to a lowercase snake_case identifier."""
    import re
    text = text.strip()
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]", "", text)
    return text.lower()


def _match_action_kind_to_template(operation: str, kind: str) -> Optional[str]:
    """Map (operation, kind) pair to an intent template key."""
    op_lower = operation.lower()
    kind_lower = kind.lower()

    if op_lower in ("get_by_id", "list") or kind_lower == "query":
        return "query_object"
    if op_lower == "create_inquiry_from_pr":
        return "create_from_object"
    if op_lower == "create_purchase_order_from_pr_and_quotation":
        return "generate_order"
    if op_lower in ("submit_award_approval", "submit_for_approval"):
        return "submit_approval"
    if op_lower == "generate_price_comparison":
        return "compare_objects"
    if op_lower in ("publish_inquiry",):
        return "publish_object"
    if op_lower in ("delete",):
        return "delete_object"
    if op_lower in ("withdraw_inquiry", "withdraw_workflow"):
        return "revoke_object_status"
    if op_lower in ("update", "process", "approve_workflow", "reject_workflow"):
        return "process_object"
    if op_lower in ("query_workflow_status", "query_workflow_history",
                    "get_purchase_order_execution_status", "collect_quotations"):
        return "query_object"
    if op_lower in ("withdraw",):
        return "revoke_object_status"
    if op_lower == "remind_supplier":
        return "recommend_object"
    return None


# ----------------------------------------------------------------------
# Rule Index Builder
# ----------------------------------------------------------------------

class _RuleIndex:
    """In-memory index of business rules keyed by entity and operation."""

    def __init__(self, rules: List[Rule]) -> None:
        self._by_action: Dict[str, List[Rule]] = {}
        self._by_entity_op: Dict[Tuple[str, str], List[Rule]] = {}
        self._all: Dict[str, Rule] = {}

        for rule in rules:
            self._all[rule.id] = rule
            when = rule.when or {}
            action_id = when.get("action")
            if action_id:
                self._by_action.setdefault(action_id, []).append(rule)
            entity = when.get("entity")
            operation = when.get("operation")
            if entity and operation:
                self._by_entity_op[(entity, operation)] = self._all.get(rule.id, [])  # type: ignore[arg-type]

    def get_rules_for_action(self, action_id: str) -> List[Rule]:
        return self._by_action.get(action_id, [])

    def get_rules_for_entity_operation(self, entity: str, operation: str) -> List[Rule]:
        return self._by_entity_op.get((entity, operation), [])

    def get_rule_description(self, rule_id: str) -> str:
        rule = self._all.get(rule_id)
        if rule is None:
            return rule_id
        return f"{rule.name}: {rule.message}"

    def get_blocking_rules(
        self, entity: str, operation: str, action_id: Optional[str] = None
    ) -> List[Rule]:
        """Return rules whose severity is 'error' and thus permanently block execution."""
        candidates: List[Rule] = []
        if action_id:
            candidates.extend(self.get_rules_for_action(action_id))
        candidates.extend(self.get_rules_for_entity_operation(entity, operation))
        return [r for r in candidates if r.severity == "error"]


# ----------------------------------------------------------------------
# Example Utterance Matcher
# ----------------------------------------------------------------------

class _UtteranceMatcher:
    """Map example utterances from PROCUREMENT_INTENTS to IntentPath objects."""

    def __init__(self) -> None:
        self._cleaner = BasicCleaner()
        # Build index: normalized_term -> list of example strings
        self._keyword_examples: Dict[str, List[str]] = {}
        self._action_examples: Dict[str, List[str]] = {}
        self._build()

    def _normalize(self, text: str) -> str:
        return self._cleaner.clean(text.lower())

    def _build(self) -> None:
        for intent in PROCUREMENT_INTENTS:
            examples = intent.examples or []
            for ex in examples:
                norm = self._normalize(ex)
                self._action_examples.setdefault(intent.action, []).append(ex)
                # Index by key noun phrases
                for keyword in intent.trigger_keywords:
                    kw_norm = self._normalize(keyword)
                    if kw_norm in norm:
                        self._keyword_examples.setdefault(kw_norm, []).append(ex)

    def get_examples_for_action(self, action_id: str) -> List[str]:
        """Return up to 3 representative examples for an action."""
        return self._action_examples.get(action_id, [])[:3]

    def get_examples_for_keywords(self, keywords: List[str]) -> List[str]:
        """Return examples matching any of the given keywords."""
        matched: Set[str] = set()
        for kw in keywords:
            norm = self._normalize(kw)
            matched.update(self._keyword_examples.get(norm, []))
        return list(matched)[:3]


# ----------------------------------------------------------------------
# Main Generator
# ----------------------------------------------------------------------

class IntentTopologyGenerator:
    """Build an IntentTopology from a ProcurementOntology.

    Parameters
    ----------
    ontology:
        Fully-loaded ProcurementOntology (from ontology_loader).
    """

    def __init__(self, ontology: ProcurementOntology) -> None:
        self.ontology = ontology
        self._rule_index = _RuleIndex(ontology.rules)
        self._utterance_matcher = _UtteranceMatcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> IntentTopology:
        """Generate the complete IntentTopology."""
        paths: List[IntentPath] = []
        object_action_map: Dict[str, List[str]] = {}

        for dataset_name, dataset in self.ontology.datasets.items():
            actions = self.ontology.get_actions_for_entity(dataset_name)
            action_ids: List[str] = []
            for action in actions:
                action_ids.append(action.id)
                path = self._generate_path(dataset, action)
                if path is not None:
                    paths.append(path)

            object_action_map[dataset_name] = action_ids

        # Build cross-path next-goal candidates
        self._attach_next_goals(paths)

        # Build rule index
        rule_index = {
            r.id: self._rule_index.get_rule_description(r.id)
            for r in self.ontology.rules
        }

        topology = IntentTopology(
            version=self.ontology.version,
            paths=paths,
            object_action_map=object_action_map,
            rule_index=rule_index,
        )
        return topology

    # ------------------------------------------------------------------
    # Path generation
    # ------------------------------------------------------------------

    def _generate_path(
        self, dataset: DataSet, action: Action
    ) -> Optional[IntentPath]:
        """Generate a single IntentPath for a (dataset, action) pair."""
        template_key = _match_action_kind_to_template(action.operation, action.kind)
        if template_key is None:
            return None

        template_def = INTENT_TEMPLATES.get(template_key, {})
        meaning = template_def.get("meaning", template_key)

        # Build path ID
        path_id = f"{dataset.name}.{_slugify(action.name)}"
        if template_key == "query_object":
            path_id = f"{dataset.name}.query.{_slugify(dataset.label)}"
        elif template_key == "create_object":
            path_id = f"{dataset.name}.create.{_slugify(dataset.label)}"
        elif template_key == "create_from_object":
            path_id = f"{dataset.name}.create_from_pr"
        elif template_key == "delete_object":
            path_id = f"{dataset.name}.delete.{_slugify(dataset.label)}"
        elif template_key == "submit_approval":
            path_id = f"{dataset.name}.submit_approval"
        elif template_key == "generate_order":
            path_id = f"{dataset.name}.generate_order"
        elif template_key == "compare_objects":
            path_id = f"{dataset.name}.compare"
        elif template_key == "publish_object":
            path_id = f"{dataset.name}.publish"
        elif template_key == "revoke_object_status":
            path_id = f"{dataset.name}.revoke"
        elif template_key == "process_object":
            path_id = f"{dataset.name}.process.{_slugify(action.name)}"

        # Extract dimensions from dataset fields
        dimensions = self._extract_dimensions(dataset)

        # Build static filters from dataset enumerated values
        filters = self._build_filters(dataset)

        # Check blocking rules
        blocking_rules = self._rule_index.get_blocking_rules(
            entity=dataset.name,
            operation=action.operation,
            action_id=action.id,
        )
        blocked_by_rule = [r.id for r in blocking_rules]

        # Required rules (warnings are informational but required for execution)
        required_rules: List[str] = []
        for rule in self.ontology.rules:
            when = rule.when or {}
            if when.get("action") == action.id:
                required_rules.append(rule.id)

        # Classify path type
        path_type = self._classify_path_type(
            template=template_key,
            object=dataset.name,
            action=action.id,
            operation=action.operation,
            filters=filters,
            required_rules=required_rules,
            blocked_by_rule=blocked_by_rule,
        )

        # Get example utterances
        examples = self._utterance_matcher.get_examples_for_action(action.id)
        if not examples:
            examples = self._utterance_matcher.get_examples_for_keywords(
                [dataset.label] + dataset.synonyms
            )

        # Build description
        desc = self._build_description(dataset, action, meaning)

        return IntentPath(
            id=path_id,
            path_type=path_type,
            template=template_key,
            object=dataset.name,
            action=action.id,
            dimensions=dimensions,
            filters=filters,
            required_rules=required_rules,
            blocked_by_rule=blocked_by_rule,
            example_utterances=examples,
            description=desc,
        )

    def _classify_path_type(
        self,
        template: str,
        object: str,
        action: str,
        operation: str,
        filters: Dict[str, Any],
        required_rules: List[str],
        blocked_by_rule: List[str],
    ) -> PathType:
        """Apply business rule logic to classify the path execution status."""
        # Rule-blocked: at least one error-severity rule applies
        if blocked_by_rule:
            return PathType.RULE_BLOCKED

        # CONFIRMATION_REQUIRED: destructive, state-transitioning, or confirmation-flagged actions
        confirmation_ops = {
            "delete",
            "publish_inquiry",
            "withdraw_inquiry",
            "close_inquiry",
            "submit_for_approval",
            "approve_workflow",
            "reject_workflow",
            "withdraw_workflow",
        }
        if operation in confirmation_ops:
            return PathType.CONFIRMATION_REQUIRED

        # CONFIRMATION_REQUIRED: submit_approval template
        if template == "submit_approval":
            return PathType.CONFIRMATION_REQUIRED

        # CONFIRMATION_REQUIRED: generate_order template
        if template == "generate_order":
            return PathType.CONFIRMATION_REQUIRED

        # CONFIRMATION_REQUIRED: publish_object template
        if template == "publish_object":
            return PathType.CONFIRMATION_REQUIRED

        # CLARIFICATION_REQUIRED: generic process_object when no specific filters set
        if template == "process_object" and not filters:
            return PathType.CLARIFICATION_REQUIRED

        # DYNAMIC_LOOKUP_REQUIRED: actions needing vendor lookup
        dynamic_lookup_ops = {
            "create_inquiry_from_pr",
            "remind_supplier",
        }
        if operation in dynamic_lookup_ops:
            return PathType.DYNAMIC_LOOKUP_REQUIRED

        # RECOMMENDATION_ONLY: compare / recommend templates
        if template in ("compare_objects", "recommend_object"):
            return PathType.RECOMMENDATION_ONLY

        # ANALYTICS: query operations with analytics prefix
        if "analytics/" in action:
            return PathType.EXECUTABLE

        # Default: executable
        return PathType.EXECUTABLE

    # ------------------------------------------------------------------
    # Dimension / Filter extraction
    # ------------------------------------------------------------------

    def _extract_dimensions(self, dataset: DataSet) -> List[str]:
        """Extract dimension field names from a dataset.

        Dimensions are fields that are commonly used to filter or group results:
        - Fields marked with ``dimension.is_time: true`` in the YAML
        - Fields with enumerated values (enum_values)
        - Key foreign-key / categorical fields
        """
        dimension_fields = [
            "apply_dep",
            "pr_type",
            "material_category",
            "company_id",
            "factory_id",
            "purgroup_id",
            "status",
            "flow_status",
            "execution_status",
            "publish_status",
            "source_type",
            "inquiry_type",
            "delivery_date",
            "inquiry_start_date",
            "inquiry_end_date",
            "price_start_date",
            "price_end_date",
        ]
        present: List[str] = []
        for field in dataset.fields:
            if field.name in dimension_fields:
                present.append(field.name)
        # Deduplicate and limit to 6 most useful
        seen: Set[str] = set()
        result: List[str] = []
        for name in present:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result[:6]

    def _build_filters(self, dataset: DataSet) -> Dict[str, Any]:
        """Build static filter conditions from dataset enumerated values.

        Only includes filters that are meaningful across all queries:
        - delete_flag = 0 (always exclude deleted records)
        - status values from ai_context.enum_values (when available)
        """
        filters: Dict[str, Any] = {}
        # Always exclude deleted records
        if dataset.name in (
            "purchase_requests",
            "purchase_inquiries",
            "purchase_quotations",
            "purchase_order_heads",
            "purchase_order_items",
        ):
            filters["delete_flag"] = "0"
        return filters

    def _attach_next_goals(self, paths: List[IntentPath]) -> None:
        """Wire candidate next-goal paths based on the S2A workflow sequence.

        The natural flow is:
          purchase_requests → purchase_inquiries → purchase_quotations →
          purchase_order_heads / purchase_order_items →
          purchase_order_receipt_history
        """
        # Maps object → canonical next object in the S2A chain
        _NEXT_OBJECT: Dict[str, str] = {
            "purchase_requests": "purchase_inquiries",
            "purchase_inquiries": "purchase_quotations",
            "purchase_quotations": "purchase_order_heads",
            "purchase_order_heads": "purchase_order_receipt_history",
            "purchase_order_items": "purchase_order_receipt_history",
        }

        # Build quick lookup: object → list of path IDs
        by_object: Dict[str, List[str]] = {}
        for path in paths:
            by_object.setdefault(path.object, []).append(path.id)

        for path in paths:
            next_object = _NEXT_OBJECT.get(path.object)
            if next_object and next_object in by_object:
                path.candidate_next_goals = by_object[next_object][:3]

    # ------------------------------------------------------------------
    # Description builder
    # ------------------------------------------------------------------

    def _build_description(
        self, dataset: DataSet, action: Action, meaning: str
    ) -> str:
        """Build a human-readable description for an IntentPath."""
        obj_label = dataset.label or dataset.name
        act_name = action.name
        parts = [meaning, obj_label, "：", act_name]
        if dataset.description:
            parts.append(f"（{dataset.description[:40]}）")
        return "".join(parts)


# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------

def generate_topology(ontology: Optional[ProcurementOntology] = None) -> IntentTopology:
    """One-liner to generate topology from the default ontology."""
    if ontology is None:
        ontology = get_procurement_ontology()
    generator = IntentTopologyGenerator(ontology)
    return generator.generate()
