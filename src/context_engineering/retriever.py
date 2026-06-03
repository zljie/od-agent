"""
Context Retriever for BeBISO Context Engineering Layer
======================================================
Retrieves candidate context from all available sources before the selector filters them.

Sources include:
- User input and preprocessing
- Ontology (ProcurementOntology from ontology_loader)
- Intent topology (IntentTopology from intent_topology.loader)
- Slot extractors (ExtractedSlots from procurement.slot_extractors)
- Session history (DialogStateManager from dialog_state)
- Tool/connector registry
- Business rules and risk policies

Each retrieved context item is annotated with:
- source: where it came from (e.g., "ProcurementSystemConnector", "ontology", "session_history")
- fetched_at: ISO timestamp
- trust_level: "high"/"medium"/"low"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RetrievedContextItem:
    """A single context item retrieved from a source."""

    key: str
    value: Any
    source: str
    fetched_at: str
    trust_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "trust_level": self.trust_level,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Container for all retrieved context items for a step."""

    step_name: str
    items: List[RetrievedContextItem] = field(default_factory=list)
    session_id: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "session_id": self.session_id,
            "filters": self.filters,
            "retrieval_time_ms": self.retrieval_time_ms,
            "items": [item.to_dict() for item in self.items],
            "context": {item.key: item.value for item in self.items},
        }

    def get(self, key: str, default: Any = None) -> Any:
        for item in self.items:
            if item.key == key:
                return item.value
        return default

    def get_by_source(self, source: str) -> List[Any]:
        return [item.value for item in self.items if item.source == source]

    def filter_by_trust(self, min_level: str) -> "RetrievalResult":
        trust_order = {"high": 3, "medium": 2, "low": 1}
        min_val = trust_order.get(min_level, 0)
        return RetrievalResult(
            step_name=self.step_name,
            session_id=self.session_id,
            filters=self.filters,
            retrieval_time_ms=self.retrieval_time_ms,
            items=[
                item
                for item in self.items
                if trust_order.get(item.trust_level, 0) >= min_val
            ],
        )


# =============================================================================
# Context Retriever
# =============================================================================


class ContextRetriever:
    """Retrieves candidate context from all available sources.

    This class is the first stage of the Context Engineering Layer.
    It collects candidate context from all sources without filtering,
    allowing the downstream Selector to make decisions about relevance.

    Usage::

        retriever = ContextRetriever(session_id="sess-123")
        result = retriever.retrieve_for_intent_recognition("帮我查一下采购需求")
        context = result.to_dict()
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self._ontology = None
        self._topology = None
        self._state_manager = None

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _annotate(
        self,
        key: str,
        value: Any,
        source: str,
        trust_level: str = "medium",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievedContextItem:
        return RetrievedContextItem(
            key=key,
            value=value,
            source=source,
            fetched_at=self._now_iso(),
            trust_level=trust_level,
            metadata=metadata or {},
        )

    def _get_ontology(self):
        if self._ontology is None:
            from src.procurement.ontology_loader import get_procurement_ontology
            self._ontology = get_procurement_ontology()
        return self._ontology

    def _get_topology(self):
        if self._topology is None:
            from src.intent_topology.loader import get_topology
            self._topology = get_topology()
        return self._topology

    def _get_state_manager(self):
        if self._state_manager is None:
            from src.dialog_state import get_state_manager
            self._state_manager = get_state_manager()
        return self._state_manager

    def _apply_filters(
        self, items: List[RetrievedContextItem], filters: Dict[str, Any]
    ) -> List[RetrievedContextItem]:
        if not filters:
            return items
        result = items
        if "sources" in filters:
            result = [i for i in result if i.source in filters["sources"]]
        if "keys" in filters:
            result = [i for i in result if i.key in filters["keys"]]
        if "min_trust" in filters:
            result = self._filter_by_trust(result, filters["min_trust"])
        if "max_items" in filters:
            result = result[: filters["max_items"]]
        return result

    def _filter_by_trust(
        self, items: List[RetrievedContextItem], min_level: str
    ) -> List[RetrievedContextItem]:
        trust_order = {"high": 3, "medium": 2, "low": 1}
        min_val = trust_order.get(min_level, 0)
        return [
            item
            for item in items
            if trust_order.get(item.trust_level, 0) >= min_val
        ]

    def _build_result(
        self,
        step_name: str,
        items: List[RetrievedContextItem],
        filters: Dict[str, Any],
    ) -> RetrievalResult:
        import time
        start = time.time()
        filtered = self._apply_filters(items, filters)
        elapsed_ms = (time.time() - start) * 1000
        return RetrievalResult(
            step_name=step_name,
            items=filtered,
            session_id=self.session_id,
            filters=filters,
            retrieval_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Step 1: Intent Recognition
    # ------------------------------------------------------------------

    def retrieve_for_intent_recognition(
        self,
        user_input: str,
        session_id: str = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve all candidate context for Step 1 (Intent Recognition).

        Sources: raw input, preprocessing, ontology summary, dialog history summary
        """
        sid = session_id or self.session_id
        filters = filters or {}
        items: List[RetrievedContextItem] = []

        # 1. Raw user input
        items.append(
            self._annotate(
                key="user_input",
                value=user_input,
                source="user_input",
                trust_level="high",
                metadata={"raw": True},
            )
        )

        # 2. Preprocessing
        try:
            from src.intent_recognition.preprocessor import InputPreprocessor
            preprocessor = InputPreprocessor()
            preprocessed = preprocessor.preprocess(user_input)
            items.append(
                self._annotate(
                    key="preprocessing_result",
                    value=preprocessed.to_dict(),
                    source="InputPreprocessor",
                    trust_level="high",
                    metadata={"original_input": preprocessed.original_input},
                )
            )
            items.append(
                self._annotate(
                    key="normalized_input",
                    value=preprocessed.normalized_input,
                    source="InputPreprocessor",
                    trust_level="high",
                )
            )
            if preprocessed.temporal_anchors:
                items.append(
                    self._annotate(
                        key="temporal_anchors",
                        value=preprocessed.temporal_anchors,
                        source="InputPreprocessor",
                        trust_level="medium",
                    )
                )
            if preprocessed.tokens:
                items.append(
                    self._annotate(
                        key="input_tokens",
                        value=preprocessed.tokens,
                        source="InputPreprocessor",
                        trust_level="medium",
                    )
                )
            if preprocessed.candidate_terms:
                items.append(
                    self._annotate(
                        key="candidate_terms",
                        value=preprocessed.candidate_terms,
                        source="InputPreprocessor",
                        trust_level="medium",
                    )
                )
        except ImportError:
            pass

        # 3. Ontology summary
        try:
            ontology = self._get_ontology()
            ontology_context = ontology.to_system_prompt_context()
            items.append(
                self._annotate(
                    key="ontology_context",
                    value=ontology_context,
                    source="ProcurementOntology",
                    trust_level="high",
                    metadata={
                        "ontology_version": ontology.version,
                        "dataset_count": len(ontology.datasets),
                        "action_count": len(ontology.actions),
                    },
                )
            )
            items.append(
                self._annotate(
                    key="ontology_datasets",
                    value=[
                        {
                            "name": ds.name,
                            "label": ds.label,
                            "description": ds.description,
                            "synonyms": ds.synonyms,
                            "keywords": ds.keywords,
                        }
                        for ds in ontology.datasets.values()
                    ],
                    source="ProcurementOntology",
                    trust_level="high",
                )
            )
            if ontology.scenarios:
                items.append(
                    self._annotate(
                        key="ontology_scenarios",
                        value=[{"name": s.name, "description": s.description} for s in ontology.scenarios],
                        source="ProcurementOntology",
                        trust_level="medium",
                    )
                )
            if ontology.instructions:
                items.append(
                    self._annotate(
                        key="ontology_instructions",
                        value=ontology.instructions,
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
        except ImportError:
            pass

        # 4. Intent topology candidates
        try:
            topology = self._get_topology()
            from src.intent_topology.loader import IntentTopologyLoader
            candidates = IntentTopologyLoader.find_candidate_paths(user_input, topology, top_k=5)
            items.append(
                self._annotate(
                    key="intent_topology_candidates",
                    value=[
                        {
                            "id": p.id,
                            "object": p.object,
                            "action": p.action,
                            "description": p.description,
                            "confidence": p.confidence,
                            "dimensions": p.dimensions,
                            "example_utterances": p.example_utterances,
                        }
                        for p in candidates
                    ],
                    source="IntentTopology",
                    trust_level="medium",
                    metadata={"topology_version": topology.version},
                )
            )
        except ImportError:
            pass

        # 5. Session history
        if sid:
            try:
                state_manager = self._get_state_manager()
                session = state_manager.get_session(sid)
                if session:
                    recent_messages = session.messages[-5:] if session.messages else []
                    items.append(
                        self._annotate(
                            key="session_history",
                            value=recent_messages,
                            source="DialogStateManager",
                            trust_level="medium",
                            metadata={
                                "session_id": sid,
                                "message_count": len(session.messages),
                                "current_state": session.current_state.value,
                            },
                        )
                    )
                    if session.context_data:
                        items.append(
                            self._annotate(
                                key="session_context_data",
                                value=session.context_data,
                                source="DialogStateManager",
                                trust_level="medium",
                            )
                        )
                    state_summary = state_manager.get_state_summary(sid)
                    if state_summary:
                        items.append(
                            self._annotate(
                                key="session_state_summary",
                                value=state_summary,
                                source="DialogStateManager",
                                trust_level="medium",
                            )
                        )
            except Exception:
                pass

        return self._build_result("intent_recognition", items, filters).to_dict()

    # ------------------------------------------------------------------
    # Step 2: Ontology Grounding
    # ------------------------------------------------------------------

    def retrieve_for_ontology_grounding(
        self,
        intent_result: Any,
        session_id: str = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve all candidate context for Step 2 (Ontology Grounding).

        Sources: detected intent, ontology subgraph, object aliases, rules
        """
        sid = session_id or self.session_id
        filters = filters or {}
        items: List[RetrievedContextItem] = []

        # 1. Intent result
        if intent_result:
            items.append(
                self._annotate(
                    key="intent_result",
                    value=intent_result if isinstance(intent_result, dict) else intent_result.to_s1_details(),
                    source="IntentRecognitionResult",
                    trust_level="high",
                )
            )
            if hasattr(intent_result, "intent"):
                intent_id = intent_result.intent
            elif isinstance(intent_result, dict):
                intent_id = intent_result.get("intent", "")
            else:
                intent_id = ""
            if hasattr(intent_result, "object_term"):
                object_term = intent_result.object_term
            elif isinstance(intent_result, dict):
                object_term = intent_result.get("object_term", "")
            else:
                object_term = ""
        else:
            intent_id = ""
            object_term = ""

        # 2. Ontology object subgraph
        try:
            ontology = self._get_ontology()
            from src.procurement.ontology_loader import OntologyQuery
            query = OntologyQuery(ontology)

            target_entity = query.find_object_by_term(object_term) if object_term else None

            if target_entity:
                items.append(
                    self._annotate(
                        key="target_entity",
                        value={
                            "name": target_entity.name,
                            "label": target_entity.label,
                            "description": target_entity.description,
                            "primary_key": target_entity.primary_key,
                            "fields": [
                                {"name": f.name, "type": f.type, "description": f.description}
                                for f in target_entity.fields
                            ],
                            "synonyms": target_entity.synonyms,
                            "keywords": target_entity.keywords,
                        },
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
                attrs = query.get_entity_attributes(target_entity.name)
                items.append(
                    self._annotate(key="entity_attributes", value=attrs, source="ProcurementOntology", trust_level="high")
                )
                actions = query.get_actions_for_entity(target_entity.name)
                items.append(
                    self._annotate(
                        key="entity_actions",
                        value=[{"id": a.id, "name": a.name, "kind": a.kind, "operation": a.operation, "description": a.description} for a in actions],
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
                related = query.get_related_entities(target_entity.name)
                items.append(
                    self._annotate(key="related_entities", value=related, source="ProcurementOntology", trust_level="medium")
                )
            else:
                items.append(
                    self._annotate(
                        key="ontology_datasets",
                        value=[{"name": ds.name, "label": ds.label, "description": ds.description} for ds in ontology.datasets.values()],
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )

            items.append(
                self._annotate(
                    key="ontology_relationships",
                    value=[{"name": r.name, "from": r.from_entity, "to": r.to_entity, "description": r.description} for r in ontology.relationships],
                    source="ProcurementOntology",
                    trust_level="medium",
                )
            )
            items.append(
                self._annotate(
                    key="ontology_actions",
                    value=[{"id": a.id, "name": a.name, "kind": a.kind, "entity_name": a.entity_name} for a in ontology.actions],
                    source="ProcurementOntology",
                    trust_level="high",
                )
            )
        except ImportError:
            pass

        # 3. Intent topology aliases
        try:
            topology = self._get_topology()
            for path in topology.paths:
                if path.id == intent_id or intent_id in path.id:
                    items.append(
                        self._annotate(
                            key="intent_path_aliases",
                            value={
                                "path_id": path.id,
                                "object": path.object,
                                "action": path.action,
                                "example_utterances": path.example_utterances,
                            },
                            source="IntentTopology",
                            trust_level="medium",
                        )
                    )
                    break
        except ImportError:
            pass

        # 4. Business rules
        try:
            from src.procurement.ontology_loader import OntologyQuery
            query = OntologyQuery(self._get_ontology())
            if intent_id:
                rules = query.get_business_rules_for_action(intent_id)
                if rules:
                    items.append(
                        self._annotate(
                            key="business_rules",
                            value=[{"id": r.id, "name": r.name, "severity": r.severity, "message": r.message, "remediation": r.remediation} for r in rules],
                            source="ProcurementOntology",
                            trust_level="high",
                        )
                    )
        except ImportError:
            pass

        # 5. Slot context
        if intent_result:
            if hasattr(intent_result, "extracted_slots") and intent_result.extracted_slots:
                items.append(
                    self._annotate(key="extracted_slots", value=intent_result.extracted_slots, source="SlotExtractor", trust_level="high")
                )
            if hasattr(intent_result, "temporal_context") and intent_result.temporal_context:
                items.append(
                    self._annotate(key="temporal_context", value=intent_result.temporal_context, source="SlotExtractor", trust_level="high")
                )

        # 6. Session context
        if sid:
            try:
                state_manager = self._get_state_manager()
                session = state_manager.get_session(sid)
                if session and session.context_data:
                    items.append(
                        self._annotate(key="session_context_data", value=session.context_data, source="DialogStateManager", trust_level="medium")
                    )
            except Exception:
                pass

        return self._build_result("ontology_grounding", items, filters).to_dict()

    # ------------------------------------------------------------------
    # Step 3: Task Planning
    # ------------------------------------------------------------------

    def retrieve_for_task_planning(
        self,
        intent_result: Any,
        ontology_result: Any,
        session_id: str = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve all candidate context for Step 3 (Task Planning).

        Sources: ontology grounding result, task templates, connectors, rules, risk policies
        """
        sid = session_id or self.session_id
        filters = filters or {}
        items: List[RetrievedContextItem] = []

        if intent_result:
            items.append(
                self._annotate(
                    key="intent_result",
                    value=intent_result if isinstance(intent_result, dict) else intent_result.to_s1_details(),
                    source="IntentRecognitionResult",
                    trust_level="high",
                )
            )

        if ontology_result:
            items.append(
                self._annotate(
                    key="ontology_result",
                    value=ontology_result if isinstance(ontology_result, dict) else ontology_result.to_s2_details(),
                    source="OntologyResolveResult",
                    trust_level="high",
                )
            )
            if hasattr(ontology_result, "available_actions"):
                available_actions = ontology_result.available_actions
            elif isinstance(ontology_result, dict):
                available_actions = ontology_result.get("available_actions", [])
            else:
                available_actions = []
        else:
            available_actions = []

        # Task templates from ontology scenarios
        try:
            ontology = self._get_ontology()
            intent_lower = ""
            if intent_result:
                intent_lower = (getattr(intent_result, "intent", "") or "").lower()

            relevant_scenarios = [
                s
                for s in ontology.scenarios
                if any(kw in s.description.lower() for kw in intent_lower.split("_") if len(kw) > 3)
            ]
            if relevant_scenarios:
                items.append(
                    self._annotate(
                        key="task_templates",
                        value=[{"name": s.name, "description": s.description, "mode": s.mode, "key_functions": s.key_functions} for s in relevant_scenarios[:3]],
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
            elif ontology.scenarios:
                items.append(
                    self._annotate(
                        key="task_templates",
                        value=[{"name": s.name, "description": s.description} for s in ontology.scenarios[:5]],
                        source="ProcurementOntology",
                        trust_level="medium",
                    )
                )
        except ImportError:
            pass

        # Connector registry
        try:
            from src.procurement.connector_registry import get_connector_registry
            registry = get_connector_registry()
            connector_info = []
            for action in available_actions:
                action_id = action.get("id", "") if isinstance(action, dict) else ""
                connector = registry.get(action_id)
                if connector:
                    connector_info.append({
                        "action_id": action_id,
                        "connector_name": connector.name,
                        "connector_type": getattr(connector, "connector_type", ""),
                    })
            if connector_info:
                items.append(
                    self._annotate(key="connectors", value=connector_info, source="ConnectorRegistry", trust_level="high")
                )
            all_connectors = registry.list_connectors()
            items.append(
                self._annotate(
                    key="available_connectors",
                    value=[c.get("name", "") for c in all_connectors],
                    source="ConnectorRegistry",
                    trust_level="medium",
                )
            )
        except ImportError:
            pass

        # Risk policies
        try:
            ontology = self._get_ontology()
            high_risk = [r for r in ontology.rules if r.severity == "error"]
            medium_risk = [r for r in ontology.rules if r.severity == "warn"]
            if high_risk:
                items.append(
                    self._annotate(
                        key="high_risk_rules",
                        value=[{"id": r.id, "name": r.name, "message": r.message, "remediation": r.remediation} for r in high_risk],
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
            if medium_risk:
                items.append(
                    self._annotate(
                        key="medium_risk_rules",
                        value=[{"id": r.id, "name": r.name, "message": r.message} for r in medium_risk],
                        source="ProcurementOntology",
                        trust_level="medium",
                    )
                )
        except ImportError:
            pass

        if available_actions:
            items.append(
                self._annotate(key="action_metadata", value=available_actions, source="OntologyResolveResult", trust_level="high")
            )

        if sid:
            try:
                state_manager = self._get_state_manager()
                session = state_manager.get_session(sid)
                if session:
                    state_summary = state_manager.get_state_summary(sid)
                    if state_summary:
                        items.append(
                            self._annotate(key="session_state", value=state_summary, source="DialogStateManager", trust_level="medium")
                        )
                    if session.context_data:
                        items.append(
                            self._annotate(key="session_context_data", value=session.context_data, source="DialogStateManager", trust_level="medium")
                        )
            except Exception:
                pass

        return self._build_result("task_planning", items, filters).to_dict()

    # ------------------------------------------------------------------
    # Step 4: Execution
    # ------------------------------------------------------------------

    def retrieve_for_execution(
        self,
        plan_result: Any,
        exec_state: Dict[str, Any],
        session_id: str = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve all candidate context for Step 4 (Execution).

        Sources: task plan, current step, intermediate results, connector results
        """
        sid = session_id or self.session_id
        filters = filters or {}
        items: List[RetrievedContextItem] = []

        if plan_result:
            items.append(
                self._annotate(
                    key="task_plan",
                    value=plan_result if isinstance(plan_result, dict) else plan_result.to_s3_details(),
                    source="TaskPlanResult",
                    trust_level="high",
                )
            )
            if hasattr(plan_result, "planned_actions"):
                planned_actions = plan_result.planned_actions
            elif isinstance(plan_result, dict):
                planned_actions = plan_result.get("planned_actions", [])
            else:
                planned_actions = []
        else:
            planned_actions = []

        if exec_state:
            items.append(
                self._annotate(key="execution_state", value=exec_state, source="ExecutionState", trust_level="high")
            )
            current_step = exec_state.get("current_step", 1)
            items.append(
                self._annotate(key="current_step", value=current_step, source="ExecutionState", trust_level="high")
            )
            intermediate = exec_state.get("intermediate_results", {})
            if intermediate:
                items.append(
                    self._annotate(key="intermediate_results", value=intermediate, source="ExecutionState", trust_level="medium")
                )

        # Connector configs
        try:
            from src.procurement.connector_registry import get_connector_registry
            registry = get_connector_registry()
            configs = []
            for action in planned_actions:
                action_id = getattr(action, "action_id", None) or (action.get("action_id") if isinstance(action, dict) else None)
                if action_id:
                    connector = registry.get(action_id)
                    if connector:
                        configs.append({
                            "action_id": action_id,
                            "connector_name": connector.name,
                            "connector_type": getattr(connector, "connector_type", ""),
                            "config": getattr(connector, "config", {}),
                        })
            if configs:
                items.append(
                    self._annotate(key="connector_configs", value=configs, source="ConnectorRegistry", trust_level="high")
                )
        except ImportError:
            pass

        if sid:
            try:
                state_manager = self._get_state_manager()
                session = state_manager.get_session(sid)
                if session and session.context_data.get("execution_history"):
                    items.append(
                        self._annotate(
                            key="execution_history",
                            value=session.context_data["execution_history"],
                            source="DialogStateManager",
                            trust_level="medium",
                        )
                    )
            except Exception:
                pass

        if plan_result:
            if hasattr(plan_result, "query_conditions"):
                query_conditions = plan_result.query_conditions
            elif isinstance(plan_result, dict):
                query_conditions = plan_result.get("query_conditions", [])
            else:
                query_conditions = []
            if query_conditions:
                items.append(
                    self._annotate(
                        key="query_conditions",
                        value=[c.to_dict() if hasattr(c, "to_dict") else c for c in query_conditions],
                        source="TaskPlanResult",
                        trust_level="high",
                    )
                )
            if hasattr(plan_result, "display_fields"):
                display_fields = plan_result.display_fields
            elif isinstance(plan_result, dict):
                display_fields = plan_result.get("display_fields", [])
            else:
                display_fields = []
            if display_fields:
                items.append(
                    self._annotate(key="display_fields", value=display_fields, source="TaskPlanResult", trust_level="high")
                )

        return self._build_result("execution", items, filters).to_dict()

    # ------------------------------------------------------------------
    # Step 5: Response Generation
    # ------------------------------------------------------------------

    def retrieve_for_response_generation(
        self,
        exec_result: Any,
        plan_result: Any,
        intent_result: Any,
        session_id: str = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve all candidate context for Step 5 (Response Generation).

        Sources: execution result, business explanation, rule results, next actions
        """
        sid = session_id or self.session_id
        filters = filters or {}
        items: List[RetrievedContextItem] = []

        if exec_result:
            items.append(
                self._annotate(
                    key="execution_result",
                    value=exec_result if isinstance(exec_result, dict) else exec_result.to_s4_details(),
                    source="ExecutionResult",
                    trust_level="high",
                )
            )
            if hasattr(exec_result, "executions"):
                executions = exec_result.executions
            elif isinstance(exec_result, dict):
                executions = exec_result.get("executions", [])
            else:
                executions = []
        else:
            executions = []

        if plan_result:
            items.append(
                self._annotate(
                    key="task_plan",
                    value=plan_result if isinstance(plan_result, dict) else plan_result.to_s3_details(),
                    source="TaskPlanResult",
                    trust_level="high",
                )
            )

        if intent_result:
            items.append(
                self._annotate(
                    key="intent_result",
                    value=intent_result if isinstance(intent_result, dict) else intent_result.to_s1_details(),
                    source="IntentRecognitionResult",
                    trust_level="high",
                )
            )

        # Execution summary
        if executions:
            total = len(executions)
            success = sum(
                1
                for e in executions
                if (hasattr(e, "status") and e.status == "success")
                or (isinstance(e, dict) and e.get("status") == "success")
            )
            failed = sum(
                1
                for e in executions
                if (hasattr(e, "status") and e.status == "failed")
                or (isinstance(e, dict) and e.get("status") == "failed")
            )
            items.append(
                self._annotate(
                    key="execution_summary",
                    value={"total": total, "success": success, "failed": failed, "success_rate": success / total if total > 0 else 0},
                    source="ExecutionResult",
                    trust_level="high",
                )
            )
            items.append(
                self._annotate(
                    key="execution_records",
                    value=[e.to_dict() if hasattr(e, "to_dict") else e for e in executions],
                    source="ExecutionResult",
                    trust_level="high",
                )
            )

        # Business explanation
        try:
            ontology = self._get_ontology()
            if ontology.instructions:
                items.append(
                    self._annotate(key="ontology_instructions", value=ontology.instructions, source="ProcurementOntology", trust_level="high")
                )
            if ontology.examples:
                items.append(
                    self._annotate(key="response_examples", value=ontology.examples, source="ProcurementOntology", trust_level="medium")
                )
        except ImportError:
            pass

        # Triggered rules
        try:
            from src.procurement.ontology_loader import OntologyQuery
            query = OntologyQuery(self._get_ontology())
            triggered_rules = []
            for exec_record in executions:
                action_id = getattr(exec_record, "action_id", None) or (exec_record.get("action_id") if isinstance(exec_record, dict) else None)
                if action_id:
                    rules = query.get_business_rules_for_action(action_id)
                    for rule in rules:
                        if rule not in triggered_rules:
                            triggered_rules.append(rule)
            if triggered_rules:
                items.append(
                    self._annotate(
                        key="triggered_rules",
                        value=[{"id": r.id, "name": r.name, "severity": r.severity, "message": r.message, "remediation": r.remediation} for r in triggered_rules],
                        source="ProcurementOntology",
                        trust_level="high",
                    )
                )
        except ImportError:
            pass

        # Next action suggestions
        try:
            from src.next_action import NextActionRecommender
            recommender = NextActionRecommender()
            intent_dict = intent_result.to_s1_details() if hasattr(intent_result, "to_s1_details") else (intent_result or {})
            exec_dict = exec_result.to_s4_details() if hasattr(exec_result, "to_s4_details") else (exec_result or {})
            suggestions = recommender.suggest_next_actions(intent=intent_dict, execution=exec_dict)
            if suggestions:
                items.append(
                    self._annotate(key="next_action_suggestions", value=suggestions, source="NextActionRecommender", trust_level="medium")
                )
        except ImportError:
            pass

        # Session history
        if sid:
            try:
                state_manager = self._get_state_manager()
                session = state_manager.get_session(sid)
                if session:
                    items.append(
                        self._annotate(
                            key="session_history",
                            value=session.messages[-10:] if session.messages else [],
                            source="DialogStateManager",
                            trust_level="medium",
                        )
                    )
                    if session.context_data:
                        items.append(
                            self._annotate(key="session_context_data", value=session.context_data, source="DialogStateManager", trust_level="medium")
                        )
            except Exception:
                pass

        return self._build_result("response_generation", items, filters).to_dict()


# =============================================================================
# Convenience Function
# =============================================================================


def retrieve_context_for_step(
    step: int,
    user_input: str = None,
    intent_result: Any = None,
    ontology_result: Any = None,
    plan_result: Any = None,
    exec_result: Any = None,
    exec_state: Dict[str, Any] = None,
    session_id: str = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to retrieve context for a specific step (1-5)."""
    retriever = ContextRetriever(session_id=session_id)
    if step == 1:
        return retriever.retrieve_for_intent_recognition(user_input, session_id, filters)
    elif step == 2:
        return retriever.retrieve_for_ontology_grounding(intent_result, session_id, filters)
    elif step == 3:
        return retriever.retrieve_for_task_planning(intent_result, ontology_result, session_id, filters)
    elif step == 4:
        return retriever.retrieve_for_execution(plan_result, exec_state or {}, session_id, filters)
    elif step == 5:
        return retriever.retrieve_for_response_generation(exec_result, plan_result, intent_result, session_id, filters)
    else:
        raise ValueError(f"Invalid step number: {step}. Must be 1-5.")
