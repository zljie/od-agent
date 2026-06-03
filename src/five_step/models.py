"""Shared data models for the 5-step pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class IntentRecognitionResult:
    """Result from Step 1: Intent Recognition."""
    intent: str
    intent_label: str
    object_term: str
    normalized_term: Optional[str] = None
    operation_type: str = "query"  # query/create/update/delete/publish/recommend
    risk_level: str = "low"
    requires_confirmation: bool = False
    requires_confirmation_reason: Optional[str] = None
    confidence: float = 1.0
    alternative_intents: List[str] = field(default_factory=list)
    # Extended fields for slot extraction
    extracted_slots: Optional[Dict[str, Any]] = None  # Extracted condition slots
    temporal_context: Optional[Dict[str, Any]] = None  # Time range info
    condition_summary: str = ""  # Human-readable condition summary
    # Phase 3: Composite pipeline result (Layers 0-7)
    composite_result: Optional[Any] = None  # CompositeIntentResult from 7-layer pipeline
    hitl_request: Optional[Any] = None  # HITLRequest if composite pipeline needs clarification
    composite_layer_results: Optional[Dict[str, Any]] = None  # Per-layer results dict
    composite_task_id: Optional[str] = None  # Task ID used by composite pipeline
    # Phase 4: Semantic Contract - the single source of truth for downstream steps
    semantic_contract: Optional["SemanticContract"] = None

    @property
    def summary(self) -> str:
        base = f"识别为【{self.intent_label}】"
        if self.condition_summary:
            base += f"，条件：{self.condition_summary}"
        return base

    def to_s1_details(self) -> Dict[str, Any]:
        details = {
            "intent": self.intent,
            "intentLabel": self.intent_label,
            "objectTerm": self.object_term,
            "normalizedTerm": self.normalized_term,
            "operationType": self.operation_type,
            "riskLevel": self.risk_level,
            "requiresConfirmation": self.requires_confirmation,
            "requiresConfirmationReason": self.requires_confirmation_reason,
            "alternativeIntents": self.alternative_intents,
            "extractedSlots": self.extracted_slots,
            "temporalContext": self.temporal_context,
            "conditionSummary": self.condition_summary,
            # Phase 3 composite pipeline fields
            "compositeLayerResults": self.composite_layer_results,
            "compositeTaskId": self.composite_task_id,
        }
        # Add Semantic Contract info if available
        if self.semantic_contract:
            details["semanticContract"] = self.semantic_contract.to_dict()
            details["semanticContractSummary"] = {
                "object": self.semantic_contract.object,
                "objectLabel": self.semantic_contract.object_label,
                "action": self.semantic_contract.action,
                "actionLabel": self.semantic_contract.action_label,
                "confidence": self.semantic_contract.confidence,
                "slotCount": len(self.semantic_contract.slots),
            }
        return details


@dataclass
class OntologyResolveResult:
    """Result from Step 2: Ontology Expansion."""
    object_type: str
    object_label: str
    hit_keywords: List[str]
    attributes: List[Dict[str, Any]]
    available_actions: List[Dict[str, Any]]
    ontology_completeness: str = "full"  # full/partial/insufficient
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    # HITL fields for LLM-assisted resolution
    llm_inferred: bool = False  # Whether this was inferred by LLM
    llm_reasoning: Optional[str] = None  # LLM reasoning for the match
    alternatives: List[Dict[str, str]] = field(default_factory=list)  # Alternative matches
    # Phase 4: Ontology expansion data
    ontology_expansion: Dict[str, Any] = field(default_factory=dict)
    # Reference to Semantic Contract (for diagnostics)
    semantic_contract_ref: Optional[Any] = None

    @property
    def summary(self) -> str:
        return f"命中【{self.object_label}】对象"

    def to_s2_details(self) -> Dict[str, Any]:
        return {
            "objectType": self.object_type,
            "objectLabel": self.object_label,
            "hitKeywords": self.hit_keywords,
            "attributes": self.attributes,
            "availableActions": [a.get("id", "") for a in self.available_actions],
            "ontologyCompleteness": self.ontology_completeness,
            "gaps": self.gaps,
            # HITL fields
            "llmInferred": self.llm_inferred,
            "llmReasoning": self.llm_reasoning,
            "alternatives": self.alternatives,
            # Phase 4: Ontology expansion
            "ontologyExpansion": self.ontology_expansion,
        }


@dataclass
class PlannedActionItem:
    """A single action in the task plan."""
    sequence: int
    action_id: str
    action_label: str
    connector: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "actionId": self.action_id,
            "actionLabel": self.action_label,
            "connector": self.connector,
            "description": self.description,
        }


@dataclass
class QueryConditionItem:
    """A query filter condition."""
    field: str
    operator: str
    value: Any
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "label": self.label,
        }


@dataclass
class TaskPlanResult:
    """Result from Step 3: Action Planning."""
    planned_actions: List[PlannedActionItem]
    query_conditions: List[QueryConditionItem] = field(default_factory=list)
    aggregation_rules: List[Dict[str, Any]] = field(default_factory=list)
    display_fields: List[str] = field(default_factory=list)
    risk_level: str = "low"
    requires_confirmation: bool = False
    alternative_plans: List[List[PlannedActionItem]] = field(default_factory=list)
    plan_summary: str = ""  # User-readable plan summary
    # Phase 4: Semantic Contract reference
    semantic_contract: Optional[Any] = None
    # Phase 4: Action drift detection
    action_drift_detected: bool = False
    drift_reason: str = ""

    @property
    def summary(self) -> str:
        if not self.planned_actions:
            return "无行动计划"
        action_labels = " → ".join(a.action_label for a in self.planned_actions)
        return f"执行计划：{action_labels}"

    def to_s3_details(self) -> Dict[str, Any]:
        details = {
            "plannedActions": [a.to_dict() for a in self.planned_actions],
            "queryConditions": [c.to_dict() for c in self.query_conditions],
            "aggregationRules": self.aggregation_rules,
            "displayFields": self.display_fields,
            "riskLevel": self.risk_level,
            "requiresConfirmation": self.requires_confirmation,
            "alternativePlans": [[a.to_dict() for a in plan] for plan in self.alternative_plans],
            "planSummary": self.plan_summary,
        }
        # Phase 4: Add action drift detection
        if self.action_drift_detected:
            details["actionDrift"] = {
                "detected": True,
                "reason": self.drift_reason,
            }
        # Phase 4: Add Semantic Contract summary
        if self.semantic_contract:
            sc = self.semantic_contract
            details["semanticContractSummary"] = {
                "object": sc.object,
                "objectLabel": sc.object_label,
                "action": sc.action,
                "actionLabel": sc.action_label,
                "confidence": sc.confidence,
                "slotCount": len(sc.slots) if sc.slots else 0,
            }
        return details


@dataclass
class ExecutionRecord:
    """A single connector execution record."""
    connector_name: str
    connector_id: str
    action_id: str
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    latency_ms: Optional[float] = None
    request_params: Optional[Dict[str, Any]] = None
    response_summary: Optional[str] = None
    result_count: Optional[int] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connectorName": self.connector_name,
            "connectorId": self.connector_id,
            "actionId": self.action_id,
            "status": self.status,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "latencyMs": self.latency_ms,
            "requestParams": self.request_params,
            "responseSummary": self.response_summary,
            "resultCount": self.result_count,
            "errorMessage": self.error_message,
            "errorCode": self.error_code,
        }


@dataclass
class ExecutionResult:
    """Result from Step 4: Execution."""
    executions: List[ExecutionRecord]
    task_id: str = ""

    @property
    def summary(self) -> str:
        if not self.executions:
            return "无执行记录"
        total = len(self.executions)
        success = sum(1 for e in self.executions if e.status == "success")
        return f"执行完成：{success}/{total} 成功"

    @property
    def all_success(self) -> bool:
        return all(e.status == "success" for e in self.executions)

    def to_s4_details(self) -> Dict[str, Any]:
        return {
            "executions": [e.to_dict() for e in self.executions],
        }


@dataclass
class SuggestedAction:
    """A suggested next action for the user."""
    id: str
    label: str
    type: str = "navigate"  # navigate/execute/export/compose
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "params": self.params,
        }


@dataclass
class SemanticContractSlot:
    """A single slot/parameter in the semantic contract."""
    name: str
    display_value: str  # Human-readable value (e.g., "销售部")
    field_name: str = ""  # Backend field name (e.g., "apply_dep")
    field_value: Any = None  # Resolved value for query
    resolved: bool = False
    source: str = "extracted"  # extracted/computed/mapped
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_value": self.display_value,
            "field_name": self.field_name,
            "field_value": self.field_value,
            "resolved": self.resolved,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class SemanticContract:
    """Semantic Contract - the single source of truth that flows through the pipeline.

    This contract is created at Step 1 (Intent Recognition) and MUST be preserved
    through Steps 2-4. No step is allowed to overwrite the core action.

    The contract contains:
    - object: The business object (e.g., "purchase_requests")
    - object_label: Human-readable label (e.g., "采购需求")
    - action: The canonical action ID (e.g., "analytics/find_unexecuted_purchase_requests")
    - slots: Key-value parameters extracted from user input
    - alternatives: Alternative candidates ranked by confidence

    This prevents "Semantic Collapse" where high-confidence intent recognition
    results get overwritten by lower-confidence downstream steps.
    """
    object: str  # Canonical object name (e.g., "purchase_requests")
    object_label: str  # Human-readable label (e.g., "采购需求")
    action: str  # Canonical action ID (e.g., "analytics/find_unexecuted_purchase_requests")
    action_label: str = ""  # Human-readable action name
    confidence: float = 0.0
    slots: Dict[str, SemanticContractSlot] = field(default_factory=dict)
    # Alternative actions (for diagnostics and fallback)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    # Expanded ontology context (populated by Step 2)
    ontology_expansion: Dict[str, Any] = field(default_factory=dict)
    # Planned parameters (populated by Step 3)
    planned_params: Dict[str, Any] = field(default_factory=dict)
    # Diagnostic info
    action_drift_detected: bool = False
    drift_reason: str = ""
    created_at: str = ""  # ISO timestamp
    # HITL info
    hitl_triggered: bool = False
    hitl_reason: str = ""
    missing_info: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.action_label:
            self.action_label = self._default_action_label()

    def _default_action_label(self) -> str:
        """Generate default action label from action ID."""
        labels = {
            "purchase_requests/list": "查询采购需求",
            "purchase_inquiries/list": "查询询价单",
            "purchase_quotations/list": "查询报价单",
            "purchase_order_heads/list": "查询采购订单",
            "analytics/find_unexecuted_purchase_requests": "查询未执行采购需求",
            "analytics/generate_price_comparison": "生成比价分析",
            "analytics/get_purchase_order_execution_status": "查询订单执行状态",
        }
        return labels.get(self.action, self.action.split("/")[-1])

    @property
    def primary_slot_names(self) -> List[str]:
        """Return list of slot names that are critical for the action."""
        return list(self.slots.keys())

    def get_slot(self, name: str) -> Optional[SemanticContractSlot]:
        return self.slots.get(name)

    def has_drift(self) -> bool:
        """Check if semantic drift has been detected."""
        return self.action_drift_detected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.object,
            "object_label": self.object_label,
            "action": self.action,
            "action_label": self.action_label,
            "confidence": self.confidence,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "alternatives": self.alternatives,
            "ontology_expansion": self.ontology_expansion,
            "planned_params": self.planned_params,
            "action_drift_detected": self.action_drift_detected,
            "drift_reason": self.drift_reason,
            "created_at": self.created_at,
            "hitl_triggered": self.hitl_triggered,
            "hitl_reason": self.hitl_reason,
            "missing_info": self.missing_info,
        }

    @classmethod
    def from_composite_result(cls, composite_result, intent_result) -> "SemanticContract":
        """Create SemanticContract from composite pipeline result.

        This is the primary factory method - it extracts the canonical action
        from the deep reasoning layer (Layer 4) which has the highest quality intent.
        """
        top = composite_result.top_candidate
        if not top:
            raise ValueError("Cannot create SemanticContract without top candidate")

        # Extract slots from top candidate params
        slots = {}
        if top.params:
            for key, value in top.params.items():
                if key not in ("object_term", "normalized_object_term", "intent_id", "intent_name"):
                    slots[key] = SemanticContractSlot(
                        name=key,
                        display_value=str(value),
                        source="extracted",
                    )

        # Add temporal context if available
        if intent_result and intent_result.temporal_context:
            for temporal_key in ("date_from", "date_to", "label"):
                if temporal_key in intent_result.temporal_context:
                    slots[f"time_{temporal_key}"] = SemanticContractSlot(
                        name=f"time_{temporal_key}",
                        display_value=str(intent_result.temporal_context[temporal_key]),
                        source="computed",
                    )

        # Build alternatives from composite candidates
        alternatives = []
        for cand in composite_result.candidates[1:5]:  # Skip first (it's the top)
            alternatives.append({
                "intent_id": cand.intent_id,
                "intent_name": cand.intent_name,
                "confidence": cand.confidence,
                "params": cand.params,
            })

        # Check deep reasoning for HITL info
        hitl_triggered = False
        hitl_reason = ""
        missing_info = []
        deep_result = composite_result.layer_results.get("layer_4_deep_reasoning", {})
        if deep_result:
            hitl_triggered = deep_result.get("recommended_decision") == "hitl"
            missing_info = deep_result.get("missing_info", [])

        # Extract object from intent_id
        # Format 1: "object/action" (e.g., "purchase_requests/list") -> object = purchase_requests
        # Format 2: "namespace/action" (e.g., "analytics/find_unexecuted_purchase_requests") -> need further parsing
        intent_id = top.intent_id or ""
        parts = intent_id.split("/")
        if len(parts) >= 2:
            object_from_id = parts[0]
            action_part = parts[1]
            # If first part is a known namespace (analytics, workflow, etc.), extract object from action name
            known_namespaces = {"analytics", "workflow", "system", "admin"}
            if object_from_id in known_namespaces:
                # Action like "find_unexecuted_purchase_requests" -> extract "purchase_requests"
                if action_part.startswith("find_unexecuted_"):
                    object_from_id = action_part.replace("find_unexecuted_", "")
                elif "_" in action_part:
                    object_from_id = action_part.split("_")[0]
        else:
            object_from_id = ""

        # Also check alternatives for a valid object/action format
        if not object_from_id or object_from_id in known_namespaces:
            for alt in alternatives:
                alt_id = alt.get("intent_id", "")
                if "/" in alt_id:
                    potential_object = alt_id.split("/")[0]
                    if potential_object not in known_namespaces:
                        object_from_id = potential_object
                        break

        return cls(
            object=top.params.get("object_term", object_from_id or intent_result.object_term if intent_result else object_from_id or "unknown"),
            object_label=intent_result.object_term if intent_result else top.params.get("object_term", object_from_id.replace("_", " ").title() if object_from_id else "业务对象"),
            action=top.intent_id,
            action_label=top.intent_name or top.intent_id,
            confidence=top.confidence,
            slots=slots,
            alternatives=alternatives,
            hitl_triggered=hitl_triggered,
            hitl_reason=hitl_reason,
            missing_info=missing_info,
        )


@dataclass
class ResponseResult:
    """Result from Step 5: Response Generation."""
    text: str
    result_summary: str = ""
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    result_definition: Optional[str] = None
    next_actions: List[SuggestedAction] = field(default_factory=list)
    ontology_improvements: List[Dict[str, Any]] = field(default_factory=list)

    def to_s5_details(self) -> Dict[str, Any]:
        return {
            "resultSummary": self.result_summary,
            "statistics": self.statistics,
            "resultDefinition": self.result_definition,
            "nextActions": [a.to_dict() for a in self.next_actions],
            "ontologyImprovements": self.ontology_improvements,
        }
