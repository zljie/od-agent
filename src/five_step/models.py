"""Shared data models for the 5-step pipeline."""

from dataclasses import dataclass, field
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

    @property
    def summary(self) -> str:
        base = f"识别为【{self.intent_label}】"
        if self.condition_summary:
            base += f"，条件：{self.condition_summary}"
        return base

    def to_s1_details(self) -> Dict[str, Any]:
        return {
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
        }


@dataclass
class OntologyResolveResult:
    """Result from Step 2: Ontology Resolution."""
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
    """Result from Step 3: Task Planning."""
    planned_actions: List[PlannedActionItem]
    query_conditions: List[QueryConditionItem] = field(default_factory=list)
    aggregation_rules: List[Dict[str, Any]] = field(default_factory=list)
    display_fields: List[str] = field(default_factory=list)
    risk_level: str = "low"
    requires_confirmation: bool = False
    alternative_plans: List[List[PlannedActionItem]] = field(default_factory=list)
    plan_summary: str = ""  # User-readable plan summary

    @property
    def summary(self) -> str:
        if not self.planned_actions:
            return "无行动计划"
        action_labels = " → ".join(a.action_label for a in self.planned_actions)
        return f"执行计划：{action_labels}"

    def to_s3_details(self) -> Dict[str, Any]:
        return {
            "plannedActions": [a.to_dict() for a in self.planned_actions],
            "queryConditions": [c.to_dict() for c in self.query_conditions],
            "aggregationRules": self.aggregation_rules,
            "displayFields": self.display_fields,
            "riskLevel": self.risk_level,
            "requiresConfirmation": self.requires_confirmation,
            "alternativePlans": [[a.to_dict() for a in plan] for plan in self.alternative_plans],
            "planSummary": self.plan_summary,
        }


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
