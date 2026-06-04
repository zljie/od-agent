"""SSE event definitions and stream utilities.

Implements the SSE streaming protocol from docs/SSE流式响应规范.md:
- think / think_done  : model reasoning (non-required)
- content / done      : final response text
- tool_call / tool_result : skill/MCP/RAG invocations

Event format (per spec):
    event: <type>
    data: <json_payload>

    <double newline>

All events are yielded as dicts ready for sse_starlette.EventSourceResponse.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SSEEventType(str, Enum):
    THINK = "think"
    THINK_DONE = "think_done"
    CONTENT = "content"
    CONTENT_DELTA = "message.delta"
    CONTENT_COMPLETED = "message.completed"
    DONE = "done"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PLAN = "plan"
    STEP_UPDATE = "step_update"
    STEP_STARTED = "runtime.step.started"
    STEP_COMPLETED = "runtime.step.completed"
    STEP_FAILED = "runtime.step.failed"
    CONFIRM_REQUEST = "confirm_request"
    SLOT_FILL_REQUEST = "slot_fill_request"
    HITL_CREATED = "hitl.created"
    HITL_RESOLVED = "hitl.resolved"
    FRAME_INPUT = "frame.input.created"
    FRAME_SEMANTIC = "frame.semantic.created"
    FRAME_DECISION = "frame.decision.created"
    FRAME_EXECUTION = "frame.execution.created"
    FRAME_RESPONSE = "frame.response.created"
    ERROR_EVENT = "error"


# ─── Payload dataclasses ───────────────────────────────────────────────────────

@dataclass
class ToolCallPayload:
    type: str          # "skill" | "mcp" | "rag"
    name: str
    input: Dict[str, Any]
    id: str
    description: Optional[str] = None

    def to_event(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "id": self.id,
        }
        if self.description:
            data["description"] = self.description
        return {"event": SSEEventType.TOOL_CALL.value, "data": json.dumps(data, ensure_ascii=False)}


@dataclass
class ToolResultPayload:
    id: str
    name: str
    status: str        # "success" | "error" | "pending"
    output: Any = None
    error: Optional[str] = None

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.TOOL_RESULT.value,
            "data": json.dumps(
                {
                    "id": self.id,
                    "name": self.name,
                    "status": self.status,
                    "output": self.output,
                    "error": self.error,
                },
                ensure_ascii=False,
            ),
        }


@dataclass
class ThinkPayload:
    content: str

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.THINK.value,
            "data": json.dumps({"content": self.content}, ensure_ascii=False),
        }


@dataclass
class ContentPayload:
    content: str

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.CONTENT.value,
            "data": json.dumps({"content": self.content}, ensure_ascii=False),
        }


@dataclass
class DonePayload:
    """Sentinel event sent when the full stream is complete."""

    def to_event(self) -> Dict[str, Any]:
        return {"event": SSEEventType.DONE.value, "data": "[DONE]"}


@dataclass
class ThinkDonePayload:
    """Marks the end of the think phase."""

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.THINK_DONE.value,
            "data": json.dumps({"status": "done"}, ensure_ascii=False),
        }


@dataclass
class PlanPayload:
    """Plan metadata emitted before tool execution, as JSON blob."""
    plan_json: Dict[str, Any]

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.PLAN.value,
            "data": json.dumps({"plan": self.plan_json}, ensure_ascii=False),
        }


# ─── 5-Step Execution Payload dataclasses ─────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    WAITING_CONFIRMATION = "waiting_confirmation"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class AttributeInfo:
    key: str
    label: str
    usage: str  # 'filter' | 'display' | 'rule' | 'audit'


@dataclass
class OntologyGap:
    gap_type: str  # 'missing_object' | 'missing_attribute' | 'missing_action' | 'missing_rule'
    description: str
    suggestion: str


@dataclass
class PlannedAction:
    sequence: int
    action_id: str
    action_label: str
    connector: Optional[str] = None
    description: str = ""


@dataclass
class QueryCondition:
    field: str
    operator: str  # '=' | '!=' | '>' | '<' | '>=' | '<=' | 'IN' | 'LIKE'
    value: Any
    label: str


@dataclass
class AggregationRule:
    type: str  # 'count' | 'sum' | 'group_by'
    fields: List[str]
    label: str


@dataclass
class RecordStat:
    label: str
    value: str | float | int
    unit: Optional[str] = None


@dataclass
class SuggestedAction:
    id: str
    label: str
    type: str  # 'navigate' | 'execute' | 'export' | 'compose'
    params: Optional[Dict[str, Any]] = None


@dataclass
class ConnectorInfo:
    name: str
    id: str
    status: str  # 'pending' | 'success' | 'error'
    result_summary: Optional[str] = None
    result_count: Optional[int] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class ExecutionRecord:
    connector_name: str
    connector_id: str
    action_id: str
    status: str  # 'pending' | 'running' | 'success' | 'error' | 'timeout'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    latency_ms: Optional[float] = None
    request_params: Optional[Dict[str, Any]] = None
    response_summary: Optional[str] = None
    result_count: Optional[int] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class Step1Details:
    intent: str
    intent_label: str
    object_term: str
    normalized_term: Optional[str] = None
    operation_type: str = "query"
    risk_level: str = "low"
    requires_confirmation: bool = False
    requires_confirmation_reason: Optional[str] = None
    alternative_intents: Optional[List[str]] = None


@dataclass
class Step2Details:
    object_type: str
    object_label: str
    hit_keywords: List[str]
    attributes: List[AttributeInfo]
    available_actions: List[str]
    ontology_completeness: str = "full"  # 'full' | 'partial' | 'insufficient'
    gaps: Optional[List[OntologyGap]] = None


@dataclass
class Step3Details:
    planned_actions: List[PlannedAction]
    query_conditions: Optional[List[QueryCondition]] = None
    aggregation_rules: Optional[List[AggregationRule]] = None
    display_fields: Optional[List[str]] = None
    risk_level: str = "low"
    requires_confirmation: bool = False
    alternative_plans: Optional[List[List[PlannedAction]]] = None


@dataclass
class Step4Details:
    executions: List[ExecutionRecord]


@dataclass
class Step5Details:
    result_summary: str
    statistics: Optional[List[RecordStat]] = None
    result_definition: Optional[str] = None
    next_actions: List[SuggestedAction] = field(default_factory=list)
    ontology_improvements: Optional[List[OntologyGap]] = None


@dataclass
class StepUpdatePayload:
    step: int  # 1-5
    step_name: str
    status: StepStatus
    summary: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    connector: Optional[ConnectorInfo] = None
    suggested_actions: Optional[List[Dict[str, Any]]] = None

    def to_event(self) -> Dict[str, Any]:
        data = {
            "step": self.step,
            "stepName": self.step_name,
            "status": self.status.value if isinstance(self.status, StepStatus) else self.status,
        }
        if self.summary:
            data["summary"] = self.summary
        if self.details:
            details_dict = {}
            for k, v in self.details.items():
                if hasattr(v, "__dataclass_fields__"):
                    details_dict[k] = {f.name: getattr(v, f.name) for f in v.__dataclass_fields__.values()}
                else:
                    details_dict[k] = v
            data["details"] = details_dict
        if self.connector:
            data["connector"] = {
                "name": self.connector.name,
                "id": self.connector.id,
                "status": self.connector.status,
            }
            if self.connector.result_summary:
                data["connector"]["resultSummary"] = self.connector.result_summary
            if self.connector.result_count is not None:
                data["connector"]["resultCount"] = self.connector.result_count
            if self.connector.latency_ms is not None:
                data["connector"]["latencyMs"] = self.connector.latency_ms
            if self.connector.error_message:
                data["connector"]["errorMessage"] = self.connector.error_message
        if self.suggested_actions:
            data["suggestedActions"] = self.suggested_actions
        return {"event": SSEEventType.STEP_UPDATE.value, "data": json.dumps(data, ensure_ascii=False)}


@dataclass
class ConfirmRequestPayload:
    step: int
    title: str
    message: str
    action: Dict[str, Any]
    cancel_action: Dict[str, Any]
    risk_level: str
    task_id: str = ""  # Real task_id for HITL session store lookup
    affected_records: Optional[List[Dict[str, Any]]] = None
    detail: Optional[str] = None  # Additional context/reasoning
    alternatives: Optional[List[Dict[str, Any]]] = None  # Alternative options

    def to_event(self) -> Dict[str, Any]:
        data = {
            "step": self.step,
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "cancelAction": self.cancel_action,
            "riskLevel": self.risk_level,
            "taskId": self.task_id,
        }
        if self.affected_records:
            data["affectedRecords"] = self.affected_records
        if self.detail:
            data["detail"] = self.detail
        if self.alternatives:
            data["alternatives"] = self.alternatives
        return {"event": SSEEventType.CONFIRM_REQUEST.value, "data": json.dumps(data, ensure_ascii=False)}


@dataclass
class SlotFillPayload:
    """Slot filling form payload for HITL clarification.

    Returned when the pipeline needs the user to fill in required slots
    before execution can proceed. Frontend renders this as a structured form.

    Example rendered output:
        ┌─────────────────────────────────────────────────┐
        │ [!] 需要澄清                                     │
        ├─────────────────────────────────────────────────┤
        │ 好的，我来帮您创建采购需求。还需要补充以下信息：   │
        │ 1. 单据编号                                      │
        │ 2. 单据类型                                      │
        ├─────────────────────────────────────────────────┤
        │ * 单据编号                                       │
        │ [________________]                              │
        │                                                 │
        │ * 单据类型                                       │
        │ [请选择单据类型 ▼________________]              │
        │                                                 │
        │ ─────────────────────────────────────────────── │
        │                    [取消]  [创建采购需求]       │
        └─────────────────────────────────────────────────┘
    """

    id: str  # Unique request ID for correlation
    step: int
    title: str
    message: str
    action: Dict[str, Any]  # {"id": "...", "label": "..."}
    cancel_action: Dict[str, Any]  # {"id": "cancel", "label": "取消"}
    risk_level: str  # low / medium / high
    slots: List[Dict[str, Any]] = field(default_factory=list)
    # Each slot: {"id": "field_name", "label": "显示名", "type": "text|select|date|number",
    #             "required": true/false, "placeholder": "...", "options": [...]}
    alternatives: Optional[List[Dict[str, Any]]] = None
    detail: Optional[str] = None  # Additional context for AI reasoning

    def to_event(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "step": self.step,
            "title": self.title,
            "message": self.message,
            "slots": self.slots,
            "action": self.action,
            "cancelAction": self.cancel_action,
            "riskLevel": self.risk_level,
            "taskId": self.id,  # taskId in SSE data matches request_id for store lookup
        }
        if self.detail:
            data["detail"] = self.detail
        if self.alternatives:
            data["alternatives"] = self.alternatives
        return {"event": SSEEventType.SLOT_FILL_REQUEST.value, "data": json.dumps(data, ensure_ascii=False)}


@dataclass
class ErrorPayload:
    code: str
    message: str
    step: Optional[int] = None
    recoverable: bool = True
    suggestions: Optional[List[str]] = None

    def to_event(self) -> Dict[str, Any]:
        data = {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.step is not None:
            data["step"] = self.step
        if self.suggestions:
            data["suggestions"] = self.suggestions
        return {"event": SSEEventType.ERROR_EVENT.value, "data": json.dumps(data, ensure_ascii=False)}


# ─── Stream event union type ───────────────────────────────────────────────────

SSEEvent = (
    ToolCallPayload
    | ToolResultPayload
    | ThinkPayload
    | ContentPayload
    | DonePayload
    | ThinkDonePayload
    | PlanPayload
    | StepUpdatePayload
    | ConfirmRequestPayload
    | ErrorPayload
)


# ─── ID generator ─────────────────────────────────────────────────────────────

_tool_call_counter: int = 0


def new_tool_id() -> str:
    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_{_tool_call_counter:03d}"


def reset_tool_counter() -> None:
    global _tool_call_counter
    _tool_call_counter = 0


# ─── Builder helpers ───────────────────────────────────────────────────────────

def think(content: str) -> Dict[str, Any]:
    return ThinkPayload(content=content).to_event()


def think_done() -> Dict[str, Any]:
    return ThinkDonePayload().to_event()


def content(text: str) -> Dict[str, Any]:
    return ContentPayload(content=text).to_event()


def done() -> Dict[str, Any]:
    return DonePayload().to_event()


def tool_call(
    name: str,
    input_data: Dict[str, Any],
    type: str = "skill",
    tool_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    payload = ToolCallPayload(
        type=type,
        name=name,
        input=input_data,
        id=tool_id or new_tool_id(),
        description=description,
    )
    return payload.to_event()


def tool_result(
    tool_id: str,
    name: str,
    status: str,
    output: Any = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload = ToolResultPayload(
        id=tool_id,
        name=name,
        status=status,
        output=output,
        error=error,
    )
    return payload.to_event()


def plan_event(plan_json: Dict[str, Any]) -> Dict[str, Any]:
    """Emit plan metadata (intent reasoning + task plan) as a JSON SSE event."""
    return PlanPayload(plan_json=plan_json).to_event()


# ─── 5-Step Execution builder helpers ─────────────────────────────────────────

def step_update(
    step: int,
    step_name: str,
    status: str | StepStatus,
    summary: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    connector: Optional[ConnectorInfo] = None,
    suggested_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    st = StepStatus(status) if isinstance(status, str) else status
    payload = StepUpdatePayload(
        step=step, step_name=step_name, status=st,
        summary=summary, details=details, connector=connector,
        suggested_actions=suggested_actions
    )
    return payload.to_event()


def confirm_request(
    step: int,
    title: str,
    message: str,
    action_label: str = "确认执行",
    cancel_label: str = "取消",
    risk_level: str = "medium",
    task_id: str = "",
    affected_records: Optional[List[Dict[str, Any]]] = None,
    detail: Optional[str] = None,
    alternatives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = ConfirmRequestPayload(
        step=step, title=title, message=message,
        action={"id": "confirm", "label": action_label},
        cancel_action={"id": "cancel", "label": cancel_label},
        risk_level=risk_level,
        task_id=task_id,
        affected_records=affected_records,
        detail=detail, alternatives=alternatives
    )
    return payload.to_event()


def slot_fill_request(
    request_id: str,
    step: int,
    title: str,
    message: str,
    slots: List[Dict[str, Any]],
    action_label: str = "确认",
    cancel_label: str = "取消",
    risk_level: str = "medium",
    action_id: str = "slot_fill_submit",
    alternatives: Optional[List[Dict[str, Any]]] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a slot_fill_request event for HITL clarification.

    Parameters
    ----------
    request_id:
        Unique ID for this slot fill request (e.g., "slot-fill-pr-001").
        Frontend uses this to correlate submitted values.
    step:
        Pipeline step (usually 1).
    title:
        Dialog title shown at the top of the form.
    message:
        Introductory text explaining what the user needs to fill in.
    slots:
        List of slot definitions. Each dict contains:
        - id: field name (e.g., "material", "quantity")
        - label: human-readable label (e.g., "物料信息")
        - type: "text" | "select" | "date" | "number"
        - required: true/false
        - placeholder: hint text for text inputs
        - options: list of {value, label} for select inputs
    action_label:
        Label for the submit button (e.g., "创建采购需求").
    cancel_label:
        Label for the cancel button.
    risk_level:
        "low" | "medium" | "high" — affects form styling.
    action_id:
        Internal action ID for the submit button.
    alternatives:
        Alternative intent options the user can switch to.

    Example
    -------
    >>> slot_fill_request(
    ...     request_id="slot-fill-pr-001",
    ...     step=1,
    ...     title="需要澄清",
    ...     message="好的，我来帮您创建采购需求。还需要补充以下信息：\n1. 单据编号\n2. 单据类型",
    ...     slots=[
    ...         {"id": "document_id", "label": "单据编号", "type": "text",
    ...          "required": True, "placeholder": "请输入单据编号"},
    ...         {"id": "document_type", "label": "单据类型", "type": "select",
    ...          "required": True, "options": [
    ...              {"value": "PR", "label": "采购需求单"},
    ...              {"value": "PO", "label": "采购订单"},
    ...          ]},
    ...     ],
    ...     action_label="创建采购需求",
    ...     action_id="create_pr",
    ...     risk_level="medium",
    ... )
    """
    payload = SlotFillPayload(
        id=request_id,
        step=step,
        title=title,
        message=message,
        slots=slots,
        action={"id": action_id, "label": action_label},
        cancel_action={"id": "cancel", "label": cancel_label},
        risk_level=risk_level,
        alternatives=alternatives,
        detail=detail,
    )
    return payload.to_event()


def error_event(
    code: str,
    message: str,
    step: Optional[int] = None,
    recoverable: bool = True,
    suggestions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = ErrorPayload(
        code=code, message=message, step=step,
        recoverable=recoverable, suggestions=suggestions
    )
    return payload.to_event()


# ─── Frame Event Builders (per PRD Frontend Architecture Section 7) ──────────

def frame_input_event(
    task_id: str,
    trace_id: str,
    raw_input: str,
    normalized_input: str,
    input_type: str = "text",
) -> Dict[str, Any]:
    """Emit frame.input.created event when Input Frame is created (Step 1).

    Per PRD Section 7.2, this is emitted after preprocessing is complete.
    """
    return {
        "event": SSEEventType.FRAME_INPUT.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "rawInput": raw_input,
                "normalizedInput": normalized_input,
                "inputType": input_type,
            }
        }, ensure_ascii=False),
    }


def frame_semantic_event(
    task_id: str,
    trace_id: str,
    action_intent: str,
    ontology_object: str,
    conditions: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit frame.semantic.created event when Semantic Frame is created (Step 2).

    Per PRD Section 7.2, this is emitted after semantic understanding.
    """
    return {
        "event": SSEEventType.FRAME_SEMANTIC.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "actionIntent": action_intent,
                "ontologyObject": ontology_object,
                "conditions": conditions or [],
                "confidence": {"overall": confidence},
                "slots": slots or {},
            }
        }, ensure_ascii=False),
    }


def frame_decision_event(
    task_id: str,
    trace_id: str,
    decision: str,
    need_hitl: bool,
    route_runtime: str,
    route_target: str,
    reason: str = "",
) -> Dict[str, Any]:
    """Emit frame.decision.created event when Decision Frame is created (Step 3).

    Per PRD Section 7.2, this is emitted after routing decision.
    """
    return {
        "event": SSEEventType.FRAME_DECISION.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "decision": decision,
                "needHitl": need_hitl,
                "route": {
                    "runtime": route_runtime,
                    "target": route_target,
                },
                "reason": reason,
            }
        }, ensure_ascii=False),
    }


def frame_execution_event(
    task_id: str,
    trace_id: str,
    runtime: str,
    target: str,
    plan: Optional[List[Dict[str, Any]]] = None,
    status: str = "running",
) -> Dict[str, Any]:
    """Emit frame.execution.created event when Execution Frame is created (Step 4).

    Per PRD Section 7.2, this is emitted when execution starts.
    """
    return {
        "event": SSEEventType.FRAME_EXECUTION.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "runtime": runtime,
                "target": target,
                "plan": plan or [],
                "status": status,
            }
        }, ensure_ascii=False),
    }


def frame_response_event(
    task_id: str,
    trace_id: str,
    message_type: str,
    content: str,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Emit frame.response.created event when Response Frame is created (Step 5).

    Per PRD Section 7.2, this is emitted when the final response is generated.
    """
    return {
        "event": SSEEventType.FRAME_RESPONSE.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "messageType": message_type,
                "content": content,
                "traceId": trace_id,
                "actions": actions or [],
            }
        }, ensure_ascii=False),
    }


def hitl_created_event(
    task_id: str,
    trace_id: str,
    hitl_task_id: str,
    title: str,
    description: str,
    risk_level: str,
    fields: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Emit hitl.created event when a HITL task is created.

    Per PRD Section 7.2.
    """
    return {
        "event": SSEEventType.HITL_CREATED.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "hitlTaskId": hitl_task_id,
                "title": title,
                "description": description,
                "riskLevel": risk_level,
                "fields": fields or [],
                "actions": actions or [],
            }
        }, ensure_ascii=False),
    }


def hitl_resolved_event(
    task_id: str,
    trace_id: str,
    hitl_task_id: str,
    action: str,
    comment: str = "",
) -> Dict[str, Any]:
    """Emit hitl.resolved event when a HITL task is resolved.

    Per PRD Section 7.2.
    """
    return {
        "event": SSEEventType.HITL_RESOLVED.value,
        "data": json.dumps({
            "taskId": task_id,
            "traceId": trace_id,
            "payload": {
                "hitlTaskId": hitl_task_id,
                "action": action,
                "comment": comment,
            }
        }, ensure_ascii=False),
    }
