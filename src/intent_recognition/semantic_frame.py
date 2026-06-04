"""
Semantic Frame Models
=====================
Standard output format for the BeBIOS intent recognition pipeline.

Based on Section 6 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Example:
    frame = SemanticFrame.from_input("帮我查一下采购需求")
    print(frame.to_dict())
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecisionType(str, Enum):
    """Decision types for the intent recognition pipeline."""
    CONTINUE = "continue"      # Proceed to routing
    CLARIFY = "clarify"       # Need user clarification
    REJECT = "reject"         # Not allowed to execute
    HITL = "hitl"             # Need human confirmation
    FALLBACK = "fallback"     # Fallback strategy
    REPLAN = "replan"         # Need re-planning


class RiskLevel(str, Enum):
    """Risk levels for actions."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RuntimeTarget(str, Enum):
    """Runtime routing targets."""
    PLANNER = "planner"
    WORKFLOW = "workflow"
    SKILL = "skill"
    TOOL = "tool"
    HITL = "hitl"
    CLARIFICATION = "clarification"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Condition Model
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """A filter condition extracted from user input."""
    property: str              # Ontology property name (e.g., "executionStatus")
    operator: str = "="       # Comparison operator
    value: Any = None         # Condition value
    raw: Optional[str] = None # Original user expression

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property,
            "operator": self.operator,
            "value": self.value,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        return cls(
            property=data.get("property", ""),
            operator=data.get("operator", "="),
            value=data.get("value"),
            raw=data.get("raw"),
        )


# ---------------------------------------------------------------------------
# Time Range Model
# ---------------------------------------------------------------------------

@dataclass
class TimeRange:
    """Time range specification."""
    start: Optional[str] = None  # ISO format or natural language
    end: Optional[str] = None
    expression: Optional[str] = None  # Natural language expression (e.g., "本月", "最近一周")
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "expression": self.expression,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# Sub-Frames
# ---------------------------------------------------------------------------

@dataclass
class InputInfo:
    """User input information."""
    raw_text: str
    normalized_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rawText": self.raw_text,
            "normalizedText": self.normalized_text,
        }


@dataclass
class SemanticInfo:
    """Semantic understanding results."""
    action_intent: str = ""                    # Normalized action (e.g., "Query")
    action_candidate: str = ""                 # Original action from LLM
    ontology_object: str = ""                  # Ontology object (e.g., "PurchaseRequirement")
    object_candidate: str = ""                 # Original object from LLM
    conditions: List[Condition] = field(default_factory=list)
    time_range: Optional[TimeRange] = None
    slots: Dict[str, Any] = field(default_factory=dict)  # Extracted slots

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actionIntent": self.action_intent,
            "actionCandidate": self.action_candidate,
            "ontologyObject": self.ontology_object,
            "objectCandidate": self.object_candidate,
            "conditions": [c.to_dict() for c in self.conditions],
            "timeRange": self.time_range.to_dict() if self.time_range else None,
            "slots": self.slots,
        }


@dataclass
class ConfidenceInfo:
    """Confidence scores from various stages."""
    llm_parse: float = 0.0   # LLM semantic parsing
    action: float = 0.0     # Action normalization
    ontology_match: float = 0.0  # Ontology matching
    slot_fill: float = 0.0   # Slot completion
    overall: float = 0.0     # Overall confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llmParse": self.llm_parse,
            "action": self.action,
            "ontologyMatch": self.ontology_match,
            "slotFill": self.slot_fill,
            "overall": self.overall,
        }


@dataclass
class GovernanceInfo:
    """Governance and policy information."""
    risk_level: RiskLevel = RiskLevel.LOW
    need_hitl: bool = False
    permission_passed: bool = True
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "riskLevel": self.risk_level.value,
            "needHitl": self.need_hitl,
            "permissionPassed": self.permission_passed,
            "blockedReason": self.blocked_reason,
        }


@dataclass
class DecisionInfo:
    """Routing decision information."""
    type: DecisionType = DecisionType.CONTINUE
    next: RuntimeTarget = RuntimeTarget.WORKFLOW
    target: str = ""                          # Target ID (e.g., "query_purchase_requirement_workflow")
    reason: str = ""                          # Decision reason
    clarify_question: Optional[str] = None    # Clarification question if needed
    clarify_options: List[str] = field(default_factory=list)  # Clarification options

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "next": self.next.value,
            "target": self.target,
            "reason": self.reason,
            "clarifyQuestion": self.clarify_question,
            "clarifyOptions": self.clarify_options,
        }


# ---------------------------------------------------------------------------
# Main Semantic Frame
# ---------------------------------------------------------------------------

@dataclass
class SemanticFrame:
    """Standard semantic frame output from intent recognition.

    This is the unified output format defined in BeBIOS architecture Section 6.

    Example:
        frame = SemanticFrame(
            input=InputInfo(raw_text="帮我查采购需求"),
            semantic=SemanticInfo(action_intent="Query", ontology_object="PurchaseRequirement"),
            confidence=ConfidenceInfo(llm_parse=0.9, overall=0.85),
            governance=GovernanceInfo(risk_level=RiskLevel.LOW),
            decision=DecisionInfo(type=DecisionType.CONTINUE, next=RuntimeTarget.WORKFLOW)
        )
    """
    input: InputInfo
    semantic: SemanticInfo = field(default_factory=SemanticInfo)
    confidence: ConfidenceInfo = field(default_factory=ConfidenceInfo)
    governance: GovernanceInfo = field(default_factory=GovernanceInfo)
    decision: DecisionInfo = field(default_factory=DecisionInfo)

    # Metadata
    task_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    layer_results: Dict[str, Any] = field(default_factory=dict)  # Debug info
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input": self.input.to_dict(),
            "semantic": self.semantic.to_dict(),
            "confidence": self.confidence.to_dict(),
            "governance": self.governance.to_dict(),
            "decision": self.decision.to_dict(),
            "taskId": self.task_id,
            "timestamp": self.timestamp,
            "layerResults": self.layer_results,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticFrame":
        """Create from dictionary."""
        input_info = InputInfo(
            raw_text=data.get("input", {}).get("rawText", ""),
            normalized_text=data.get("input", {}).get("normalizedText", ""),
        )

        sem_data = data.get("semantic", {})
        conditions = [Condition.from_dict(c) for c in sem_data.get("conditions", [])]
        time_range_data = sem_data.get("timeRange")
        time_range = TimeRange(**time_range_data) if time_range_data else None

        semantic = SemanticInfo(
            action_intent=sem_data.get("actionIntent", ""),
            action_candidate=sem_data.get("actionCandidate", ""),
            ontology_object=sem_data.get("ontologyObject", ""),
            object_candidate=sem_data.get("objectCandidate", ""),
            conditions=conditions,
            time_range=time_range,
            slots=sem_data.get("slots", {}),
        )

        conf_data = data.get("confidence", {})
        confidence = ConfidenceInfo(
            llm_parse=conf_data.get("llmParse", 0.0),
            action=conf_data.get("action", 0.0),
            ontology_match=conf_data.get("ontologyMatch", 0.0),
            slot_fill=conf_data.get("slotFill", 0.0),
            overall=conf_data.get("overall", 0.0),
        )

        gov_data = data.get("governance", {})
        governance = GovernanceInfo(
            risk_level=RiskLevel(gov_data.get("riskLevel", "Low")),
            need_hitl=gov_data.get("needHitl", False),
            permission_passed=gov_data.get("permissionPassed", True),
            blocked_reason=gov_data.get("blockedReason"),
        )

        dec_data = data.get("decision", {})
        decision = DecisionInfo(
            type=DecisionType(dec_data.get("type", "continue")),
            next=RuntimeTarget(dec_data.get("next", "workflow")),
            target=dec_data.get("target", ""),
            reason=dec_data.get("reason", ""),
            clarify_question=dec_data.get("clarifyQuestion"),
            clarify_options=dec_data.get("clarifyOptions", []),
        )

        return cls(
            input=input_info,
            semantic=semantic,
            confidence=confidence,
            governance=governance,
            decision=decision,
            task_id=data.get("taskId", ""),
            timestamp=data.get("timestamp", ""),
            layer_results=data.get("layerResults", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_input(cls, raw_text: str, normalized_text: str = "") -> "SemanticFrame":
        """Create a basic semantic frame from input text."""
        return cls(
            input=InputInfo(raw_text=raw_text, normalized_text=normalized_text or raw_text),
        )

    # Convenience properties
    @property
    def action(self) -> str:
        return self.semantic.action_intent

    @property
    def object(self) -> str:
        return self.semantic.ontology_object

    @property
    def is_ready(self) -> bool:
        """Check if the frame is ready for execution."""
        return (
            self.decision.type == DecisionType.CONTINUE
            and self.governance.permission_passed
            and self.governance.blocked_reason is None
        )

    @property
    def needs_clarification(self) -> bool:
        """Check if clarification is needed."""
        return self.decision.type == DecisionType.CLARIFY

    @property
    def needs_hitl(self) -> bool:
        """Check if HITL is needed."""
        return self.governance.need_hitl or self.decision.type == DecisionType.HITL


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------

@dataclass
class IntentCandidate:
    """Backward-compatible intent candidate (from old models)."""
    rank: int = 1
    intent_id: str = ""
    intent_name: str = ""
    confidence: float = 0.0
    matched_path_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)


@dataclass
class ClarificationOption:
    """Clarification option."""
    text: str
    value: str
    description: Optional[str] = None


@dataclass
class HITLRequest:
    """Backward-compatible HITL request model."""
    question: str
    options: List[ClarificationOption] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": [
                {"text": o.text, "value": o.value, "description": o.description}
                for o in self.options
            ],
            "context": self.context,
        }
