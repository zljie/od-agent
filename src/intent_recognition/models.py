"""
Intent Recognition Data Models
==============================
Phase 2a core dataclasses for the BeBISO 7-layer intent recognition framework.

Defines all result types produced by each layer:
- Layer 0: PreprocessingResult
- Layer 1: LLMLightResult
- Layer 2: OntologyMatchResult
- Layer 3/5: ConfidenceScore
- Layer 4: DeepReasoningResult
- Layer 6: HITLRequest / ClarificationOption
- Layer 7: ResolvedValue
- Composite: IntentCandidate / CompositeIntentResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Lazy forward reference to avoid circular import
# All references to IntentPath in docstrings point to src.intent_topology.models


# ---------------------------------------------------------------------------
# Layer 0 — Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingResult:
    """Layer 0 output: normalized and corrected user input."""

    original_input: str
    normalized_input: str
    corrections: List[Dict[str, str]] = field(default_factory=list)
    temporal_anchors: List[str] = field(default_factory=list)
    masked_terms: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Extended fields populated by InputPreprocessor
    tokens: List[str] = field(default_factory=list)
    candidate_terms: Dict[str, List[str]] = field(default_factory=dict)
    dynamic_resolution_tasks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_input": self.original_input,
            "normalized_input": self.normalized_input,
            "corrections": self.corrections,
            "temporal_anchors": self.temporal_anchors,
            "masked_terms": self.masked_terms,
            "metadata": self.metadata,
            "tokens": self.tokens,
            "candidate_terms": self.candidate_terms,
            "dynamic_resolution_tasks": self.dynamic_resolution_tasks,
        }


# ---------------------------------------------------------------------------
# Layer 1 — Light LLM Reasoning
# ---------------------------------------------------------------------------


@dataclass
class ObjectMatch:
    """A single ontology object matched from the user input."""

    object_name: str = ""
    matched_phrase: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_name": self.object_name,
            "matched_phrase": self.matched_phrase,
            "confidence": self.confidence,
        }

    @classmethod
    def from_topology_matcher(
        cls,
        object: str = "",
        aliases: Optional[List[str]] = None,
        score: float = 0.0,
    ) -> "ObjectMatch":
        """Factory: construct from topology_matcher.py's output fields."""
        return cls(
            object_name=object,
            matched_phrase=object,
            confidence=score,
            # Store aliases in a special metadata key
        )

    @property
    def object(self) -> str:
        return self.object_name

    @property
    def aliases(self) -> List[str]:
        return self.metadata.get("aliases", [])


@dataclass
class ActionMatch:
    """A single ontology action matched from the user input."""

    action_id: str = ""
    action_name: str = ""
    matched_phrase: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "matched_phrase": self.matched_phrase,
            "confidence": self.confidence,
        }

    @classmethod
    def from_topology_matcher(
        cls,
        raw_action: str = "",
        mapped_action: str = "",
        mapped_to: Optional[List[str]] = None,
        score: float = 0.0,
    ) -> "ActionMatch":
        """Factory: construct from topology_matcher.py's output fields."""
        return cls(
            action_id=mapped_action,
            action_name=raw_action,
            matched_phrase=raw_action,
            confidence=score,
        )

    @property
    def raw_action(self) -> str:
        return self.action_name

    @property
    def mapped_action(self) -> str:
        return self.action_id

    @property
    def mapped_to(self) -> List[str]:
        return self.metadata.get("mapped_to", [])


@dataclass
class DimensionMatch:
    """A single dimension / slot value matched from the user input."""

    dimension_name: str = ""
    value: str = ""
    matched_phrase: str = ""
    confidence: float = 0.0
    need_dynamic_lookup: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "value": self.value,
            "matched_phrase": self.matched_phrase,
            "confidence": self.confidence,
            "need_dynamic_lookup": self.need_dynamic_lookup,
        }

    @classmethod
    def from_topology_matcher(
        cls,
        raw_value: str = "",
        field: str = "",
        need_dynamic_lookup: bool = False,
        score: float = 0.0,
    ) -> "DimensionMatch":
        """Factory: construct from topology_matcher.py's output fields."""
        return cls(
            dimension_name=field,
            value=raw_value,
            matched_phrase=raw_value,
            confidence=score,
            need_dynamic_lookup=need_dynamic_lookup,
        )

    @property
    def raw_value(self) -> str:
        return self.value

    @property
    def field(self) -> str:
        return self.dimension_name


@dataclass
class LLMLightResult:
    """Layer 1 output: lightweight LLM reasoning over preprocessed input."""

    primary_intent: str = ""
    secondary_intents: List[str] = field(default_factory=list)
    object_matches: List[ObjectMatch] = field(default_factory=list)
    action_matches: List[ActionMatch] = field(default_factory=list)
    dimension_matches: List[DimensionMatch] = field(default_factory=list)
    raw_reasoning: str = ""
    confidence_a: float = 0.0
    missing_slots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "secondary_intents": self.secondary_intents,
            "object_matches": [m.to_dict() for m in self.object_matches],
            "action_matches": [m.to_dict() for m in self.action_matches],
            "dimension_matches": [m.to_dict() for m in self.dimension_matches],
            "raw_reasoning": self.raw_reasoning,
            "confidence_a": self.confidence_a,
            "missing_slots": self.missing_slots,
            "metadata": self.metadata,
        }

    @property
    def intent_template(self) -> str:
        return self.primary_intent

    @property
    def object_candidate(self) -> str:
        if self.object_matches:
            return self.object_matches[0].object_name
        return ""

    @property
    def object_label(self) -> str:
        if self.object_matches:
            return self.object_matches[0].matched_phrase
        return ""

    @property
    def action_candidate(self) -> str:
        if self.action_matches:
            return self.action_matches[0].action_name
        return ""

    @property
    def filters(self) -> Dict[str, Any]:
        return {
            d.dimension_name: d.value
            for d in self.dimension_matches
            if d.dimension_name
        }

    @classmethod
    def from_light_reasoner_output(
        cls,
        intent_template: str = "",
        object_candidate: str = "",
        object_label: str = "",
        action_candidate: str = "",
        filters: Optional[Dict[str, Any]] = None,
        ambiguities: Optional[List[str]] = None,
        candidate_next_goals: Optional[List[str]] = None,
        missing_info: Optional[List[str]] = None,
        confidence_breakdown: Optional[Dict[str, float]] = None,
        A_score: float = 0.0,
        raw_reasoning: str = "",
        secondary_intents: Optional[List[str]] = None,
    ) -> "LLMLightResult":
        """Factory: construct from light_reasoner.py's output fields."""
        object_match = None
        if object_candidate:
            object_match = ObjectMatch(
                object_name=object_candidate,
                matched_phrase=object_label or object_candidate,
                confidence=confidence_breakdown.get("object_score", 0.0) if confidence_breakdown else 0.0,
            )
        action_match = None
        if action_candidate:
            action_match = ActionMatch(
                action_name=action_candidate,
                matched_phrase=action_candidate,
                confidence=confidence_breakdown.get("action_score", 0.0) if confidence_breakdown else 0.0,
            )
        dimension_matches = [
            DimensionMatch(dimension_name=k, value=str(v))
            for k, v in (filters or {}).items()
        ]
        return cls(
            primary_intent=intent_template,
            secondary_intents=secondary_intents or [],
            object_matches=[object_match] if object_match else [],
            action_matches=[action_match] if action_match else [],
            dimension_matches=dimension_matches,
            raw_reasoning=raw_reasoning,
            confidence_a=A_score,
            missing_slots=missing_info or [],
            metadata={
                "ambiguities": ambiguities or [],
                "candidate_next_goals": candidate_next_goals or [],
                "confidence_breakdown": confidence_breakdown or {},
            },
        )


    @property
    def filters(self) -> Dict[str, Any]:
        return {
            d.dimension_name: d.value
            for d in self.dimension_matches
            if d.dimension_name
        }


# ---------------------------------------------------------------------------
# Layer 2 — Ontology Topology Match
# ---------------------------------------------------------------------------


@dataclass
class OntologyMatchResult:
    """Layer 2 output: ontology topology match results.

    Matches Layer-1 output against the intent topology and procurement ontology.
    Produces the B_score used in Layer-3 confidence fusion.
    """

    object_match: Optional[ObjectMatch] = None
    action_match: Optional[ActionMatch] = None
    dimension_matches: List[DimensionMatch] = field(default_factory=list)
    matched_path_ids: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    confidence_b: float = 0.0
    missing_slots: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_match": self.object_match.to_dict() if self.object_match else None,
            "action_match": self.action_match.to_dict() if self.action_match else None,
            "dimension_matches": [d.to_dict() for d in self.dimension_matches],
            "matched_path_ids": self.matched_path_ids,
            "gaps": self.gaps,
            "confidence_b": self.confidence_b,
            "missing_slots": self.missing_slots,
            "reason": self.reason,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Layers 3 & 5 — Confidence Scoring
# ---------------------------------------------------------------------------


class ConfidenceDecision(str, Enum):
    """Decision outcome from confidence fusion."""

    EXECUTE = "execute"
    CONTINUE = "continue"
    DEEP_REASONING = "deep_reasoning"
    HITL = "hitl"
    BLOCKED = "blocked"


@dataclass
class ConfidenceScore:
    """Aggregated confidence result after fusion."""

    score_ab: float = 0.0
    score_abc: float = 0.0
    final_decision: ConfidenceDecision = ConfidenceDecision.CONTINUE
    threshold_ab: float = 0.75
    threshold_abc: float = 0.75
    confidence_a: float = 0.0
    confidence_b: float = 0.0
    confidence_c: float = 0.0
    deep_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_ab": self.score_ab,
            "score_abc": self.score_abc,
            "final_decision": self.final_decision.value,
            "threshold_ab": self.threshold_ab,
            "threshold_abc": self.threshold_abc,
            "confidence_a": self.confidence_a,
            "confidence_b": self.confidence_b,
            "confidence_c": self.confidence_c,
            "deep_triggered": self.deep_triggered,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Layer 4 — Deep Reasoning
# ---------------------------------------------------------------------------


@dataclass
class DeepReasoningResult:
    """Layer 4 output: deep reasoning result from heavy LLM."""

    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    recommended_decision: str = ""
    confidence_c: float = 0.0
    reasoning_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranked_candidates": self.ranked_candidates,
            "missing_info": self.missing_info,
            "recommended_decision": self.recommended_decision,
            "confidence_c": self.confidence_c,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Layer 6 — HITL
# ---------------------------------------------------------------------------


@dataclass
class ClarificationOption:
    """A single clarification option presented to the user."""

    option_id: str
    label: str
    description: str = ""
    intent_path_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HITLRequest:
    """Layer 6 output: request for human-in-the-loop clarification."""

    question: str = ""
    options: List[ClarificationOption] = field(default_factory=list)
    recommended_default: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": [
                {
                    "option_id": o.option_id,
                    "label": o.label,
                    "description": o.description,
                    "intent_path_id": o.intent_path_id,
                    "params": o.params,
                }
                for o in self.options
            ],
            "recommended_default": self.recommended_default,
            "context": self.context,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Layer 7 — Dynamic Resolution
# ---------------------------------------------------------------------------


class ResolutionStatus(str, Enum):
    """Status of a dynamic term resolution."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass
class ResolvedValue:
    """Layer 7 output: resolved value for a dynamic term."""

    term: str
    term_type: str
    status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "term_type": self.term_type,
            "status": self.status.value,
            "candidates": self.candidates,
            "selected": self.selected,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Layer 2.5 — Slot Completion
# ---------------------------------------------------------------------------


@dataclass
class SlotCheckResult:
    """Layer 2.5 output: slot readiness check for action execution gate.

    Produced by SlotCompletionEngine; consumed by fusion_engine.fuse_abc()
    to override confidence-based decisions when required slots are missing.
    """

    intent_template: str = ""
    object_name: str = ""
    action_readiness_score: float = 0.0          # 0.0–100.0
    decision: str = "execute"                     # clarify | suggest | execute
    filled_required: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    filled_optional: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    schema_found: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_template": self.intent_template,
            "object_name": self.object_name,
            "action_readiness_score": self.action_readiness_score,
            "decision": self.decision,
            "filled_required": self.filled_required,
            "missing_required": self.missing_required,
            "filled_optional": self.filled_optional,
            "missing_optional": self.missing_optional,
            "schema_found": self.schema_found,
            "metadata": self.metadata,
        }

    @property
    def has_missing_required(self) -> bool:
        return len(self.missing_required) > 0


# ---------------------------------------------------------------------------
# Composite Result
# ---------------------------------------------------------------------------


@dataclass
class IntentCandidate:
    """A single ranked intent candidate."""

    rank: int
    intent_id: str
    intent_name: str
    confidence: float
    matched_path_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeIntentResult:
    """Final composite result from the full 7-layer pipeline."""

    raw_input: str = ""
    normalized_input: str = ""
    candidates: List[IntentCandidate] = field(default_factory=list)
    top_candidate: Optional[IntentCandidate] = None
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore)
    hitl_request: Optional[HITLRequest] = None
    resolved_values: List[ResolvedValue] = field(default_factory=list)
    layer_results: Dict[str, Any] = field(default_factory=dict)
    final_decision: ConfidenceDecision = ConfidenceDecision.CONTINUE
    execution_ready: bool = False
    slot_check_result: Optional["SlotCheckResult"] = None
    task_id: str = ""
    ontology_version: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "normalized_input": self.normalized_input,
            "candidates": [
                {
                    "rank": c.rank,
                    "intent_id": c.intent_id,
                    "intent_name": c.intent_name,
                    "confidence": c.confidence,
                    "matched_path_id": c.matched_path_id,
                    "params": c.params,
                    "missing_slots": c.missing_slots,
                    "metadata": c.metadata,
                }
                for c in self.candidates
            ],
            "top_candidate": (
                {
                    "rank": self.top_candidate.rank,
                    "intent_id": self.top_candidate.intent_id,
                    "intent_name": self.top_candidate.intent_name,
                    "confidence": self.top_candidate.confidence,
                    "matched_path_id": self.top_candidate.matched_path_id,
                    "params": self.top_candidate.params,
                    "missing_slots": self.top_candidate.missing_slots,
                }
                if self.top_candidate
                else None
            ),
            "confidence": self.confidence.to_dict(),
            "hitl_request": self.hitl_request.to_dict() if self.hitl_request else None,
            "resolved_values": [rv.to_dict() for rv in self.resolved_values],
            "layer_results": self.layer_results,
            "final_decision": self.final_decision.value,
            "execution_ready": self.execution_ready,
            "slot_check_result": self.slot_check_result.to_dict() if self.slot_check_result else None,
            "task_id": self.task_id,
            "ontology_version": self.ontology_version,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
