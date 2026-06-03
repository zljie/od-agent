"""
Context Engineering Layer — Data Models
=======================================
BeBISO 9-layer context model (L1–L9) and the standard LLMContextPackage.

This module defines all dataclasses used to construct the context package
passed to downstream LLM calls. Each dataclass provides a `to_dict()` method
for serialization.

Reference: BeBISO_上下文工程层_Context_Engineering_标准方案.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrustLevel(str, Enum):
    """Overall trust level of the context package."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContextPurpose(str, Enum):
    """Purpose / stage of the LLM call receiving this context package."""

    INTENT_RECOGNITION = "intent_recognition"
    ONTOLOGY_GROUNDING = "ontology_grounding"
    TASK_PLANNING = "task_planning"
    EXECUTION_ASSIST = "execution_assist"
    RESPONSE_GENERATION = "response_generation"


class RiskLevel(str, Enum):
    """Risk level for a tool definition."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    """Severity of a policy rule."""

    INFO = "info"
    WARNING = "warning"
    MANDATORY = "mandatory"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContextMeta:
    """Metadata header for the context package."""

    purpose: ContextPurpose = ContextPurpose.INTENT_RECOGNITION
    task_id: str = ""
    ontology_version: str = ""
    context_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "task_id": self.task_id,
            "ontology_version": self.ontology_version,
            "context_version": self.context_version,
        }


@dataclass
class ContextInstructions:
    """System and developer instructions for the LLM."""

    system_instruction: str = ""
    developer_instruction: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_instruction": self.system_instruction,
            "developer_instruction": self.developer_instruction,
        }


@dataclass
class TokenBudget:
    """Token budget constraints for the context package."""

    max_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    hard_limit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "hard_limit": self.hard_limit,
        }


@dataclass
class SlotDefinition:
    """A resolved slot (dimension/key-value pair) in the context."""

    name: str
    value: Any
    source: str = ""  # e.g. "user_input", "ontology_inference", "dynamic_lookup"
    confidence: float = 1.0
    raw_phrase: str = ""  # The phrase from user input that resolved to this slot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "raw_phrase": self.raw_phrase,
        }


@dataclass
class ToolDefinition:
    """A single tool available to the LLM."""

    name: str
    type: str = "function"  # function|search|calculation|...
    risk_level: RiskLevel = RiskLevel.LOW
    description: str = ""
    confirmation_required: bool = False
    use_condition: str = ""  # Human-readable condition for when to use this tool
    parameters: Dict[str, Any] = field(default_factory=dict)  # OpenAPI-style schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "confirmation_required": self.confirmation_required,
            "use_condition": self.use_condition,
            "parameters": self.parameters,
        }


@dataclass
class RuleDefinition:
    """A single policy / safety rule."""

    id: str
    severity: Severity = Severity.INFO
    description: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "description": self.description,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# L1–L9 Layer dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContextUserInput:
    """L1 — Raw user input and L0 preprocessing results."""

    raw_input: str = ""
    normalized_input: str = ""
    corrections: List[Dict[str, str]] = field(default_factory=list)  # [{from, to, reason}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "normalized_input": self.normalized_input,
            "corrections": self.corrections,
        }


@dataclass
class ContextPreprocessing:
    """L2 — Tokenization and term analysis from preprocessing layer."""

    tokens: List[str] = field(default_factory=list)
    dynamic_terms: List[str] = field(default_factory=list)  # Terms needing dynamic resolution
    candidate_terms: List[str] = field(default_factory=list)  # Disambiguation candidates
    temporal_anchors: List[str] = field(default_factory=list)  # Time references (e.g. "this month")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "dynamic_terms": self.dynamic_terms,
            "candidate_terms": self.candidate_terms,
            "temporal_anchors": self.temporal_anchors,
        }


@dataclass
class ContextIntent:
    """L3 — Detected intent, slots, and confidence from intent recognition."""

    detected_intent: str = ""
    slots: List[SlotDefinition] = field(default_factory=list)
    confidence: float = 0.0
    ambiguities: List[str] = field(default_factory=list)  # Ambiguous terms flagged for disambiguation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_intent": self.detected_intent,
            "slots": [s.to_dict() for s in self.slots],
            "confidence": self.confidence,
            "ambiguities": self.ambiguities,
        }


@dataclass
class ContextOntology:
    """L4 — Ontology grounding: resolved object, action, slots, relationships, rules."""

    resolved_object: str = ""
    resolved_action: str = ""
    resolved_slots: List[SlotDefinition] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)  # [{from, to, type}]
    rules: List[str] = field(default_factory=list)  # Applicable ontology rule IDs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved_object": self.resolved_object,
            "resolved_action": self.resolved_action,
            "resolved_slots": [s.to_dict() for s in self.resolved_slots],
            "relationships": self.relationships,
            "rules": self.rules,
        }


@dataclass
class ContextBusiness:
    """L5 — Dynamic business values fetched from external sources."""

    # Keyed by type, e.g. "user_context", "catalog", "recent_orders", "budget"
    dynamic_values: Dict[str, Any] = field(default_factory=dict)
    last_query_result: Optional[Dict[str, Any]] = None
    fetched_at: Optional[str] = None  # ISO 8601 timestamp
    source: str = ""  # e.g. "erp", "crm", "catalog"
    trust_level: TrustLevel = TrustLevel.HIGH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dynamic_values": self.dynamic_values,
            "last_query_result": self.last_query_result,
            "fetched_at": self.fetched_at,
            "source": self.source,
            "trust_level": self.trust_level.value,
        }


@dataclass
class ContextDialog:
    """L6 — Dialog state: history summary and pending clarifications."""

    previous_task_summary: str = ""
    last_selected_objects: List[str] = field(default_factory=list)
    unresolved_clarifications: List[str] = field(default_factory=list)  # Pending clarification questions
    user_confirmations: List[Dict[str, Any]] = field(default_factory=list)  # Confirmed values from user

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_task_summary": self.previous_task_summary,
            "last_selected_objects": self.last_selected_objects,
            "unresolved_clarifications": self.unresolved_clarifications,
            "user_confirmations": self.user_confirmations,
        }


@dataclass
class ContextTool:
    """L7 — Available tools and their risk/condition metadata."""

    available_tools: List[ToolDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_tools": [t.to_dict() for t in self.available_tools],
        }


@dataclass
class ContextPolicy:
    """L8 — Policy rules, permissions, and safety constraints."""

    rules: List[RuleDefinition] = field(default_factory=list)
    permissions: Dict[str, Any] = field(default_factory=dict)  # e.g. {can_export: true, max_results: 100}
    safety_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "permissions": self.permissions,
            "safety_constraints": self.safety_constraints,
        }


@dataclass
class ContextOutput:
    """L9 — Output constraints for the LLM response."""

    format: str = "text"  # text|markdown|json|table|...
    schema: Optional[Dict[str, Any]] = None  # Expected JSON schema if format=json
    max_tokens: int = 0
    language: str = "auto"  # auto|zh|en|...
    style: str = "concise"  # concise|detailed|technical|...
    forbidden: List[str] = field(default_factory=list)  # Phrases/patterns to avoid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "schema": self.schema,
            "max_tokens": self.max_tokens,
            "language": self.language,
            "style": self.style,
            "forbidden": self.forbidden,
        }


# ---------------------------------------------------------------------------
# LLMContextPackage — the standard context package
# ---------------------------------------------------------------------------


@dataclass
class LLMContextPackage:
    """Standard 9-layer context package for LLM calls.

    Composes all L1–L9 layers plus metadata, instructions, audit fields,
    and trust/freshness signals into a single structure.
    """

    meta: ContextMeta = field(default_factory=ContextMeta)
    instructions: ContextInstructions = field(default_factory=ContextInstructions)
    user_input: ContextUserInput = field(default_factory=ContextUserInput)
    preprocessing: ContextPreprocessing = field(default_factory=ContextPreprocessing)
    intent_context: ContextIntent = field(default_factory=ContextIntent)
    ontology_context: ContextOntology = field(default_factory=ContextOntology)
    business_context: ContextBusiness = field(default_factory=ContextBusiness)
    dialog_context: ContextDialog = field(default_factory=ContextDialog)
    tool_context: ContextTool = field(default_factory=ContextTool)
    policy_context: ContextPolicy = field(default_factory=ContextPolicy)
    output_constraints: ContextOutput = field(default_factory=ContextOutput)

    # Audit fields
    included_sources: List[str] = field(default_factory=list)  # Which data sources were included
    excluded_sources: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{source, reason}] entries for excluded sources
    token_budget: Optional[TokenBudget] = None

    # Trust and freshness
    trust_level: TrustLevel = TrustLevel.HIGH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "instructions": self.instructions.to_dict(),
            "user_input": self.user_input.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "intent_context": self.intent_context.to_dict(),
            "ontology_context": self.ontology_context.to_dict(),
            "business_context": self.business_context.to_dict(),
            "dialog_context": self.dialog_context.to_dict(),
            "tool_context": self.tool_context.to_dict(),
            "policy_context": self.policy_context.to_dict(),
            "output_constraints": self.output_constraints.to_dict(),
            "included_sources": self.included_sources,
            "excluded_sources": self.excluded_sources,
            "token_budget": self.token_budget.to_dict() if self.token_budget else None,
            "trust_level": self.trust_level.value,
        }
