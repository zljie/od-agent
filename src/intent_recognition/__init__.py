"""
Intent Recognition Module
========================
Phase 2a (core) and Phase 2b (layers 3-7) of the BeBISO intent recognition upgrade.

Implements the 7-layer intent recognition framework:
- Layer 0: InputPreprocessor (preprocessing)
- Layer 1: LightIntentReasoner (light LLM)
- Layer 2: OntologyTopologyMatcher (ontology match)
- Layer 3: ConfidenceFusionEngine (AB fusion)
- Layer 4: DeepIntentReasoner (deep reasoning)
- Layer 5: ConfidenceFusionEngine (ABC fusion)
- Layer 6: HITLClarificationEngine (human-in-the-loop)
- Layer 7: DynamicSlotResolver (dynamic resolution)

Exports:
    PreprocessingResult, LLMLightResult, ObjectMatch, ActionMatch,
    DimensionMatch, OntologyMatchResult, ConfidenceScore, IntentCandidate,
    DeepReasoningResult, ClarificationOption, HITLRequest, ResolvedValue,
    CompositeIntentResult, ConfidenceDecision, ResolutionStatus, IntentLayerEvent,
    IntentRecognitionAuditRecord, IntentRecognitionAuditLogger
    InputPreprocessor, LightIntentReasoner, OntologyTopologyMatcher,
    ConfidenceFusionEngine, DeepIntentReasoner, HITLClarificationEngine,
    DynamicSlotResolver, get_intent_audit_logger
"""

from .models import (
    PreprocessingResult,
    LLMLightResult,
    ObjectMatch,
    ActionMatch,
    DimensionMatch,
    OntologyMatchResult,
    ConfidenceScore,
    IntentCandidate,
    DeepReasoningResult,
    ClarificationOption,
    HITLRequest,
    ResolvedValue,
    CompositeIntentResult,
    ConfidenceDecision,
    ResolutionStatus,
)
from .audit import (
    IntentLayerEvent,
    IntentRecognitionAuditRecord,
    IntentRecognitionAuditLogger,
    get_intent_audit_logger,
)
from .preprocessor import InputPreprocessor
from .light_reasoner import LightIntentReasoner
from .topology_matcher import OntologyTopologyMatcher
from .pipeline import CompositeIntentPipeline
from .fusion_engine import ConfidenceFusionEngine
from .deep_reasoner import DeepIntentReasoner
from .hitl_engine import HITLClarificationEngine
from .dynamic_resolver import DynamicSlotResolver

__all__ = [
    # Models
    "PreprocessingResult",
    "LLMLightResult",
    "ObjectMatch",
    "ActionMatch",
    "DimensionMatch",
    "OntologyMatchResult",
    "ConfidenceScore",
    "IntentCandidate",
    "DeepReasoningResult",
    "ClarificationOption",
    "HITLRequest",
    "ResolvedValue",
    "CompositeIntentResult",
    "ConfidenceDecision",
    "ResolutionStatus",
    # Engines
    "ConfidenceFusionEngine",
    "DeepIntentReasoner",
    "HITLClarificationEngine",
    "DynamicSlotResolver",
    # Preprocessors
    "InputPreprocessor",
    "LightIntentReasoner",
    "OntologyTopologyMatcher",
    "CompositeIntentPipeline",
    # Audit
    "IntentLayerEvent",
    "IntentRecognitionAuditRecord",
    "IntentRecognitionAuditLogger",
    "get_intent_audit_logger",
]
