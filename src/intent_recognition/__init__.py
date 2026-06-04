"""
Intent Recognition Module
========================
Based on BeBIOS Intent Processing Architecture v1.0

Implements the simplified 6-stage intent recognition framework:
- Input Preprocessor
- LLM Semantic Parser
- Ontology Matcher
- Action Intent Normalizer
- Slot Extractor
- Decision Fusion

The new simplified pipeline provides:
- Unified SemanticFrame output format
- Configuration-driven behavior
- Route Policy for routing decisions
- Clarification Policy for user interactions

For backward compatibility, the old 7-layer pipeline is still available
but the new SimplifiedIntentPipeline is recommended.
"""

# ============================================================================
# Old exports (backward compatibility)
# ============================================================================

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
from .fusion_engine import ConfidenceFusionEngine
from .deep_reasoner import DeepIntentReasoner
from .hitl_engine import HITLClarificationEngine
from .dynamic_resolver import DynamicSlotResolver


# ============================================================================
# New simplified pipeline exports
# ============================================================================

from .config import (
    IntentRecognitionConfig,
    get_default_config,
    set_default_config,
)

from .semantic_frame import (
    SemanticFrame,
    InputInfo,
    SemanticInfo,
    ConfidenceInfo,
    GovernanceInfo,
    DecisionInfo,
    Condition,
    TimeRange,
    DecisionType,
    RiskLevel,
    RuntimeTarget,
    IntentCandidate,  # Re-export for compatibility
    ClarificationOption as ClarificationOptionNew,
    HITLRequest as HITLRequestNew,
)

from .action_normalizer import (
    ActionIntentNormalizer,
    ActionIntentNormalizer,
    normalize_action,
    get_default_normalizer,
)

from .slot_extractor import (
    SlotExtractor,
    SlotExtractor,
    SlotDefinition,
    SlotExtractionResult,
    extract_slots,
    get_default_extractor,
)

from .decision_fusion import (
    DecisionFusion,
    FusionWeights,
    FusionThresholds,
    FusionResult,
    fuse_decision,
    get_default_fusion,
)

from .policies import (
    RoutePolicy,
    RoutePolicyEngine,
    get_default_route_policy,
    ClarificationPolicy,
    ClarificationEngine,
    get_default_clarification,
)

from .core_pipeline import (
    CoreIntentPipeline,
    PipelineConfig,
    PipelineResult,
    get_default_pipeline,
    recognize_intent,
)

# Alias for backward compatibility
SimplifiedIntentPipeline = CoreIntentPipeline

# Import old pipeline for backward compatibility (used by step1_intent.py)
from .pipeline import CompositeIntentPipeline as OldCompositePipeline


# Alias for backward compatibility
CompositeIntentPipeline = OldCompositePipeline

__all__ = [
    # ========================================================================
    # Configuration
    # ========================================================================
    "IntentRecognitionConfig",
    "get_default_config",
    "set_default_config",

    # ========================================================================
    # Semantic Frame (standard output)
    # ========================================================================
    "SemanticFrame",
    "InputInfo",
    "SemanticInfo",
    "ConfidenceInfo",
    "GovernanceInfo",
    "DecisionInfo",
    "Condition",
    "TimeRange",

    # Enums
    "DecisionType",
    "RiskLevel",
    "RuntimeTarget",

    # ========================================================================
    # Core Pipeline
    # ========================================================================
    "CoreIntentPipeline",
    "PipelineConfig",
    "PipelineResult",
    "recognize_intent",
    "get_default_pipeline",

    # Backward compatibility
    "SimplifiedIntentPipeline",

    # ========================================================================
    # Core Components
    # ========================================================================
    "ActionIntentNormalizer",
    "SlotExtractor",
    "SlotDefinition",
    "SlotExtractionResult",
    "DecisionFusion",

    # ========================================================================
    # Policies
    # ========================================================================
    "RoutePolicy",
    "RoutePolicyEngine",
    "get_default_route_policy",
    "ClarificationPolicy",
    "ClarificationEngine",
    "get_default_clarification",

    # ========================================================================
    # Backward Compatibility (old exports)
    # ========================================================================
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
    "ClarificationOption",  # Old version
    "HITLRequest",  # Old version
    "ResolvedValue",
    "CompositeIntentResult",
    "ConfidenceDecision",
    "ResolutionStatus",

    # Engines
    "ConfidenceFusionEngine",
    "DeepIntentReasoner",
    "HITLClarificationEngine",
    "DynamicSlotResolver",

    # Pipeline
    "OldCompositePipeline",  # Old pipeline (from pipeline.py)
    "CompositeIntentPipeline",  # Alias for backward compatibility

    # Audit
    "IntentLayerEvent",
    "IntentRecognitionAuditRecord",
    "IntentRecognitionAuditLogger",
    "get_intent_audit_logger",
]
