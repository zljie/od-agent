"""
Core Intent Recognition Pipeline
================================
Unified 6-stage intent recognition pipeline.

Based on BeBIOS Intent Processing Architecture v1.0

Architecture:
    Input → Preprocess → LLM Parse → Ontology Match → Action Normalize → 
    Slot Extract → Decision Fusion → Semantic Frame

This is the SINGLE canonical intent recognition pipeline.
Use this instead of the deprecated 7-layer pipeline (pipeline.py).
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
)
from .preprocessor import InputPreprocessor
from .light_reasoner import LightIntentReasoner
from .topology_matcher import OntologyTopologyMatcher
from .action_normalizer import ActionIntentNormalizer
from .slot_extractor import SlotExtractor, SlotExtractionResult
from .decision_fusion import DecisionFusion, FusionResult
from .policies import RoutePolicyEngine, ClarificationEngine


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the simplified pipeline."""

    # Layer enable/disable flags
    use_ontology_match: bool = True
    use_action_normalize: bool = True
    use_slot_extraction: bool = True
    use_routing: bool = True

    # Decision thresholds
    continue_threshold: float = 0.70
    clarify_threshold: float = 0.50
    high_confidence_threshold: float = 0.85

    # HITL settings
    force_hitl_on_create: bool = True
    force_hitl_on_high_risk: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_ontology_match": self.use_ontology_match,
            "use_action_normalize": self.use_action_normalize,
            "use_slot_extraction": self.use_slot_extraction,
            "use_routing": self.use_routing,
            "continue_threshold": self.continue_threshold,
            "clarify_threshold": self.clarify_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "force_hitl_on_create": self.force_hitl_on_create,
            "force_hitl_on_high_risk": self.force_hitl_on_high_risk,
        }


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result from the simplified pipeline."""
    frame: SemanticFrame
    layer_results: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "layerResults": self.layer_results,
            "durationMs": self.duration_ms,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Simplified Pipeline
# ---------------------------------------------------------------------------

class CoreIntentPipeline:
    """Core 6-stage intent recognition pipeline.

    This is the SINGLE canonical intent recognition pipeline for BeBIOS.
    Use this instead of the deprecated 7-layer pipeline (pipeline.py).

    Example:
        pipeline = CoreIntentPipeline()
        result = pipeline.run("帮我查一下采购需求")
        print(result.frame.decision.type)  # DecisionType.CONTINUE
        print(result.frame.decision.target)  # "workflow.query_purchase_requirement"
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        route_policy: Optional[RoutePolicyEngine] = None,
        clarification: Optional[ClarificationEngine] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
            route_policy: Route policy engine
            clarification: Clarification engine
        """
        self.config = config or PipelineConfig()
        self.task_id = self._generate_task_id()

        # Initialize components
        self._preprocessor: Optional[InputPreprocessor] = None
        self._light_reasoner: Optional[LightIntentReasoner] = None
        self._topology_matcher: Optional[OntologyTopologyMatcher] = None
        self._action_normalizer: Optional[ActionIntentNormalizer] = None
        self._slot_extractor: Optional[SlotExtractor] = None
        self._decision_fusion: Optional[DecisionFusion] = None
        self._route_policy = route_policy
        self._clarification = clarification

    # -------------------------------------------------------------------------
    # Component properties (lazy initialization)
    # -------------------------------------------------------------------------

    @property
    def preprocessor(self) -> InputPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = InputPreprocessor()
        return self._preprocessor

    @property
    def light_reasoner(self) -> LightIntentReasoner:
        if self._light_reasoner is None:
            self._light_reasoner = LightIntentReasoner()
        return self._light_reasoner

    @property
    def topology_matcher(self) -> OntologyTopologyMatcher:
        if self._topology_matcher is None:
            self._topology_matcher = OntologyTopologyMatcher()
        return self._topology_matcher

    @property
    def action_normalizer(self) -> ActionIntentNormalizer:
        if self._action_normalizer is None:
            self._action_normalizer = ActionIntentNormalizer()
        return self._action_normalizer

    @property
    def slot_extractor(self) -> SlotExtractor:
        if self._slot_extractor is None:
            self._slot_extractor = SlotExtractor()
        return self._slot_extractor

    @property
    def decision_fusion(self) -> DecisionFusion:
        if self._decision_fusion is None:
            self._decision_fusion = DecisionFusion()
        return self._decision_fusion

    @property
    def route_policy(self) -> RoutePolicyEngine:
        if self._route_policy is None:
            self._route_policy = RoutePolicyEngine()
        return self._route_policy

    @property
    def clarification(self) -> ClarificationEngine:
        if self._clarification is None:
            self._clarification = ClarificationEngine()
        return self._clarification

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self, raw_input: str) -> PipelineResult:
        """Run the simplified intent recognition pipeline.

        Args:
            raw_input: User input text

        Returns:
            PipelineResult with SemanticFrame and layer results
        """
        start_time = time.time()
        layer_results: Dict[str, Any] = {}

        try:
            print(f"[SimplifiedPipeline][INFO] Pipeline启动 | task_id={self.task_id} | input='{raw_input}'")

            # Stage 1: Preprocessing
            preprocessed = self._stage_preprocess(raw_input)
            layer_results["preprocess"] = preprocessed.to_dict()

            # Stage 2: LLM Semantic Parsing
            llm_result = self._stage_llm_parse(raw_input, preprocessed)
            layer_results["llm_parse"] = llm_result.to_dict()

            # Stage 3: Ontology Matching
            ontology_result = None
            if self.config.use_ontology_match:
                ontology_result = self._stage_ontology_match(llm_result, preprocessed)
                layer_results["ontology_match"] = ontology_result.to_dict()

            # Stage 4: Action Normalization
            action_result = None
            if self.config.use_action_normalize:
                action_result = self._stage_action_normalize(raw_input, llm_result)
                layer_results["action_normalize"] = action_result.to_dict()

            # Extract action and object
            action = action_result.action_intent if action_result else llm_result.primary_intent or "Query"
            object_name = llm_result.object_matches[0].object_name if llm_result.object_matches else ""
            if not object_name and ontology_result and ontology_result.object_match:
                object_name = ontology_result.object_match.object_name

            # Stage 5: Slot Extraction
            slot_result = None
            if self.config.use_slot_extraction:
                slot_result = self._stage_slot_extract(
                    raw_input, action, object_name, llm_result
                )
                layer_results["slot_extract"] = slot_result.to_dict()

            # Stage 6: Decision Fusion
            fusion_result = self._stage_decision_fusion(
                llm_result=llm_result,
                action_result=action_result,
                ontology_result=ontology_result,
                slot_result=slot_result,
            )
            layer_results["decision_fusion"] = fusion_result.to_dict()

            # Determine routing target
            if self.config.use_routing and fusion_result.decision_type == DecisionType.CONTINUE:
                route_result = self.route_policy.route(action, object_name)
                fusion_result.next_target = route_result.target_runtime
                if route_result.target_id:
                    # Update target in the result
                    pass

            # Build SemanticFrame
            frame = self._build_frame(
                raw_input=raw_input,
                normalized_input=preprocessed.normalized_input,
                llm_result=llm_result,
                action_result=action_result,
                ontology_result=ontology_result,
                slot_result=slot_result,
                fusion_result=fusion_result,
            )

            # Update routing target
            if self.config.use_routing and fusion_result.decision_type == DecisionType.CONTINUE:
                route_result = self.route_policy.route(action, object_name)
                frame.decision.target = route_result.target_id

            duration_ms = (time.time() - start_time) * 1000
            print(f"[SimplifiedPipeline][INFO] Pipeline完成 | decision={frame.decision.type.value} | target={frame.decision.target} | duration_ms={duration_ms:.1f}")

            return PipelineResult(
                frame=frame,
                layer_results=layer_results,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            print(f"[SimplifiedPipeline][ERROR] Pipeline错误: {e}")
            return PipelineResult(
                frame=SemanticFrame.from_input(raw_input),
                layer_results=layer_results,
                duration_ms=duration_ms,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # Stage implementations
    # -------------------------------------------------------------------------

    def _stage_preprocess(self, raw_input: str):
        """Stage 1: Input preprocessing."""
        return self.preprocessor.preprocess(raw_input)

    def _stage_llm_parse(self, raw_input: str, preprocessed):
        """Stage 2: LLM semantic parsing."""
        return self.light_reasoner.reason(raw_input, preprocessed)

    def _stage_ontology_match(self, llm_result, preprocessed):
        """Stage 3: Ontology matching."""
        return self.topology_matcher.match(llm_result, preprocessed)

    def _stage_action_normalize(self, raw_input: str, llm_result):
        """Stage 4: Action intent normalization."""
        # Use LLM's action candidate if available
        candidate = None
        if llm_result.action_matches:
            candidate = llm_result.action_matches[0].action_name if hasattr(llm_result.action_matches[0], 'action_name') else str(llm_result.action_matches[0])

        return self.action_normalizer.normalize(raw_input, candidate)

    def _stage_slot_extract(
        self,
        raw_input: str,
        action: str,
        object_name: str,
        llm_result,
    ) -> SlotExtractionResult:
        """Stage 5: Slot extraction."""
        # Merge filters from LLM result
        existing_slots = {}
        for dim in llm_result.dimension_matches:
            if dim.value:
                existing_slots[dim.dimension_name] = dim.value

        # Extract object name for slot definitions
        object_for_slots = object_name
        if not object_for_slots and llm_result.object_matches:
            object_for_slots = llm_result.object_matches[0].object_name

        return self.slot_extractor.extract(
            text=raw_input,
            action=action,
            object=object_for_slots or "PurchaseRequirement",
            existing_slots=existing_slots,
        )

    def _stage_decision_fusion(
        self,
        llm_result,
        action_result,
        ontology_result,
        slot_result,
    ) -> FusionResult:
        """Stage 6: Decision fusion."""
        # Calculate confidence scores
        llm_confidence = llm_result.confidence_a
        action_confidence = action_result.confidence if action_result else 0.8
        ontology_confidence = ontology_result.confidence_b if ontology_result else 0.8

        # Determine risk level and HITL requirement
        risk_level = RiskLevel.LOW
        need_hitl = False
        is_create_action = False

        if action_result:
            risk_level = self.action_normalizer.get_risk_level(action_result.action_intent)
            need_hitl = self.action_normalizer.requires_hitl(action_result.action_intent)
            is_create_action = action_result.action_intent == "Create"

        # Perform fusion
        return self.decision_fusion.fuse(
            llm_confidence=llm_confidence,
            action_confidence=action_confidence,
            ontology_confidence=ontology_confidence,
            slot_result=slot_result,
            risk_level=risk_level,
            need_hitl=need_hitl,
            is_create_action=is_create_action,
            permission_passed=True,
        )

    # -------------------------------------------------------------------------
    # Frame building
    # -------------------------------------------------------------------------

    def _build_frame(
        self,
        raw_input: str,
        normalized_input: str,
        llm_result,
        action_result,
        ontology_result,
        slot_result,
        fusion_result: FusionResult,
    ) -> SemanticFrame:
        """Build the final SemanticFrame."""
        # Build SemanticInfo
        semantic = SemanticInfo(
            action_intent=action_result.action_intent if action_result else llm_result.primary_intent or "Query",
            action_candidate=action_result.action_candidate if action_result else "",
            ontology_object=(
                ontology_result.object_match.object_name
                if ontology_result and ontology_result.object_match
                else (llm_result.object_matches[0].object_name if llm_result.object_matches else "")
            ),
            object_candidate=(
                llm_result.object_matches[0].matched_phrase
                if llm_result.object_matches else ""
            ),
            conditions=[],  # TODO: Extract from LLM result
            slots=slot_result.slots if slot_result else {},
        )

        # Build GovernanceInfo
        governance = GovernanceInfo(
            risk_level=fusion_result.confidence.governance.risk_level if hasattr(fusion_result.confidence, 'governance') else RiskLevel.LOW,
            need_hitl=fusion_result.decision_type == DecisionType.HITL,
            permission_passed=True,
        )

        # Build DecisionInfo
        decision = DecisionInfo(
            type=fusion_result.decision_type,
            next=fusion_result.next_target,
            target="",  # Will be set by router
            reason=fusion_result.reason,
            clarify_question=fusion_result.clarify_question,
            clarify_options=fusion_result.clarify_options,
        )

        # Build SemanticFrame
        frame = SemanticFrame(
            input=InputInfo(raw_text=raw_input, normalized_text=normalized_input),
            semantic=semantic,
            confidence=fusion_result.confidence,
            governance=governance,
            decision=decision,
            task_id=self.task_id,
        )

        # Generate clarification if needed
        if fusion_result.decision_type == DecisionType.CLARIFY and not fusion_result.clarify_question:
            frame.decision.clarify_question = self._generate_clarify_question(
                raw_input, slot_result, action_result
            )

        return frame

    def _generate_clarify_question(
        self,
        raw_input: str,
        slot_result: Optional[SlotExtractionResult],
        action_result,
    ) -> str:
        """Generate clarification question."""
        if slot_result and slot_result.missing_slots:
            return self.slot_extractor.generate_clarification_question(
                slot_result,
                slot_result.object
            )

        # Generic clarification
        return f"关于「{raw_input[:30]}」，能再详细说明一下吗？"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _generate_task_id(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:6].upper()
        return f"TASK-{date_str}-{short_uuid}"


# ---------------------------------------------------------------------------
# Backward Compatibility Aliases
# ---------------------------------------------------------------------------

# Aliases for backward compatibility with old code
CompositeIntentPipeline = CoreIntentPipeline
SimplifiedIntentPipeline = CoreIntentPipeline


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_pipeline: Optional[CoreIntentPipeline] = None


def get_default_pipeline() -> CoreIntentPipeline:
    """Get the default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = CoreIntentPipeline()
    return _default_pipeline


def recognize_intent(raw_input: str) -> PipelineResult:
    """Convenience function to run intent recognition."""
    return get_default_pipeline().run(raw_input)
