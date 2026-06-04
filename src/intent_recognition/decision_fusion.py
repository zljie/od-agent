"""
Decision Fusion
==============
Single-stage decision fusion layer for intent recognition.

Based on Section 5.6 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Replaces the old Layer 3 + Layer 5 fusion with a single, simpler fusion.

Decision Types:
- continue: Proceed to routing
- clarify: Need user clarification
- reject: Not allowed to execute
- hitl: Need human confirmation
- fallback: Fallback strategy
- replan: Need re-planning
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .semantic_frame import (
    SemanticFrame,
    InputInfo,
    SemanticInfo,
    ConfidenceInfo,
    GovernanceInfo,
    DecisionInfo,
    DecisionType,
    RiskLevel,
    RuntimeTarget,
)
from .slot_extractor import SlotExtractionResult
from .action_normalizer import ActionIntentNormalizer


# ---------------------------------------------------------------------------
# Fusion Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionWeights:
    """Weights for decision fusion."""
    llm_parse: float = 0.30   # LLM semantic parsing confidence
    action: float = 0.20      # Action normalization confidence
    ontology_match: float = 0.25  # Ontology matching confidence
    slot_fill: float = 0.25   # Slot completion confidence


@dataclass
class FusionThresholds:
    """Thresholds for decision making."""
    continue_threshold: float = 0.70  # >= this → continue
    clarify_threshold: float = 0.50   # >= this → clarify, < this → reject
    high_confidence: float = 0.85    # >= this → skip HITL for medium risk


# ---------------------------------------------------------------------------
# Fusion Result
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Result of decision fusion."""
    decision_type: DecisionType
    next_target: RuntimeTarget
    confidence: ConfidenceInfo
    reason: str
    clarify_question: Optional[str] = None
    clarify_options: List[str] = field(default_factory=list)
    override_hitl: bool = False  # Force skip HITL despite policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisionType": self.decision_type.value,
            "nextTarget": self.next_target.value,
            "confidence": self.confidence.to_dict(),
            "reason": self.reason,
            "clarifyQuestion": self.clarify_question,
            "clarifyOptions": self.clarify_options,
            "overrideHitl": self.override_hitl,
        }


# ---------------------------------------------------------------------------
# Decision Fusion
# ---------------------------------------------------------------------------

class DecisionFusion:
    """Single-stage decision fusion layer.

    This implements Section 5.6 of the BeBIOS architecture.
    It replaces the old Layer 3 + Layer 5 two-stage fusion with a simpler,
    more maintainable single-stage approach.

    Example:
        fusion = DecisionFusion()
        result = fusion.fuse(
            llm_confidence=0.85,
            action_confidence=0.90,
            ontology_confidence=0.88,
            slot_result=slot_result,
            risk_level=RiskLevel.LOW,
            need_hitl=False,
        )
    """

    def __init__(
        self,
        weights: Optional[FusionWeights] = None,
        thresholds: Optional[FusionThresholds] = None,
        action_normalizer: Optional[ActionIntentNormalizer] = None,
    ):
        """Initialize the decision fusion.

        Args:
            weights: Custom fusion weights
            thresholds: Custom decision thresholds
            action_normalizer: Action normalizer for risk level lookups
        """
        self.weights = weights or FusionWeights()
        self.thresholds = thresholds or FusionThresholds()
        self.action_normalizer = action_normalizer or ActionIntentNormalizer()

    def fuse(
        self,
        llm_confidence: float,
        action_confidence: float,
        ontology_confidence: float,
        slot_result: SlotExtractionResult,
        risk_level: RiskLevel = RiskLevel.LOW,
        need_hitl: bool = False,
        is_create_action: bool = False,
        permission_passed: bool = True,
        additional_factors: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """Fuse confidence scores and make routing decision.

        Args:
            llm_confidence: LLM semantic parsing confidence [0.0, 1.0]
            action_confidence: Action normalization confidence [0.0, 1.0]
            ontology_confidence: Ontology matching confidence [0.0, 1.0]
            slot_result: Slot extraction result
            risk_level: Action risk level
            need_hitl: Whether HITL is required by governance
            is_create_action: Whether this is a create action
            permission_passed: Whether user has permission
            additional_factors: Additional confidence factors to consider

        Returns:
            FusionResult with decision and routing target
        """
        # Calculate weighted confidence
        confidence = ConfidenceInfo(
            llm_parse=llm_confidence,
            action=action_confidence,
            ontology_match=ontology_confidence,
            slot_fill=slot_result.confidence if slot_result else 0.0,
        )

        # Calculate overall confidence
        overall = self._calculate_overall(confidence, additional_factors)
        confidence.overall = overall

        # Check permission first
        if not permission_passed:
            return FusionResult(
                decision_type=DecisionType.REJECT,
                next_target=RuntimeTarget.REJECT,
                confidence=confidence,
                reason="用户权限不足",
            )

        # Check for missing slots
        has_missing_slots = slot_result and not slot_result.is_complete

        # Build decision reasons
        reasons = []
        if overall >= self.thresholds.continue_threshold:
            reasons.append(f"整体置信度较高({overall:.2f})")
        if not has_missing_slots:
            reasons.append("槽位完整")
        if risk_level == RiskLevel.LOW:
            reasons.append("风险等级低")

        # Make primary decision
        if has_missing_slots:
            # Missing slots → clarify
            reason_str = "、".join(reasons) if reasons else "槽位缺失"
            return FusionResult(
                decision_type=DecisionType.CLARIFY,
                next_target=RuntimeTarget.CLARIFICATION,
                confidence=confidence,
                reason=f"{reason_str}，但{slot_result.missing_slots}缺失",
                clarify_question=self._generate_clarify_question(slot_result),
            )

        if overall < self.thresholds.clarify_threshold:
            # Low confidence → reject
            return FusionResult(
                decision_type=DecisionType.REJECT,
                next_target=RuntimeTarget.REJECT,
                confidence=confidence,
                reason=f"整体置信度过低({overall:.2f})",
            )

        if overall >= self.thresholds.continue_threshold:
            # High confidence → determine next target

            # Check if HITL is required
            if self._requires_hitl(risk_level, need_hitl, is_create_action, overall):
                reason_str = "、".join(reasons) if reasons else "风险控制要求"
                return FusionResult(
                    decision_type=DecisionType.HITL,
                    next_target=RuntimeTarget.HITL,
                    confidence=confidence,
                    reason=f"{reason_str}，需要人工确认",
                )

            # Continue to routing
            reason_str = "、".join(reasons) if reasons else "置信度达标"
            return FusionResult(
                decision_type=DecisionType.CONTINUE,
                next_target=self._determine_next_target(risk_level, is_create_action),
                confidence=confidence,
                reason=reason_str,
            )

        # Medium confidence → clarify
        reason_str = "、".join(reasons) if reasons else "置信度中等"
        return FusionResult(
            decision_type=DecisionType.CLARIFY,
            next_target=RuntimeTarget.CLARIFICATION,
            confidence=confidence,
            reason=f"{reason_str}，请确认",
        )

    def _calculate_overall(
        self,
        confidence: ConfidenceInfo,
        additional_factors: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate overall confidence score."""
        weights = self.weights

        overall = (
            confidence.llm_parse * weights.llm_parse
            + confidence.action * weights.action
            + confidence.ontology_match * weights.ontology_match
            + confidence.slot_fill * weights.slot_fill
        )

        # Apply additional factors if provided
        if additional_factors:
            factor_count = len(additional_factors)
            if factor_count > 0:
                factor_avg = sum(additional_factors.values()) / factor_count
                overall = overall * 0.7 + factor_avg * 0.3

        return round(min(max(overall, 0.0), 1.0), 4)

    def _requires_hitl(
        self,
        risk_level: RiskLevel,
        need_hitl: bool,
        is_create_action: bool,
        overall: float,
    ) -> bool:
        """Determine if HITL is required."""
        # Explicit HITL requirement
        if need_hitl:
            return True

        # High risk actions always need HITL
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return True

        # Create actions typically need HITL
        if is_create_action and overall < self.thresholds.high_confidence:
            return True

        return False

    def _determine_next_target(
        self,
        risk_level: RiskLevel,
        is_create_action: bool,
    ) -> RuntimeTarget:
        """Determine the next runtime target."""
        if risk_level == RiskLevel.HIGH or risk_level == RiskLevel.CRITICAL:
            return RuntimeTarget.WORKFLOW  # Workflow for high risk
        if is_create_action:
            return RuntimeTarget.WORKFLOW
        return RuntimeTarget.WORKFLOW  # Default to workflow

    def _generate_clarify_question(self, slot_result: SlotExtractionResult) -> str:
        """Generate clarification question for missing slots."""
        if not slot_result or not slot_result.missing_slots:
            return "请提供更多信息"

        # Get labels for missing slots
        missing_info = []
        for name in slot_result.missing_slots:
            definition = next((d for d in slot_result.definitions if d.name == name), None)
            if definition:
                label = definition.label or definition.name
                placeholder = definition.placeholder or f"请输入{label}"
                missing_info.append(f"{label}（{placeholder}）")

        if missing_info:
            return f"请补充以下信息：\n" + "\n".join(f"{i+1}. {info}" for i, info in enumerate(missing_info))

        return "请提供更多信息"

    def fuse_and_build_frame(
        self,
        raw_input: str,
        normalized_input: str,
        semantic: SemanticInfo,
        llm_confidence: float,
        ontology_confidence: float,
        slot_result: SlotExtractionResult,
        risk_level: RiskLevel = RiskLevel.LOW,
        need_hitl: bool = False,
        is_create_action: bool = False,
        permission_passed: bool = True,
    ) -> SemanticFrame:
        """Fuse decisions and build complete SemanticFrame.

        Args:
            raw_input: Original user input
            normalized_input: Preprocessed input
            semantic: Semantic understanding result
            llm_confidence: LLM parsing confidence
            ontology_confidence: Ontology matching confidence
            slot_result: Slot extraction result
            risk_level: Action risk level
            need_hitl: Whether HITL is required
            is_create_action: Whether this is a create action
            permission_passed: Whether user has permission

        Returns:
            Complete SemanticFrame ready for routing
        """
        # Perform fusion
        fusion_result = self.fuse(
            llm_confidence=llm_confidence,
            action_confidence=semantic.action_intent and 1.0 or 0.0,
            ontology_confidence=ontology_confidence,
            slot_result=slot_result,
            risk_level=risk_level,
            need_hitl=need_hitl,
            is_create_action=is_create_action,
            permission_passed=permission_passed,
        )

        # Build GovernanceInfo
        governance = GovernanceInfo(
            risk_level=risk_level,
            need_hitl=fusion_result.decision_type == DecisionType.HITL,
            permission_passed=permission_passed,
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
        return SemanticFrame(
            input=InputInfo(raw_text=raw_input, normalized_text=normalized_input),
            semantic=semantic,
            confidence=fusion_result.confidence,
            governance=governance,
            decision=decision,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_fusion: Optional[DecisionFusion] = None


def get_default_fusion() -> DecisionFusion:
    """Get the default fusion instance."""
    global _default_fusion
    if _default_fusion is None:
        _default_fusion = DecisionFusion()
    return _default_fusion


def fuse_decision(
    llm_confidence: float,
    action_confidence: float,
    ontology_confidence: float,
    slot_result: SlotExtractionResult,
    **kwargs,
) -> FusionResult:
    """Convenience function to fuse decision."""
    return get_default_fusion().fuse(
        llm_confidence=llm_confidence,
        action_confidence=action_confidence,
        ontology_confidence=ontology_confidence,
        slot_result=slot_result,
        **kwargs,
    )
