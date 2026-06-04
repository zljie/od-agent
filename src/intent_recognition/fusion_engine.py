"""
Confidence Fusion Engine
========================
Layers 3 and 5 of the BeBISO 7-layer intent recognition framework.

Layer 3 — fuses confidence scores A (LLM light) and B (ontology match):
    Score_AB = A × 0.45 + B × 0.55
    Decision thresholds:
        >= 0.85 → "execute"   (high confidence, proceed directly)
        >= 0.75 → "continue"  (proceed to dynamic slot resolution)
        <  0.75 → "deep_reasoning" (trigger Layer 4)

Layer 5 — fuses A, B, and C (deep reasoning) scores:
    Score_ABC = A × 0.25 + B × 0.35 + C × 0.40
    Decision thresholds:
        >= 0.80 → "execute"   (slots complete AND high confidence)
        >= 0.70 → "continue"  (slots complete, proceed)
        <  0.70 → "clarify"  (slots missing OR low confidence → HITL)

Priority Rule: missing_required_slots > 0 → "clarify"
This overrides ALL confidence-based decisions. High confidence ≠ executable.
"""

from typing import Any, Dict, List, Optional, Tuple

from .models import ConfidenceDecision, ConfidenceScore, SlotCheckResult


class ConfidenceFusionEngine:
    """Fuses multi-source confidence scores to produce routing decisions.

    Parameters follow the BeBISO specification:
    - WEIGHT_A / WEIGHT_B: layer-3 weights for A (LLM-light) and B (ontology)
    - WEIGHT_A_ABC / WEIGHT_B_ABC / WEIGHT_C_ABC: layer-5 weights
    - THRESHOLD_AB: decision boundary for layer-3 fusion
    - THRESHOLD_ABC: decision boundary for layer-5 fusion
    """

    WEIGHT_A: float = 0.45
    WEIGHT_B: float = 0.55
    WEIGHT_A_ABC: float = 0.25
    WEIGHT_B_ABC: float = 0.35
    WEIGHT_C_ABC: float = 0.40
    THRESHOLD_AB: float = 0.75
    THRESHOLD_ABC: float = 0.75

    # Absolute high-confidence threshold for direct execution
    EXECUTE_THRESHOLD: float = 0.85

    def fuse_ab(self, A: float, B: float) -> Tuple[float, str]:
        """Layer 3 fusion: combine LLM-light (A) and ontology-match (B) scores.

        Parameters
        ----------
        A:
            Layer-1 confidence score [0.0, 1.0] from light LLM reasoning.
        B:
            Layer-2 confidence score [0.0, 1.0] from ontology topology match.

        Returns
        -------
        Tuple[float, str]
            (score_ab, decision) where decision is one of:
            - "execute"   : score_ab >= 0.85 — high confidence, direct execution
            - "continue" : 0.75 <= score_ab < 0.85 — proceed to dynamic resolution
            - "deep_reasoning" : score_ab < 0.75 — trigger Layer 4
        """
        score_ab = A * self.WEIGHT_A + B * self.WEIGHT_B

        if score_ab >= self.EXECUTE_THRESHOLD:
            decision = "execute"
        elif score_ab >= self.THRESHOLD_AB:
            decision = "continue"
        else:
            decision = "deep_reasoning"

        return score_ab, decision

    def fuse_abc(
        self,
        A: float,
        B: float,
        C: float,
        slot_check: Optional[SlotCheckResult] = None,
    ) -> Tuple[float, str]:
        """Layer 5 fusion: combine A, B, and deep-reasoning (C) scores.

        Priority rule: if any required slots are missing, decision is always
        "clarify" regardless of confidence scores. High confidence does NOT
        mean executable — slots must also be complete.

        Parameters
        ----------
        A:
            Layer-1 confidence score [0.0, 1.0].
        B:
            Layer-2 confidence score [0.0, 1.0].
        C:
            Layer-4 confidence score [0.0, 1.0] from deep reasoning.
        slot_check:
            Layer-2.5 slot check result. If provided and has missing required
            slots, overrides confidence-based decision with "clarify".

        Returns
        -------
        Tuple[float, str]
            (score_abc, decision) where decision is one of:
            - "execute" : slots complete AND score_abc >= 0.80
            - "continue" : slots complete AND 0.70 <= score_abc < 0.80
            - "clarify" : missing required slots OR score_abc < 0.70
        """
        # HARD OVERRIDE: missing required slots → clarify regardless of confidence
        if slot_check is not None and slot_check.has_missing_required:
            return 0.0, "clarify"

        score_abc = (
            A * self.WEIGHT_A_ABC
            + B * self.WEIGHT_B_ABC
            + C * self.WEIGHT_C_ABC
        )

        if score_abc >= 0.80:
            decision = "execute"
        elif score_abc >= 0.70:
            decision = "continue"
        else:
            decision = "clarify"

        return score_abc, decision

    def score_decision(
        self,
        A: float = 0.0,
        B: float = 0.0,
        C: float = 0.0,
        deep_triggered: bool = False,
        slot_check: Optional[SlotCheckResult] = None,
        action_id: str = "",
    ) -> ConfidenceScore:
        """Orchestrate the full two-stage fusion pipeline.

        Parameters
        ----------
        A:
            Layer-1 (LLM-light) confidence score.
        B:
            Layer-2 (ontology-match) confidence score.
        C:
            Layer-4 (deep reasoning) confidence score.
        deep_triggered:
            Whether Layer 4 deep reasoning was triggered.
        slot_check:
            Layer-2.5 slot check result. If provided and has missing required
            slots, forces decision="clarify" regardless of confidence scores.
        action_id:
            The action ID being evaluated. If it contains "create", the action
            is marked as requiring human confirmation regardless of confidence.
        """
        # HARD OVERRIDE: create operations always require human confirmation
        is_create_action = "create" in action_id.lower() if action_id else False
        if is_create_action:
            print(f"[ConfidenceFusionEngine][INFO] 创建操作拦截 | action={action_id} | 强制HITL确认")
            return ConfidenceScore(
                score_ab=A * self.WEIGHT_A + B * self.WEIGHT_B,
                score_abc=A * self.WEIGHT_A_ABC + B * self.WEIGHT_B_ABC + C * self.WEIGHT_C_ABC,
                final_decision=ConfidenceDecision.HITL,
                threshold_ab=self.THRESHOLD_AB,
                threshold_abc=self.THRESHOLD_ABC,
                confidence_a=A,
                confidence_b=B,
                confidence_c=C,
                deep_triggered=deep_triggered,
                metadata={"create_action_override": True},
            )

        score_ab, decision_ab = self.fuse_ab(A, B)

        if decision_ab == "execute":
            return ConfidenceScore(
                score_ab=score_ab,
                score_abc=0.0,
                final_decision=ConfidenceDecision.EXECUTE,
                threshold_ab=self.THRESHOLD_AB,
                threshold_abc=self.THRESHOLD_ABC,
                confidence_a=A,
                confidence_b=B,
                confidence_c=0.0,
                deep_triggered=False,
            )

        if decision_ab == "continue":
            if slot_check is not None and slot_check.has_missing_required:
                return ConfidenceScore(
                    score_ab=score_ab,
                    score_abc=0.0,
                    final_decision=ConfidenceDecision.HITL,
                    threshold_ab=self.THRESHOLD_AB,
                    threshold_abc=self.THRESHOLD_ABC,
                    confidence_a=A,
                    confidence_b=B,
                    confidence_c=0.0,
                    deep_triggered=False,
                    metadata={"slot_check_override": True},
                )
            return ConfidenceScore(
                score_ab=score_ab,
                score_abc=0.0,
                final_decision=ConfidenceDecision.CONTINUE,
                threshold_ab=self.THRESHOLD_AB,
                threshold_abc=self.THRESHOLD_ABC,
                confidence_a=A,
                confidence_b=B,
                confidence_c=0.0,
                deep_triggered=False,
            )

        # decision_ab == "deep_reasoning"
        if not deep_triggered:
            return ConfidenceScore(
                score_ab=score_ab,
                score_abc=0.0,
                final_decision=ConfidenceDecision.DEEP_REASONING,
                threshold_ab=self.THRESHOLD_AB,
                threshold_abc=self.THRESHOLD_ABC,
                confidence_a=A,
                confidence_b=B,
                confidence_c=0.0,
                deep_triggered=False,
            )

        score_abc, decision_abc = self.fuse_abc(A, B, C, slot_check)

        if decision_abc == "clarify":
            final = ConfidenceDecision.HITL
        elif decision_abc == "continue":
            final = ConfidenceDecision.CONTINUE
        else:
            final = ConfidenceDecision.HITL

        return ConfidenceScore(
            score_ab=score_ab,
            score_abc=score_abc,
            final_decision=final,
            threshold_ab=self.THRESHOLD_AB,
            threshold_abc=self.THRESHOLD_ABC,
            confidence_a=A,
            confidence_b=B,
            confidence_c=C,
            deep_triggered=True,
            metadata={"slot_check_override": slot_check is not None and slot_check.has_missing_required},
        )
