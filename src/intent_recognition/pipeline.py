"""
Composite Intent Recognition Pipeline (DEPRECATED)
===============================================
Phase 3: End-to-end 7-layer intent recognition pipeline.

.. deprecated::
    This pipeline is deprecated. Use CoreIntentPipeline from core_pipeline.py instead.

Orchestrates Layers 0-7 (+ Layer 2.5 Slot Completion):
    Layer 0:  InputPreprocessor        (normalization + tokenization)
    Layer 1:  LightIntentReasoner       (fast LLM, thinking_budget=500)
    Layer 2:  OntologyTopologyMatcher   (ontology validation)
    Layer 2.5: SlotCompletionEngine      (execution gate — slot readiness check)
    Layer 3:  ConfidenceFusionEngine    (A×0.45 + B×0.55)
    Layer 4:  DeepIntentReasoner        (heavy LLM, thinking_budget=1200, conditional)
    Layer 5:  ConfidenceFusionEngine    (A×0.25 + B×0.35 + C×0.40, conditional)
    Layer 6:  HITLClarificationEngine   (human-in-the-loop, conditional)
    Layer 7:  DynamicSlotResolver       (master-data lookup)

Priority rule: missing_required_slots > 0 → decision=clarify (blocks execution).

Migration:
    Replace CompositeIntentPipeline with CoreIntentPipeline from core_pipeline.py
"""

import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..procurement.ontology_loader import get_procurement_ontology
from ..intent_topology import IntentTopologyLoader, IntentPath, PathType

from .models import (
    PreprocessingResult,
    LLMLightResult,
    OntologyMatchResult,
    ConfidenceScore,
    DeepReasoningResult,
    HITLRequest,
    ResolvedValue,
    CompositeIntentResult,
    IntentCandidate,
    ConfidenceDecision,
    SlotCheckResult,
)
from .preprocessor import InputPreprocessor
from .light_reasoner import LightIntentReasoner
from .topology_matcher import OntologyTopologyMatcher
from .fusion_engine import ConfidenceFusionEngine
from .deep_reasoner import DeepIntentReasoner
from .hitl_engine import HITLClarificationEngine
from .dynamic_resolver import DynamicSlotResolver
from .slot_completion_engine import SlotCompletionEngine


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

DEFAULT_DEEP_REASONING_THRESHOLD = 0.75
DEFAULT_HITL_THRESHOLD = 0.75
DEFAULT_EXECUTE_THRESHOLD = 0.85

# Block list: generic templates must NOT be executed as final action
BLOCKED_FINAL_ACTIONS = {"query_object", "create_object", "update_object", "delete_object"}

# Cancel keywords that override pending HITL session
CANCEL_KEYWORDS = {"取消", "重新开始", "不是这个", "算了", "换话题", "quit", "cancel", "start over", "wrong"}

# Slot-fill protocol pattern: [slot-fill] <task_id> | <slot_name>: <value>
SLOT_FILL_PATTERN = r"^\[slot-fill\]\s*([^\s|]+)\s*\|?\s*(.+)?$"


class SlotFillParser:
    """Pre-Router: parses [slot-fill] protocol input.

    Protocol format: [slot-fill] <task_id> | slot_name: value [slot_name: value ...]

    This is the highest-priority router. When detected, it:
    1. Extracts the pending task ID
    2. Parses the filled slot(s)
    3. Returns a SlotFillEvent WITHOUT running intent recognition
    """

    @staticmethod
    def is_slot_fill_input(raw_input: str) -> bool:
        """Check if input starts with [slot-fill] protocol marker."""
        return raw_input.strip().lower().startswith("[slot-fill]")

    @staticmethod
    def parse(raw_input: str) -> Optional[Dict[str, Any]]:
        """Parse slot-fill protocol input.

        Returns:
            Dict with:
            - task_id: the pending task ID
            - filled_slots: dict of {slot_name: value}
            Or None if not valid slot-fill format.
        """
        import re

        match = re.match(SLOT_FILL_PATTERN, raw_input.strip(), re.IGNORECASE)
        if not match:
            return None

        task_id = match.group(1).strip()
        slots_str = match.group(2) or ""

        # Parse slot_name: value pairs
        filled_slots = {}
        if slots_str:
            # Split by | or & for multiple slots
            slot_pairs = re.split(r"[|&]", slots_str)
            for pair in slot_pairs:
                pair = pair.strip()
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    filled_slots[key.strip()] = value.strip()

        return {
            "task_id": task_id,
            "filled_slots": filled_slots,
            "raw_input": raw_input,
        }


@dataclass
class SlotFillEvent:
    """Result of parsing a slot-fill protocol input."""
    task_id: str
    filled_slots: Dict[str, Any]
    raw_input: str
    pending_task: Optional[Dict[str, Any]] = None  # The pending HITL task to update


class CompositeIntentPipeline:
    """End-to-end composite intent recognition pipeline.

    .. deprecated::
        This class is deprecated. Use CoreIntentPipeline from core_pipeline.py instead.

    Run ``run()`` with a user input string to get a ``CompositeIntentResult``
    with the resolved intent and any HITL request.

    Migration
    ---------
    Replace with CoreIntentPipeline:

        from intent_recognition import CoreIntentPipeline

        pipeline = CoreIntentPipeline()
        result = pipeline.run(user_input)

    Attributes
    ----------
    use_deep_reasoning:
        Whether to invoke Layer 4 (DeepIntentReasoner) when fusion score
        is below threshold. Default True.
    use_hitl:
        Whether to invoke Layer 6 (HITLClarificationEngine) when
        deep reasoning also fails to reach threshold. Default True.
    audit_logger:
        Optional audit logger for per-layer instrumentation.
    """

    def __init__(
        self,
        use_deep_reasoning: bool = True,
        use_hitl: bool = True,
        task_id: Optional[str] = None,
        audit_logger=None,
        pending_hitl_task: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id or self._generate_task_id()
        self.use_deep_reasoning = use_deep_reasoning
        self.use_hitl = use_hitl
        self.audit_logger = audit_logger
        # Pending HITL task from previous round: slot filling mode
        # {
        #   "action": "create_purchase_requests",
        #   "object": "purchase_requests",
        #   "missing_slots": ["material", "quantity", "delivery_date"],
        #   "filled_slots": {"apply_dep": "采购部"},
        #   "task_id": "TASK-xxx",
        # }
        self.pending_hitl_task = pending_hitl_task

        # Lazy-initialized layer instances
        self._preprocessor: Optional[InputPreprocessor] = None
        self._light_reasoner: Optional[LightIntentReasoner] = None
        self._topology_matcher: Optional[OntologyTopologyMatcher] = None
        self._fusion_engine: Optional[ConfidenceFusionEngine] = None
        self._deep_reasoner: Optional[DeepIntentReasoner] = None
        self._hitl_engine: Optional[HITLClarificationEngine] = None
        self._dynamic_resolver: Optional[DynamicSlotResolver] = None
        self._slot_completion_engine: Optional[SlotCompletionEngine] = None

        # Cached topology
        self._topology = None
        self._ontology_version = ""

    @property
    def in_slot_filling_mode(self) -> bool:
        """True if there is an active pending HITL task from a previous round."""
        return self.pending_hitl_task is not None and len(self.pending_hitl_task) > 0

    # ------------------------------------------------------------------
    # Lazy layer initialization
    # ------------------------------------------------------------------

    @property
    def preprocessor(self) -> InputPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = InputPreprocessor()
        return self._preprocessor

    @property
    def light_reasoner(self) -> LightIntentReasoner:
        if self._light_reasoner is None:
            self._light_reasoner = LightIntentReasoner()
            self._light_reasoner.set_ontology_summary(
                self.topology_matcher.get_ontology_summary()
            )
        return self._light_reasoner

    @property
    def topology_matcher(self) -> OntologyTopologyMatcher:
        if self._topology_matcher is None:
            self._topology_matcher = OntologyTopologyMatcher(topology=self.topology)
        return self._topology_matcher

    @property
    def fusion_engine(self) -> ConfidenceFusionEngine:
        if self._fusion_engine is None:
            self._fusion_engine = ConfidenceFusionEngine()
        return self._fusion_engine

    @property
    def deep_reasoner(self) -> DeepIntentReasoner:
        if self._deep_reasoner is None:
            self._deep_reasoner = DeepIntentReasoner()
        return self._deep_reasoner

    @property
    def hitl_engine(self) -> HITLClarificationEngine:
        if self._hitl_engine is None:
            self._hitl_engine = HITLClarificationEngine()
        return self._hitl_engine

    @property
    def dynamic_resolver(self) -> DynamicSlotResolver:
        if self._dynamic_resolver is None:
            self._dynamic_resolver = DynamicSlotResolver()
        return self._dynamic_resolver

    @property
    def slot_completion_engine(self) -> SlotCompletionEngine:
        if self._slot_completion_engine is None:
            self._slot_completion_engine = SlotCompletionEngine()
        return self._slot_completion_engine

    @property
    def topology(self):
        if self._topology is None:
            self._topology = IntentTopologyLoader.get_topology()
            try:
                self._ontology_version = get_procurement_ontology().version
            except Exception:
                self._ontology_version = "unknown"
        return self._topology

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raw_input: str) -> CompositeIntentResult:
        """Run the full 7-layer composite intent recognition pipeline.

        Parameters
        ----------
        raw_input:
            The raw user input string.

        Returns
        -------
        CompositeIntentResult
            Complete result with resolved intent, confidence scores,
            and optional HITL request.
        """
        print(f"[CompositeIntentPipeline][INFO] ===== Pipeline启动 | task_id={self.task_id} | input='{raw_input}'")
        start_time = time.time()
        layer_results: Dict[str, Any] = {}

        # =========================================================================
        # PRE-ROUTER: Highest priority - check for slot-fill protocol
        # If input starts with [slot-fill], route directly to slot fill handler.
        # This MUST bypass all intent recognition layers.
        # =========================================================================
        if SlotFillParser.is_slot_fill_input(raw_input):
            print(f"[CompositeIntentPipeline][INFO] ===== PRE-ROUTER: 检测到[slot-fill]协议输入")
            return self._run_slot_fill_protocol(raw_input, start_time, layer_results)

        # =========================================================================
        # RULE 1: Slot Filling Mode — if pending HITL task exists, route directly
        # to slot completion without re-running global intent detection.
        # Unless user explicitly cancels.
        # =========================================================================
        if self.in_slot_filling_mode:
            user_canceled = any(kw in raw_input.lower() for kw in CANCEL_KEYWORDS)
            if user_canceled:
                print(f"[CompositeIntentPipeline][INFO] 用户取消HITL任务，清除pending状态")
                self.pending_hitl_task = None
                # Fall through to normal pipeline
            else:
                print(f"[CompositeIntentPipeline][INFO] ===== 进入Slot Filling模式 | pending_action={self.pending_hitl_task.get('action')} | 跳过全局意图识别")
                return self._run_slot_filling(raw_input, start_time, layer_results)

        # =========================================================================
        # Layer 0: Preprocessing
        # =========================================================================
        preprocessing = self._run_layer_0(raw_input)
        layer_results["layer_0_preprocessing"] = preprocessing.to_dict()

        # Layer 1: Light LLM reasoning  (and Layer 2: topology match — parallel in concept,
        # sequential in sync implementation; both use the same preprocessing result)
        llm_result = self._run_layer_1(raw_input, preprocessing)
        layer_results["layer_1_light_reasoning"] = llm_result.to_dict()

        # Layer 2: Ontology topology matching
        ontology_match = self._run_layer_2(llm_result, preprocessing)
        layer_results["layer_2_ontology_match"] = ontology_match.to_dict()

        # Layer 2.5: Slot Completion check (execution gate)
        # Run BEFORE fusion so missing slots can block execution early
        deep_candidates = []  # will be populated if deep reasoning is triggered
        slot_check = self._run_layer_2_5_slot_completion(
            llm_result, ontology_match, intent_id=""
        )
        layer_results["layer_2_5_slot_completion"] = slot_check.to_dict()
        print(f"[CompositeIntentPipeline][INFO] Layer-2.5槽位检查 | score={slot_check.action_readiness_score:.1f}% | decision={slot_check.decision} | missing={slot_check.missing_required}")

        # Layer 3: First confidence fusion (A × 0.45 + B × 0.55)
        confidence_ab = self.fusion_engine.fuse_ab(
            llm_result.confidence_a, ontology_match.confidence_b
        )
        score_ab, decision_ab = confidence_ab
        layer_results["layer_3_fusion_ab"] = {
            "score_ab": score_ab,
            "decision_ab": decision_ab,
            "confidence_a": llm_result.confidence_a,
            "confidence_b": ontology_match.confidence_b,
        }

        if decision_ab == "execute":
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=None,
                hitl_request=None,
                decision=ConfidenceDecision.EXECUTE,
                execution_ready=True,
                layer_results=layer_results,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=EXECUTE | score_ab={score_ab:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        if decision_ab == "continue":
            # Slot check override: missing required slots → HITL clarification
            if slot_check.has_missing_required:
                print(f"[CompositeIntentPipeline][INFO] Layer-3 Slots检查拦截 | missing_required={slot_check.missing_required}")
                hitl_request = self._run_layer_6_hitl(
                    llm_result, ontology_match, slot_check=slot_check
                )
                layer_results["layer_6_hitl"] = hitl_request.to_dict()
                result = self._build_result(
                    raw_input=raw_input,
                    preprocessing=preprocessing,
                    llm_result=llm_result,
                    ontology_match=ontology_match,
                    confidence_ab=score_ab,
                    deep_result=None,
                    hitl_request=hitl_request,
                    decision=ConfidenceDecision.HITL,
                    execution_ready=False,
                    layer_results=layer_results,
                    duration_ms=(time.time() - start_time) * 1000,
                    slot_check_result=slot_check,
                )
                print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL(slot_block) | score_ab={score_ab:.4f} | missing={slot_check.missing_required} | duration_ms={(time.time()-start_time)*1000:.1f}")
                return result
            resolved = self._run_layer_7_dynamic(
                llm_result, ontology_match, preprocessing
            )
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=None,
                hitl_request=None,
                decision=ConfidenceDecision.CONTINUE,
                execution_ready=True,
                layer_results=layer_results,
                resolved_values=resolved,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=CONTINUE | score_ab={score_ab:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        # decision_ab == "deep_reasoning" — proceed to Layer 4
        print(f"[CompositeIntentPipeline][INFO] Layer-3融合结果 | score_ab={score_ab:.4f} | decision=DEEP_REASONING | 进入Layer-4")
        if not self.use_deep_reasoning:
            # Skip deep reasoning; go directly to HITL if slots are missing
            if slot_check.has_missing_required:
                hitl_request = self._run_layer_6_hitl(
                    llm_result, ontology_match, slot_check=slot_check
                )
                layer_results["layer_6_hitl"] = hitl_request.to_dict()
                result = self._build_result(
                    raw_input=raw_input,
                    preprocessing=preprocessing,
                    llm_result=llm_result,
                    ontology_match=ontology_match,
                    confidence_ab=score_ab,
                    deep_result=None,
                    hitl_request=hitl_request,
                    decision=ConfidenceDecision.HITL,
                    execution_ready=False,
                    layer_results=layer_results,
                    duration_ms=(time.time() - start_time) * 1000,
                    slot_check_result=slot_check,
                )
                print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL(slot_block,skip_deep) | score_ab={score_ab:.4f} | duration_ms={(time.time()-start_time)*1000:.1f}")
                return result
            hitl_request = self._run_layer_6_hitl(llm_result, ontology_match)
            layer_results["layer_6_hitl"] = hitl_request.to_dict()
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=None,
                hitl_request=hitl_request,
                decision=ConfidenceDecision.HITL,
                execution_ready=False,
                layer_results=layer_results,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL(skip_deep) | score_ab={score_ab:.4f} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        # Layer 4: Deep reasoning
        print(f"[CompositeIntentPipeline][INFO] Layer-4深度推理开始 | task_id={self.task_id}")
        deep_result = self._run_layer_4_deep(
            raw_input, preprocessing, llm_result, ontology_match
        )
        layer_results["layer_4_deep_reasoning"] = deep_result.to_dict()
        print(f"[CompositeIntentPipeline][INFO] Layer-4深度推理完成 | C_score={deep_result.confidence_c:.4f}")

        # Extract top intent_id for more precise slot schema lookup
        top_intent_id = ""
        if deep_result.ranked_candidates:
            top_intent_id = deep_result.ranked_candidates[0].get("intent_id", "")
        # Re-run slot check with intent_id from deep reasoning
        slot_check = self._run_layer_2_5_slot_completion(
            llm_result, ontology_match, intent_id=top_intent_id
        )
        layer_results["layer_2_5_slot_completion"] = slot_check.to_dict()
        print(f"[CompositeIntentPipeline][INFO] Layer-2.5(重检) | score={slot_check.action_readiness_score:.1f}% | decision={slot_check.decision} | missing={slot_check.missing_required}")

        # Layer 5: Second fusion (A × 0.25 + B × 0.35 + C × 0.40)
        score_abc, decision_abc = self.fusion_engine.fuse_abc(
            llm_result.confidence_a,
            ontology_match.confidence_b,
            deep_result.confidence_c,
            slot_check=slot_check,
        )
        layer_results["layer_5_fusion_abc"] = {
            "score_abc": score_abc,
            "decision_abc": decision_abc,
            "confidence_c": deep_result.confidence_c,
        }

        if decision_abc == "execute":
            # High confidence after deep reasoning - proceed directly
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=deep_result,
                hitl_request=None,
                decision=ConfidenceDecision.CONTINUE,
                execution_ready=True,
                layer_results=layer_results,
                resolved_values=None,
                confidence_abc=score_abc,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=EXECUTE(ABC) | score_abc={score_abc:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        if decision_abc == "continue":
            resolved = self._run_layer_7_dynamic(
                llm_result, ontology_match, preprocessing
            )
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=deep_result,
                hitl_request=None,
                decision=ConfidenceDecision.CONTINUE,
                execution_ready=True,
                layer_results=layer_results,
                resolved_values=resolved,
                confidence_abc=score_abc,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=CONTINUE(ABC) | score_abc={score_abc:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        # decision_abc == "clarify" (slot-missing or low confidence) — Layer 6
        print(f"[CompositeIntentPipeline][INFO] Layer-5融合结果 | score_abc={score_abc:.4f} | decision=CLARIFY | slot_override={slot_check.has_missing_required}")
        if not self.use_hitl:
            # HITL disabled; return blocked result
            result = self._build_result(
                raw_input=raw_input,
                preprocessing=preprocessing,
                llm_result=llm_result,
                ontology_match=ontology_match,
                confidence_ab=score_ab,
                deep_result=deep_result,
                hitl_request=None,
                decision=ConfidenceDecision.HITL,
                execution_ready=False,
                layer_results=layer_results,
                confidence_abc=score_abc,
                duration_ms=(time.time() - start_time) * 1000,
                slot_check_result=slot_check,
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL(disabled) | score_abc={score_abc:.4f} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        hitl_request = self._run_layer_6_hitl(
            llm_result, ontology_match, slot_check=slot_check
        )
        layer_results["layer_6_hitl"] = hitl_request.to_dict()

        result = self._build_result(
            raw_input=raw_input,
            preprocessing=preprocessing,
            llm_result=llm_result,
            ontology_match=ontology_match,
            confidence_ab=score_ab,
            deep_result=deep_result,
            hitl_request=hitl_request,
            decision=ConfidenceDecision.HITL,
            execution_ready=False,
            layer_results=layer_results,
            confidence_abc=score_abc,
            duration_ms=(time.time() - start_time) * 1000,
            slot_check_result=slot_check,
        )
        print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL | score_abc={score_abc:.4f} | missing={slot_check.missing_required} | duration_ms={(time.time()-start_time)*1000:.1f}")
        return result

    # ------------------------------------------------------------------
    # Layer runners
    # ------------------------------------------------------------------

    def _run_layer_0(self, raw_input: str) -> PreprocessingResult:
        return self.preprocessor.preprocess(raw_input)

    def _run_layer_1(
        self, raw_input: str, preprocessing: PreprocessingResult
    ) -> LLMLightResult:
        return self.light_reasoner.reason(raw_input, preprocessing)

    def _run_layer_2(
        self, llm_result: LLMLightResult, preprocessing: PreprocessingResult
    ) -> OntologyMatchResult:
        return self.topology_matcher.match(llm_result, preprocessing)

    def _run_layer_4_deep(
        self,
        raw_input: str,
        preprocessing: PreprocessingResult,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
    ) -> DeepReasoningResult:
        ontology_ctx = self._get_full_ontology_context()
        return self.deep_reasoner.reason(
            raw_input=raw_input,
            preprocessing=preprocessing,
            llm_light=llm_result,
            ontology_match=ontology_match,
            full_ontology_context=ontology_ctx,
        )

    def _run_layer_2_5_slot_completion(
        self,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        intent_id: str = "",
    ) -> SlotCheckResult:
        """Run Layer 2.5: check slot completeness before execution.

        This is the execution gate — if required slots are missing,
        the pipeline is forced into clarification mode regardless of
        confidence scores.
        """
        return self.slot_completion_engine.check(
            llm_result=llm_result,
            ontology_match=ontology_match,
            intent_id=intent_id,
        )

    def _run_slot_fill_protocol(
        self,
        raw_input: str,
        start_time: float,
        layer_results: Dict[str, Any],
    ) -> CompositeIntentResult:
        """Handle [slot-fill] protocol input - highest priority Pre-Router.

        Protocol format: [slot-fill] <task_id> | slot_name: value

        This bypasses ALL intent recognition layers. It:
        1. Parses the protocol to extract task_id and filled slots
        2. Finds the matching pending HITL task
        3. Updates the task's filled_slots
        4. Returns a HITL response asking for remaining slots OR execute-ready result
        """
        from .models import IntentCandidate, HITLRequest

        # Parse the slot-fill protocol
        parsed = SlotFillParser.parse(raw_input)
        if not parsed:
            print(f"[CompositeIntentPipeline][WARN] SlotFill协议解析失败 | input='{raw_input}'")
            # Return a generic error HITL
            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=raw_input,
                candidates=[],
                top_candidate=None,
                confidence=ConfidenceScore(
                    score_ab=0.0, score_abc=0.0,
                    final_decision=ConfidenceDecision.HITL,
                    confidence_a=0.0, confidence_b=0.0, confidence_c=0.0,
                    deep_triggered=False,
                ),
                hitl_request=HITLRequest(
                    question="无法解析slot-fill请求。请使用正确的格式：[slot-fill] <任务ID> | 槽位名: 值",
                    options=[],
                    context={"source": "slot_fill_protocol", "error": "parse_failed"},
                ),
                layer_results={},
                final_decision=ConfidenceDecision.HITL,
                execution_ready=False,
                task_id=self.task_id,
                duration_ms=(time.time() - start_time) * 1000,
            )

        slot_task_id = parsed["task_id"]
        filled_slots = parsed["filled_slots"]

        print(f"[CompositeIntentPipeline][INFO] SlotFill协议解析 | task_id={slot_task_id} | slots={list(filled_slots.keys())}")

        # Find the matching pending task
        pending = self._find_pending_task_by_id(slot_task_id)
        if not pending:
            print(f"[CompositeIntentPipeline][WARN] 未找到PendingTask | task_id={slot_task_id}")
            # Return error HITL
            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=raw_input,
                candidates=[],
                top_candidate=None,
                confidence=ConfidenceScore(
                    score_ab=0.0, score_abc=0.0,
                    final_decision=ConfidenceDecision.HITL,
                    confidence_a=0.0, confidence_b=0.0, confidence_c=0.0,
                    deep_triggered=False,
                ),
                hitl_request=HITLRequest(
                    question=f"未找到任务 {slot_task_id}。该任务可能已过期或不存在。",
                    options=[],
                    context={"source": "slot_fill_protocol", "error": "task_not_found", "task_id": slot_task_id},
                ),
                layer_results={},
                final_decision=ConfidenceDecision.HITL,
                execution_ready=False,
                task_id=self.task_id,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Merge filled slots into pending task
        action = pending.get("action", "")
        object_name = pending.get("object", "")
        original_missing = pending.get("missing_slots", [])
        original_filled = pending.get("filled_slots", {})

        # Validate and normalize filled slots
        validated_slots = self._validate_slot_values(filled_slots, original_missing)

        updated_filled = {**original_filled, **validated_slots}

        # Determine which slots are still missing
        still_missing = [s for s in original_missing if s not in updated_filled or not updated_filled.get(s)]

        print(f"[CompositeIntentPipeline][INFO] SlotFill更新 | 已填={list(validated_slots.keys())} | 仍缺={still_missing}")

        # Update the pending task
        updated_pending = {
            **pending,
            "filled_slots": updated_filled,
            "missing_slots": still_missing,
        }
        self.pending_hitl_task = updated_pending

        # =================================================================
        # Phase 4.1: Persist updated task to PendingTaskStore
        # This ensures slot-fill submissions can continue across HTTP requests
        # =================================================================
        from .pending_task_store import update_pending_task
        update_pending_task(
            slot_task_id,  # task_id is the first positional arg
            filled_slots=updated_filled,
            missing_slots=still_missing,
            status="slots_partial" if still_missing else "slots_complete",
        )

        layer_results["slot_fill_protocol"] = {
            "task_id": slot_task_id,
            "filled_slots": validated_slots,
            "still_missing": still_missing,
            "was_validated": True,
        }

        if still_missing:
            # Still missing slots - generate clarification for remaining ones
            readiness_score = (len(original_missing) - len(still_missing)) / len(original_missing) * 100 if original_missing else 100.0

            # Generate slot-fill question for remaining missing slots
            slot_question = self._generate_slot_fill_confirmation(validated_slots, still_missing, object_name)

            hitl_request = HITLRequest(
                question=slot_question,
                options=[],
                context={
                    "source": "slot_fill_protocol",
                    "mode": "slot_fill",
                    "pending_task": updated_pending,
                    "missing_slots": still_missing,
                    "readiness_score": readiness_score,
                    "newly_filled": list(validated_slots.keys()),
                },
            )
            layer_results["layer_6_hitl"] = hitl_request.to_dict()

            candidates = [
                IntentCandidate(
                    rank=1,
                    intent_id=action,
                    intent_name=action,
                    confidence=1.0,  # We know the intent from pending task
                    params=updated_filled,
                    missing_slots=still_missing,
                )
            ]

            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=raw_input,
                candidates=candidates,
                top_candidate=candidates[0],
                confidence=ConfidenceScore(
                    score_ab=1.0, score_abc=1.0,
                    final_decision=ConfidenceDecision.HITL,
                    confidence_a=1.0, confidence_b=1.0, confidence_c=1.0,
                    deep_triggered=False,
                ),
                hitl_request=hitl_request,
                resolved_values=[],
                layer_results=layer_results,
                final_decision=ConfidenceDecision.HITL,
                execution_ready=False,
                slot_check_result=SlotCheckResult(
                    intent_template=action,
                    object_name=object_name,
                    action_readiness_score=readiness_score,
                    decision="clarify",
                    filled_required=list(updated_filled.keys()),
                    missing_required=still_missing,
                    schema_found=True,
                    metadata={"mode": "slot_fill_protocol"},
                ),
                task_id=self.task_id,
                ontology_version=self._ontology_version,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"slot_fill_protocol": True, "pending_task": updated_pending},
            )
        else:
            # All slots filled - ready to execute
            print(f"[CompositeIntentPipeline][INFO] SlotFill完成 | 所有槽位已填满，准备执行")

            candidates = [
                IntentCandidate(
                    rank=1,
                    intent_id=action,
                    intent_name=action,
                    confidence=1.0,
                    params=updated_filled,
                    missing_slots=[],
                )
            ]

            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=raw_input,
                candidates=candidates,
                top_candidate=candidates[0],
                confidence=ConfidenceScore(
                    score_ab=1.0, score_abc=1.0,
                    final_decision=ConfidenceDecision.EXECUTE,
                    confidence_a=1.0, confidence_b=1.0, confidence_c=1.0,
                    deep_triggered=False,
                ),
                hitl_request=None,
                resolved_values=[],
                layer_results=layer_results,
                final_decision=ConfidenceDecision.EXECUTE,
                execution_ready=True,
                slot_check_result=SlotCheckResult(
                    intent_template=action,
                    object_name=object_name,
                    action_readiness_score=100.0,
                    decision="execute",
                    filled_required=list(updated_filled.keys()),
                    missing_required=[],
                    schema_found=True,
                    metadata={"mode": "slot_fill_protocol", "completed": True},
                ),
                task_id=self.task_id,
                ontology_version=self._ontology_version,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"slot_fill_protocol": True, "slots_completed": True},
            )

    def _find_pending_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Find a pending HITL task by its task_id.

        Uses the persistent PendingTaskStore to find the task across HTTP requests.
        This is critical for slot-fill protocol: the frontend submits [slot-fill] with
        the task_id, and the backend must load the corresponding pending task.
        """
        # First check in-memory pending_hitl_task (for same-session slot filling)
        if self.pending_hitl_task and self.pending_hitl_task.get("task_id") == task_id:
            return self.pending_hitl_task

        # Then check the persistent task store (for cross-request slot filling)
        from .pending_task_store import load_pending_task
        pending = load_pending_task(task_id)
        if pending:
            # Convert PendingTask to dict format
            return {
                "task_id": pending.task_id,
                "action": pending.action,
                "object": pending.object,
                "missing_slots": pending.missing_slots,
                "filled_slots": pending.filled_slots,
                "status": pending.status,
                "confidence": pending.confidence,
                "metadata": pending.metadata,
            }

        print(f"[CompositeIntentPipeline][WARN] PendingTask未找到 | task_id={task_id}")
        return None

    def _validate_slot_values(
        self, filled_slots: Dict[str, Any], expected_slots: List[str]
    ) -> Dict[str, Any]:
        """Validate and normalize slot values.

        Parameters
        ----------
        filled_slots:
            The slot values to validate.
        expected_slots:
            The list of slot names that should be filled.

        Returns
        -------
        Dict[str, Any]
            Validated and normalized slot values.
        """
        validated = {}
        for slot_name, value in filled_slots.items():
            if not value:
                continue

            # Normalize slot name
            normalized_name = self.slot_completion_engine._normalize_slot_name(slot_name)

            # Date validation - delivery_date must be a valid future date
            if normalized_name == "delivery_date":
                if self._is_valid_delivery_date(value):
                    validated[normalized_name] = value
                    validated[slot_name] = value  # Keep original too
                else:
                    print(f"[CompositeIntentPipeline][WARN] 日期验证失败 | slot=delivery_date | value={value}")
                    # Still accept it but flag it
                    validated[normalized_name] = value
                    validated[slot_name] = value
            else:
                validated[normalized_name] = value
                validated[slot_name] = value

        return validated

    def _is_valid_delivery_date(self, value: str) -> bool:
        """Check if a delivery_date value is valid.

        Valid if:
        1. It can be parsed as a date
        2. The date is today or in the future (for procurement delivery dates)

        Invalid if:
        - Cannot be parsed as a date
        - Is in the past (past dates are not valid delivery dates)
        """
        from datetime import datetime, date

        if not value:
            return False

        # Try to parse various date formats
        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y年%m月%d日",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]

        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(value, fmt).date()
                break
            except ValueError:
                continue

        if parsed_date is None:
            print(f"[CompositeIntentPipeline][WARN] 无法解析日期 | value={value}")
            return False

        today = date.today()

        # For delivery dates, we accept today or future dates
        if parsed_date >= today:
            print(f"[CompositeIntentPipeline][INFO] 日期有效 | value={value} | parsed={parsed_date} | is_future=True")
            return True
        else:
            print(f"[CompositeIntentPipeline][WARN] 日期在过去 | value={value} | parsed={parsed_date} | today={today}")
            return False

    def _generate_slot_fill_confirmation(
        self,
        filled_slots: Dict[str, Any],
        still_missing: List[str],
        object_name: str,
    ) -> str:
        """Generate a confirmation message showing what was filled and what is still needed."""
        lines = []

        object_label = self.slot_completion_engine._get_object_label(object_name)

        # Show what was filled
        for slot_name, value in filled_slots.items():
            slot_label = self.slot_completion_engine._get_slot_label(slot_name)
            lines.append(f"已记录 {slot_label}：{value}")

        # Show what's still needed
        if still_missing:
            lines.append("")
            lines.append("还需要补充以下信息：")
            for i, slot in enumerate(still_missing, 1):
                slot_label = self.slot_completion_engine._get_slot_label(slot)
                slot_placeholder = self.slot_completion_engine._get_placeholder(slot)
                lines.append(f"{i}. {slot_label}（{slot_placeholder}）")

        return "\n".join(lines)

    def _run_slot_filling(
        self,
        raw_input: str,
        start_time: float,
        layer_results: Dict[str, Any],
    ) -> CompositeIntentResult:
        """Handle slot filling mode when pending HITL task exists.

        This is RULE 1: if session.pending_hitl_task exists, route to slot_completion
        and skip global intent detection.
        """
        # Layer 0: Still preprocess the input (to extract slot values)
        preprocessing = self._run_layer_0(raw_input)
        layer_results["layer_0_preprocessing"] = preprocessing.to_dict()

        # Layer 1: Light reasoner extracts dimension matches (slots)
        llm_result = self._run_layer_1(raw_input, preprocessing)
        layer_results["layer_1_light_reasoning"] = llm_result.to_dict()

        # Layer 2: Ontology match for validation
        ontology_match = self._run_layer_2(llm_result, preprocessing)
        layer_results["layer_2_ontology_match"] = ontology_match.to_dict()

        # Get pending task info
        pending = self.pending_hitl_task
        action = pending.get("action", "")
        object_name = pending.get("object", "")
        missing_slots = pending.get("missing_slots", [])
        filled_slots = pending.get("filled_slots", {})

        # Extract newly provided values from this input
        extracted = self.slot_completion_engine._build_extracted_slots(llm_result)

        # Merge extracted slots into filled_slots
        updated_filled = dict(filled_slots)
        for slot_name, value in extracted.items():
            if value and value != "":
                updated_filled[slot_name] = value

        # Determine which of the originally missing slots are now filled
        still_missing = [s for s in missing_slots if s not in updated_filled]
        newly_filled = [s for s in missing_slots if s in updated_filled]

        print(f"[CompositeIntentPipeline][INFO] Slot Filling | 已填槽={newly_filled} | 仍缺槽={still_missing} | 总已填={list(updated_filled.keys())}")

        # Build slot check result for this round
        from .models import SlotCheckResult
        slot_check = SlotCheckResult(
            intent_template=action,
            object_name=object_name,
            action_readiness_score=0.0 if still_missing else 100.0,
            decision="clarify" if still_missing else "execute",
            filled_required=list(updated_filled.keys()),
            missing_required=still_missing,
            schema_found=True,
            metadata={
                "mode": "slot_filling",
                "pending_task_id": pending.get("task_id", ""),
                "newly_filled": newly_filled,
                "updated_filled_slots": updated_filled,
            },
        )
        layer_results["layer_2_5_slot_completion"] = slot_check.to_dict()

        # Update pending task with new slot values
        self.pending_hitl_task = {
            **pending,
            "filled_slots": updated_filled,
            "missing_slots": still_missing,
        }

        if still_missing:
            # Still missing slots — generate clarification question
            slot_question = self.slot_completion_engine.generate_clarification_question(
                slot_check, object_name
            )
            from .models import HITLRequest
            hitl_request = HITLRequest(
                question=slot_question,
                options=[],
                context={
                    "source": "slot_completion",
                    "mode": "slot_filling",
                    "pending_task": self.pending_hitl_task,
                    "missing_slots": still_missing,
                    "readiness_score": slot_check.action_readiness_score,
                    "newly_filled": newly_filled,
                },
            )
            layer_results["layer_6_hitl"] = hitl_request.to_dict()

            # Build result: same action, updated slots, still need clarification
            candidates = [
                IntentCandidate(
                    rank=1,
                    intent_id=action,
                    intent_name=action,
                    confidence=1.0,  # We know the intent from pending task
                    params=updated_filled,
                    missing_slots=still_missing,
                )
            ]

            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=preprocessing.normalized_input,
                candidates=candidates,
                top_candidate=candidates[0],
                confidence=ConfidenceScore(
                    score_ab=1.0,
                    score_abc=1.0,
                    final_decision=ConfidenceDecision.HITL,
                    confidence_a=1.0,
                    confidence_b=1.0,
                    confidence_c=1.0,
                    deep_triggered=False,
                ),
                hitl_request=hitl_request,
                resolved_values=[],
                layer_results=layer_results,
                final_decision=ConfidenceDecision.HITL,
                execution_ready=False,
                slot_check_result=slot_check,
                task_id=self.task_id,
                ontology_version=self._ontology_version,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"slot_filling_mode": True, "pending_task": self.pending_hitl_task},
            )
        else:
            # All slots filled — ready to execute
            print(f"[CompositeIntentPipeline][INFO] Slot Filling完成 | 所有槽位已填满，准备执行")
            self.pending_hitl_task = None  # Clear pending

            candidates = [
                IntentCandidate(
                    rank=1,
                    intent_id=action,
                    intent_name=action,
                    confidence=1.0,
                    params=updated_filled,
                    missing_slots=[],
                )
            ]

            return CompositeIntentResult(
                raw_input=raw_input,
                normalized_input=preprocessing.normalized_input,
                candidates=candidates,
                top_candidate=candidates[0],
                confidence=ConfidenceScore(
                    score_ab=1.0,
                    score_abc=1.0,
                    final_decision=ConfidenceDecision.EXECUTE,
                    confidence_a=1.0,
                    confidence_b=1.0,
                    confidence_c=1.0,
                    deep_triggered=False,
                ),
                hitl_request=None,
                resolved_values=[],
                layer_results=layer_results,
                final_decision=ConfidenceDecision.EXECUTE,
                execution_ready=True,
                slot_check_result=slot_check,
                task_id=self.task_id,
                ontology_version=self._ontology_version,
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"slot_filling_mode": True, "slots_completed": True},
            )

    def _run_layer_6_hitl(
        self,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        slot_check: Optional[SlotCheckResult] = None,
    ) -> HITLRequest:
        known_facts = self._extract_known_facts(llm_result, ontology_match)
        unclear_points = list(ontology_match.gaps) + list(llm_result.missing_slots)
        candidate_paths = self._get_candidate_paths(ontology_match)

        # If slot check shows missing required slots, use slot-specific question
        if slot_check is not None and slot_check.has_missing_required:
            slot_question = self.slot_completion_engine.generate_clarification_question(
                slot_check, ontology_match.object_match.object_name if ontology_match.object_match else ""
            )
            if slot_question:
                return HITLRequest(
                    question=slot_question,
                    options=[],
                    context={
                        "source": "slot_completion",
                        "missing_slots": slot_check.missing_required,
                        "readiness_score": slot_check.action_readiness_score,
                    },
                )

        return self.hitl_engine.generate(known_facts, unclear_points, candidate_paths)

    def _run_layer_7_dynamic(
        self,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        preprocessing: PreprocessingResult,
    ) -> List[ResolvedValue]:
        tasks = preprocessing.dynamic_resolution_tasks or []
        if not tasks:
            # Extract dynamic terms from dimension matches
            tasks = [
                {"term": d.value, "term_type": "organization"}
                for d in ontology_match.dimension_matches
                if d.value
            ]
        if not tasks:
            return []

        return self.dynamic_resolver.resolve_batch(tasks)

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        raw_input: str,
        preprocessing: PreprocessingResult,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        confidence_ab: float,
        deep_result: Optional[DeepReasoningResult],
        hitl_request: Optional[HITLRequest],
        decision: ConfidenceDecision,
        execution_ready: bool,
        layer_results: Dict[str, Any],
        resolved_values: List[ResolvedValue] = None,
        confidence_abc: float = 0.0,
        duration_ms: float = 0.0,
        slot_check_result: Optional[SlotCheckResult] = None,
    ) -> CompositeIntentResult:
        candidates = self._build_candidates(llm_result, ontology_match, deep_result)

        # =========================================================================
        # RULE 4: If deep reasoning top candidate confidence >> current top,
        # promote it to the primary candidate (update semantic contract).
        # =========================================================================
        top = candidates[0] if candidates else None
        if deep_result and deep_result.ranked_candidates:
            deep_top = deep_result.ranked_candidates[0]
            deep_conf = deep_top.get("confidence", 0.0)
            deep_id = deep_top.get("intent_id", "")
            # If deep reasoning is significantly better (>=0.05 margin) and not a generic template
            if top and deep_conf > top.confidence + 0.05:
                if deep_id not in BLOCKED_FINAL_ACTIONS:
                    print(f"[CompositeIntentPipeline][INFO] Deep推理升级 | {top.intent_id}({top.confidence:.2f}) → {deep_id}({deep_conf:.2f})")
                    top = IntentCandidate(
                        rank=top.rank,
                        intent_id=deep_id,
                        intent_name=deep_top.get("intent_name", deep_id),
                        confidence=deep_conf,
                        matched_path_id=deep_top.get("matched_path_id", ""),
                        params=deep_top.get("params", {}),
                        missing_slots=deep_top.get("missing_slots", []),
                    )
                    # Rebuild candidates with new top at position 1
                    candidates = [top] + [c for c in candidates[1:] if c.intent_id != deep_id]

        # =========================================================================
        # RULE 5: Block generic templates from being final action.
        # query_object / create_object can only exist as alternatives.
        # =========================================================================
        if top and top.intent_id in BLOCKED_FINAL_ACTIONS:
            # Look for a better alternative
            better_candidates = [c for c in candidates[1:] if c.intent_id not in BLOCKED_FINAL_ACTIONS]
            if better_candidates:
                top = better_candidates[0]
                candidates = [top] + [c for c in candidates if c != top]
                print(f"[CompositeIntentPipeline][INFO] 通用模板被拦截 | blocked={candidates[1].intent_id if len(candidates) > 1 else 'N/A'} | promoted={top.intent_id}")
            else:
                # No valid alternative — force HITL
                decision = ConfidenceDecision.HITL
                execution_ready = False
                print(f"[CompositeIntentPipeline][WARN] 无有效候选 | {top.intent_id}被阻止，无替代方案")

        # Auto-derive execution_ready from slot check if available
        if slot_check_result is not None:
            actual_ready = execution_ready and not slot_check_result.has_missing_required
        else:
            actual_ready = execution_ready

        conf = ConfidenceScore(
            score_ab=confidence_ab,
            score_abc=confidence_abc,
            final_decision=decision,
            confidence_a=llm_result.confidence_a,
            confidence_b=ontology_match.confidence_b,
            confidence_c=deep_result.confidence_c if deep_result else 0.0,
            deep_triggered=deep_result is not None,
        )

        return CompositeIntentResult(
            raw_input=raw_input,
            normalized_input=preprocessing.normalized_input,
            candidates=candidates,
            top_candidate=top,
            confidence=conf,
            hitl_request=hitl_request,
            resolved_values=resolved_values or [],
            layer_results=layer_results,
            final_decision=decision,
            execution_ready=actual_ready,
            slot_check_result=slot_check_result,
            task_id=self.task_id,
            ontology_version=self._ontology_version,
            duration_ms=duration_ms,
        )

    def _build_candidates(
        self,
        llm_result: LLMLightResult,
        ontology_match: OntologyMatchResult,
        deep_result: Optional[DeepReasoningResult],
    ) -> List[IntentCandidate]:
        candidates: List[IntentCandidate] = []

        # Primary candidate from LLM result + ontology match
        top_obj = ""
        if llm_result.object_matches:
            top_obj = llm_result.object_matches[0].object_name
        if not top_obj and ontology_match.object_match:
            top_obj = ontology_match.object_match.object_name

        # Build params with object_term and dimension matches
        params = {d.dimension_name: d.value for d in llm_result.dimension_matches}
        if top_obj:
            params["object_term"] = top_obj

        candidates.append(
            IntentCandidate(
                rank=1,
                intent_id=llm_result.primary_intent or "unknown",
                intent_name=llm_result.primary_intent or "未识别意图",
                confidence=llm_result.confidence_a,
                matched_path_id=(
                    ontology_match.matched_path_ids[0]
                    if ontology_match.matched_path_ids
                    else ""
                ),
                params=params,
                missing_slots=list(llm_result.missing_slots),
            )
        )

        # Deep reasoning ranked candidates
        if deep_result and deep_result.ranked_candidates:
            for i, cand in enumerate(deep_result.ranked_candidates[:5]):
                candidates.append(
                    IntentCandidate(
                        rank=i + 2,
                        intent_id=cand.get("intent_id", ""),
                        intent_name=cand.get("intent_name", cand.get("intent_id", "")),
                        confidence=cand.get("confidence", 0.0),
                        matched_path_id=cand.get("matched_path_id", ""),
                        params=cand.get("params", {}),
                        missing_slots=cand.get("missing_slots", []),
                    )
                )

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_task_id(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:6].upper()
        return f"TASK-{date_str}-{short_uuid}"

    def _get_full_ontology_context(self) -> str:
        try:
            from ..procurement.ontology_loader import OntologyQuery

            query = OntologyQuery()
            return query.to_context_for_llm()
        except Exception:
            return ""

    def _extract_known_facts(
        self, llm_result: LLMLightResult, ontology_match: OntologyMatchResult
    ) -> List[str]:
        facts: List[str] = []

        if llm_result.primary_intent:
            facts.append(f"意图模板: {llm_result.primary_intent}")

        if llm_result.object_matches:
            obj = llm_result.object_matches[0]
            facts.append(f"对象: {obj.object_name or obj.matched_phrase}")

        if ontology_match.object_match and ontology_match.object_match.object_name:
            facts.append(f"本体对象: {ontology_match.object_match.object_name}")

        if llm_result.dimension_matches:
            dim_strs = [
                f"{d.dimension_name}={d.value}"
                for d in llm_result.dimension_matches
                if d.dimension_name and d.value
            ]
            if dim_strs:
                facts.append(f"提取的过滤条件: {', '.join(dim_strs)}")

        return facts

    def _get_candidate_paths(
        self, ontology_match: OntologyMatchResult
    ) -> List[IntentPath]:
        """Resolve matched path IDs back to IntentPath objects from the topology."""
        topo = self.topology
        path_map = {p.id: p for p in topo.paths}

        results: List[IntentPath] = []
        for path_id in ontology_match.matched_path_ids[:5]:
            if path_id in path_map:
                results.append(path_map[path_id])

        # Fall back: return paths for the matched object
        if not results and ontology_match.object_match:
            obj = ontology_match.object_match.object_name
            results = topo.get_paths_for_object(obj)[:5]

        return results
