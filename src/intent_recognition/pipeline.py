"""
Composite Intent Recognition Pipeline
====================================
Phase 3: End-to-end 7-layer intent recognition pipeline.

Orchestrates Layers 0-7:
    Layer 0: InputPreprocessor       (normalization + tokenization)
    Layer 1: LightIntentReasoner    (fast LLM, thinking_budget=500)
    Layer 2: OntologyTopologyMatcher (ontology validation, parallel with Layer 1)
    Layer 3: ConfidenceFusionEngine  (A×0.45 + B×0.55)
    Layer 4: DeepIntentReasoner      (heavy LLM, thinking_budget=1200, conditional)
    Layer 5: ConfidenceFusionEngine  (A×0.25 + B×0.35 + C×0.40, conditional)
    Layer 6: HITLClarificationEngine  (human-in-the-loop, conditional)
    Layer 7: DynamicSlotResolver      (master-data lookup)

Each layer emits structured results; fusion layers decide whether to
continue, trigger deep reasoning, or escalate to HITL.
"""

import time
import uuid
from datetime import datetime
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
)
from .preprocessor import InputPreprocessor
from .light_reasoner import LightIntentReasoner
from .topology_matcher import OntologyTopologyMatcher
from .fusion_engine import ConfidenceFusionEngine
from .deep_reasoner import DeepIntentReasoner
from .hitl_engine import HITLClarificationEngine
from .dynamic_resolver import DynamicSlotResolver


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

DEFAULT_DEEP_REASONING_THRESHOLD = 0.75
DEFAULT_HITL_THRESHOLD = 0.75
DEFAULT_EXECUTE_THRESHOLD = 0.85


class CompositeIntentPipeline:
    """End-to-end composite intent recognition pipeline.

    Run ``run()`` with a user input string to get a ``CompositeIntentResult``
    with the resolved intent and any HITL request.

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
    ):
        self.task_id = task_id or self._generate_task_id()
        self.use_deep_reasoning = use_deep_reasoning
        self.use_hitl = use_hitl
        self.audit_logger = audit_logger

        # Lazy-initialized layer instances
        self._preprocessor: Optional[InputPreprocessor] = None
        self._light_reasoner: Optional[LightIntentReasoner] = None
        self._topology_matcher: Optional[OntologyTopologyMatcher] = None
        self._fusion_engine: Optional[ConfidenceFusionEngine] = None
        self._deep_reasoner: Optional[DeepIntentReasoner] = None
        self._hitl_engine: Optional[HITLClarificationEngine] = None
        self._dynamic_resolver: Optional[DynamicSlotResolver] = None

        # Cached topology
        self._topology = None
        self._ontology_version = ""

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

        # Layer 0: Preprocessing
        preprocessing = self._run_layer_0(raw_input)
        layer_results["layer_0_preprocessing"] = preprocessing.to_dict()

        # Layer 1: Light LLM reasoning  (and Layer 2: topology match — parallel in concept,
        # sequential in sync implementation; both use the same preprocessing result)
        llm_result = self._run_layer_1(raw_input, preprocessing)
        layer_results["layer_1_light_reasoning"] = llm_result.to_dict()

        # Layer 2: Ontology topology matching
        ontology_match = self._run_layer_2(llm_result, preprocessing)
        layer_results["layer_2_ontology_match"] = ontology_match.to_dict()

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
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=EXECUTE | score_ab={score_ab:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        if decision_ab == "continue":
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
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=CONTINUE | score_ab={score_ab:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        # decision_ab == "deep_reasoning" — proceed to Layer 4
        print(f"[CompositeIntentPipeline][INFO] Layer-3融合结果 | score_ab={score_ab:.4f} | decision=DEEP_REASONING | 进入Layer-4")
        if not self.use_deep_reasoning:
            # Skip deep reasoning; go directly to HITL
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

        # Layer 5: Second fusion (A × 0.25 + B × 0.35 + C × 0.40)
        score_abc, decision_abc = self.fusion_engine.fuse_abc(
            llm_result.confidence_a,
            ontology_match.confidence_b,
            deep_result.confidence_c,
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
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=CONTINUE(ABC) | score_abc={score_abc:.4f} | intent={result.top_candidate.intent_id if result.top_candidate else 'N/A'} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        # decision_abc == "hitl" — Layer 6
        print(f"[CompositeIntentPipeline][INFO] Layer-5融合结果 | score_abc={score_abc:.4f} | decision=HITL")
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
            )
            print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL(disabled) | score_abc={score_abc:.4f} | duration_ms={(time.time()-start_time)*1000:.1f}")
            return result

        hitl_request = self._run_layer_6_hitl(llm_result, ontology_match)
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
        )
        print(f"[CompositeIntentPipeline][INFO] Pipeline结束 | decision=HITL | score_abc={score_abc:.4f} | duration_ms={(time.time()-start_time)*1000:.1f}")
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

    def _run_layer_6_hitl(
        self, llm_result: LLMLightResult, ontology_match: OntologyMatchResult
    ) -> HITLRequest:
        known_facts = self._extract_known_facts(llm_result, ontology_match)
        unclear_points = list(ontology_match.gaps) + list(llm_result.missing_slots)
        candidate_paths = self._get_candidate_paths(ontology_match)
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
    ) -> CompositeIntentResult:
        candidates = self._build_candidates(llm_result, ontology_match, deep_result)
        top = candidates[0] if candidates else None

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
            execution_ready=execution_ready,
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
