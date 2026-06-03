"""Step 1: Intent Recognition for the 5-step pipeline."""

from typing import List, Optional, TYPE_CHECKING
from .models import IntentRecognitionResult, SemanticContract, SemanticContractSlot

if TYPE_CHECKING:
    from ..intent_recognition.models import CompositeIntentResult

_COMPOSITE_AVAILABLE = True


class Step1Recognizer:
    """Step 1: Intent Recognition.

    Recognizes user intent from the input message. First tries the
    ProcurementIntentRouter, then falls back to the SkillManager classifier.

    When the composite pipeline (Phase 2-3) is available, it is used as the
    primary recognition engine. The legacy router serves as a fallback.

    Enhanced with slot extraction for:
    - Time range (上周、本月等)
    - Business conditions (采购类型、物料分类、部门等)
    """

    def __init__(self):
        self.procurement_router = None  # Lazy import
        self.skill_manager = None
        self._slot_extractor = None  # Lazy import
        self._composite_pipeline = None  # Lazy import

    def _get_composite_pipeline(self, task_id: str = ""):
        """Get the composite intent recognition pipeline (Layer 0-7)."""
        if not _COMPOSITE_AVAILABLE:
            return None
        if self._composite_pipeline is None:
            from ..intent_recognition import CompositeIntentPipeline
            self._composite_pipeline = CompositeIntentPipeline(
                use_deep_reasoning=True,
                use_hitl=True,
                task_id=task_id,
            )
        elif task_id:
            # Update task_id on existing pipeline
            self._composite_pipeline.task_id = task_id
        return self._composite_pipeline

    def _get_procurement_router(self):
        if self.procurement_router is None:
            from ..procurement.intent_router import IntentRouter
            self.procurement_router = IntentRouter()
        return self.procurement_router

    def _get_slot_extractor(self):
        """Get the slot extractor for condition extraction."""
        if self._slot_extractor is None:
            from ..procurement.slot_extractors import ProcurementSlotExtractor
            self._slot_extractor = ProcurementSlotExtractor()
        return self._slot_extractor

    def _parse_temporal(self, user_input: str) -> dict:
        """Parse temporal expressions in user input.
        
        Returns:
            dict with temporal context: {"label": "上周", "date_from": "2026-05-26", "date_to": "2026-06-01"}
        """
        from ..temporal.temporal_parser import TemporalParser
        
        parser = TemporalParser()
        ctx = parser.parse(user_input)
        
        # Get resolved dates
        temporal_result = {}
        if ctx.dates:
            # Get the week range if "上周" was detected
            for label in ctx.dates:
                if "周" in label or label in ["今天", "昨天", "明天"]:
                    d = ctx.dates[label]
                    temporal_result["label"] = label
                    temporal_result["resolved_date"] = d.isoformat()
                    
            # If no specific weekday found but we have dates, create range
            if "label" not in temporal_result and ctx.dates:
                # Use the first date's week range
                pass
        
        return temporal_result

    def _create_semantic_contract(self, composite_result, intent_id: str, intent_name: str, intent_term: str, slots_dict: dict, temporal_ctx: dict) -> SemanticContract:
        """Create SemanticContract from composite pipeline result."""
        top = composite_result.top_candidate
        if not top:
            raise ValueError("Cannot create SemanticContract without top candidate")

        # Extract slots from top candidate params
        slots = {}
        if top.params:
            for key, value in top.params.items():
                if key not in ("object_term", "normalized_object_term", "intent_id", "intent_name"):
                    slots[key] = SemanticContractSlot(
                        name=key,
                        display_value=str(value),
                        source="extracted",
                    )

        # Add temporal context if available
        if temporal_ctx:
            for temporal_key in ("date_from", "date_to", "label"):
                if temporal_key in temporal_ctx:
                    slots[f"time_{temporal_key}"] = SemanticContractSlot(
                        name=f"time_{temporal_key}",
                        display_value=str(temporal_ctx[temporal_key]),
                        source="computed",
                    )

        # Build alternatives from composite candidates
        alternatives = []
        for cand in composite_result.candidates[1:5]:  # Skip first (it's the top)
            alternatives.append({
                "intent_id": cand.intent_id,
                "intent_name": cand.intent_name,
                "confidence": cand.confidence,
                "params": cand.params,
            })

        # Check deep reasoning for HITL info
        hitl_triggered = False
        missing_info = []
        deep_result = composite_result.layer_results.get("layer_4_deep_reasoning", {})
        if deep_result:
            hitl_triggered = deep_result.get("recommended_decision") == "hitl"
            missing_info = deep_result.get("missing_info", [])

        return SemanticContract(
            object=top.params.get("object_term", intent_term or "unknown"),
            object_label=intent_term or top.params.get("object_term", "业务对象"),
            action=top.intent_id,
            action_label=top.intent_name or top.intent_id,
            confidence=top.confidence,
            slots=slots,
            alternatives=alternatives,
            hitl_triggered=hitl_triggered,
            missing_info=missing_info,
        )

    def recognize(self, user_input: str, task_id: str = "") -> IntentRecognitionResult:
        """Recognize intent from user input.

        Returns IntentRecognitionResult with intent details and extracted slots.
        """
        print(f"[Step1Recognizer][INFO] ===== Step1识别开始 | input='{user_input}'")
        # 1. Extract condition slots first (time range, business conditions)
        extractor = self._get_slot_extractor()
        extracted_slots = extractor.extract(user_input)
        slots_dict = extracted_slots.to_dict()
        
        # Build temporal context
        temporal_ctx = {}
        if extracted_slots.time_range:
            temporal_ctx = {
                "label": extracted_slots.time_range.label,
                "date_from": extracted_slots.time_range.date_from,
                "date_to": extracted_slots.time_range.date_to,
            }
        
        # Build condition summary for display
        condition_parts = []
        if extracted_slots.time_range:
            condition_parts.append(f"{extracted_slots.time_range.label}({extracted_slots.time_range.date_from}至{extracted_slots.time_range.date_to})")
        if extracted_slots.pr_type:
            condition_parts.append(f"采购类型={extracted_slots.pr_type.display_value}")
        if extracted_slots.material_category:
            condition_parts.append(f"物料分类={extracted_slots.material_category.display_value}")
        if extracted_slots.department:
            condition_parts.append(f"部门={extracted_slots.department.display_value}")
        if extracted_slots.execution_status:
            condition_parts.append(f"状态={extracted_slots.execution_status.display_value}")
        condition_summary = "、".join(condition_parts) if condition_parts else ""
        
        # 2. Try composite intent pipeline first (Phase 2-3 enhanced recognition)
        composite_pipeline = self._get_composite_pipeline(task_id=task_id)
        if composite_pipeline:
            composite_result = composite_pipeline.run(user_input)
            if composite_result.top_candidate:
                top = composite_result.top_candidate
                operation_type = self._map_action_to_operation(top.intent_id)
                risk_level = self._assess_risk(operation_type)
                print(f"[Step1Recognizer][INFO] 复合管道识别结果 | intent={top.intent_id} | intent_name={top.intent_name} | confidence={top.confidence:.4f} | decision={composite_result.final_decision.value}")
                
                # Create Semantic Contract - the single source of truth
                semantic_contract = self._create_semantic_contract(
                    composite_result, top.intent_id, top.intent_name,
                    top.params.get("object_term", ""), slots_dict, temporal_ctx
                )
                
                return IntentRecognitionResult(
                    intent=top.intent_id,
                    intent_label=top.intent_name or top.intent_id,
                    object_term=top.params.get("object_term", ""),
                    normalized_term=top.params.get("normalized_object_term"),
                    operation_type=operation_type,
                    risk_level=risk_level,
                    requires_confirmation=(composite_result.final_decision.value == "hitl"),
                    confidence=top.confidence,
                    alternative_intents=[c.intent_id for c in composite_result.candidates[1:4]],
                    extracted_slots=slots_dict,
                    temporal_context=temporal_ctx,
                    condition_summary=condition_summary,
                    # Phase 3: composite pipeline result
                    composite_result=composite_result,
                    hitl_request=composite_result.hitl_request,
                    composite_layer_results=composite_result.layer_results,
                    composite_task_id=composite_result.task_id,
                    # Phase 4: Semantic Contract
                    semantic_contract=semantic_contract,
                )

        # 3. Try procurement router (legacy path)
        print(f"[Step1Recognizer][INFO] 复合管道未命中，进入旧路由")
        router = self._get_procurement_router()
        match = router.route(user_input)

        if match:
            # Determine operation type from action
            operation_type = self._map_action_to_operation(match.intent.action)

            # Determine risk level from operation type
            risk_level = self._assess_risk(operation_type)

            # Extract object term from user input
            object_term = self._extract_object_term(user_input)

            # Check if normalization happened
            normalized_term = None
            synonyms = self._get_synonyms(object_term)
            if object_term not in synonyms:
                normalized_term = object_term

            return IntentRecognitionResult(
                intent=match.intent.id,
                intent_label=match.intent.name,
                object_term=object_term,
                normalized_term=normalized_term,
                operation_type=operation_type,
                risk_level=risk_level,
                requires_confirmation=match.intent.required_slots and len(match.missing_slots) > 0,
                confidence=match.confidence / 100.0 if hasattr(match, 'confidence') else 1.0,
                alternative_intents=[],
                extracted_slots=slots_dict,
                temporal_context=temporal_ctx,
                condition_summary=condition_summary,
            )

        # 4. Fallback: use skill manager
        print(f"[Step1Recognizer][INFO] 所有路径未命中，进入Fallback")
        result = self._fallback_recognition(user_input, slots_dict, temporal_ctx, condition_summary)
        print(f"[Step1Recognizer][INFO] Fallback识别结果 | intent={result.intent} | confidence={result.confidence:.4f}")
        return result

    def _map_action_to_operation(self, action: str) -> str:
        """Map action ID to operation type."""
        action_lower = action.lower()
        if "create" in action_lower or "add" in action_lower:
            return "create"
        if "update" in action_lower or "approve" in action_lower:
            return "update"
        if "delete" in action_lower:
            return "delete"
        if "publish" in action_lower:
            return "publish"
        if "compare" in action_lower or "recommend" in action_lower:
            return "recommend"
        return "query"

    def _assess_risk(self, operation_type: str) -> str:
        """Assess risk level based on operation type."""
        risk_map = {
            "query": "low",
            "recommend": "medium",
            "create": "medium",
            "update": "medium",
            "publish": "high",
            "delete": "high",
        }
        return risk_map.get(operation_type, "low")

    def _extract_object_term(self, user_input: str) -> str:
        """Extract the main object term from user input."""
        # Simple keyword-based extraction
        keywords = ["采购需求", "采购计划", "询价单", "报价单", "采购订单", "PO",
                    "物料", "供应商", "订单", "采购"]
        for kw in keywords:
            if kw in user_input:
                return kw
        return "业务对象"

    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        synonym_map = {
            "采购需求": ["采购计划", "物料需求", "需求单", "PR"],
            "采购订单": ["PO", "采购单", "订单"],
            "询价单": ["RFQ", "询价"],
            "报价单": ["QUO", "报价"],
        }
        base = [term]
        base.extend(synonym_map.get(term, []))
        return base

    def _fallback_recognition(
        self,
        user_input: str,
        slots_dict: dict = None,
        temporal_ctx: dict = None,
        condition_summary: str = ""
    ) -> IntentRecognitionResult:
        """Fallback intent recognition using LLM-based classification."""
        # If we have extracted slots, this is likely a procurement query with conditions
        if slots_dict and any([
            slots_dict.get('time_range'),
            slots_dict.get('pr_type'),
            slots_dict.get('material_category'),
            slots_dict.get('department'),
        ]):
            return IntentRecognitionResult(
                intent="procurement/query_with_conditions",
                intent_label="带条件的采购需求查询",
                object_term=self._extract_object_term(user_input),
                operation_type="query",
                risk_level="low",
                requires_confirmation=False,
                confidence=0.85,
                extracted_slots=slots_dict or {},
                temporal_context=temporal_ctx or {},
                condition_summary=condition_summary,
            )

        return IntentRecognitionResult(
            intent="general_query",
            intent_label="通用查询",
            object_term=self._extract_object_term(user_input),
            operation_type="query",
            risk_level="low",
            requires_confirmation=False,
            confidence=0.3,
            extracted_slots=slots_dict or {},
            temporal_context=temporal_ctx or {},
            condition_summary=condition_summary,
        )

    def get_slot_extraction_result(self, user_input: str) -> dict:
        """Get detailed slot extraction result for debugging/display.

        Returns:
            dict with extracted slot information for the given user input.
        """
        from ..procurement.slot_extractors import ProcurementSlotExtractor

        extractor = ProcurementSlotExtractor()
        slots = extractor.extract(user_input)

        return {
            "user_input": user_input,
            "slots": slots.to_dict(),
            "connector_params": slots.to_connector_params(),
            "display_text": extractor.extract_for_display(user_input),
        }

    def run_composite(self, user_input: str, task_id: Optional[str] = None):
        """Run the composite intent pipeline (Layers 0-7) directly.

        Returns the raw CompositeIntentResult for use by Step2 and the pipeline.

        Parameters
        ----------
        user_input:
            Raw user input string.
        task_id:
            Optional task ID for audit tracing.

        Returns
        -------
        CompositeIntentResult or None
            The full composite result, or None if the pipeline is unavailable.
        """
        pipeline = self._get_composite_pipeline()
        if pipeline is None:
            return None
        if task_id:
            pipeline.task_id = task_id
        return pipeline.run(user_input)

    def get_composite_hitl(self, user_input: str, task_id: Optional[str] = None):
        """Check if composite pipeline needs HITL for this input.

        Returns the HITLRequest if HITL is needed, None otherwise.
        This is a lightweight alternative to run_composite() for
        checking whether the user needs to be asked for clarification.

        Returns
        -------
        HITLRequest or None
        """
        result = self.run_composite(user_input, task_id)
        if result is None:
            return None
        return result.hitl_request
