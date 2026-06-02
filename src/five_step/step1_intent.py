"""Step 1: Intent Recognition for the 5-step pipeline."""

from typing import List, Optional
from .models import IntentRecognitionResult


class Step1Recognizer:
    """Step 1: Intent Recognition.

    Recognizes user intent from the input message. First tries the
    ProcurementIntentRouter, then falls back to the SkillManager classifier.
    
    Enhanced with slot extraction for:
    - Time range (上周、本月等)
    - Business conditions (采购类型、物料分类、部门等)
    """

    def __init__(self):
        self.procurement_router = None  # Lazy import
        self.skill_manager = None
        self._slot_extractor = None  # Lazy import

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

    def recognize(self, user_input: str) -> IntentRecognitionResult:
        """Recognize intent from user input.

        Returns IntentRecognitionResult with intent details and extracted slots.
        """
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
        
        # 2. Try procurement router first
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

        # 3. Fallback: use skill manager
        return self._fallback_recognition(user_input, slots_dict, temporal_ctx, condition_summary)

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
