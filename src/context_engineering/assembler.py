"""Context Assembler — assembles LLMContextPackage from selected/filtered context."""

from typing import Any, Dict, List

from src.context_engineering.models import (
    ContextIntent,
    ContextMeta,
    ContextOntology,
    ContextPurpose,
    ContextUserInput,
    ContextInstructions,
    LLMContextPackage,
    TokenBudget,
)


class ContextAssembler:
    """Assembles a full LLMContextPackage from selected/filtered context per step."""

    def assemble_for_X(
        self,
        purpose: ContextPurpose,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Generic assembly dispatch based on purpose.

        Routes to the appropriate step-specific assembly method.
        """
        dispatch = {
            ContextPurpose.INTENT_RECOGNITION: self.assemble_for_intent,
            ContextPurpose.ONTOLOGY_GROUNDING: self.assemble_for_ontology,
            ContextPurpose.TASK_PLANNING: self.assemble_for_planning,
            ContextPurpose.EXECUTION_ASSIST: self.assemble_for_execution,
            ContextPurpose.RESPONSE_GENERATION: self.assemble_for_response,
        }
        method = dispatch.get(purpose, self.assemble_for_intent)
        return method(selected, user_input, step1_result, step2_result, step3_result, step4_result, task_id)

    def assemble_for_intent(
        self,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Assemble context package for Step 1 (Intent Recognition)."""
        package = LLMContextPackage(
            meta=ContextMeta(
                purpose=ContextPurpose.INTENT_RECOGNITION,
                task_id=task_id,
            ),
            instructions=ContextInstructions(
                system_instruction="You are a procurement assistant. Analyze user intent from natural language.",
            ),
            user_input=ContextUserInput(
                raw_input=user_input,
                normalized_input=self._safe_get_value(selected, "normalized_input", "value", user_input),
            ),
            included_sources=self._extract_included_sources(selected),
            excluded_sources=[],
            token_budget=TokenBudget(max_tokens=4000),
        )
        return package

    def assemble_for_ontology(
        self,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Assemble context package for Step 2 (Ontology Grounding)."""
        package = LLMContextPackage(
            meta=ContextMeta(
                purpose=ContextPurpose.ONTOLOGY_GROUNDING,
                task_id=task_id,
            ),
            instructions=ContextInstructions(
                system_instruction="You are a procurement assistant. Ground the intent to ontology objects.",
            ),
            user_input=ContextUserInput(raw_input=user_input),
            intent_context=ContextIntent(
                detected_intent=getattr(step1_result, "intent", "") if step1_result else "",
                confidence=getattr(step1_result, "confidence", 0.0) if step1_result else 0.0,
            ),
            ontology_context=ContextOntology(
                resolved_object=getattr(step1_result, "object_term", "") if step1_result else "",
            ),
            included_sources=self._extract_included_sources(selected),
            excluded_sources=[],
            token_budget=TokenBudget(max_tokens=6000),
        )
        return package

    def assemble_for_planning(
        self,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Assemble context package for Step 3 (Task Planning)."""
        package = LLMContextPackage(
            meta=ContextMeta(
                purpose=ContextPurpose.TASK_PLANNING,
                task_id=task_id,
            ),
            instructions=ContextInstructions(
                system_instruction="You are a procurement assistant. Create an execution plan for the resolved intent.",
            ),
            intent_context=ContextIntent(
                detected_intent=getattr(step1_result, "intent", "") if step1_result else "",
                confidence=getattr(step1_result, "confidence", 0.0) if step1_result else 0.0,
            ),
            ontology_context=ContextOntology(
                resolved_object=getattr(step2_result, "object_type", "") if step2_result else "",
                resolved_action=getattr(step2_result, "object_label", "") if step2_result else "",
            ),
            included_sources=self._extract_included_sources(selected),
            excluded_sources=[],
            token_budget=TokenBudget(max_tokens=6000),
        )
        return package

    def assemble_for_execution(
        self,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Assemble context package for Step 4 (Execution)."""
        package = LLMContextPackage(
            meta=ContextMeta(
                purpose=ContextPurpose.EXECUTION_ASSIST,
                task_id=task_id,
            ),
            instructions=ContextInstructions(
                system_instruction="Execute the planned actions using the appropriate connectors.",
            ),
            included_sources=self._extract_included_sources(selected),
            excluded_sources=[],
            token_budget=TokenBudget(max_tokens=8000),
        )
        return package

    def assemble_for_response(
        self,
        selected: Dict[str, Any],
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
        task_id: str = "",
    ) -> LLMContextPackage:
        """Assemble context package for Step 5 (Response Generation)."""
        package = LLMContextPackage(
            meta=ContextMeta(
                purpose=ContextPurpose.RESPONSE_GENERATION,
                task_id=task_id,
            ),
            instructions=ContextInstructions(
                system_instruction="Generate a clear, helpful response for the user based on execution results.",
            ),
            included_sources=self._extract_included_sources(selected),
            excluded_sources=[],
            token_budget=TokenBudget(max_tokens=4000),
        )
        return package

    def _extract_included_sources(self, selected: Dict[str, Any]) -> List[str]:
        """Extract list of source names from selected context."""
        sources = []
        for key, item in selected.items():
            if isinstance(item, dict) and "source" in item:
                sources.append(f"{key}:{item['source']}")
            else:
                sources.append(key)
        return sources

    def _safe_get_value(
        self, data: Dict[str, Any], key: str, sub_key: str, default: Any
    ) -> Any:
        """Safely extract nested value, handling non-dict intermediate values."""
        value = data.get(key)
        if isinstance(value, dict):
            return value.get(sub_key, default)
        return default
