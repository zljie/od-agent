"""
Output Constraint Injector
==========================
Injects output format and constraint context for each pipeline step.

Each step has different output requirements:
- Intent recognition: structured classification output
- Ontology grounding: structured entity resolution output
- Task planning: structured plan output
- Execution: raw data or structured execution report
- Response generation: adapts style based on execution result type
"""

from typing import Any, Dict, List, Optional

from src.context_engineering.models import ContextOutput


# ---------------------------------------------------------------------------
# Default constraints per purpose
# ---------------------------------------------------------------------------

_DEFAULT_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "intent_recognition": {
        "format": "json",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "intentLabel": {"type": "string"},
                "objectTerm": {"type": "string"},
                "normalizedTerm": {"type": "string"},
                "operationType": {"type": "string"},
                "confidence": {"type": "number"},
                "alternativeIntents": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intent", "objectTerm"],
        },
        "language": "zh",
        "style": "concise",
        "forbidden": ["I think", "I believe", "maybe", "perhaps", "not sure"],
    },
    "ontology_grounding": {
        "format": "json",
        "schema": {
            "type": "object",
            "properties": {
                "objectType": {"type": "string"},
                "objectLabel": {"type": "string"},
                "attributes": {"type": "array"},
                "availableActions": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["objectType"],
        },
        "language": "zh",
        "style": "concise",
        "forbidden": ["I think", "I'm not certain"],
    },
    "task_planning": {
        "format": "json",
        "schema": {
            "type": "object",
            "properties": {
                "plannedActions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sequence": {"type": "integer"},
                            "actionId": {"type": "string"},
                            "actionLabel": {"type": "string"},
                            "connector": {"type": "string"},
                        },
                        "required": ["sequence", "actionId"],
                    },
                },
                "queryConditions": {"type": "array"},
                "displayFields": {"type": "array", "items": {"type": "string"}},
                "riskLevel": {"type": "string"},
            },
            "required": ["plannedActions"],
        },
        "language": "zh",
        "style": "technical",
        "forbidden": ["maybe", "possibly", "you could try"],
    },
    "execution_assist": {
        "format": "json",
        "schema": {
            "type": "object",
            "properties": {
                "executions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "connectorName": {"type": "string"},
                            "status": {"type": "string"},
                            "resultCount": {"type": "integer"},
                            "errorMessage": {"type": "string"},
                        },
                    },
                },
            },
        },
        "language": "zh",
        "style": "concise",
        "forbidden": [],
    },
    "response_generation": {
        "format": "markdown",
        "language": "zh",
        "style": "detailed",
        "max_tokens": 2000,
        "forbidden": [
            "internal error",
            "stack trace",
            "connector",
            "Step 1",
            "Step 2",
            "pipeline",
        ],
    },
}


# ---------------------------------------------------------------------------
# Style adaptation based on execution result
# ---------------------------------------------------------------------------

_STYLE_ADAPTATIONS: Dict[str, Dict[str, Any]] = {
    "empty": {
        "style": "concise",
        "format": "text",
        "message": "用户未提供足够信息，无法生成有效回复。",
    },
    "success": {
        "style": "detailed",
        "format": "markdown",
        "message": "操作已成功完成。",
    },
    "partial": {
        "style": "detailed",
        "format": "markdown",
        "message": "操作部分完成，部分数据无法获取。",
    },
    "error": {
        "style": "concise",
        "format": "text",
        "message": "操作失败，请检查输入参数或稍后重试。",
    },
    "confirmation_needed": {
        "style": "concise",
        "format": "text",
        "message": "此操作需要您确认后继续。",
    },
}


# ---------------------------------------------------------------------------
# OutputConstraintInjector
# ---------------------------------------------------------------------------


class OutputConstraintInjector:
    """Injects output format and constraint context.

    Provides step-specific output schemas and constraints so the LLM
    produces correctly structured responses at every pipeline stage.
    """

    def inject_for_intent_recognition(self) -> ContextOutput:
        """Output constraints for intent recognition.

        Returns a JSON schema constrained output for structured classification.
        """
        return self._get_default_constraints("intent_recognition")

    def inject_for_ontology_grounding(self) -> ContextOutput:
        """Output constraints for ontology grounding.

        Returns a JSON schema for structured entity resolution.
        """
        return self._get_default_constraints("ontology_grounding")

    def inject_for_task_planning(self) -> ContextOutput:
        """Output constraints for task planning.

        Returns a technical JSON schema for the structured plan.
        """
        return self._get_default_constraints("task_planning")

    def inject_for_execution(self) -> ContextOutput:
        """Output constraints for execution assist.

        Returns a concise JSON format for execution records.
        """
        return self._get_default_constraints("execution_assist")

    def inject_for_response_generation(self, exec_result: Any) -> ContextOutput:
        """Output constraints for response generation.

        Style adapts based on execution result type:
        - Success: detailed markdown with statistics
        - Error: concise text with error message
        - Empty: concise text asking for clarification
        - Confirmation needed: concise text with question
        """
        style_key = self._classify_execution_result(exec_result)
        return self._get_default_constraints("response_generation", override_style=style_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_default_constraints(
        self, purpose: str, override_style: str = None
    ) -> ContextOutput:
        """Get default output constraints for a purpose."""
        defaults = _DEFAULT_CONSTRAINTS.get(purpose, _DEFAULT_CONSTRAINTS["response_generation"])

        style = override_style if override_style else defaults.get("style", "concise")

        return ContextOutput(
            format=defaults.get("format", "text"),
            schema=defaults.get("schema"),
            max_tokens=defaults.get("max_tokens", 0),
            language=defaults.get("language", "zh"),
            style=style,
            forbidden=defaults.get("forbidden", []),
        )

    def _adapt_style(self, exec_result: Any) -> str:
        """Adapt response style based on execution result.

        Returns a style key from _STYLE_ADAPTATIONS.
        """
        return self._classify_execution_result(exec_result)

    def _classify_execution_result(self, exec_result: Any) -> str:
        """Classify the execution result into a style category.

        Categories:
        - success: all executions succeeded
        - partial: some succeeded, some failed
        - error: all failed or critical error
        - empty: no executions or no data
        - confirmation_needed: pending user confirmation
        """
        if exec_result is None:
            return "empty"

        try:
            all_success = getattr(exec_result, "all_success", None)
            if all_success is True:
                return "success"
            if all_success is False:
                return "error"

            # Check executions list
            executions = getattr(exec_result, "executions", [])
            if not executions:
                return "empty"

            statuses = [e.status for e in executions]
            success_count = statuses.count("success")
            total = len(statuses)

            if success_count == 0:
                return "error"
            if success_count < total:
                return "partial"
            return "success"

        except (TypeError, AttributeError):
            pass

        # Check if requires confirmation
        requires_conf = getattr(exec_result, "requires_confirmation", False)
        if requires_conf:
            return "confirmation_needed"

        return "empty"
