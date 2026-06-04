"""
Clarification Policy
==================
Manages clarification strategies for intent recognition.

Based on Section 10.3 of BeBIOS_Intent_Processing_Architecture_v1.0.md

Example:
    policy = ClarificationEngine()
    result = policy.generate(
        trigger_type="missing_object",
        context={"raw_input": "帮我查一下单子"}
    )
    print(result.question)  # "你说的单子是指..."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Clarification Policy Model
# ---------------------------------------------------------------------------

@dataclass
class ClarificationPolicy:
    """Clarification policy definition.

    Defines how to generate clarification questions for different trigger types.
    """
    trigger_type: str                        # Trigger type (e.g., "missing_object")
    threshold: float = 0.5                  # Confidence threshold
    question_template: str = ""              # Question template
    options_source: Optional[str] = None     # Options source (e.g., "ontology_objects")
    max_options: int = 4                    # Max number of options
    max_turns: int = 3                      # Max clarification turns
    fallback: str = "generic_clarify"       # Fallback strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggerType": self.trigger_type,
            "threshold": self.threshold,
            "questionTemplate": self.question_template,
            "optionsSource": self.options_source,
            "maxOptions": self.max_options,
            "maxClarifyTurns": self.max_turns,
            "fallback": self.fallback,
        }


@dataclass
class ClarificationResult:
    """Result of clarification generation."""
    question: str
    trigger_type: str
    options: List[Dict[str, str]] = field(default_factory=list)  # [{"text": ..., "value": ...}]
    context: Dict[str, Any] = field(default_factory=dict)
    is_generic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": self.options,
            "triggerType": self.trigger_type,
            "context": self.context,
            "isGeneric": self.is_generic,
        }


# ---------------------------------------------------------------------------
# Default Clarification Policies
# ---------------------------------------------------------------------------

DEFAULT_CLARIFICATION_POLICIES: List[ClarificationPolicy] = [
    # Missing object clarification
    ClarificationPolicy(
        trigger_type="missing_object",
        threshold=0.5,
        question_template="{object}是指什么？",
        options_source="ontology_objects",
        max_options=4,
        fallback="ask_object",
    ),
    # Ambiguous object clarification
    ClarificationPolicy(
        trigger_type="ambiguous_object",
        threshold=0.5,
        question_template="你说的{object}是指哪个？",
        options_source="related_objects",
        max_options=4,
        fallback="list_objects",
    ),
    # Missing action clarification
    ClarificationPolicy(
        trigger_type="missing_action",
        threshold=0.5,
        question_template="你希望我怎么处理这个{object}？",
        options_source="available_actions",
        max_options=4,
        fallback="ask_action",
    ),
    # Ambiguous action clarification
    ClarificationPolicy(
        trigger_type="ambiguous_action",
        threshold=0.5,
        question_template="你的意思是{action}还是其他操作？",
        options_source="alternative_actions",
        max_options=3,
        fallback="ask_action",
    ),
    # Missing slots clarification
    ClarificationPolicy(
        trigger_type="missing_slots",
        threshold=0.5,
        question_template="创建{object}还需要以下信息：",
        options_source="required_slots",
        max_options=6,
        fallback="sequential_fill",
    ),
    # Low confidence clarification
    ClarificationPolicy(
        trigger_type="low_confidence",
        threshold=0.5,
        question_template="我不太确定你的意思，能再说明一下吗？",
        options_source=None,
        max_options=0,
        fallback="rephrase",
    ),
    # Multiple candidates clarification
    ClarificationPolicy(
        trigger_type="multiple_candidates",
        threshold=0.5,
        question_template="你是指哪一个？",
        options_source="candidates",
        max_options=5,
        fallback="ask_which",
    ),
]


# ---------------------------------------------------------------------------
# Object Mapping for Clarification
# ---------------------------------------------------------------------------

OBJECT_CANDIDATE_MAP: Dict[str, List[str]] = {
    "单子": ["采购需求", "采购订单", "询价单", "报价单"],
    "单": ["采购需求", "采购订单", "询价单", "报价单"],
    "订单": ["采购订单", "询价单", "报价单"],
    "单据": ["采购需求", "采购订单", "询价单", "报价单", "合同"],
}


# ---------------------------------------------------------------------------
# Clarification Engine
# ---------------------------------------------------------------------------

class ClarificationEngine:
    """Generates clarification questions based on policies.

    Example:
        engine = ClarificationEngine()
        result = engine.generate(
            trigger_type="missing_object",
            context={"raw_input": "帮我查一下单子"}
        )
        print(result.question)  # "你说的单子是指采购需求、采购订单，还是询价单？"
    """

    def __init__(self, policies: Optional[List[ClarificationPolicy]] = None):
        """Initialize the clarification engine.

        Args:
            policies: Custom clarification policies. Uses default if not provided.
        """
        self.policies = {p.trigger_type: p for p in (policies or DEFAULT_CLARIFICATION_POLICIES)}

    def generate(
        self,
        trigger_type: str,
        context: Dict[str, Any],
        custom_template: Optional[str] = None,
        custom_options: Optional[List[Dict[str, str]]] = None,
    ) -> ClarificationResult:
        """Generate a clarification question.

        Args:
            trigger_type: Type of clarification trigger
            context: Context including raw_input, candidates, etc.
            custom_template: Override question template
            custom_options: Override options

        Returns:
            ClarificationResult with question and options
        """
        policy = self.policies.get(trigger_type)

        if not policy and not custom_template:
            # Fallback to generic clarification
            return self._generic_clarify(context)

        # Get template
        template = custom_template or (policy.question_template if policy else "{info}")

        # Get options
        if custom_options:
            options = custom_options
        elif policy and policy.options_source:
            options = self._get_options(policy.options_source, context)
        else:
            options = []

        # Build question
        question = self._build_question(template, context)

        return ClarificationResult(
            question=question,
            options=options[:policy.max_options] if policy else options,
            trigger_type=trigger_type,
            context=context,
        )

    def _get_options(
        self,
        source: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Get options from a source."""
        if source == "ontology_objects":
            return self._get_object_options(context)
        elif source == "related_objects":
            return self._get_related_object_options(context)
        elif source == "available_actions":
            return self._get_action_options(context)
        elif source == "candidates":
            return context.get("candidates", [])
        elif source == "required_slots":
            return self._get_slot_options(context)
        return []

    def _get_object_options(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get object clarification options."""
        raw_input = context.get("raw_input", "")

        # Check for ambiguous object words
        for key, candidates in OBJECT_CANDIDATE_MAP.items():
            if key in raw_input:
                return [{"text": c, "value": c} for c in candidates]

        # Default procurement objects
        default_objects = [
            {"text": "采购需求", "value": "PurchaseRequirement"},
            {"text": "采购订单", "value": "PurchaseOrder"},
            {"text": "询价单", "value": "Inquiry"},
            {"text": "报价单", "value": "Quotation"},
            {"text": "合同", "value": "Contract"},
        ]
        return default_objects

    def _get_related_object_options(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get related object options based on current object."""
        current_object = context.get("object", "")
        # In production, this would look up related objects from ontology
        return self._get_object_options(context)

    def _get_action_options(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get action clarification options."""
        return [
            {"text": "查询", "value": "Query"},
            {"text": "创建", "value": "Create"},
            {"text": "更新", "value": "Update"},
            {"text": "分析", "value": "Analyze"},
        ]

    def _get_slot_options(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get slot clarification options."""
        missing_slots = context.get("missing_slots", [])
        return [{"text": slot, "value": slot} for slot in missing_slots]

    def _build_question(self, template: str, context: Dict[str, Any]) -> str:
        """Build question from template using context."""
        question = template

        # Replace placeholders
        replacements = {
            "{object}": context.get("object", ""),
            "{action}": context.get("action", ""),
            "{info}": context.get("info", ""),
        }

        for placeholder, value in replacements.items():
            question = question.replace(placeholder, value)

        return question

    def _generic_clarify(self, context: Dict[str, Any]) -> ClarificationResult:
        """Generate a generic clarification question."""
        raw_input = context.get("raw_input", "")

        # Try to extract what we're unclear about
        unclear = context.get("unclear", "")

        if unclear:
            question = f"你说的{unclear}是指什么？能详细说明一下吗？"
        else:
            question = f"关于「{raw_input[:50]}」，能再详细说明一下吗？"

        return ClarificationResult(
            question=question,
            options=[],
            trigger_type="generic",
            context=context,
            is_generic=True,
        )

    def generate_slot_clarification(
        self,
        object: str,
        missing_slots: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> ClarificationResult:
        """Generate clarification for missing slots.

        This is a convenience method for slot-related clarification.
        """
        ctx = context or {}
        ctx["object"] = object
        ctx["missing_slots"] = missing_slots

        return self.generate(
            trigger_type="missing_slots",
            context=ctx,
        )

    def generate_object_clarification(
        self,
        ambiguous_word: str,
        candidates: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> ClarificationResult:
        """Generate clarification for ambiguous object.

        This is a convenience method for object ambiguity clarification.
        """
        ctx = context or {}
        ctx["object"] = ambiguous_word
        ctx["candidates"] = [{"text": c, "value": c} for c in candidates]

        return self.generate(
            trigger_type="ambiguous_object",
            context=ctx,
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default_engine: Optional[ClarificationEngine] = None


def get_default_clarification() -> ClarificationEngine:
    """Get the default clarification engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = ClarificationEngine()
    return _default_engine
