"""Intent routing configuration.

Defines the binding between an intent type and its execution strategy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Strategy(Enum):
    """Five execution strategies for an intent binding.

    REJECT:         Directly reject the request (security / out-of-scope)
    HITL_CONFIRM:   Pause and wait for human confirmation
    FIXED_SKILL:    Execute a single predetermined skill DAG
    SKILL_WHITELIST:LLM freely orchestrates within a skill whitelist
    LLM_FREE:       No constraints, delegate everything to LLM
    """

    REJECT = "reject"
    HITL_CONFIRM = "hitl_confirm"
    FIXED_SKILL = "fixed_skill"
    SKILL_WHITELIST = "skill_whitelist"
    LLM_FREE = "llm_free"


@dataclass
class IntentBinding:
    """Binding from intent type to execution strategy.

    Attributes:
        intent_type:        The intent type this binding applies to
        strategy:           Which strategy to use
        skill_id:           FIXED_SKILL: the single skill to invoke
        skill_whitelist:     SKILL_WHITELIST: allowed skill IDs
        required_slots:      Entity slots that must be extracted; if missing → HITL_CONFIRM
        confidence_floor:    If classification confidence < this → HITL_CONFIRM
        data_sensitivity:    PUBLIC / INTERNAL / PII (for audit)
        reject_message:      Text returned to user when REJECT
        hitl_prompt:         Confirmation question shown to user for HITL_CONFIRM
        depends_on_skills:   Skill IDs that must execute before this one (DAG dependency)
        composite_meta_intent: Optional label grouping related bindings into one composite intent
    """

    intent_type: str
    strategy: Strategy
    skill_id: Optional[str] = None
    skill_whitelist: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    confidence_floor: float = 0.35
    data_sensitivity: str = "PUBLIC"
    reject_message: str = "抱歉，我无法处理这个请求。"
    hitl_prompt: str = ""
    depends_on_skills: List[str] = field(default_factory=list)
    composite_meta_intent: Optional[str] = None

    @classmethod
    def fixed_skill(
        cls,
        intent_type: str,
        skill_id: str,
        *,
        required_slots: Optional[List[str]] = None,
        confidence_floor: float = 0.35,
        depends_on_skills: Optional[List[str]] = None,
        composite_meta_intent: Optional[str] = None,
    ) -> "IntentBinding":
        return cls(
            intent_type=intent_type,
            strategy=Strategy.FIXED_SKILL,
            skill_id=skill_id,
            required_slots=required_slots or [],
            confidence_floor=confidence_floor,
            depends_on_skills=depends_on_skills or [],
            composite_meta_intent=composite_meta_intent,
        )

    @classmethod
    def composite(
        cls,
        intent_type: str,
        skill_id: str,
        meta_intent: str,
        *,
        required_slots: Optional[List[str]] = None,
        confidence_floor: float = 0.35,
        depends_on_skills: Optional[List[str]] = None,
    ) -> "IntentBinding":
        """Create a composite intent binding that depends on other skills.

        Used for compound scenarios like "travel journey" where Math Teacher
        needs the result from Time Converter before computing the average.
        """
        return cls(
            intent_type=intent_type,
            strategy=Strategy.FIXED_SKILL,
            skill_id=skill_id,
            required_slots=required_slots or [],
            confidence_floor=confidence_floor,
            depends_on_skills=depends_on_skills or [],
            composite_meta_intent=meta_intent,
        )

    @classmethod
    def llm_free(cls, intent_type: str) -> "IntentBinding":
        return cls(intent_type=intent_type, strategy=Strategy.LLM_FREE)

    @classmethod
    def reject(cls, intent_type: str, message: str) -> "IntentBinding":
        return cls(intent_type=intent_type, strategy=Strategy.REJECT, reject_message=message)

    @classmethod
    def hitl_confirm(cls, intent_type: str, prompt: str) -> "IntentBinding":
        return cls(intent_type=intent_type, strategy=Strategy.HITL_CONFIRM, hitl_prompt=prompt)

    def missing_slots(self, entities: dict) -> List[str]:
        """Return list of required slots that are absent in entities."""
        return [slot for slot in self.required_slots if slot not in entities or not entities[slot]]


@dataclass
class IntentBindingTable:
    """Registry of all intent → binding mappings.

    Default fallback is REJECT (whitelist security model).
    """

    bindings: List[IntentBinding] = field(default_factory=list)
    _by_type: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        for b in self.bindings:
            self._by_type[b.intent_type] = b

    def resolve(self, intent_type: str) -> IntentBinding:
        """Look up binding for intent_type. Falls back to REJECT if unknown."""
        if intent_type == "UNKNOWN":
            return self._default_reject()
        return self._by_type.get(intent_type, self._default_reject())

    def _default_reject(self) -> IntentBinding:
        return IntentBinding(
            intent_type="UNKNOWN",
            strategy=Strategy.REJECT,
            reject_message="抱歉，我无法理解您的请求。",
        )

    def add(self, binding: IntentBinding) -> None:
        self.bindings.append(binding)
        self._by_type[binding.intent_type] = binding
