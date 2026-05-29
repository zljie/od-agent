"""Task planning: TaskPlan and TaskNode DAG representation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..intent.intent_classification import IntentClassification


class Decision(Enum):
    """Gate decision from the Planner."""

    REJECT = "reject"            # Refuse to execute
    CLARIFY = "clarify"          # Need user clarification (ambiguous)
    HITL_CONFIRM = "hitl_confirm" # Require human confirmation
    EXECUTE = "execute"           # Execute TaskPlan DAG
    DELEGATE_LLM = "delegate_llm" # No constraints, hand to LLM
    SLOT_MISSING = "slot_missing" # Required entity slots are absent


@dataclass
class TaskNode:
    """A single node in the TaskPlan DAG.

    Represents one skill invocation with resolved parameters.
    """

    node_id: str
    skill_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    rationale: str = ""
    hitl: bool = False
    risk_tag: str = "LOW"
    user_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "skill_id": self.skill_id,
            "params": self.params,
            "depends_on": self.depends_on,
            "rationale": self.rationale,
            "hitl": self.hitl,
            "risk_tag": self.risk_tag,
            "user_message": self.user_message,
        }


@dataclass
class TaskPlan:
    """Planning result: how to execute an intent.

    The DAG of TaskNodes is topologically ordered before execution.
    """

    intent: IntentClassification
    bound_intent_type: str
    decision: Decision
    tasks: List[TaskNode] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    rejected_reason: str = ""
    hitl_prompt: str = ""
    warnings: List[str] = field(default_factory=list)

    def topological_order(self) -> List[TaskNode]:
        """Topologically sort tasks; raises ValueError on cycle."""
        in_degree: Dict[str, int] = {t.node_id: 0 for t in self.tasks}
        deps_map: Dict[str, List[str]] = {t.node_id: list(t.depends_on) for t in self.tasks}

        for task in self.tasks:
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.node_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered: List[TaskNode] = []
        node_map: Dict[str, TaskNode] = {t.node_id: t for t in self.tasks}

        while queue:
            tid = queue.pop(0)
            ordered.append(node_map[tid])
            for other_id, deps in deps_map.items():
                if tid in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(ordered) != len(self.tasks):
            raise ValueError("Cycle detected in TaskPlan DAG")

        return ordered

    def is_executable(self) -> bool:
        return self.decision == Decision.EXECUTE

    @classmethod
    def reject(
        cls, classification: IntentClassification, reason: str
    ) -> "TaskPlan":
        return cls(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.REJECT,
            rejected_reason=reason,
        )

    @classmethod
    def clarify(cls, classification: IntentClassification, prompt: str) -> "TaskPlan":
        return cls(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.CLARIFY,
            hitl_prompt=prompt,
        )

    @classmethod
    def hitl_confirm(
        cls, classification: IntentClassification, prompt: str, warnings: Optional[List[str]] = None
    ) -> "TaskPlan":
        return cls(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.HITL_CONFIRM,
            hitl_prompt=prompt,
            warnings=warnings or [],
        )

    @classmethod
    def slot_missing(
        cls, classification: IntentClassification, missing: List[str]
    ) -> "TaskPlan":
        return cls(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.SLOT_MISSING,
            warnings=[f"缺少必需槽位: {', '.join(missing)}"],
        )

    @classmethod
    def delegate_llm(cls, classification: IntentClassification) -> "TaskPlan":
        return cls(
            intent=classification,
            bound_intent_type=classification.intent_type,
            decision=Decision.DELEGATE_LLM,
        )
