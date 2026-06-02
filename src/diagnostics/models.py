"""Data models for Agent diagnostics and testing."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class TestRunStatus(str, Enum):
    """Status of a test run."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    BLOCKED = "blocked"


class ResourceStatus(str, Enum):
    """Status of a resource usage."""
    USED = "used"
    NOT_USED = "not_used"
    FAILED = "failed"
    NOT_CONFIGURED = "not_configured"


class StepStatus(str, Enum):
    """Status of a plan step."""
    SUCCESS = "success"
    WAITING = "waiting"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CandidateIntent:
    """Represents a candidate intent during intent classification."""
    intent: str
    confidence: float
    trigger_reason: str = ""


@dataclass
class IntentTrace:
    """Trace of intent classification for a test run."""
    test_run_id: str
    selected_intent: str
    confidence: float
    threshold: float
    candidates: List[CandidateIntent] = field(default_factory=list)
    routing_result: str = ""
    is_selected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_run_id": self.test_run_id,
            "selected_intent": self.selected_intent,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "candidates": [
                {
                    "intent": c.intent,
                    "confidence": c.confidence,
                    "trigger_reason": c.trigger_reason,
                }
                for c in self.candidates
            ],
            "routing_result": self.routing_result,
            "is_selected": self.is_selected,
        }


@dataclass
class PlanStep:
    """A single step in the task execution plan."""
    step: int
    name: str
    status: StepStatus = StepStatus.SUCCESS
    result: Any = None
    latency_ms: int = 0
    missing_slots: List[str] = field(default_factory=list)
    skill_name: str = ""
    tool_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "status": self.status.value if isinstance(self.status, StepStatus) else self.status,
            "result": self.result,
            "latency_ms": self.latency_ms,
            "missing_slots": self.missing_slots,
            "skill_name": self.skill_name,
            "tool_name": self.tool_name,
            "details": self.details,
        }


@dataclass
class PlanTrace:
    """Trace of the task planning and execution for a test run."""
    test_run_id: str
    steps: List[PlanStep] = field(default_factory=list)
    decision: str = ""
    rejected_reason: str = ""
    hitl_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_run_id": self.test_run_id,
            "steps": [s.to_dict() for s in self.steps],
            "decision": self.decision,
            "rejected_reason": self.rejected_reason,
            "hitl_prompt": self.hitl_prompt,
        }


@dataclass
class PromptFragment:
    """A fragment of the prompt used in a test run."""
    name: str
    version: str = ""
    tokens: int = 0
    status: ResourceStatus = ResourceStatus.USED
    content_preview: str = ""


@dataclass
class IntentUsage:
    """Usage record for an intent."""
    name: str
    status: ResourceStatus
    confidence: float = 0.0
    details: str = ""


@dataclass
class SkillUsage:
    """Usage record for a skill."""
    name: str
    status: ResourceStatus
    called: bool = False
    details: str = ""
    not_called_reason: str = ""


@dataclass
class KnowledgeUsage:
    """Usage record for knowledge base."""
    name: str = ""
    status: ResourceStatus = ResourceStatus.NOT_USED
    hit_count: int = 0
    details: str = ""


@dataclass
class SemanticMappingUsage:
    """Usage record for semantic mapping."""
    mappings: List[Dict[str, str]] = field(default_factory=list)
    status: ResourceStatus = ResourceStatus.NOT_USED
    details: str = ""


@dataclass
class PolicyUsage:
    """Usage record for policy checks."""
    name: str
    status: ResourceStatus
    passed: bool = True
    details: str = ""


@dataclass
class ResourceUsage:
    """Comprehensive resource usage record for a test run."""
    test_run_id: str
    prompt_fragments: List[PromptFragment] = field(default_factory=list)
    intents: List[IntentUsage] = field(default_factory=list)
    skills: List[SkillUsage] = field(default_factory=list)
    knowledge: List[KnowledgeUsage] = field(default_factory=list)
    semantic_mappings: List[SemanticMappingUsage] = field(default_factory=list)
    policies: List[PolicyUsage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_run_id": self.test_run_id,
            "prompt_fragments": [
                {
                    "name": p.name,
                    "version": p.version,
                    "tokens": p.tokens,
                    "status": p.status.value if isinstance(p.status, ResourceStatus) else p.status,
                    "content_preview": p.content_preview[:100] if p.content_preview else "",
                }
                for p in self.prompt_fragments
            ],
            "intents": [
                {
                    "name": i.name,
                    "status": i.status.value if isinstance(i.status, ResourceStatus) else i.status,
                    "confidence": i.confidence,
                    "details": i.details,
                }
                for i in self.intents
            ],
            "skills": [
                {
                    "name": s.name,
                    "status": s.status.value if isinstance(s.status, ResourceStatus) else s.status,
                    "called": s.called,
                    "details": s.details,
                    "not_called_reason": s.not_called_reason,
                }
                for s in self.skills
            ],
            "knowledge": [
                {
                    "name": k.name,
                    "status": k.status.value if isinstance(k.status, ResourceStatus) else k.status,
                    "hit_count": k.hit_count,
                    "details": k.details,
                }
                for k in self.knowledge
            ],
            "semantic_mappings": [
                {
                    "mappings": m.mappings,
                    "status": m.status.value if isinstance(m.status, ResourceStatus) else m.status,
                    "details": m.details,
                }
                for m in self.semantic_mappings
            ],
            "policies": [
                {
                    "name": p.name,
                    "status": p.status.value if isinstance(p.status, ResourceStatus) else p.status,
                    "passed": p.passed,
                    "details": p.details,
                }
                for p in self.policies
            ],
        }


@dataclass
class ToolCallTrace:
    """Trace of a tool/skill call during a test run."""
    test_run_id: str
    skill_name: str
    tool_name: str
    status: TestRunStatus = TestRunStatus.SUCCESS
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    latency_ms: int = 0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_run_id": self.test_run_id,
            "skill_name": self.skill_name,
            "tool_name": self.tool_name,
            "status": self.status.value if isinstance(self.status, TestRunStatus) else self.status,
            "input_data": self._sanitize_data(self.input_data),
            "output_data": self._sanitize_data(self.output_data) if isinstance(self.output_data, dict) else str(self.output_data)[:500],
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize sensitive data from the trace."""
        if isinstance(data, dict):
            sensitive_keys = {"password", "secret", "api_key", "apikey", "token", "phone", "id_card", "身份证"}
            result = {}
            for k, v in data.items():
                key_lower = k.lower()
                if any(s in key_lower for s in sensitive_keys):
                    result[k] = "***REDACTED***"
                else:
                    result[k] = self._sanitize_data(v)
            return result
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            import re
            phone_pattern = r'1[3-9]\d{9}'
            data = re.sub(phone_pattern, '***REDACTED_PHONE***', data)
            id_pattern = r'\d{17}[\dXx]'
            data = re.sub(id_pattern, '***REDACTED_ID***', data)
            return data
        return data


@dataclass
class Suggestion:
    """Configuration suggestion generated from diagnostics."""
    id: str
    category: str
    title: str
    description: str
    priority: int = 1
    actionable: bool = True
    config_key: str = ""
    suggested_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "actionable": self.actionable,
            "config_key": self.config_key,
            "suggested_value": self.suggested_value,
        }


@dataclass
class EvalResult:
    """Evaluation result for a test run."""
    test_run_id: str
    overall_score: int
    overall_conclusion: str
    intent_score: float = 0.0
    plan_score: float = 0.0
    skill_score: float = 0.0
    resource_score: float = 0.0
    response_quality_score: float = 0.0
    risk_check_passed: bool = True
    suggestions: List[Suggestion] = field(default_factory=list)
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_run_id": self.test_run_id,
            "overall_score": self.overall_score,
            "overall_conclusion": self.overall_conclusion,
            "intent_score": self.intent_score,
            "plan_score": self.plan_score,
            "skill_score": self.skill_score,
            "resource_score": self.resource_score,
            "response_quality_score": self.response_quality_score,
            "risk_check_passed": self.risk_check_passed,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "summary": self.summary,
            "details": self.details,
        }


@dataclass
class TestRun:
    """Complete record of a test run with all diagnostic data."""
    id: str
    agent_id: str
    input: str
    agent_version: str = "draft"
    scenario_id: str = ""
    output: str = ""
    status: TestRunStatus = TestRunStatus.SUCCESS
    latency_ms: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    intent_trace: Optional[IntentTrace] = None
    plan_trace: Optional[PlanTrace] = None
    resource_usage: Optional[ResourceUsage] = None
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    eval_result: Optional[EvalResult] = None
    preprocessing_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "scenario_id": self.scenario_id,
            "input": self.input,
            "output": self.output,
            "status": self.status.value if isinstance(self.status, TestRunStatus) else self.status,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "intent_trace": self.intent_trace.to_dict() if self.intent_trace else None,
            "plan_trace": self.plan_trace.to_dict() if self.plan_trace else None,
            "resource_usage": self.resource_usage.to_dict() if self.resource_usage else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "eval_result": self.eval_result.to_dict() if self.eval_result else None,
            "preprocessing_result": self.preprocessing_result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRun":
        """Create a TestRun from a dictionary."""
        test_run = cls(
            id=data["id"],
            agent_id=data["agent_id"],
            agent_version=data.get("agent_version", "draft"),
            scenario_id=data.get("scenario_id", ""),
            input=data["input"],
            output=data.get("output", ""),
            status=TestRunStatus(data.get("status", "success")),
            latency_ms=data.get("latency_ms", 0),
            model=data.get("model", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )

        if "intent_trace" in data and data["intent_trace"]:
            test_run.intent_trace = IntentTrace(**data["intent_trace"])
        if "plan_trace" in data and data["plan_trace"]:
            test_run.plan_trace = PlanTrace(**data["plan_trace"])
        if "resource_usage" in data and data["resource_usage"]:
            test_run.resource_usage = ResourceUsage(**data["resource_usage"])
        if "tool_calls" in data:
            test_run.tool_calls = [ToolCallTrace(**tc) for tc in data["tool_calls"]]
        if "eval_result" in data and data["eval_result"]:
            test_run.eval_result = EvalResult(**data["eval_result"])

        return test_run


@dataclass
class TestCase:
    """Test case definition for batch testing."""
    id: str
    agent_id: str
    name: str
    input: str
    scenario_id: str = ""
    description: str = ""
    user_profile: str = ""
    expected_intent: str = ""
    expected_skill: str = ""
    expected_response_keywords: List[str] = field(default_factory=list)
    status: str = "active"
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "input": self.input,
            "user_profile": self.user_profile,
            "expected_intent": self.expected_intent,
            "expected_skill": self.expected_skill,
            "expected_response_keywords": self.expected_response_keywords,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create a TestCase from a dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            scenario_id=data.get("scenario_id", ""),
            name=data["name"],
            description=data.get("description", ""),
            input=data["input"],
            user_profile=data.get("user_profile", ""),
            expected_intent=data.get("expected_intent", ""),
            expected_skill=data.get("expected_skill", ""),
            expected_response_keywords=data.get("expected_response_keywords", []),
            status=data.get("status", "active"),
            created_by=data.get("created_by", "system"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


@dataclass
class TestCaseResult:
    """Result of running a test case."""
    test_case_id: str
    test_run_id: str
    passed: bool
    intent_match: bool = False
    skill_match: bool = False
    response_match: bool = False
    executed_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "test_run_id": self.test_run_id,
            "passed": self.passed,
            "intent_match": self.intent_match,
            "skill_match": self.skill_match,
            "response_match": self.response_match,
            "executed_at": self.executed_at.isoformat() if isinstance(self.executed_at, datetime) else str(self.executed_at),
            "details": self.details,
        }
