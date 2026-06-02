"""Diagnostics Collector - collects diagnostic data during Agent execution."""

import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import (
    TestRun,
    TestRunStatus,
    IntentTrace,
    CandidateIntent,
    PlanTrace,
    PlanStep,
    StepStatus,
    ResourceUsage,
    PromptFragment,
    IntentUsage,
    SkillUsage,
    KnowledgeUsage,
    SemanticMappingUsage,
    PolicyUsage,
    ResourceStatus,
    ToolCallTrace,
    EvalResult,
    Suggestion,
)

if TYPE_CHECKING:
    pass


def generate_run_id() -> str:
    """Generate a unique test run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:6]
    return f"run_{timestamp}_{short_uuid}"


class DiagnosticsCollector:
    """Collects diagnostic data during Agent execution.

    This class intercepts and records:
    - Intent classification results
    - Task planning steps
    - Resource usage
    - Tool/skill calls
    - Final evaluation
    """

    def __init__(self, agent_id: str = "agent_1", agent_version: str = "draft"):
        self.run_id = generate_run_id()
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

        self.intent_traces: List[IntentTrace] = []
        self.plan_steps: List[PlanStep] = []
        self.resource_usage: ResourceUsage = ResourceUsage(test_run_id=self.run_id)
        self.tool_calls: List[ToolCallTrace] = []
        self.preprocessing_result: Optional[Dict[str, Any]] = None

        self.user_input: str = ""
        self.agent_output: str = ""
        self.model: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.latency_ms: int = 0

        self._step_counter: int = 0

    def record_preprocessing(self, result: Dict[str, Any]) -> None:
        """Record preprocessing results (corrections, protected terms, etc.)."""
        self.preprocessing_result = result

    def record_intent(
        self,
        selected_intent: str,
        confidence: float,
        threshold: float,
        candidates: List[Dict[str, Any]],
        routing_result: str = "",
    ) -> None:
        """Record intent classification result."""
        candidate_list = [
            CandidateIntent(
                intent=c.get("intent", ""),
                confidence=c.get("confidence", 0.0),
                trigger_reason=c.get("trigger_reason", ""),
            )
            for c in candidates
        ]

        trace = IntentTrace(
            test_run_id=self.run_id,
            selected_intent=selected_intent,
            confidence=confidence,
            threshold=threshold,
            candidates=candidate_list,
            routing_result=routing_result,
            is_selected=True,
        )
        self.intent_traces.append(trace)

        self.resource_usage.intents.append(IntentUsage(
            name=selected_intent,
            status=ResourceStatus.USED,
            confidence=confidence,
            details=f"命中阈值 {threshold}，置信度 {confidence:.2f}",
        ))

    def record_plan_step(
        self,
        step_name: str,
        status: StepStatus = StepStatus.SUCCESS,
        result: Any = None,
        latency_ms: int = 0,
        skill_name: str = "",
        tool_name: str = "",
        missing_slots: List[str] = None,
        details: Dict[str, Any] = None,
    ) -> None:
        """Record a task planning step."""
        self._step_counter += 1
        step = PlanStep(
            step=self._step_counter,
            name=step_name,
            status=status,
            result=result,
            latency_ms=latency_ms,
            skill_name=skill_name,
            tool_name=tool_name,
            missing_slots=missing_slots or [],
            details=details or {},
        )
        self.plan_steps.append(step)

    def record_resource_used(
        self,
        resource_type: str,
        name: str,
        status: ResourceStatus,
        details: str = "",
        **kwargs,
    ) -> None:
        """Record resource usage."""
        if resource_type == "prompt":
            self.resource_usage.prompt_fragments.append(PromptFragment(
                name=name,
                version=kwargs.get("version", ""),
                tokens=kwargs.get("tokens", 0),
                status=status,
                content_preview=kwargs.get("content_preview", ""),
            ))
        elif resource_type == "skill":
            self.resource_usage.skills.append(SkillUsage(
                name=name,
                status=status,
                called=kwargs.get("called", False),
                details=details,
                not_called_reason=kwargs.get("not_called_reason", ""),
            ))
        elif resource_type == "knowledge":
            self.resource_usage.knowledge.append(KnowledgeUsage(
                name=name,
                status=status,
                hit_count=kwargs.get("hit_count", 0),
                details=details,
            ))
        elif resource_type == "semantic_mapping":
            self.resource_usage.semantic_mappings.append(SemanticMappingUsage(
                mappings=kwargs.get("mappings", []),
                status=status,
                details=details,
            ))
        elif resource_type == "policy":
            self.resource_usage.policies.append(PolicyUsage(
                name=name,
                status=status,
                passed=status == ResourceStatus.USED,
                details=details,
            ))

    def record_tool_call(
        self,
        skill_name: str,
        tool_name: str,
        status: TestRunStatus,
        input_data: Dict[str, Any],
        output_data: Any = None,
        latency_ms: int = 0,
        error_message: str = "",
    ) -> None:
        """Record a tool/skill call."""
        trace = ToolCallTrace(
            test_run_id=self.run_id,
            skill_name=skill_name,
            tool_name=tool_name,
            status=status,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            error_message=error_message,
        )
        self.tool_calls.append(trace)

    def finalize(
        self,
        user_input: str,
        output: str,
        model: str = "",
        latency_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        status: TestRunStatus = TestRunStatus.SUCCESS,
    ) -> TestRun:
        """Finalize the test run and return a complete TestRun record."""
        self.end_time = datetime.now()
        self.user_input = user_input
        self.agent_output = output
        self.model = model
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        intent_trace = self.intent_traces[-1] if self.intent_traces else None
        plan_trace = PlanTrace(
            test_run_id=self.run_id,
            steps=self.plan_steps,
        )

        test_run = TestRun(
            id=self.run_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            input=user_input,
            output=output,
            status=status,
            latency_ms=latency_ms,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            created_at=self.start_time,
            intent_trace=intent_trace,
            plan_trace=plan_trace,
            resource_usage=self.resource_usage,
            tool_calls=self.tool_calls,
            preprocessing_result=self.preprocessing_result,
        )

        test_run.eval_result = self.calculate_eval_result(test_run)

        return test_run

    def calculate_score(self) -> int:
        """Calculate overall diagnostic score (0-100)."""
        score = 100

        if not self.intent_traces:
            score -= 30
        elif self.intent_traces:
            latest = self.intent_traces[-1]
            if latest.confidence < latest.threshold:
                score -= 20

        failed_steps = [s for s in self.plan_steps if s.status == StepStatus.FAILED]
        score -= len(failed_steps) * 10

        uncalled_skills = [s for s in self.resource_usage.skills if not s.called]
        if uncalled_skills and any(s.not_called_reason for s in uncalled_skills):
            score -= 15

        if self.tool_calls:
            failed_calls = [t for t in self.tool_calls if t.status == TestRunStatus.ERROR]
            score -= len(failed_calls) * 20

        return max(0, min(100, score))

    def generate_suggestions(self) -> List[Suggestion]:
        """Generate configuration suggestions based on diagnostic data."""
        suggestions = []
        suggestion_id = 1

        if self.intent_traces:
            intent = self.intent_traces[-1]
            if intent.confidence < intent.threshold * 1.2:
                suggestions.append(Suggestion(
                    id=f"sug_{suggestion_id}",
                    category="intent",
                    title="意图置信度偏低",
                    description=f"当前命中意图 '{intent.selected_intent}' 置信度为 {intent.confidence:.2f}，"
                                f"接近阈值 {intent.threshold}。建议添加更多关键词或同义词以提高区分度。",
                    priority=1,
                    actionable=True,
                    config_key="intent_keywords",
                ))
                suggestion_id += 1

            if len(intent.candidates) > 1:
                top_two = intent.candidates[:2]
                if len(top_two) >= 2 and abs(top_two[0].confidence - top_two[1].confidence) < 0.1:
                    suggestions.append(Suggestion(
                        id=f"sug_{suggestion_id}",
                        category="intent",
                        title="意图区分度不足",
                        description=f"意图 '{top_two[0].intent}' 与 '{top_two[1].intent}' "
                                    f"置信度接近 ({top_two[0].confidence:.2f} vs {top_two[1].confidence:.2f})，"
                                    f"存在语义重叠。建议优化关键词或调整优先级。",
                        priority=2,
                        actionable=True,
                    ))
                    suggestion_id += 1

        missing_slot_steps = [s for s in self.plan_steps if s.missing_slots]
        if missing_slot_steps:
            for step in missing_slot_steps:
                suggestions.append(Suggestion(
                    id=f"sug_{suggestion_id}",
                    category="slot_handling",
                    title=f"缺少必要字段：{', '.join(step.missing_slots)}",
                    description=f"任务 '{step.name}' 缺少必要信息 {step.missing_slots}。"
                                f"建议配置追问模板或提供默认值。",
                    priority=2,
                    actionable=True,
                ))
                suggestion_id += 1

        uncalled_skills = [s for s in self.resource_usage.skills if not s.called and s.not_called_reason]
        if uncalled_skills:
            for skill in uncalled_skills:
                suggestions.append(Suggestion(
                    id=f"sug_{suggestion_id}",
                    category="skill",
                    title=f"技能 '{skill.name}' 未被调用",
                    description=f"该技能已匹配但未执行。原因：{skill.not_called_reason}。"
                                f"请检查必要参数配置。",
                    priority=3,
                    actionable=True,
                ))
                suggestion_id += 1

        unused_knowledge = [k for k in self.resource_usage.knowledge if k.status == ResourceStatus.NOT_USED]
        if not self.resource_usage.knowledge and self.user_input:
            suggestions.append(Suggestion(
                id=f"sug_{suggestion_id}",
                category="knowledge",
                title="建议开启知识检索",
                description="当前对话未命中知识库。如需根据业务数据回答，请配置语义映射和知识路由。",
                priority=4,
                actionable=True,
            ))

        return suggestions

    def calculate_eval_result(self, test_run: TestRun) -> EvalResult:
        """Calculate evaluation result based on collected traces."""
        overall_score = self.calculate_score()
        suggestions = self.generate_suggestions()

        if overall_score >= 85:
            conclusion = "优秀"
        elif overall_score >= 70:
            conclusion = "良好"
        elif overall_score >= 50:
            conclusion = "一般"
        else:
            conclusion = "需优化"

        intent_score = 0.0
        if test_run.intent_trace:
            intent_score = min(1.0, test_run.intent_trace.confidence)

        plan_score = 1.0
        if test_run.plan_trace and test_run.plan_trace.steps:
            successful_steps = sum(1 for s in test_run.plan_trace.steps if s.status == StepStatus.SUCCESS)
            plan_score = successful_steps / len(test_run.plan_trace.steps)

        skill_score = 1.0
        if test_run.resource_usage.skills:
            called_skills = sum(1 for s in test_run.resource_usage.skills if s.called)
            skill_score = called_skills / len(test_run.resource_usage.skills)

        resource_score = 1.0
        total_resources = (
            len(test_run.resource_usage.prompt_fragments) +
            len(test_run.resource_usage.intents) +
            len(test_run.resource_usage.skills)
        )
        used_resources = sum(
            1 for p in test_run.resource_usage.prompt_fragments if p.status == ResourceStatus.USED
        ) + sum(
            1 for i in test_run.resource_usage.intents if i.status == ResourceStatus.USED
        ) + sum(
            1 for s in test_run.resource_usage.skills if s.status == ResourceStatus.USED
        )
        if total_resources > 0:
            resource_score = used_resources / total_resources

        failed_policies = [p for p in test_run.resource_usage.policies if not p.passed]
        risk_passed = len(failed_policies) == 0

        summary_parts = []
        if test_run.intent_trace:
            summary_parts.append(f"意图命中: {test_run.intent_trace.selected_intent} ({test_run.intent_trace.confidence:.0%})")
        summary_parts.append(f"任务链路: {conclusion}")
        if test_run.resource_usage.skills:
            called = sum(1 for s in test_run.resource_usage.skills if s.called)
            summary_parts.append(f"Skill 调用: {called}/{len(test_run.resource_usage.skills)}")
        summary_parts.append(f"风险检查: {'通过' if risk_passed else '未通过'}")

        return EvalResult(
            test_run_id=self.run_id,
            overall_score=overall_score,
            overall_conclusion=conclusion,
            intent_score=intent_score,
            plan_score=plan_score,
            skill_score=skill_score,
            resource_score=resource_score,
            response_quality_score=0.5,
            risk_check_passed=risk_passed,
            suggestions=suggestions,
            summary=" | ".join(summary_parts),
            details={
                "total_steps": len(self.plan_steps),
                "failed_steps": sum(1 for s in self.plan_steps if s.status == StepStatus.FAILED),
                "tool_calls": len(self.tool_calls),
                "missing_slots": [s for s in self.plan_steps if s.missing_slots],
            },
        )


class DiagnosticContext:
    """Context manager for wrapping agent operations with diagnostics."""

    def __init__(self, collector: DiagnosticsCollector):
        self.collector = collector
        self.start_ms: int = 0

    def __enter__(self) -> DiagnosticsCollector:
        self.start_ms = int(time.time() * 1000)
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
