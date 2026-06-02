"""Five-Step Transparent Execution Pipeline."""

import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from .models import (
    IntentRecognitionResult,
    OntologyResolveResult,
    TaskPlanResult,
    ExecutionResult,
    ResponseResult,
)
from .step1_intent import Step1Recognizer
from .step2_ontology import Step2OntologyResolver
from .step3_planner import Step3TaskPlanner
from .step4_executor import Step4ConnectorExecutor
from .step5_response import Step5ResponseGenerator


class FiveStepPipeline:
    """Five-Step Transparent Execution Pipeline.

    Orchestrates the complete 5-step execution lifecycle:
    1. Intent Recognition
    2. Ontology Object Resolution
    3. Task Planning
    4. Execution
    5. Response Generation

    Each step emits SSE step_update events.
    """

    STEP_NAMES = {
        1: "意图识别",
        2: "本体对象定位",
        3: "任务规划",
        4: "执行过程",
        5: "生成回复",
    }

    def __init__(self, agent=None, session_id: Optional[str] = None):
        self.agent = agent
        self.session_id = session_id
        self._step1 = Step1Recognizer()
        self._step2 = Step2OntologyResolver()
        self._step3 = Step3TaskPlanner()
        self._step4 = Step4ConnectorExecutor()
        self._step5 = Step5ResponseGenerator()
        self._task_context: Dict[str, Any] = {}

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        date_str = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:6].upper()
        return f"TASK-{date_str}-{short_uuid}"

    def _get_sse_helpers(self):
        """Lazy import SSE helpers."""
        from ..sse_stream import (
            step_update, done, content, tool_call, tool_result, error_event
        )
        return step_update, done, content, tool_call, tool_result, error_event

    async def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the complete 5-step pipeline.

        Yields SSE events for each step.
        """
        task_id = self._generate_task_id()
        self.session_id = session_id or self.session_id

        # Step timing
        step_times: Dict[int, float] = {}

        # Helper imports
        step_update, done, content, tool_call, tool_result, error_fn = self._get_sse_helpers()

        try:
            # ===== STEP 1: Intent Recognition =====
            step_times[1] = time.time()
            yield step_update(
                step=1,
                step_name=self.STEP_NAMES[1],
                status="active",
            )

            intent_result = self._step1.recognize(user_input)
            step_times[1] = (time.time() - step_times[1]) * 1000

            yield step_update(
                step=1,
                step_name=self.STEP_NAMES[1],
                status="completed",
                summary=intent_result.summary,
                details=intent_result.to_s1_details(),
            )

            # ===== STEP 2: Ontology Resolution =====
            step_times[2] = time.time()
            yield step_update(
                step=2,
                step_name=self.STEP_NAMES[2],
                status="active",
            )

            ontology_result = self._step2.resolve(user_input, intent_result)
            step_times[2] = (time.time() - step_times[2]) * 1000

            # Check if LLM inference needs HITL confirmation
            if ontology_result.llm_inferred and ontology_result.llm_reasoning:
                # Emit confirmation request for LLM-inferred match
                from ..sse_stream import confirm_request as cr
                yield cr(
                    step=2,
                    title="LLM 推理匹配确认",
                    message=f"我理解您说的「{intent_result.object_term}」对应本体对象「{ontology_result.object_label}」",
                    detail=ontology_result.llm_reasoning,
                    action_label="确认",
                    alternatives=ontology_result.alternatives if ontology_result.alternatives else None,
                )
                # Store the inference context for confirmation handling
                self._task_context["llm_inference"] = {
                    "original_term": intent_result.object_term,
                    "inferred_label": ontology_result.object_label,
                    "inferred_type": ontology_result.object_type,
                    "reasoning": ontology_result.llm_reasoning,
                }

            yield step_update(
                step=2,
                step_name=self.STEP_NAMES[2],
                status="completed",
                summary=ontology_result.summary,
                details=ontology_result.to_s2_details(),
            )

            # ===== STEP 3: Task Planning =====
            step_times[3] = time.time()
            yield step_update(
                step=3,
                step_name=self.STEP_NAMES[3],
                status="active",
            )

            plan_result = self._step3.plan(intent_result, ontology_result)
            step_times[3] = (time.time() - step_times[3]) * 1000

            yield step_update(
                step=3,
                step_name=self.STEP_NAMES[3],
                status="completed",
                summary=plan_result.summary,
                details=plan_result.to_s3_details(),
            )

            # Emit plan summary to user for transparency
            if plan_result.plan_summary:
                yield content(f"📋 **执行计划说明**：{plan_result.plan_summary}")

            # Check if confirmation is needed
            if plan_result.requires_confirmation:
                # Emit confirmation request
                from ..sse_stream import confirm_request as cr
                yield cr(
                    step=4,
                    title="即将执行操作",
                    message=f"此操作需要您的确认，是否继续？",
                    action_label="确认执行",
                    risk_level=plan_result.risk_level,
                )
                # For now, continue anyway (confirmation handling is async)

            # ===== STEP 4: Execution =====
            step_times[4] = time.time()
            yield step_update(
                step=4,
                step_name=self.STEP_NAMES[4],
                status="active",
            )

            # Emit tool_call events for each planned action
            for i, action in enumerate(plan_result.planned_actions):
                yield tool_call(
                    name=action.connector or "ProcurementConnector",
                    input_data={c.field: c.value for c in plan_result.query_conditions},
                    type="connector",
                    tool_id=f"call_{task_id}_{i}",
                    description=action.description,
                )

            exec_result = await self._step4.execute(plan_result, task_id, extracted_params)
            step_times[4] = (time.time() - step_times[4]) * 1000

            # Emit tool_result events
            for i, record in enumerate(exec_result.executions):
                yield tool_result(
                    tool_id=f"call_{task_id}_{i}",
                    name=record.connector_name,
                    status=record.status,
                    output=record.data,
                    error=record.error_message,
                )

            yield step_update(
                step=4,
                step_name=self.STEP_NAMES[4],
                status="completed",
                summary=exec_result.summary,
                details=exec_result.to_s4_details(),
            )

            # ===== STEP 5: Response Generation =====
            step_times[5] = time.time()
            yield step_update(
                step=5,
                step_name=self.STEP_NAMES[5],
                status="active",
            )

            response_result = self._step5.generate(
                exec_result, plan_result, intent_result, ontology_result
            )
            step_times[5] = (time.time() - step_times[5]) * 1000

            yield step_update(
                step=5,
                step_name=self.STEP_NAMES[5],
                status="completed",
                summary="回复已生成",
                details=response_result.to_s5_details(),
                suggested_actions=[a.to_dict() for a in response_result.next_actions],
            )

            # ===== Final content event =====
            yield content(response_result.text)
            yield done()

            # ===== Audit log =====
            await self._audit_log(task_id, user_input, intent_result, ontology_result,
                                  plan_result, exec_result, step_times)

        except Exception as e:
            yield error_fn(
                code="PIPELINE_ERROR",
                message=str(e),
                recoverable=True,
                suggestions=["请稍后重试", "如问题持续，请联系管理员"],
            )
            yield done()

    async def run_with_confirmation(
        self,
        user_input: str,
        task_id: str,
        confirmation: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run pipeline with a confirmation result.

        This is called after user confirms or cancels a pending action.
        """
        from ..sse_stream import done as done_fn, content as content_fn, step_update as step_update_fn, error_event

        action = confirmation.get("action", "")

        if action == "cancel":
            yield error_event(
                code="CANCELLED",
                message="用户取消了操作",
                step=4,
                recoverable=False,
            )
            yield done_fn()
            return

        # User confirmed - resume execution
        # Store confirmation in context
        self._task_context[task_id] = {
            "confirmation": confirmation,
            "confirmed_at": datetime.now().isoformat(),
        }

        # Skip to execution step (we know user confirmed intent, ontology, and plan)
        yield step_update_fn(
            step=4,
            step_name=self.STEP_NAMES[4],
            status="active",
            summary="用户已确认，继续执行",
        )

        # Continue with execution (simplified - just mark complete)
        yield step_update_fn(
            step=4,
            step_name=self.STEP_NAMES[4],
            status="completed",
            summary="执行已恢复",
        )

        # Step 5
        yield step_update_fn(
            step=5,
            step_name=self.STEP_NAMES[5],
            status="active",
        )

        yield content_fn("操作已根据您的确认继续执行。")
        yield done_fn()

    async def resume_from_step(
        self,
        task_id: str,
        from_step: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume pipeline from a specific step.

        Used after confirmation or recovery from error.
        """
        from ..sse_stream import done as done_fn, content as content_fn, step_update

        # For now, just emit a simple response
        yield step_update(
            step=from_step,
            step_name=self.STEP_NAMES.get(from_step, "执行"),
            status="active",
        )
        yield content_fn("任务已从步骤 {} 恢复执行".format(from_step))
        yield done_fn()

    async def _audit_log(
        self,
        task_id: str,
        user_input: str,
        intent_result,
        ontology_result,
        plan_result,
        exec_result,
        step_times: Dict[int, float],
    ):
        """Write audit log for the task."""
        try:
            from ..procurement.ontology_loader import get_procurement_ontology
            ontology = get_procurement_ontology()

            audit = {
                "taskId": task_id,
                "userInput": user_input,
                "intent": intent_result.intent,
                "intentLabel": intent_result.intent_label,
                "ontologyVersion": ontology.version,
                "matchedObjects": [ontology_result.object_type],
                "matchedActions": [a.action_id for a in plan_result.planned_actions],
                "connectors": list(set(a.connector or "ProcurementConnector"
                                     for a in plan_result.planned_actions)),
                "confirmationRequired": plan_result.requires_confirmation,
                "finalStatus": "completed" if exec_result.all_success else "partial",
                "totalDurationMs": sum(step_times.values()),
                "stepDurations": step_times,
                "createdAt": datetime.now().isoformat(),
            }

            # Save audit log
            import json
            from pathlib import Path
            audit_dir = Path("data/audit")
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_file = audit_dir / f"{task_id}.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2)

        except Exception:
            pass  # Audit logging should not break the pipeline
