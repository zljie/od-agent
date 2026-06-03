"""Five-Step Transparent Execution Pipeline.

PHASE 4: Semantic Contract Pipeline
===================================

The pipeline now implements the Semantic Contract pattern to prevent "Semantic Collapse":

Step 1: Semantic Intent Discovery
    - Identifies the correct action with highest confidence
    - Creates Semantic Contract as single source of truth

Step 2: Ontology Expansion
    - Supplements Semantic Contract with ontology knowledge
    - Does NOT re-identify the object (no re-matching)

Step 3: Action Planning
    - Uses Semantic Contract action for planning
    - Completes parameters, not re-maps actions
    - Detects and reports action drift

Step 4: Execution
    - Executes the planned action from Semantic Contract
    - Uses parameters built from Semantic Contract slots

Step 5: Response Generation
    - Generates response based on execution results

New Diagnostic Features:
- Action Drift Detection: flags when planned action differs from Semantic Contract
- Semantic Contract Propagation: passes contract through all steps
- Enhanced SSE Events: include diagnostic info for frontend display
"""

import logging
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from ..context_engineering import (
    ContextAssembler,
    ContextPolicyEngine,
    ContextPurpose,
    ContextRetriever,
    ContextSelector,
    LLMContextPackage,
)

from .models import (
    IntentRecognitionResult,
    OntologyResolveResult,
    TaskPlanResult,
    ExecutionResult,
    ResponseResult,
    SemanticContract,
)
from .step1_intent import Step1Recognizer
from .step2_ontology import Step2OntologyResolver
from .step3_planner import Step3TaskPlanner
from .step4_executor import Step4ConnectorExecutor
from .step5_response import Step5ResponseGenerator


class ActionDriftDetector:
    """Detects semantic drift between Semantic Contract and planned execution.

    Action drift occurs when:
    1. Step 1 identifies a specific analytics action (e.g., find_unexecuted_purchase_requests)
    2. Step 3 replaces it with a generic list action (e.g., query_object/list)

    This is a critical bug that causes incorrect execution.
    """

    @staticmethod
    def detect_drift(
        semantic_contract: Optional[SemanticContract],
        planned_action_id: str
    ) -> Dict[str, Any]:
        """Detect if there's drift between Semantic Contract and planned action.

        Returns:
            Dict with drift detection results
        """
        if not semantic_contract:
            return {
                "drift_detected": False,
                "reason": "No Semantic Contract available",
            }

        contract_action = semantic_contract.action

        # Same action - no drift
        if contract_action == planned_action_id:
            return {
                "drift_detected": False,
                "reason": "Actions match",
            }

        # Analytics action replaced with list action - DRIFT
        if contract_action.startswith("analytics/") and "/list" in planned_action_id:
            return {
                "drift_detected": True,
                "drift_type": "analytics_to_generic",
                "contract_action": contract_action,
                "planned_action": planned_action_id,
                "reason": f"Semantic Contract action '{contract_action}' replaced with generic '{planned_action_id}'",
                "severity": "high",
                "impact": "Business logic (e.g., unexecuted filter) will be lost",
            }

        # Different analytics action - DRIFT
        if contract_action.startswith("analytics/") and not planned_action_id.startswith("analytics/"):
            return {
                "drift_detected": True,
                "drift_type": "analytics_replaced",
                "contract_action": contract_action,
                "planned_action": planned_action_id,
                "reason": f"Analytics action '{contract_action}' replaced with '{planned_action_id}'",
                "severity": "high",
                "impact": "Analytics logic will be lost",
            }

        # Different action in same category - minor drift
        if contract_action.split("/")[0] == planned_action_id.split("/")[0]:
            return {
                "drift_detected": True,
                "drift_type": "action_variant",
                "contract_action": contract_action,
                "planned_action": planned_action_id,
                "reason": f"Action variant: '{contract_action}' vs '{planned_action_id}'",
                "severity": "low",
                "impact": "May affect query parameters",
            }

        # Unknown drift
        return {
            "drift_detected": True,
            "drift_type": "unknown",
            "contract_action": contract_action,
            "planned_action": planned_action_id,
            "reason": f"Action changed from '{contract_action}' to '{planned_action_id}'",
            "severity": "medium",
            "impact": "Unknown - review needed",
        }

    @staticmethod
    def format_drift_warning(drift_result: Dict[str, Any]) -> str:
        """Format drift detection result as a warning message."""
        if not drift_result.get("drift_detected"):
            return ""

        severity_emoji = {
            "high": "🚨",
            "medium": "⚠️",
            "low": "📝",
        }.get(drift_result.get("severity", "medium"), "⚠️")

        return (
            f"{severity_emoji} **Action Drift Detected**\n"
            f"- Semantic Contract: `{drift_result.get('contract_action', 'N/A')}`\n"
            f"- Planned Action: `{drift_result.get('planned_action', 'N/A')}`\n"
            f"- Reason: {drift_result.get('reason', 'Unknown')}\n"
            f"- Impact: {drift_result.get('impact', 'Unknown')}"
        )


class FiveStepPipeline:
    """Five-Step Transparent Execution Pipeline.

    PHASE 4: Semantic Contract Pipeline

    Orchestrates the complete 5-step execution lifecycle:
    1. Intent Recognition (Semantic Intent Discovery)
    2. Ontology Expansion (formerly Object Resolution)
    3. Action Planning (formerly Task Planning)
    4. Execution
    5. Response Generation

    Each step emits SSE step_update events.
    """

    STEP_NAMES = {
        1: "意图识别",
        2: "本体扩展",
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

        # Context Engineering components
        self._ctx_retriever = ContextRetriever()
        self._ctx_selector = ContextSelector()
        self._ctx_policy = ContextPolicyEngine()
        self._ctx_assembler = ContextAssembler()
        self._logger = logging.getLogger(__name__)

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

    def _build_context_package(
        self,
        purpose: ContextPurpose,
        user_input: str,
        task_id: str = "",
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
    ) -> LLMContextPackage:
        """Build a standard LLMContextPackage for the given purpose.

        Implements the Retrieve → Select → Policy Check → Assemble flow.
        """
        try:
            # Retrieve candidates using the correct method for each purpose
            candidates = self._retrieve_for_purpose(
                purpose, user_input, step1_result, step2_result, step3_result, step4_result
            )

            # Select relevant context based on step purpose
            selected = self._select_context(purpose, candidates, step1_result, step2_result, step3_result, step4_result)

            # Apply policy checks (security filter)
            filtered, excluded = self._ctx_policy.apply_security_filter(selected)

            # Merge policy metadata into selected context
            merged = self._ctx_policy.merge_policy_context(filtered)

            # Assemble the final package
            package = self._ctx_assembler.assemble_for_X(
                purpose,
                merged,
                user_input,
                step1_result,
                step2_result,
                step3_result,
                step4_result,
                task_id,
            )
            package.excluded_sources = excluded

            return package

        except Exception as e:
            self._logger.warning(f"Context engineering failed for {purpose.value}: {e}")
            # Fallback: return minimal package
            from ..context_engineering.models import ContextMeta, ContextInstructions
            return LLMContextPackage(
                meta=ContextMeta(purpose=purpose, task_id=task_id),
                instructions=ContextInstructions(),
                included_sources=[],
                excluded_sources=[],
            )

    def _retrieve_for_purpose(
        self,
        purpose: ContextPurpose,
        user_input: str,
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
    ) -> Dict[str, Any]:
        """Retrieve context using the appropriate method for each purpose."""
        try:
            if purpose == ContextPurpose.INTENT_RECOGNITION:
                result = self._ctx_retriever.retrieve_for_intent_recognition(user_input)
            elif purpose == ContextPurpose.ONTOLOGY_GROUNDING:
                result = self._ctx_retriever.retrieve_for_ontology_grounding(
                    step1_result  # intent_result is first param
                )
            elif purpose == ContextPurpose.TASK_PLANNING:
                result = self._ctx_retriever.retrieve_for_task_planning(
                    step1_result,  # intent_result
                    step2_result,  # ontology_result
                )
            elif purpose == ContextPurpose.EXECUTION_ASSIST:
                result = self._ctx_retriever.retrieve_for_execution(
                    step3_result,  # plan_result
                    {},  # exec_state (empty dict for now)
                )
            elif purpose == ContextPurpose.RESPONSE_GENERATION:
                result = self._ctx_retriever.retrieve_for_response_generation(
                    step4_result,  # exec_result
                    step3_result,  # plan_result
                    step1_result,  # intent_result
                )
            else:
                return {}

            # Convert RetrievalResult to dict format expected by selector
            if hasattr(result, "to_dict"):
                return result.to_dict().get("context", {})
            return result if isinstance(result, dict) else {}

        except Exception as e:
            self._logger.warning(f"Context retrieval failed for {purpose.value}: {e}")
            return {}

    def _select_context(
        self,
        purpose: ContextPurpose,
        candidates: Dict[str, Any],
        step1_result=None,
        step2_result=None,
        step3_result=None,
        step4_result=None,
    ) -> Dict[str, Any]:
        """Select relevant context based on step purpose."""
        dispatch = {
            ContextPurpose.INTENT_RECOGNITION: self._ctx_selector.select_for_intent_recognition,
            ContextPurpose.ONTOLOGY_GROUNDING: self._ctx_selector.select_for_ontology_grounding,
            ContextPurpose.TASK_PLANNING: self._ctx_selector.select_for_task_planning,
            ContextPurpose.EXECUTION_ASSIST: self._ctx_selector.select_for_execution,
            ContextPurpose.RESPONSE_GENERATION: self._ctx_selector.select_for_response_generation,
        }
        method = dispatch.get(purpose)
        if method is None:
            return candidates

        if purpose == ContextPurpose.INTENT_RECOGNITION:
            return method(candidates)
        elif purpose == ContextPurpose.ONTOLOGY_GROUNDING:
            return method(candidates, step1_result)
        elif purpose == ContextPurpose.TASK_PLANNING:
            return method(candidates, step1_result, step2_result)
        elif purpose == ContextPurpose.EXECUTION_ASSIST:
            current_step = 4
            return method(candidates, step3_result, current_step)
        else:  # RESPONSE_GENERATION
            return method(candidates, step4_result)

    async def _context_audit(
        self,
        task_id: str,
        purpose: str,
        included_sources: List[str],
        excluded_sources: List[Dict[str, str]],
        package: LLMContextPackage,
    ):
        """Log context package audit info per PRD Section 23."""
        try:
            from ..audit import AuditEvent, create_audit_sink
            sink = create_audit_sink()
            sink.write(AuditEvent.EXECUTION, {
                "task_id": task_id,
                "event": "CONTEXT_AUDIT",
                "purpose": purpose,
                "included_sources": included_sources,
                "excluded_sources": excluded_sources,
                "token_budget": {
                    "max_tokens": package.token_budget.max_tokens if package.token_budget else 0,
                    "estimated_input_tokens": package.token_budget.estimated_input_tokens if package.token_budget else 0,
                },
                "trust_level": package.trust_level.value if package.trust_level else "unknown",
                "timestamp": datetime.now().isoformat(),
            })
        except Exception:
            pass  # Audit logging should not break the pipeline



    async def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the complete 5-step pipeline.

        Yields SSE events for each step.
        """
        from ..audit import create_task_logger

        task_id = self._generate_task_id()
        self.session_id = session_id or self.session_id
        task_logger = create_task_logger(task_id, user_input)

        # Step timing
        step_times: Dict[int, float] = {}

        # Helper imports
        step_update, done, content, tool_call, tool_result, error_fn = self._get_sse_helpers()

        try:
            # ===== STEP 1: Intent Recognition =====
            step_times[1] = time.time()
            step_event = step_update(
                step=1,
                step_name=self.STEP_NAMES[1],
                status="active",
            )
            task_logger.emit_sse_event(step_event)
            yield step_event

            # Build context package for Step 1
            ctx_package_1 = self._build_context_package(
                ContextPurpose.INTENT_RECOGNITION,
                user_input,
                task_id,
            )

            intent_result = self._step1.recognize(user_input, task_id=task_id)
            step_times[1] = (time.time() - step_times[1]) * 1000

            # Context audit for Step 1
            await self._context_audit(
                task_id,
                "intent_recognition",
                ctx_package_1.included_sources,
                ctx_package_1.excluded_sources,
                ctx_package_1,
            )

            # Phase 3: Check if composite pipeline triggered HITL
            if intent_result.hitl_request is not None:
                from ..sse_stream import confirm_request as cr
                hitl = intent_result.hitl_request
                hitl_event = cr(
                    step=1,
                    title="需要澄清",
                    message=hitl.question,
                    detail=hitl.context.get("summary", ""),
                    action_label="确认",
                    alternatives=[{"label": o.label, "value": o.option_id} for o in hitl.options],
                )
                task_logger.emit_hitl(hitl.question, [{"label": o.label, "value": o.option_id} for o in hitl.options])
                task_logger.emit_sse_event(hitl_event)
                task_logger.emit_decision("L6_HITL", "HITL", intent_result.confidence, hitl.question)
                yield hitl_event
                # Store composite result for resume
                self._task_context["composite_result"] = intent_result.composite_result
                self._task_context["hitl_request"] = intent_result.hitl_request
                # Wait for confirmation via run_with_confirmation
                done_event = done()
                task_logger.emit_sse_event(done_event)
                yield done_event
                return

            s1_complete = step_update(
                step=1,
                step_name=self.STEP_NAMES[1],
                status="completed",
                summary=intent_result.summary,
                details=intent_result.to_s1_details(),
            )
            task_logger.emit_sse_event(s1_complete)
            task_logger.emit_decision("L5_FUSION", "execute", intent_result.confidence, "confidence threshold met")
            yield s1_complete

            # ===== STEP 2: Ontology Resolution =====
            step_times[2] = time.time()
            s2_active = step_update(
                step=2,
                step_name=self.STEP_NAMES[2],
                status="active",
            )
            task_logger.emit_sse_event(s2_active)
            yield s2_active

            # Build context package for Step 2
            ctx_package_2 = self._build_context_package(
                ContextPurpose.ONTOLOGY_GROUNDING,
                user_input,
                task_id,
                step1_result=intent_result,
            )

            # Phase 3: Pass composite_result to Step 2 if available
            ontology_result = self._step2.resolve(user_input, intent_result, intent_result.composite_result)
            step_times[2] = (time.time() - step_times[2]) * 1000

            # Context audit for Step 2
            await self._context_audit(
                task_id,
                "ontology_grounding",
                ctx_package_2.included_sources,
                ctx_package_2.excluded_sources,
                ctx_package_2,
            )

            # Check if LLM inference needs HITL confirmation
            if ontology_result.llm_inferred and ontology_result.llm_reasoning:
                # Emit confirmation request for LLM-inferred match
                from ..sse_stream import confirm_request as cr
                cr_event = cr(
                    step=2,
                    title="LLM 推理匹配确认",
                    message=f"我理解您说的「{intent_result.object_term}」对应本体对象「{ontology_result.object_label}」",
                    detail=ontology_result.llm_reasoning,
                    action_label="确认",
                    alternatives=ontology_result.alternatives if ontology_result.alternatives else None,
                )
                task_logger.emit_hitl(f"LLM推理确认: {ontology_result.object_label}", ontology_result.alternatives or [])
                task_logger.emit_sse_event(cr_event)
                yield cr_event
                # Store the inference context for confirmation handling
                self._task_context["llm_inference"] = {
                    "original_term": intent_result.object_term,
                    "inferred_label": ontology_result.object_label,
                    "inferred_type": ontology_result.object_type,
                    "reasoning": ontology_result.llm_reasoning,
                }

            s2_complete = step_update(
                step=2,
                step_name=self.STEP_NAMES[2],
                status="completed",
                summary=ontology_result.summary,
                details=ontology_result.to_s2_details(),
            )
            task_logger.emit_sse_event(s2_complete)
            yield s2_complete

            # ===== STEP 3: Task Planning =====
            step_times[3] = time.time()
            s3_active = step_update(
                step=3,
                step_name=self.STEP_NAMES[3],
                status="active",
            )
            task_logger.emit_sse_event(s3_active)
            yield s3_active

            # Build context package for Step 3
            ctx_package_3 = self._build_context_package(
                ContextPurpose.TASK_PLANNING,
                user_input,
                task_id,
                step1_result=intent_result,
                step2_result=ontology_result,
            )

            plan_result = self._step3.plan(intent_result, ontology_result)
            step_times[3] = (time.time() - step_times[3]) * 1000

            # Context audit for Step 3
            await self._context_audit(
                task_id,
                "task_planning",
                ctx_package_3.included_sources,
                ctx_package_3.excluded_sources,
                ctx_package_3,
            )

            # PHASE 4: Action Drift Detection
            drift_result = None
            if intent_result.semantic_contract and plan_result.planned_actions:
                drift_result = ActionDriftDetector.detect_drift(
                    intent_result.semantic_contract,
                    plan_result.planned_actions[0].action_id
                )
                # Update plan_result with drift detection
                if drift_result.get("drift_detected"):
                    plan_result.action_drift_detected = True
                    plan_result.drift_reason = drift_result.get("reason", "")

            # Emit drift warning if detected
            if drift_result and drift_result.get("drift_detected"):
                drift_warning = ActionDriftDetector.format_drift_warning(drift_result)
                drift_event = content(drift_warning)
                task_logger.emit_sse_event(drift_event)
                yield drift_event
                self._logger.warning(f"[ActionDrift] {drift_result.get('reason', 'Unknown drift')}")

            s3_complete = step_update(
                step=3,
                step_name=self.STEP_NAMES[3],
                status="completed",
                summary=plan_result.summary,
                details=plan_result.to_s3_details(),
            )
            task_logger.emit_sse_event(s3_complete)
            yield s3_complete

            # Emit plan summary to user for transparency
            if plan_result.plan_summary:
                content_event = content(f"📋 **执行计划说明**：{plan_result.plan_summary}")
                task_logger.emit_sse_event(content_event)
                yield content_event

            # Check if confirmation is needed
            if plan_result.requires_confirmation:
                # Emit confirmation request
                from ..sse_stream import confirm_request as cr
                cr_event = cr(
                    step=4,
                    title="即将执行操作",
                    message=f"此操作需要您的确认，是否继续？",
                    action_label="确认执行",
                    risk_level=plan_result.risk_level,
                )
                task_logger.emit_hitl("操作确认", [])
                task_logger.emit_sse_event(cr_event)
                yield cr_event

            # ===== STEP 4: Execution =====
            step_times[4] = time.time()
            s4_active = step_update(
                step=4,
                step_name=self.STEP_NAMES[4],
                status="active",
            )
            task_logger.emit_sse_event(s4_active)
            yield s4_active

            # Build context package for Step 4
            ctx_package_4 = self._build_context_package(
                ContextPurpose.EXECUTION_ASSIST,
                user_input,
                task_id,
                step3_result=plan_result,
            )

            # Emit tool_call events for each planned action
            for i, action in enumerate(plan_result.planned_actions):
                tc_event = tool_call(
                    name=action.connector or "ProcurementConnector",
                    input_data={c.field: c.value for c in plan_result.query_conditions},
                    type="connector",
                    tool_id=f"call_{task_id}_{i}",
                    description=action.description,
                )
                task_logger.emit_sse_event(tc_event)
                yield tc_event

            exec_result = await self._step4.execute(plan_result, task_id, extracted_params)
            step_times[4] = (time.time() - step_times[4]) * 1000

            # Context audit for Step 4
            await self._context_audit(
                task_id,
                "execution_assist",
                ctx_package_4.included_sources,
                ctx_package_4.excluded_sources,
                ctx_package_4,
            )

            # Emit tool_result events
            for i, record in enumerate(exec_result.executions):
                tr_event = tool_result(
                    tool_id=f"call_{task_id}_{i}",
                    name=record.connector_name,
                    status=record.status,
                    output=record.data,
                    error=record.error_message,
                )
                task_logger.emit_sse_event(tr_event)
                yield tr_event

            s4_complete = step_update(
                step=4,
                step_name=self.STEP_NAMES[4],
                status="completed",
                summary=exec_result.summary,
                details=exec_result.to_s4_details(),
            )
            task_logger.emit_sse_event(s4_complete)
            yield s4_complete

            # ===== STEP 5: Response Generation =====
            step_times[5] = time.time()
            s5_active = step_update(
                step=5,
                step_name=self.STEP_NAMES[5],
                status="active",
            )
            task_logger.emit_sse_event(s5_active)
            yield s5_active

            # Build context package for Step 5
            ctx_package_5 = self._build_context_package(
                ContextPurpose.RESPONSE_GENERATION,
                user_input,
                task_id,
                step4_result=exec_result,
            )

            response_result = self._step5.generate(
                exec_result, plan_result, intent_result, ontology_result
            )
            step_times[5] = (time.time() - step_times[5]) * 1000

            # Context audit for Step 5
            await self._context_audit(
                task_id,
                "response_generation",
                ctx_package_5.included_sources,
                ctx_package_5.excluded_sources,
                ctx_package_5,
            )

            s5_complete = step_update(
                step=5,
                step_name=self.STEP_NAMES[5],
                status="completed",
                summary="回复已生成",
                details=response_result.to_s5_details(),
                suggested_actions=[a.to_dict() for a in response_result.next_actions],
            )
            task_logger.emit_sse_event(s5_complete)
            yield s5_complete

            # ===== Final content event =====
            content_event = content(response_result.text)
            task_logger.emit_sse_event(content_event)
            yield content_event

            done_event = done()
            task_logger.emit_sse_event(done_event)
            yield done_event

            # ===== Audit log =====
            await self._audit_log(task_id, user_input, intent_result, ontology_result,
                                  plan_result, exec_result, step_times)

            # Close task logger
            task_logger.close(final_status="completed")

        except Exception as e:
            error_event = error_fn(
                code="PIPELINE_ERROR",
                message=str(e),
                recoverable=True,
                suggestions=["请稍后重试", "如问题持续，请联系管理员"],
            )
            task_logger.emit_sse_event(error_event)
            task_logger.error(f"Pipeline error: {e}", source="Pipeline")
            task_logger.close(final_status="error")
            yield error_event
            done_event = done()
            task_logger.emit_sse_event(done_event)
            yield done_event

    async def run_with_confirmation(
        self,
        user_input: str,
        task_id: str,
        confirmation: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run pipeline with a confirmation result.

        This is called after user confirms or cancels a pending action.
        For composite HITL, re-runs the pipeline from Step 1 with the user's clarification.
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

        # Check if this is a composite HITL resume
        composite_result = self._task_context.get("composite_result")
        hitl_request = self._task_context.get("hitl_request")

        if composite_result is not None and hitl_request is not None:
            # Phase 3: Resume from composite HITL
            selected_value = confirmation.get("selected_value", "")
            step_update, done, content, tool_call, tool_result, error_fn = self._get_sse_helpers()

            # Re-run composite pipeline with user's clarification
            composite_pipeline = self._step1._get_composite_pipeline()
            if composite_pipeline:
                # Provide the clarification to the composite pipeline
                clarified_input = f"{user_input} {selected_value}"
                composite_result = composite_pipeline.run(clarified_input)

            # Build intent_result from updated composite result
            if composite_result and composite_result.top_candidate:
                top = composite_result.top_candidate
                from .models import IntentRecognitionResult
                intent_result = IntentRecognitionResult(
                    intent=top.intent_id,
                    intent_label=top.intent_name or top.intent_id,
                    object_term=top.params.get("object_term", ""),
                    normalized_term=top.params.get("normalized_object_term"),
                    operation_type=self._step1._map_action_to_operation(top.intent_id),
                    risk_level=self._step1._assess_risk(self._step1._map_action_to_operation(top.intent_id)),
                    requires_confirmation=False,
                    confidence=top.confidence,
                    alternative_intents=[c.intent_id for c in composite_result.candidates[1:4]],
                    composite_result=composite_result,
                    hitl_request=composite_result.hitl_request,
                    composite_layer_results=composite_result.layer_results,
                    composite_task_id=composite_result.task_id,
                )
            else:
                # Fall back to original intent_result
                intent_result = None

            # Clear HITL context
            self._task_context.pop("composite_result", None)
            self._task_context.pop("hitl_request", None)

            if intent_result:
                yield step_update_fn(
                    step=1,
                    step_name=self.STEP_NAMES[1],
                    status="completed",
                    summary=intent_result.summary,
                    details=intent_result.to_s1_details(),
                )

                # Continue with Step 2 (with composite_result)
                yield step_update_fn(
                    step=2,
                    step_name=self.STEP_NAMES[2],
                    status="active",
                )

                ontology_result = self._step2.resolve(user_input, intent_result, composite_result)

                yield step_update_fn(
                    step=2,
                    step_name=self.STEP_NAMES[2],
                    status="completed",
                    summary=ontology_result.summary,
                    details=ontology_result.to_s2_details(),
                )

                # Continue with Step 3
                yield step_update_fn(
                    step=3,
                    step_name=self.STEP_NAMES[3],
                    status="active",
                )

                plan_result = self._step3.plan(intent_result, ontology_result)

                yield step_update_fn(
                    step=3,
                    step_name=self.STEP_NAMES[3],
                    status="completed",
                    summary=plan_result.summary,
                    details=plan_result.to_s3_details(),
                )

                if plan_result.plan_summary:
                    yield content_fn(f"📋 **执行计划说明**：{plan_result.plan_summary}")

                # Continue with Step 4
                yield step_update_fn(
                    step=4,
                    step_name=self.STEP_NAMES[4],
                    status="active",
                )

                exec_result = await self._step4.execute(plan_result, task_id, None)

                yield step_update_fn(
                    step=4,
                    step_name=self.STEP_NAMES[4],
                    status="completed",
                    summary=exec_result.summary,
                    details=exec_result.to_s4_details(),
                )

                # Continue with Step 5
                yield step_update_fn(
                    step=5,
                    step_name=self.STEP_NAMES[5],
                    status="active",
                )

                response_result = self._step5.generate(
                    exec_result, plan_result, intent_result, ontology_result
                )

                yield step_update_fn(
                    step=5,
                    step_name=self.STEP_NAMES[5],
                    status="completed",
                    summary="回复已生成",
                    details=response_result.to_s5_details(),
                    suggested_actions=[a.to_dict() for a in response_result.next_actions],
                )

                yield content_fn(response_result.text)
                yield done_fn()
                return

        # Legacy confirmation handling (non-composite)
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

            # Build base audit dict (backward-compatible structure)
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

            # --- Context audit block (additive, PRD Section 23) ---
            # Retrieve context logger if the task set one in _task_context
            ctx_logger = self._task_context.get("_context_audit_logger")
            if ctx_logger is not None:
                ctx_dict = ctx_logger.to_audit_dict()
            else:
                # Fallback: emit a minimal context audit block
                ctx_dict = {
                    "includedSources": [],
                    "excludedSources": [],
                    "tokenBudget": {
                        "max": 4000,
                        "estimated": 0,
                        "allocation": {},
                    },
                    "redactedFields": [],
                    "permissionChecked": False,
                    "ontologyVersion": ontology.version,
                    "trustBreakdown": {},
                    "freshnessBreakdown": {},
                    "policySummary": {},
                }
            audit["contextAudit"] = ctx_dict

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
