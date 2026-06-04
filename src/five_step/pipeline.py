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
    SemanticContractSlot,
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

        # Phase 4.1: HITL Session State — pending task for multi-round slot filling
        self._pending_hitl_task: Optional[Dict[str, Any]] = None
        
        # Flag to skip confirmation when resuming from confirmation
        self._skip_confirmation: bool = False

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

    def _save_and_emit_hitl(
        self,
        task_id: str,
        hitl_event: Dict[str, Any],
        intent_result,
        ontology_result,
        plan_result,
        composite_result,
        hitl_request,
        pending_hitl_task: Optional[Dict[str, Any]],
        task_logger,
        hitl_phase: str,
        user_input: str = "",
    ) -> bool:
        """Save HITL session to global store and emit the HITL event, then yield done.

        Returns True if HITL was emitted (caller should return).
        The session is saved BEFORE yielding so that run_with_confirmation
        can load it on the next HTTP request.
        """
        from .hitl_session_store import save_hitl_session

        print(f"[FiveStepPipeline][DEBUG] _save_and_emit_hitl called | task_id={task_id} | hitl_phase={hitl_phase}")

        # Save full context to global HITLSessionStore
        ctx = save_hitl_session(
            task_id=task_id,
            session_id=self.session_id or "",
            intent_result=intent_result,
            ontology_result=ontology_result,
            plan_result=plan_result,
            composite_result=composite_result,
            hitl_request=hitl_request,
            pending_hitl_task=pending_hitl_task,
            semantic_contract=getattr(intent_result, 'semantic_contract', None) if intent_result else None,
            original_user_input=user_input,
            hitl_phase=hitl_phase,
        )

        print(f"[FiveStepPipeline][DEBUG] HITL session saved | task_id={task_id} | ctx.expires_at={ctx.expires_at}")

        # Also update the in-memory pending task for same-session access
        if pending_hitl_task:
            self._pending_hitl_task = pending_hitl_task

        # Log HITL emission
        hitl_question = getattr(hitl_request, 'question', '') if hitl_request else ''
        hitl_options = getattr(hitl_request, 'options', []) if hitl_request else []
        task_logger.emit_hitl(hitl_question, hitl_options)
        task_logger.emit_sse_event(hitl_event)
        return True

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

            # Phase 4.1: Pass pending HITL task to Step1 recognizer
            # This enables slot_filling mode on the next composite pipeline run
            self._step1._pending_hitl_task = self._pending_hitl_task

            intent_result = self._step1.recognize(user_input, task_id=task_id)
            step_times[1] = (time.time() - step_times[1]) * 1000

            # Sync back updated pending_hitl_task from Step1 recognizer
            # This is critical for slot-fill protocol: the composite pipeline updates
            # the pending task internally, and we need to reflect those changes here
            if self._step1._pending_hitl_task is not None:
                self._pending_hitl_task = self._step1._pending_hitl_task
                print(f"[FiveStepPipeline][INFO] PendingHITL同步更新 | action={self._pending_hitl_task.get('action')} | 缺槽={self._pending_hitl_task.get('missing_slots')}")

            # Context audit for Step 1
            await self._context_audit(
                task_id,
                "intent_recognition",
                ctx_package_1.included_sources,
                ctx_package_1.excluded_sources,
                ctx_package_1,
            )

            # Phase 3/4: Check if composite pipeline triggered HITL
            if intent_result.hitl_request is not None:
                from ..sse_stream import slot_fill_request
                hitl = intent_result.hitl_request
                task_logger.emit_hitl(hitl.question, [])

                # Build slot form fields for the HITL request
                composite = intent_result.composite_result
                slot_check = None
                missing_slots = []

                if composite and composite.layer_results:
                    # Check slot-fill protocol first (highest priority for slot-fill inputs)
                    if "slot_fill_protocol" in composite.layer_results:
                        sf_data = composite.layer_results["slot_fill_protocol"]
                        missing_slots = sf_data.get("still_missing", [])
                        slot_check = composite.layer_results.get("layer_2_5_slot_completion", {})
                    else:
                        # Fall back to normal slot check
                        slot_check = composite.layer_results.get("layer_2_5_slot_completion", {})
                        if slot_check:
                            missing_slots = slot_check.get("missing_required", [])
                        if not missing_slots:
                            deep = composite.layer_results.get("layer_4_deep_reasoning", {})
                            missing_slots = deep.get("missing_info", [])

                # Generate slot form fields from slot definitions
                slot_form_fields = []
                if missing_slots:
                    slot_form_fields = self._step1._get_slot_form_fields(
                        missing_slots, intent_result.object_term
                    )

                object_label = {
                    "purchase_requests": "采购需求",
                    "purchase_orders": "采购订单",
                    "purchase_inquiries": "询价单",
                    "purchase_quotations": "报价单",
                }.get(intent_result.object_term, intent_result.object_term)

                # Use slot_fill_request if we have slot fields, else fall back to confirm_request
                if slot_form_fields:
                    action_label = intent_result.intent_label or intent_result.intent
                    hitl_event = slot_fill_request(
                        request_id=task_id,
                        step=1,
                        title="需要澄清",
                        message=hitl.question,
                        slots=slot_form_fields,
                        action_label=f"创建{object_label}" if "create" in intent_result.intent.lower() else f"确认{object_label}",
                        action_id=intent_result.intent,
                        risk_level=intent_result.risk_level,
                        alternatives=[{"label": o.label, "value": o.option_id} for o in hitl.options] if hitl.options else None,
                        detail=hitl.context.get("summary", ""),
                    )
                    task_logger.emit_decision("L6_HITL", "SLOT_FILL", intent_result.confidence, hitl.question)
                else:
                    from ..sse_stream import confirm_request as cr
                    hitl_event = cr(
                        step=1,
                        title="需要澄清",
                        message=hitl.question,
                        detail=hitl.context.get("summary", ""),
                        action_label="确认",
                        task_id=task_id,
                        alternatives=[{"label": o.label, "value": o.option_id} for o in hitl.options],
                    )
                    task_logger.emit_decision("L6_HITL", "HITL", intent_result.confidence, hitl.question)

                # Build pending_hitl_task for slot filling
                composite_meta = intent_result.composite_result.metadata if intent_result.composite_result else {}
                updated_pending = composite_meta.get("pending_task")
                if updated_pending:
                    self._pending_hitl_task = updated_pending
                else:
                    self._pending_hitl_task = {
                        "action": intent_result.intent,
                        "object": intent_result.object_term,
                        "missing_slots": missing_slots,
                        "filled_slots": {},
                        "task_id": task_id,
                        "confidence": intent_result.confidence,
                    }

                # Also persist to PendingTaskStore for slot-fill protocol
                from ..intent_recognition.pending_task_store import save_pending_task
                save_pending_task(
                    task_id=self._pending_hitl_task.get("task_id", task_id),
                    action=self._pending_hitl_task.get("action", intent_result.intent),
                    object=self._pending_hitl_task.get("object", intent_result.object_term),
                    missing_slots=self._pending_hitl_task.get("missing_slots", missing_slots),
                    filled_slots=self._pending_hitl_task.get("filled_slots", {}),
                    confidence=self._pending_hitl_task.get("confidence", intent_result.confidence),
                    metadata={
                        "intent_label": intent_result.intent_label,
                        "slot_form_fields": slot_form_fields,
                    },
                )

                # CRITICAL: Save full HITL session to global store so run_with_confirmation can resume
                self._save_and_emit_hitl(
                    task_id=task_id,
                    hitl_event=hitl_event,
                    intent_result=intent_result,
                    ontology_result=None,
                    plan_result=None,
                    composite_result=intent_result.composite_result,
                    hitl_request=hitl,
                    pending_hitl_task=self._pending_hitl_task,
                    task_logger=task_logger,
                    hitl_phase="slot_fill",
                    user_input=user_input,
                )
                # Emit the HITL event to frontend (hitl_event was saved but not yielded yet)
                yield hitl_event
                task_logger.emit_sse_event(done())
                yield done()
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
                from ..sse_stream import confirm_request as cr
                cr_event = cr(
                    step=2,
                    title="LLM 推理匹配确认",
                    message=f"我理解您说的「{intent_result.object_term}」对应本体对象「{ontology_result.object_label}」",
                    detail=ontology_result.llm_reasoning,
                    action_label="确认",
                    risk_level=intent_result.risk_level,
                    task_id=task_id,
                    alternatives=ontology_result.alternatives if ontology_result.alternatives else None,
                )

                # CRITICAL: Save full HITL session to global store for run_with_confirmation
                self._save_and_emit_hitl(
                    task_id=task_id,
                    hitl_event=cr_event,
                    intent_result=intent_result,
                    ontology_result=ontology_result,
                    plan_result=None,
                    composite_result=intent_result.composite_result,
                    hitl_request=None,
                    pending_hitl_task=self._pending_hitl_task,
                    task_logger=task_logger,
                    hitl_phase="ontology_confirm",
                    user_input=user_input,
                )
                # Emit the HITL event to frontend
                yield cr_event
                task_logger.emit_sse_event(done())
                yield done()
                return

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
            # PHASE 4.1: Create operations ALWAYS require human confirmation
            requires_confirmation = plan_result.requires_confirmation
            operation_type = getattr(intent_result, 'operation_type', '')
            hitl_triggered = getattr(intent_result.semantic_contract, 'hitl_triggered', False) if intent_result.semantic_contract else False

            # Strategy 1: operation_type == "create" → always require confirmation
            if operation_type == "create":
                requires_confirmation = True
                plan_result.risk_level = "high"

            # Strategy 2: HITL was triggered by intent recognition (missing slots) → require confirmation
            if hitl_triggered:
                requires_confirmation = True

            # Strategy 3: intent recognition result has hitl_request → require confirmation
            if intent_result.hitl_request is not None:
                requires_confirmation = True
            
            if requires_confirmation and not self._skip_confirmation:
                confirmation_message = self._build_confirmation_message(intent_result, plan_result)

                from ..sse_stream import confirm_request as cr
                cr_event = cr(
                    step=4,
                    title="即将执行操作",
                    message=confirmation_message,
                    action_label="确认执行",
                    risk_level=plan_result.risk_level,
                    task_id=task_id,
                )

                # CRITICAL: Save full HITL session to global store for run_with_confirmation
                self._save_and_emit_hitl(
                    task_id=task_id,
                    hitl_event=cr_event,
                    intent_result=intent_result,
                    ontology_result=ontology_result,
                    plan_result=plan_result,
                    composite_result=intent_result.composite_result,
                    hitl_request=None,
                    pending_hitl_task=self._pending_hitl_task,
                    task_logger=task_logger,
                    hitl_phase="create_confirm",
                    user_input=user_input,
                )
                # Emit the HITL event to frontend
                yield cr_event
                task_logger.emit_sse_event(done())
                yield done()
                return

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
        """Resume pipeline after user confirms a HITL request.

        IMPORTANT: This is called on a NEW FiveStepPipeline instance (different from the
        one that emitted the HITL). Therefore, we MUST load the saved context from
        HITLSessionStore — instance fields like self._task_context are ALWAYS empty.

        The flow is:
        1. Load saved HITL session from HITLSessionStore (keyed by task_id)
        2. Based on hitl_phase, determine which step to resume from
        3. Continue execution with the loaded context
        4. Clear the session on completion
        """
        from ..sse_stream import done as done_fn, content as content_fn, step_update as step_update_fn, error_event
        from .hitl_session_store import load_hitl_session, clear_hitl_session
        from ..audit import get_task_logger

        action = confirmation.get("action", "")

        if action == "cancel":
            clear_hitl_session(task_id)
            err = error_event(
                code="CANCELLED",
                message="用户取消了操作",
                step=4,
                recoverable=False,
            )
            yield err
            yield done_fn()
            return

        # CRITICAL: Load saved HITL session from global store
        hitl_ctx = load_hitl_session(task_id)

        # Recover or create task logger for this task
        task_logger = get_task_logger(task_id)
        if task_logger is None:
            task_logger = get_task_logger(hitl_ctx.task_id) if hitl_ctx else None
        if task_logger is None:
            from ..audit import create_task_logger
            task_logger = create_task_logger(task_id, hitl_ctx.original_user_input if hitl_ctx else "")

        if hitl_ctx is None:
            # No HITL session found — try to recover from PendingTaskStore as fallback
            print(f"[FiveStepPipeline][WARN] run_with_confirmation: 未找到HITL会话 {task_id}，尝试从PendingTaskStore恢复")
            from ..intent_recognition.pending_task_store import load_pending_task
            pending_task = load_pending_task(task_id)
            if pending_task:
                print(f"[FiveStepPipeline][INFO] 从PendingTaskStore恢复 | action={pending_task.action} | 已填槽={list(pending_task.filled_slots.keys())} | 缺槽={pending_task.missing_slots}")
                # Restore to pending_hitl_task for later use
                self._pending_hitl_task = {
                    "action": pending_task.action,
                    "object": pending_task.object,
                    "missing_slots": pending_task.missing_slots,
                    "filled_slots": pending_task.filled_slots,
                    "task_id": task_id,
                    "confidence": pending_task.confidence,
                }
                # Emit a message indicating the session was partially recovered
                c = content_fn(f"已恢复任务 {task_id}，但部分上下文丢失。请重新提交表单。")
                task_logger.emit_sse_event(c); yield c
                d = done_fn()
                task_logger.emit_sse_event(d)
                yield d
                return
            # If both stores are empty, use legacy path
            print(f"[FiveStepPipeline][WARN] run_with_confirmation: 所有Store都未找到 {task_id}，使用legacy路径")
            s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
            task_logger.emit_sse_event(s4a); yield s4a
            s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed")
            task_logger.emit_sse_event(s4c); yield s4c
            s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
            task_logger.emit_sse_event(s5a); yield s5a
            c = content_fn("操作已根据您的确认继续执行。")
            task_logger.emit_sse_event(c); yield c
            d = done_fn()
            task_logger.emit_sse_event(d)
            yield d
            return

        print(f"[FiveStepPipeline][INFO] run_with_confirmation: 从HITLSessionStore加载 | phase={hitl_ctx.hitl_phase}")

        step_update, done, content, tool_call, tool_result, error_fn = self._get_sse_helpers()

        # Restore context from saved session
        intent_result = hitl_ctx.intent_result
        ontology_result = hitl_ctx.ontology_result
        plan_result = hitl_ctx.plan_result
        composite_result = hitl_ctx.composite_result
        self._pending_hitl_task = hitl_ctx.pending_hitl_task

        # Set skip flag so we don't ask for confirmation again
        self._skip_confirmation = True

        # Route based on which phase triggered HITL
        hitl_phase = hitl_ctx.hitl_phase

        if hitl_phase == "slot_fill":
            # Frontend sends filled_slots as {id: value, ...} dict
            # Also support slots as [{"id": "...", "value": "..."}, ...] list for compatibility
            submitted_slots: Dict[str, Any] = {}
            # Primary: filled_slots dict
            filled = confirmation.get("filled_slots", {})
            if isinstance(filled, dict):
                submitted_slots = {k: v for k, v in filled.items() if v is not None and v != ""}
            # Fallback: slots list
            if not submitted_slots:
                slots_list = confirmation.get("slots", [])
                if isinstance(slots_list, list):
                    for slot in slots_list:
                            if isinstance(slot, dict) and slot.get("id") and slot.get("value"):
                                submitted_slots[slot["id"]] = slot["value"]

            pending = self._pending_hitl_task
            if pending:
                original_filled = pending.get("filled_slots", {})
                original_missing = pending.get("missing_slots", [])
                updated_filled = {**original_filled, **submitted_slots}
                still_missing = [s for s in original_missing if not updated_filled.get(s)]
                self._pending_hitl_task = {
                    **pending,
                    "filled_slots": updated_filled,
                    "missing_slots": still_missing,
                }
                print(f"[FiveStepPipeline][INFO] SlotFill表单数据 | submitted={list(submitted_slots.keys())} | 已填槽={list(updated_filled.keys())} | 仍缺槽={still_missing}")

            # If slots are still incomplete, emit another HITL requesting remaining ones
            still_missing = self._pending_hitl_task.get("missing_slots", []) if self._pending_hitl_task else []
            if still_missing:
                from .models import IntentRecognitionResult
                from .hitl_session_store import save_hitl_session
                from ..sse_stream import slot_fill_request as sfr

                # Rebuild intent_result from original (intent is already known)
                orig_intent = hitl_ctx.intent_result
                top_intent_id = getattr(orig_intent, 'intent', 'create_object')
                top_object_term = getattr(orig_intent, 'object_term', '')
                operation_type = self._step1._map_action_to_operation(top_intent_id) if self._step1 else "create"
                risk_level = self._step1._assess_risk(operation_type) if self._step1 else "medium"

                intent_result = IntentRecognitionResult(
                    intent=top_intent_id,
                    intent_label=getattr(orig_intent, 'intent_label', top_intent_id),
                    object_term=top_object_term,
                    normalized_term=getattr(orig_intent, 'normalized_term', ''),
                    operation_type=operation_type,
                    risk_level=risk_level,
                    requires_confirmation=False,
                    confidence=getattr(orig_intent, 'confidence', 1.0),
                    composite_result=composite_result,
                )

                slot_form_fields = []
                if self._step1:
                    slot_form_fields = self._step1._get_slot_form_fields(still_missing, top_object_term)

                object_label = {
                    "purchase_requests": "采购需求",
                    "purchase_orders": "采购订单",
                    "purchase_inquiries": "询价单",
                    "purchase_quotations": "报价单",
                }.get(top_object_term, top_object_term)

                hitl_event = sfr(
                    request_id=task_id,
                    step=1,
                    title="需要补充更多信息",
                    message="请继续补充以下信息：",
                    slots=slot_form_fields,
                    action_label=f"创建{object_label}" if "create" in top_intent_id.lower() else f"确认{object_label}",
                    action_id=top_intent_id,
                    risk_level=risk_level,
                )

                save_hitl_session(
                    task_id=task_id,
                    session_id=self.session_id or "",
                    intent_result=hitl_ctx.intent_result,
                    ontology_result=hitl_ctx.ontology_result,
                    plan_result=hitl_ctx.plan_result,
                    composite_result=hitl_ctx.composite_result,
                    hitl_request=None,
                    pending_hitl_task=self._pending_hitl_task,
                    semantic_contract=getattr(hitl_ctx.intent_result, 'semantic_contract', None) if hitl_ctx.intent_result else None,
                    original_user_input=hitl_ctx.original_user_input,
                    hitl_phase="slot_fill",
                )

                yield hitl_event
                task_logger.emit_sse_event(hitl_event)
                d = done_fn()
                task_logger.emit_sse_event(d)
                yield d
                return

            # All slots filled — proceed to Step 2+3+4+5
            from .models import IntentRecognitionResult
            orig_intent = hitl_ctx.intent_result
            top_intent_id = getattr(orig_intent, 'intent', 'create_object')
            top_object_term = getattr(orig_intent, 'object_term', '')
            top_intent_label = getattr(orig_intent, 'intent_label', top_intent_id)
            top_confidence = getattr(orig_intent, 'confidence', 1.0)
            normalized_term = getattr(orig_intent, 'normalized_term', '')
            operation_type = self._step1._map_action_to_operation(top_intent_id) if self._step1 else "create"
            risk_level = self._step1._assess_risk(operation_type) if self._step1 else "medium"

            # CRITICAL: Build semantic contract with filled slots for Step 3 planner
            filled = self._pending_hitl_task.get("filled_slots", {}) if self._pending_hitl_task else {}

            intent_result = IntentRecognitionResult(
                intent=top_intent_id,
                intent_label=top_intent_label,
                object_term=top_object_term,
                normalized_term=normalized_term,
                operation_type=operation_type,
                risk_level=risk_level,
                requires_confirmation=False,
                confidence=top_confidence,
                composite_result=composite_result,
                # Pass filled_slots via semantic_contract for planner to use
                semantic_contract=self._build_semantic_contract_with_slots(
                    top_intent_id, top_object_term, filled
                ) if filled else None,
            )

            # Skip to Step 2+3 since intent is known
            s1c = step_update_fn(step=1, step_name=self.STEP_NAMES[1], status="completed",
                               summary=intent_result.summary, details=intent_result.to_s1_details())
            task_logger.emit_sse_event(s1c); yield s1c

            s2a = step_update_fn(step=2, step_name=self.STEP_NAMES[2], status="active")
            task_logger.emit_sse_event(s2a); yield s2a
            ontology_result = self._step2.resolve(hitl_ctx.original_user_input, intent_result, composite_result)
            s2c = step_update_fn(step=2, step_name=self.STEP_NAMES[2], status="completed",
                               summary=ontology_result.summary, details=ontology_result.to_s2_details())
            task_logger.emit_sse_event(s2c); yield s2c

            s3a = step_update_fn(step=3, step_name=self.STEP_NAMES[3], status="active")
            task_logger.emit_sse_event(s3a); yield s3a
            plan_result = self._step3.plan(intent_result, ontology_result)
            s3c = step_update_fn(step=3, step_name=self.STEP_NAMES[3], status="completed",
                               summary=plan_result.summary, details=plan_result.to_s3_details())
            task_logger.emit_sse_event(s3c); yield s3c

            if plan_result.plan_summary:
                c = content_fn(f"📋 **执行计划说明**：{plan_result.plan_summary}")
                task_logger.emit_sse_event(c); yield c

            # Check if execution is ready
            filled = self._pending_hitl_task.get("filled_slots", {}) if self._pending_hitl_task else {}
            if not plan_result.requires_confirmation and getattr(intent_result, 'operation_type', '') != "create":
                # Ready to execute
                s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
                task_logger.emit_sse_event(s4a); yield s4a
                exec_result = await self._step4.execute(plan_result, task_id, filled)
                s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed",
                                   summary=exec_result.summary, details=exec_result.to_s4_details())
                task_logger.emit_sse_event(s4c); yield s4c
                s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
                task_logger.emit_sse_event(s5a); yield s5a
                response_result = self._step5.generate(exec_result, plan_result, intent_result, ontology_result)
                s5c = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="completed",
                                   summary="回复已生成", details=response_result.to_s5_details(),
                                   suggested_actions=[a.to_dict() for a in response_result.next_actions])
                task_logger.emit_sse_event(s5c); yield s5c
                c = content_fn(response_result.text)
                task_logger.emit_sse_event(c); yield c
            else:
                # Create operation: need additional confirmation
                operation_type = getattr(intent_result, 'operation_type', '')
                if operation_type == "create":
                    confirmation_msg = self._build_confirmation_message(intent_result, plan_result)
                    c = content_fn(f"✅ **您已确认**，即将执行以下操作：\n\n{confirmation_msg}")
                    task_logger.emit_sse_event(c); yield c

                s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
                task_logger.emit_sse_event(s4a); yield s4a
                exec_result = await self._step4.execute(plan_result, task_id, filled)
                s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed",
                                   summary=exec_result.summary, details=exec_result.to_s4_details())
                task_logger.emit_sse_event(s4c); yield s4c
                s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
                task_logger.emit_sse_event(s5a); yield s5a
                response_result = self._step5.generate(exec_result, plan_result, intent_result, ontology_result)
                s5c = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="completed",
                                   summary="回复已生成", details=response_result.to_s5_details(),
                                   suggested_actions=[a.to_dict() for a in response_result.next_actions])
                task_logger.emit_sse_event(s5c); yield s5c
                c = content_fn(response_result.text)
                task_logger.emit_sse_event(c); yield c
        elif hitl_phase == "ontology_confirm":
            # Ontology confirmation: user confirmed LLM-inferred object
            # Continue from Step 3
            s1c = step_update_fn(step=1, step_name=self.STEP_NAMES[1], status="completed",
                               summary=getattr(intent_result, 'summary', ''))
            task_logger.emit_sse_event(s1c); yield s1c
            s2c = step_update_fn(step=2, step_name=self.STEP_NAMES[2], status="completed",
                               summary=ontology_result.summary if ontology_result else '')
            task_logger.emit_sse_event(s2c); yield s2c
            s3a = step_update_fn(step=3, step_name=self.STEP_NAMES[3], status="active")
            task_logger.emit_sse_event(s3a); yield s3a
            plan_result = self._step3.plan(intent_result, ontology_result)
            s3c = step_update_fn(step=3, step_name=self.STEP_NAMES[3], status="completed",
                               summary=plan_result.summary, details=plan_result.to_s3_details())
            task_logger.emit_sse_event(s3c); yield s3c

            if plan_result.plan_summary:
                c = content_fn(f"📋 **执行计划说明**：{plan_result.plan_summary}")
                task_logger.emit_sse_event(c); yield c

            operation_type = getattr(intent_result, 'operation_type', '')
            if operation_type == "create":
                confirmation_msg = self._build_confirmation_message(intent_result, plan_result)
                c = content_fn(f"✅ **您已确认**，即将执行以下操作：\n\n{confirmation_msg}")
                task_logger.emit_sse_event(c); yield c

            s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
            task_logger.emit_sse_event(s4a); yield s4a
            exec_result = await self._step4.execute(plan_result, task_id, None)
            s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed",
                               summary=exec_result.summary, details=exec_result.to_s4_details())
            task_logger.emit_sse_event(s4c); yield s4c
            s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
            task_logger.emit_sse_event(s5a); yield s5a
            response_result = self._step5.generate(exec_result, plan_result, intent_result, ontology_result)
            s5c = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="completed",
                               summary="回复已生成", details=response_result.to_s5_details(),
                               suggested_actions=[a.to_dict() for a in response_result.next_actions])
            task_logger.emit_sse_event(s5c); yield s5c
            c = content_fn(response_result.text)
            task_logger.emit_sse_event(c); yield c
        elif hitl_phase == "create_confirm":
            # Create confirmation: user confirmed the create operation
            # Skip to Step 4 execution directly
            s1c = step_update_fn(step=1, step_name=self.STEP_NAMES[1], status="completed",
                               summary=getattr(intent_result, 'summary', ''))
            task_logger.emit_sse_event(s1c); yield s1c
            s2c = step_update_fn(step=2, step_name=self.STEP_NAMES[2], status="completed",
                               summary=ontology_result.summary if ontology_result else '')
            task_logger.emit_sse_event(s2c); yield s2c
            s3c = step_update_fn(step=3, step_name=self.STEP_NAMES[3], status="completed",
                               summary=plan_result.summary if plan_result else '')
            task_logger.emit_sse_event(s3c); yield s3c

            if plan_result and plan_result.plan_summary:
                c = content_fn(f"📋 **执行计划说明**：{plan_result.plan_summary}")
                task_logger.emit_sse_event(c); yield c

            confirmation_msg = self._build_confirmation_message(intent_result, plan_result)
            c = content_fn(f"✅ **您已确认**，即将执行以下操作：\n\n{confirmation_msg}")
            task_logger.emit_sse_event(c); yield c

            s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
            task_logger.emit_sse_event(s4a); yield s4a
            exec_result = await self._step4.execute(plan_result, task_id, None)
            s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed",
                               summary=exec_result.summary, details=exec_result.to_s4_details())
            task_logger.emit_sse_event(s4c); yield s4c
            s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
            task_logger.emit_sse_event(s5a); yield s5a
            response_result = self._step5.generate(exec_result, plan_result, intent_result, ontology_result)
            s5c = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="completed",
                               summary="回复已生成", details=response_result.to_s5_details(),
                               suggested_actions=[a.to_dict() for a in response_result.next_actions])
            task_logger.emit_sse_event(s5c); yield s5c
            c = content_fn(response_result.text)
            task_logger.emit_sse_event(c); yield c
        else:
            # Unknown phase — fallback to simple execution
            print(f"[FiveStepPipeline][WARN] run_with_confirmation: 未知phase={hitl_phase}")
            if plan_result:
                s4a = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="active")
                task_logger.emit_sse_event(s4a); yield s4a
                exec_result = await self._step4.execute(plan_result, task_id, None)
                s4c = step_update_fn(step=4, step_name=self.STEP_NAMES[4], status="completed",
                                   summary=exec_result.summary, details=exec_result.to_s4_details())
                task_logger.emit_sse_event(s4c); yield s4c
                s5a = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="active")
                task_logger.emit_sse_event(s5a); yield s5a
                response_result = self._step5.generate(exec_result, plan_result, intent_result, ontology_result)
                s5c = step_update_fn(step=5, step_name=self.STEP_NAMES[5], status="completed",
                                   summary="回复已生成", details=response_result.to_s5_details())
                task_logger.emit_sse_event(s5c); yield s5c
                c = content_fn(response_result.text)
                task_logger.emit_sse_event(c); yield c
            else:
                c = content_fn("操作已完成。")
                task_logger.emit_sse_event(c); yield c

        # Clear the HITL session after successful completion
        clear_hitl_session(task_id)
        d = done_fn()
        task_logger.emit_sse_event(d)
        task_logger.close(final_status="completed")
        yield d


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

    def _build_confirmation_message(self, intent_result, plan_result) -> str:
        """Build a detailed confirmation message for create operations.
        
        Shows:
        - What action will be performed
        - What data will be created
        - All filled slots with their values
        """
        lines = []
        
        # Get action label
        action_label = getattr(intent_result, 'intent_label', '创建记录')
        lines.append(f"即将执行：【{action_label}】")
        lines.append("")
        
        # Get semantic contract slots
        semantic_contract = getattr(plan_result, 'semantic_contract', None)
        if semantic_contract and semantic_contract.slots:
            lines.append("📝 **将要创建的数据：**")
            for slot_name, slot in semantic_contract.slots.items():
                display_value = slot.display_value if hasattr(slot, 'display_value') else str(slot)
                slot_label = self._get_slot_label(slot_name)
                lines.append(f"  • {slot_label}：{display_value}")
            lines.append("")
        
        # Show query conditions if any
        if plan_result.query_conditions:
            lines.append("🔍 **查询条件：**")
            for cond in plan_result.query_conditions:
                lines.append(f"  • {cond.label}：{cond.value}")
            lines.append("")
        
        lines.append("请确认以上信息是否正确，点击「确认执行」继续。")
        
        return "\n".join(lines)
    
    def _get_slot_label(self, slot_name: str) -> str:
        """Get human-readable label for a slot name."""
        labels = {
            "material": "物料信息",
            "material_id": "物料编码",
            "material_d": "物料描述",
            "quantity": "采购数量",
            "delivery_date": "需求日期",
            "apply_dep": "申请部门",
            "pr_type": "采购类型",
            "factory_id": "工厂",
            "company_id": "公司",
            "unit_id": "单位",
            "source_type": "来源类型",
            "material_category": "物料分类",
        }
        return labels.get(slot_name, slot_name)

    def _build_semantic_contract_with_slots(
        self, action: str, object_term: str, filled_slots: Dict[str, Any]
    ) -> SemanticContract:
        """Build a SemanticContract with filled slots for planner/executor to use."""
        slots = {}
        for key, value in filled_slots.items():
            slots[key] = SemanticContractSlot(
                name=key,
                display_value=str(value),
                source="user_filled",
            )
        return SemanticContract(
            object=object_term,
            object_label=object_term,
            action=action,
            action_label=action,
            confidence=1.0,
            slots=slots,
            alternatives=[],
            hitl_triggered=False,
            missing_info=[],
        )

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
