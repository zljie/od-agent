"""Task-level logging: one file per task with console output + SSE events."""

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import StringIO


class TaskLogger:
    """Per-task logger that captures console output and SSE events to a single file.
    
    Usage:
        logger = TaskLogger(task_id="TASK-20260603-ABC123")
        logger.info("Starting step 1")
        logger.emit_sse_event({"event": "step_update", "data": {...}})
        logger.close()
    """

    LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, task_id: str, user_input: str = ""):
        self.task_id = task_id
        self.user_input = user_input
        self.started_at = datetime.now().isoformat()
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Setup file path
        self.logs_dir = self.LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.logs_dir / f"{task_id}.log"
        
        # Create log file with header
        self._write_raw(f"# Task Log: {task_id}")
        self._write_raw(f"# Started: {self.started_at}")
        self._write_raw(f"# User Input: {user_input[:200]}{'...' if len(user_input) > 200 else ''}")
        self._write_raw("#" + "=" * 60)
        
        # Redirect stdout for this task
        self._original_stdout = sys.stdout
        self._capture_buffer = StringIO()
        self._capture_mode = False

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from text."""
        return self.ANSI_ESCAPE.sub("", text)

    def _write_raw(self, line: str) -> None:
        """Write raw line to log file (no prefix)."""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _format_line(self, level: str, source: str, message: str) -> str:
        """Format a log line with timestamp and source."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        clean_msg = self._strip_ansi(message)
        return f"[{ts}] [{level}] [{source}] {clean_msg}"

    def info(self, message: str, source: str = "TaskLogger") -> None:
        """Log an info message."""
        line = self._format_line("INFO", source, message)
        self._write_raw(line)

    def debug(self, message: str, source: str = "TaskLogger") -> None:
        """Log a debug message."""
        line = self._format_line("DEBUG", source, message)
        self._write_raw(line)

    def warning(self, message: str, source: str = "TaskLogger") -> None:
        """Log a warning message."""
        line = self._format_line("WARN", source, message)
        self._write_raw(line)

    def error(self, message: str, source: str = "TaskLogger") -> None:
        """Log an error message."""
        line = self._format_line("ERROR", source, message)
        self._write_raw(line)

    def emit_sse_event(self, event: Dict[str, Any]) -> None:
        """Record an SSE event (step_update, confirm_request, etc.)."""
        with self._lock:
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "sse_event",
                "event": event,
            })
            # Write to file
            try:
                event_str = json.dumps(event, ensure_ascii=False, default=str)
                self._write_raw(f"[SSE] {event_str}")
            except Exception:
                pass

    def emit_llm_call(self, step: str, prompt_len: int, response_len: int, elapsed_ms: float) -> None:
        """Record an LLM API call."""
        with self._lock:
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "llm_call",
                "step": step,
                "prompt_len": prompt_len,
                "response_len": response_len,
                "elapsed_ms": elapsed_ms,
            })
            self._write_raw(f"[LLM] step={step} prompt_len={prompt_len} response_len={response_len} elapsed_ms={elapsed_ms:.1f}")

    def emit_decision(self, layer: str, decision: str, score: float, reason: str = "") -> None:
        """Record a pipeline decision (HITL, execute, etc.)."""
        with self._lock:
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "decision",
                "layer": layer,
                "decision": decision,
                "score": score,
                "reason": reason,
            })
            self._write_raw(f"[DECISION] layer={layer} decision={decision} score={score:.4f} reason={reason}")

    def emit_hitl(self, question: str, options: List[Dict]) -> None:
        """Record a HITL (Human-In-The-Loop) trigger."""
        with self._lock:
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "type": "hitl",
                "question": question,
                "options": options,
            })
            self._write_raw(f"[HITL] question={question[:100]} options_count={len(options)}")

    def close(self, final_status: str = "completed") -> None:
        """Close the logger and write final summary."""
        ended_at = datetime.now().isoformat()
        duration = (datetime.fromisoformat(ended_at) - datetime.fromisoformat(self.started_at)).total_seconds()
        
        self._write_raw("#" + "=" * 60)
        self._write_raw(f"# Task Log Ended: {ended_at}")
        self._write_raw(f"# Duration: {duration:.2f}s")
        self._write_raw(f"# Final Status: {final_status}")
        self._write_raw(f"# Total Events: {len(self.events)}")
        self._write_raw("#" + "=" * 60)


# Global task logger registry (for cross-module access)
_task_loggers: Dict[str, TaskLogger] = {}
_logger_lock = threading.Lock()


def get_task_logger(task_id: str) -> Optional[TaskLogger]:
    """Get existing task logger by task_id."""
    with _logger_lock:
        return _task_loggers.get(task_id)


def create_task_logger(task_id: str, user_input: str = "") -> TaskLogger:
    """Create and register a new task logger."""
    with _logger_lock:
        logger = TaskLogger(task_id=task_id, user_input=user_input)
        _task_loggers[task_id] = logger
        return logger


def remove_task_logger(task_id: str) -> None:
    """Remove and close a task logger."""
    with _logger_lock:
        if task_id in _task_loggers:
            _task_loggers[task_id].close()
            del _task_loggers[task_id]

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class AuditEvent(str, Enum):
    """Standard audit event types."""

    REQUEST = "REQUEST"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    HITL_TRIGGER = "HITL_TRIGGER"
    ONTOLOGY_LOOKUP = "ONTOLOGY_LOOKUP"
    RULE_EVAL = "RULE_EVAL"
    CONFIRMATION = "CONFIRMATION"
    PREPROCESSING = "PREPROCESSING"
    EXECUTION = "EXECUTION"
    ERROR = "ERROR"

    # Phase 2b — intent recognition layer events
    LAYER_0_PREPROCESSING = "LAYER_0_PREPROCESSING"
    LAYER_1_LIGHT_REASONING = "LAYER_1_LIGHT_REASONING"
    LAYER_2_ONTOLOGY_MATCH = "LAYER_2_ONTOLOGY_MATCH"
    LAYER_3_CONFIDENCE_FUSION_AB = "LAYER_3_CONFIDENCE_FUSION_AB"
    LAYER_4_DEEP_REASONING = "LAYER_4_DEEP_REASONING"
    LAYER_5_CONFIDENCE_FUSION_ABC = "LAYER_5_CONFIDENCE_FUSION_ABC"
    LAYER_6_HITL = "LAYER_6_HITL"
    LAYER_7_DYNAMIC_RESOLUTION = "LAYER_7_DYNAMIC_RESOLUTION"

    # Phase 5 — context engineering layer events (PRD Section 23)
    CONTEXT_RETRIEVE = "CONTEXT_RETRIEVE"
    CONTEXT_SELECT = "CONTEXT_SELECT"
    CONTEXT_POLICY_CHECK = "CONTEXT_POLICY_CHECK"
    CONTEXT_ASSEMBLE = "CONTEXT_ASSEMBLE"
    CONTEXT_AUDIT = "CONTEXT_AUDIT"


@dataclass
class ExecutionAuditRecord:
    """Complete execution audit record per BeBISO specification."""

    # Core fields
    request_id: str = ""                           # 请求ID
    user_input: str = ""                          # 用户原始输入
    session_id: str = ""                           # 会话ID
    user_id: str = ""                            # 用户ID

    # Intent recognition
    intent: str = ""                              # 识别的意图
    intent_confidence: float = 0.0               # 意图置信度
    intent_decision: str = ""                     # 意图决策

    # Ontology resolution
    ontology_version: str = ""                     # 本体版本
    ontology_hit_objects: List[str] = field(default_factory=list)  # 命中的本体对象
    ontology_hit_attributes: List[str] = field(default_factory=list)  # 命中的属性
    ontology_hit_relationships: List[str] = field(default_factory=list)  # 命中的关系

    # Rule evaluation
    rules_triggered: List[str] = field(default_factory=list)   # 触发的规则
    rules_blocked: List[str] = field(default_factory=list)    # 阻断的规则

    # Connector execution
    connectors_called: List[str] = field(default_factory=list)  # 调用的连接器
    connector_results: Dict[str, Any] = field(default_factory=dict)  # 连接器结果

    # User confirmation
    user_confirmation: str = ""                    # 用户确认状态 (confirmed/rejected/pending)
    confirmation_timestamp: str = ""              # 确认时间

    # Execution result
    execution_action: str = ""                   # 执行的动作
    execution_result: str = ""                   # 执行结果 (success/blocked/failed)
    error_reason: str = ""                       # 错误原因
    error_type: str = ""                         # 错误类型

    # Metadata
    timestamp: str = ""                          # 时间戳
    duration_ms: float = 0.0                    # 执行耗时

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "user_input": self.user_input,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_decision": self.intent_decision,
            "ontology_version": self.ontology_version,
            "ontology_hit_objects": self.ontology_hit_objects,
            "ontology_hit_attributes": self.ontology_hit_attributes,
            "ontology_hit_relationships": self.ontology_hit_relationships,
            "rules_triggered": self.rules_triggered,
            "rules_blocked": self.rules_blocked,
            "connectors_called": self.connectors_called,
            "connector_results": self.connector_results,
            "user_confirmation": self.user_confirmation,
            "confirmation_timestamp": self.confirmation_timestamp,
            "execution_action": self.execution_action,
            "execution_result": self.execution_result,
            "error_reason": self.error_reason,
            "error_type": self.error_type,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class ContextAuditRecord:
    """Context audit record per PRD Section 23.

    Captures the context retrieval, selection, policy check, and assembly
    decisions for a single task/purpose.
    """

    context_id: str = ""
    task_id: str = ""
    purpose: str = ""  # intent_recognition|ontology_grounding|task_planning|execution|response
    ontology_version: str = ""

    # Timestamps
    generated_at: str = ""

    # Sources
    included_sources: List[str] = field(default_factory=list)  # What context was used
    excluded_sources: List[Dict[str, str]] = field(
        default_factory=list
    )  # What was excluded and why

    # Token budget
    token_budget: Dict[str, Any] = field(default_factory=dict)  # {max_tokens, estimated_tokens, allocation}

    # Security
    redacted_fields: List[str] = field(default_factory=list)
    permission_checked: bool = False

    # Trust and freshness summary
    trust_breakdown: Dict[str, int] = field(default_factory=dict)
    freshness_breakdown: Dict[str, int] = field(default_factory=dict)

    # Policy check results
    policy_summary: Dict[str, Any] = field(default_factory=dict)

    # Context assembly metadata
    retrieval_result: Dict[str, Any] = field(default_factory=dict)
    selection_result: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            self.context_id = uuid.uuid4().hex[:16].upper() if "uuid" in dir() else datetime.now().strftime("%Y%m%d%H%M%S")
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "context_id": self.context_id,
            "task_id": self.task_id,
            "purpose": self.purpose,
            "ontology_version": self.ontology_version,
            "generated_at": self.generated_at,
            "included_sources": self.included_sources,
            "excluded_sources": self.excluded_sources,
            "token_budget": self.token_budget,
            "redacted_fields": self.redacted_fields,
            "permission_checked": self.permission_checked,
            "trust_breakdown": self.trust_breakdown,
            "freshness_breakdown": self.freshness_breakdown,
            "policy_summary": self.policy_summary,
            "retrieval_result": self.retrieval_result,
            "selection_result": self.selection_result,
            "metadata": self.metadata,
        }


def create_context_audit_record(
    task_id: str,
    purpose: str,
    included_sources: List[str],
    excluded_sources: List[Dict[str, str]],
    token_info: Dict[str, Any],
    redacted_fields: Optional[List[str]] = None,
    ontology_version: str = "",
    trust_breakdown: Optional[Dict[str, int]] = None,
    freshness_breakdown: Optional[Dict[str, int]] = None,
    policy_summary: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    selection_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextAuditRecord:
    """Factory to create a ContextAuditRecord with all fields populated."""
    import uuid
    return ContextAuditRecord(
        context_id=uuid.uuid4().hex[:16].upper(),
        task_id=task_id,
        purpose=purpose,
        ontology_version=ontology_version,
        generated_at=datetime.now().isoformat(),
        included_sources=included_sources,
        excluded_sources=excluded_sources,
        token_budget=token_info,
        redacted_fields=redacted_fields or [],
        permission_checked=True,
        trust_breakdown=trust_breakdown or {},
        freshness_breakdown=freshness_breakdown or {},
        policy_summary=policy_summary or {},
        retrieval_result=retrieval_result or {},
        selection_result=selection_result or {},
        metadata=metadata or {},
    )


def log_context_audit(
    task_id: str,
    purpose: str,
    included_sources: List[str],
    excluded_sources: List[Dict[str, str]],
    token_info: Dict[str, Any],
    redacted_fields: Optional[List[str]] = None,
    ontology_version: str = "",
    trust_breakdown: Optional[Dict[str, int]] = None,
    freshness_breakdown: Optional[Dict[str, int]] = None,
    policy_summary: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    selection_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    audit_sink: Optional["AuditSink"] = None,
) -> None:
    """Create and write a context audit record to the configured sink.

    If no audit_sink is provided, uses the default from ``create_audit_sink()``.
    """
    record = create_context_audit_record(
        task_id=task_id,
        purpose=purpose,
        included_sources=included_sources,
        excluded_sources=excluded_sources,
        token_info=token_info,
        redacted_fields=redacted_fields,
        ontology_version=ontology_version,
        trust_breakdown=trust_breakdown,
        freshness_breakdown=freshness_breakdown,
        policy_summary=policy_summary,
        retrieval_result=retrieval_result,
        selection_result=selection_result,
        metadata=metadata,
    )
    sink = audit_sink or create_audit_sink()
    sink.write(AuditEvent.CONTEXT_AUDIT, record.to_dict())


class AuditSink(ABC):
    """SPI interface for audit event sinks."""

    @abstractmethod
    def write(self, event_type: AuditEvent, detail: Dict[str, Any]) -> None:
        """Write an audit event with structured detail."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Flush and close any resources."""
        pass

    def record_request(
        self,
        user_input: str,
        intent_type: str,
        confidence: float,
        decision: str,
        rag_hit_count: int = 0,
        temporal_anchors: Optional[List[str]] = None,
        preprocessing_corrections: Optional[List[Dict]] = None,
    ) -> None:
        """Convenience method to record a REQUEST event."""
        self.write(AuditEvent.REQUEST, {
            "user_input": user_input,
            "intent_type": intent_type,
            "confidence": confidence,
            "decision": decision,
            "rag_hit_count": rag_hit_count,
            "temporal_anchors": temporal_anchors or [],
            "preprocessing_corrections": preprocessing_corrections or [],
            "timestamp": datetime.now().isoformat(),
        })

    def record_tool_call(
        self,
        skill_id: str,
        action_id: str,
        params: Dict[str, Any],
        node_id: str = "",
    ) -> None:
        """Convenience method to record a TOOL_CALL event."""
        self.write(AuditEvent.TOOL_CALL, {
            "skill_id": skill_id,
            "action_id": action_id,
            "params": params,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
        })

    def record_tool_result(
        self,
        skill_id: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        node_id: str = "",
    ) -> None:
        """Convenience method to record a TOOL_RESULT event."""
        self.write(AuditEvent.TOOL_RESULT, {
            "skill_id": skill_id,
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
        })

    def record_ontology_lookup(
        self,
        objects: List[str],
        attributes: List[str],
        relationships: List[str],
        ontology_version: str = "",
    ) -> None:
        """Convenience method to record ONTOLOGY_LOOKUP event."""
        self.write(AuditEvent.ONTOLOGY_LOOKUP, {
            "objects": objects,
            "attributes": attributes,
            "relationships": relationships,
            "ontology_version": ontology_version,
            "timestamp": datetime.now().isoformat(),
        })

    def record_rule_eval(
        self,
        action: str,
        rules_triggered: List[str],
        rules_blocked: List[str],
        blocked_reason: Optional[str] = None,
    ) -> None:
        """Convenience method to record RULE_EVAL event."""
        self.write(AuditEvent.RULE_EVAL, {
            "action": action,
            "rules_triggered": rules_triggered,
            "rules_blocked": rules_blocked,
            "blocked_reason": blocked_reason,
            "timestamp": datetime.now().isoformat(),
        })

    def record_confirmation(
        self,
        action: str,
        confirmed: bool,
        user_input: str = "",
    ) -> None:
        """Convenience method to record CONFIRMATION event."""
        self.write(AuditEvent.CONFIRMATION, {
            "action": action,
            "confirmed": confirmed,
            "user_input": user_input,
            "timestamp": datetime.now().isoformat(),
        })

    def record_execution(
        self,
        record: ExecutionAuditRecord,
    ) -> None:
        """Convenience method to record a full EXECUTION event."""
        self.write(AuditEvent.EXECUTION, record.to_dict())


class Slf4jAuditSink(AuditSink):
    """Writes audit events to the standard logger (INFO level)."""

    def __init__(self, logger_name: str = "od_agent.audit"):
        self._logger = logging.getLogger(logger_name)

    def write(self, event_type: AuditEvent, detail: Dict[str, Any]) -> None:
        self._logger.info(
            "[AUDIT] type=%s detail=%s",
            event_type.value,
            json.dumps(detail, ensure_ascii=False),
        )

    def close(self) -> None:
        pass


class StdoutAuditSink(AuditSink):
    """Writes audit events as formatted JSON lines to stdout."""

    def write(self, event_type: AuditEvent, detail: Dict[str, Any]) -> None:
        line = json.dumps(
            {"event": event_type.value, **detail},
            ensure_ascii=False,
            default=str,
        )
        print(f"[AUDIT] {line}")

    def close(self) -> None:
        pass


class JsonFileAuditSink(AuditSink):
    """Appends audit events as JSON lines to a rolling file."""

    def __init__(
        self,
        path: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        import logging.handlers

        if path is None:
            path = os.environ.get(
                "AUDIT_LOG_PATH",
                str(Path(__file__).parent.parent.parent / "logs" / "audit.log"),
            )

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("od_agent.file_audit")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        handler = logging.handlers.RotatingFileHandler(
            str(self._path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._handler = handler

    def write(self, event_type: AuditEvent, detail: Dict[str, Any]) -> None:
        self._logger.info(
            json.dumps({"event": event_type.value, **detail}, ensure_ascii=False, default=str)
        )

    def close(self) -> None:
        self._handler.close()


class NoOpAuditSink(AuditSink):
    """Disabled audit sink (drop all events)."""

    def write(self, event_type: AuditEvent, detail: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def load_audit_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load audit configuration from JSON file."""
    import json
    import os

    if path is None:
        path = os.environ.get(
            "AUDIT_CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "config" / "audit_config.json"),
        )

    defaults: Dict[str, Any] = {
        "enabled": True,
        "sink": "stdout",
        "events": [
            "REQUEST",
            "TOOL_CALL",
            "TOOL_RESULT",
            "HITL_TRIGGER",
            "ONTOLOGY_LOOKUP",
            "RULE_EVAL",
            "CONFIRMATION",
            "EXECUTION",
            "ERROR",
        ],
    }

    if not os.path.exists(path):
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            return {**defaults, **json.load(f)}
    except Exception:
        return defaults


def create_audit_sink(config: Optional[Dict[str, Any]] = None) -> AuditSink:
    """Factory: create an AuditSink from a config dict."""
    if config is None:
        config = load_audit_config()

    if not config.get("enabled", True):
        return NoOpAuditSink()

    sink_type = config.get("sink", "stdout")
    if sink_type == "slf4j":
        return Slf4jAuditSink()
    elif sink_type == "json_file":
        return JsonFileAuditSink(
            path=config.get("path"),
            max_bytes=config.get("max_bytes", 10 * 1024 * 1024),
            backup_count=config.get("backup_count", 5),
        )
    elif sink_type == "stdout":
        return StdoutAuditSink()
    else:
        return StdoutAuditSink()
