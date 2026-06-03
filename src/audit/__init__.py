"""AuditSink: structured event logging for the agent pipeline.

Records key lifecycle events:
- REQUEST:     Every incoming user message (intent type, confidence, decision, RAG hits)
- TOOL_CALL:   Each skill/tool invocation (skill_id, action_id, params)
- TOOL_RESULT: Each tool result (success, duration_ms, error)
- HITL_TRIGGER: When HITL is triggered (action, reason)
- ONTOLOGY_LOOKUP: When ontology objects are resolved
- RULE_EVAL:   When rules are evaluated (triggered rules)
- CONFIRMATION: When user confirms or rejects an action
- ERROR:       Any exception in the pipeline
- EXECUTION:   Full execution trace with all audit fields

Supports pluggable sinks: Slf4j (default), Stdout, JSON file.

Audit fields according to BeBISO specification:
- user_input: User's original input
- intent: Recognized intent
- ontology_version: Ontology version used
- ontology_hit_objects: Objects resolved from ontology
- rules_triggered: Rules that were triggered during execution
- connectors_called: Connectors invoked
- user_confirmation: User confirmation status
- execution_action: Action being executed
- execution_result: Execution result (success/blocked/failed)
- error_reason: Error reason if failed
"""

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
