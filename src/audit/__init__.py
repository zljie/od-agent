"""AuditSink: structured event logging for the agent pipeline.

Records key lifecycle events:
- REQUEST:     Every incoming user message (intent type, confidence, decision, RAG hits)
- TOOL_CALL:   Each skill/tool invocation (skill_id, action_id, params)
- TOOL_RESULT: Each tool result (success, duration_ms, error)
- HITL_TRIGGER: When HITL is triggered (action, reason)
- ERROR:       Any exception in the pipeline

Supports pluggable sinks: Slf4j (default), Stdout, JSON file.
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


class AuditEvent(str, Enum):
    """Standard audit event types."""

    REQUEST = "REQUEST"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    HITL_TRIGGER = "HITL_TRIGGER"
    ONTOLOGY_LOOKUP = "ONTOLOGY_LOOKUP"
    PREPROCESSING = "PREPROCESSING"
    ERROR = "ERROR"


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
