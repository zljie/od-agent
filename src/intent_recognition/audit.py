"""
Intent Recognition Audit Module
==============================
Phase 4 of the BeBISO intent recognition upgrade.

Extends the core ``AuditEvent`` enum with layer-specific events and provides
the ``IntentRecognitionAuditRecord`` and ``IntentRecognitionAuditLogger`` for
capturing the full intent recognition pipeline execution.

Usage::

    from src.intent_recognition.audit import IntentRecognitionAuditLogger

    logger = IntentRecognitionAuditLogger()
    logger.log_layer("LAYER_1_LIGHT_REASONING", task_id, llm_light.to_dict())
    logger.log_full_recognition(record)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.audit import AuditEvent, AuditSink, create_audit_sink


# ---------------------------------------------------------------------------
# Layer-specific audit events
# ---------------------------------------------------------------------------


class IntentLayerEvent(str, Enum):
    """Layer-specific audit events for the intent recognition pipeline."""

    LAYER_0_PREPROCESSING = "LAYER_0_PREPROCESSING"
    LAYER_1_LIGHT_REASONING = "LAYER_1_LIGHT_REASONING"
    LAYER_2_ONTOLOGY_MATCH = "LAYER_2_ONTOLOGY_MATCH"
    LAYER_3_CONFIDENCE_FUSION_AB = "LAYER_3_CONFIDENCE_FUSION_AB"
    LAYER_4_DEEP_REASONING = "LAYER_4_DEEP_REASONING"
    LAYER_5_CONFIDENCE_FUSION_ABC = "LAYER_5_CONFIDENCE_FUSION_ABC"
    LAYER_6_HITL = "LAYER_6_HITL"
    LAYER_7_DYNAMIC_RESOLUTION = "LAYER_7_DYNAMIC_RESOLUTION"


# ---------------------------------------------------------------------------
# Full composite audit record
# ---------------------------------------------------------------------------


@dataclass
class IntentRecognitionAuditRecord:
    """Complete audit record for a full intent recognition pipeline run.

    Captures the raw input, all intermediate layer results, the final
    decision, and execution metadata per the BeBISO audit specification.
    """

    task_id: str
    raw_input: str
    normalized_input: str
    preprocessing_result: Dict[str, Any] = field(default_factory=dict)
    llm_light_result: Dict[str, Any] = field(default_factory=dict)
    ontology_match_result: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)
    deep_reasoning_result: Dict[str, Any] = field(default_factory=dict)
    hitl_request: Dict[str, Any] = field(default_factory=dict)
    dynamic_resolution_result: Dict[str, Any] = field(default_factory=dict)
    final_intent: Dict[str, Any] = field(default_factory=dict)
    ontology_version: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12].upper()
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record to a dictionary for JSON logging."""
        return {
            "task_id": self.task_id,
            "raw_input": self.raw_input,
            "normalized_input": self.normalized_input,
            "preprocessing_result": self.preprocessing_result,
            "llm_light_result": self.llm_light_result,
            "ontology_match_result": self.ontology_match_result,
            "confidence": self.confidence,
            "deep_reasoning_result": self.deep_reasoning_result,
            "hitl_request": self.hitl_request,
            "dynamic_resolution_result": self.dynamic_resolution_result,
            "final_intent": self.final_intent,
            "ontology_version": self.ontology_version,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


class IntentRecognitionAuditLogger:
    """Logs intent recognition events to an ``AuditSink``.

    Parameters
    ----------
    audit_sink:
        The sink to write events to. Defaults to ``create_audit_sink()``.
    """

    def __init__(self, audit_sink: Optional[AuditSink] = None):
        self._sink = audit_sink or create_audit_sink()

    @property
    def sink(self) -> AuditSink:
        """The underlying audit sink."""
        return self._sink

    def log_layer(
        self,
        layer: str,
        task_id: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single layer's result to the audit sink.

        Parameters
        ----------
        layer:
            One of the ``IntentLayerEvent`` values (or its string value).
        task_id:
            Unique identifier for the intent recognition task.
        result:
            The layer's output. Will be converted to dict via ``to_dict()``
            if available, otherwise used as-is.
        metadata:
            Optional additional fields to include in the log entry.
        """
        # Convert result to dict if it has a to_dict method
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"value": str(result)}

        detail: Dict[str, Any] = {
            "task_id": task_id,
            "layer": layer,
            "result": result_dict,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            detail["metadata"] = metadata

        # Map to a core AuditEvent for the sink
        event = self._layer_to_audit_event(layer)
        self._sink.write(event, detail)

    def log_full_recognition(
        self,
        record: IntentRecognitionAuditRecord,
    ) -> None:
        """Log the complete composite intent recognition result.

        Parameters
        ----------
        record:
            The full ``IntentRecognitionAuditRecord`` from the pipeline.
        """
        self._sink.write(AuditEvent.EXECUTION, record.to_dict())

    def log_hitl_trigger(
        self,
        task_id: str,
        hitl_request: Any,
        reason: str = "",
    ) -> None:
        """Log a HITL trigger event.

        Parameters
        ----------
        task_id:
            Unique identifier for the intent recognition task.
        hitl_request:
            The ``HITLRequest`` that was generated.
        reason:
            Human-readable reason for triggering HITL.
        """
        if hasattr(hitl_request, "to_dict"):
            request_dict = hitl_request.to_dict()
        elif isinstance(hitl_request, dict):
            request_dict = hitl_request
        else:
            request_dict = {}

        self._sink.write(AuditEvent.HITL_TRIGGER, {
            "task_id": task_id,
            "reason": reason,
            "hitl_request": request_dict,
            "timestamp": datetime.now().isoformat(),
        })

    def log_ontology_lookup(
        self,
        task_id: str,
        objects: List[str],
        attributes: List[str],
        relationships: List[str],
        ontology_version: str = "",
    ) -> None:
        """Log an ontology lookup event.

        Delegates to the sink's ``record_ontology_lookup`` convenience method.
        """
        self._sink.record_ontology_lookup(
            objects=objects,
            attributes=attributes,
            relationships=relationships,
            ontology_version=ontology_version,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_to_audit_event(self, layer: str) -> AuditEvent:
        """Map a layer identifier to the corresponding ``AuditEvent``."""
        mapping = {
            IntentLayerEvent.LAYER_0_PREPROCESSING.value: AuditEvent.PREPROCESSING,
            IntentLayerEvent.LAYER_1_LIGHT_REASONING.value: AuditEvent.REQUEST,
            IntentLayerEvent.LAYER_2_ONTOLOGY_MATCH.value: AuditEvent.ONTOLOGY_LOOKUP,
            IntentLayerEvent.LAYER_3_CONFIDENCE_FUSION_AB.value: AuditEvent.REQUEST,
            IntentLayerEvent.LAYER_4_DEEP_REASONING.value: AuditEvent.REQUEST,
            IntentLayerEvent.LAYER_5_CONFIDENCE_FUSION_ABC.value: AuditEvent.REQUEST,
            IntentLayerEvent.LAYER_6_HITL.value: AuditEvent.HITL_TRIGGER,
            IntentLayerEvent.LAYER_7_DYNAMIC_RESOLUTION.value: AuditEvent.ONTOLOGY_LOOKUP,
        }
        return mapping.get(layer, AuditEvent.REQUEST)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


_default_logger: Optional[IntentRecognitionAuditLogger] = None


def get_intent_audit_logger() -> IntentRecognitionAuditLogger:
    """Return the global ``IntentRecognitionAuditLogger`` instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = IntentRecognitionAuditLogger()
    return _default_logger
