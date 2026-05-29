"""SSE event definitions and stream utilities.

Implements the SSE streaming protocol from docs/SSE流式响应规范.md:
- think / think_done  : model reasoning (non-required)
- content / done      : final response text
- tool_call / tool_result : skill/MCP/RAG invocations

Event format (per spec):
    event: <type>
    data: <json_payload>

    <double newline>

All events are yielded as dicts ready for sse_starlette.EventSourceResponse.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SSEEventType(str, Enum):
    THINK = "think"
    THINK_DONE = "think_done"
    CONTENT = "content"
    DONE = "done"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# ─── Payload dataclasses ───────────────────────────────────────────────────────

@dataclass
class ToolCallPayload:
    type: str          # "skill" | "mcp" | "rag"
    name: str
    input: Dict[str, Any]
    id: str
    description: Optional[str] = None

    def to_event(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "id": self.id,
        }
        if self.description:
            data["description"] = self.description
        return {"event": SSEEventType.TOOL_CALL.value, "data": json.dumps(data, ensure_ascii=False)}


@dataclass
class ToolResultPayload:
    id: str
    name: str
    status: str        # "success" | "error" | "pending"
    output: Any = None
    error: Optional[str] = None

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.TOOL_RESULT.value,
            "data": json.dumps(
                {
                    "id": self.id,
                    "name": self.name,
                    "status": self.status,
                    "output": self.output,
                    "error": self.error,
                },
                ensure_ascii=False,
            ),
        }


@dataclass
class ThinkPayload:
    content: str

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.THINK.value,
            "data": json.dumps({"content": self.content}, ensure_ascii=False),
        }


@dataclass
class ContentPayload:
    content: str

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.CONTENT.value,
            "data": json.dumps({"content": self.content}, ensure_ascii=False),
        }


@dataclass
class DonePayload:
    """Sentinel event sent when the full stream is complete."""

    def to_event(self) -> Dict[str, Any]:
        return {"event": SSEEventType.DONE.value, "data": "[DONE]"}


@dataclass
class ThinkDonePayload:
    """Marks the end of the think phase."""

    def to_event(self) -> Dict[str, Any]:
        return {
            "event": SSEEventType.THINK_DONE.value,
            "data": json.dumps({"status": "done"}, ensure_ascii=False),
        }


# ─── Stream event union type ───────────────────────────────────────────────────

SSEEvent = (
    ToolCallPayload
    | ToolResultPayload
    | ThinkPayload
    | ContentPayload
    | DonePayload
    | ThinkDonePayload
)


# ─── ID generator ─────────────────────────────────────────────────────────────

_tool_call_counter: int = 0


def new_tool_id() -> str:
    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_{_tool_call_counter:03d}"


def reset_tool_counter() -> None:
    global _tool_call_counter
    _tool_call_counter = 0


# ─── Builder helpers ───────────────────────────────────────────────────────────

def think(content: str) -> Dict[str, Any]:
    return ThinkPayload(content=content).to_event()


def think_done() -> Dict[str, Any]:
    return ThinkDonePayload().to_event()


def content(text: str) -> Dict[str, Any]:
    return ContentPayload(content=text).to_event()


def done() -> Dict[str, Any]:
    return DonePayload().to_event()


def tool_call(
    name: str,
    input_data: Dict[str, Any],
    type: str = "skill",
    tool_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    payload = ToolCallPayload(
        type=type,
        name=name,
        input=input_data,
        id=tool_id or new_tool_id(),
        description=description,
    )
    return payload.to_event()


def tool_result(
    tool_id: str,
    name: str,
    status: str,
    output: Any = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    payload = ToolResultPayload(
        id=tool_id,
        name=name,
        status=status,
        output=output,
        error=error,
    )
    return payload.to_event()
