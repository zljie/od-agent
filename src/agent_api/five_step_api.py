"""
Five-Step Runtime API
====================
REST API endpoints for the Five-Step runtime per PRD Frontend Architecture.

API Endpoints (per PRD Section 13):
------------------------------------

## Chat API
- POST /api/chat/sessions          - Create new chat session
- GET  /api/chat/sessions           - List chat sessions
- GET  /api/chat/sessions/{id}      - Get chat session
- POST /api/chat/sessions/{id}/messages  - Send message
- POST /api/chat/stream             - SSE stream for chat
- POST /api/chat/messages/{id}/retry  - Retry message

## Runtime API
- GET  /api/runtime/tasks/{taskId}           - Get task status
- POST /api/runtime/tasks/{taskId}/cancel     - Cancel task
- GET  /api/runtime/tasks/{taskId}/events    - Get task events
- GET  /api/runtime/tasks/{taskId}/events/stream  - SSE stream for events

## HITL API
- GET  /api/runtime/hitl/tasks/{id}           - Get HITL task
- POST /api/runtime/hitl/tasks/{id}/actions   - Submit HITL action

## Trace API
- GET  /api/traces/{traceId}           - Get trace
- GET  /api/traces/{traceId}/steps     - Get trace steps

## Agent Config API
- GET  /api/agents              - List agents
- POST /api/agents              - Create agent
- GET  /api/agents/{id}        - Get agent
- PUT  /api/agents/{id}        - Update agent
- POST /api/agents/{id}/resources  - Bind resources
- POST /api/agents/{id}/runtime-config  - Configure runtime
- POST /api/agents/{id}/chatbot-config  - Configure chatbot
- POST /api/agents/{id}/test    - Test agent
- POST /api/agents/{id}/publish - Publish agent
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime


# ─── Request/Response Models ────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    """Create chat session request."""
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class ChatSession(BaseModel):
    """Chat session model."""
    id: str
    agent_id: str
    user_id: Optional[str]
    created_at: str
    updated_at: str
    message_count: int = 0
    metadata: Optional[dict] = None


class MessageSend(BaseModel):
    """Send message request."""
    content: str
    session_id: Optional[str] = None
    attachments: Optional[list] = None
    context: Optional[dict] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    id: str
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    type: str  # "text" | "markdown" | "table" | "form" | "chart" | "approval_card"
    created_at: str
    metadata: Optional[dict] = None


class HITLActionRequest(BaseModel):
    """HITL action request (per PRD Section 8.3)."""
    action: str  # "approve" | "reject" | "edit"
    comment: Optional[str] = None
    form_data: Optional[dict] = None
    filled_slots: Optional[dict] = None  # For slot-fill protocol


class HITLActionResponse(BaseModel):
    """HITL action response."""
    status: str  # "accepted" | "rejected" | "modified"
    task_id: str
    resume: bool
    message: Optional[str] = None


class TaskStatus(BaseModel):
    """Runtime task status."""
    task_id: str
    session_id: Optional[str]
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    current_step: int  # 1-5
    created_at: str
    updated_at: str
    frames: Optional[dict] = None  # All 5 frames if completed


class TraceInfo(BaseModel):
    """Trace information."""
    trace_id: str
    task_id: str
    session_id: Optional[str]
    steps: list
    created_at: str


# ─── Router ───────────────────────────────────────────────────────────────────

# Create the Five-Step API router
router = APIRouter(prefix="/api", tags=["Five-Step Runtime"])


# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ─── API Documentation Note ────────────────────────────────────────────────────

"""
NOTE: The actual endpoint implementations are in src/app.py.

This module documents the Five-Step Runtime API per PRD Frontend Architecture.
The endpoints should be implemented as follows:

## Chat Endpoints (to be added to app.py)

### POST /api/chat/sessions
Create a new chat session.

Request:
{
    "agent_id": "purchase-agent",
    "user_id": "user123"
}

Response:
{
    "id": "session_xxx",
    "agent_id": "purchase-agent",
    "user_id": "user123",
    "created_at": "2026-06-04T10:00:00Z"
}

### POST /api/chat/stream
SSE stream for chat (main interaction endpoint).

Request (query params):
- session_id: str (optional)
- enable_five_step: bool (default: true)

Response: SSE events per src/sse_stream.py

### GET /api/chat/sessions/{session_id}/messages
Get messages in a session.

Response:
{
    "messages": [
        {
            "id": "msg_xxx",
            "role": "user",
            "content": "帮我查一下采购需求",
            "created_at": "2026-06-04T10:00:00Z"
        },
        {
            "id": "msg_yyy",
            "role": "assistant",
            "type": "markdown",
            "content": "已查询到12条采购需求...",
            "created_at": "2026-06-04T10:00:01Z"
        }
    ]
}

## Runtime Endpoints (to be added)

### GET /api/runtime/tasks/{task_id}
Get task status with all frames.

Response:
{
    "task_id": "task_xxx",
    "session_id": "session_xxx",
    "status": "completed",
    "current_step": 5,
    "frames": {
        "input": {...},
        "semantic": {...},
        "decision": {...},
        "execution": {...},
        "response": {...}
    }
}

### GET /api/runtime/tasks/{task_id}/events/stream
SSE stream for task events.

## HITL Endpoints (already partially implemented)

### POST /api/runtime/hitl/tasks/{task_id}/actions
Submit HITL action (confirm/cancel/edit).

Request:
{
    "action": "approve",
    "comment": "同意创建询价单",
    "filled_slots": {
        "material": "M-001",
        "quantity": 100
    }
}

Response:
{
    "status": "accepted",
    "task_id": "task_xxx",
    "resume": true
}

## Trace Endpoints

### GET /api/traces/{trace_id}
Get full trace.

### GET /api/traces/{trace_id}/steps
Get trace steps.
"""
