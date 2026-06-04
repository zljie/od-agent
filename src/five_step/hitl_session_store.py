"""HITL Session Store.

Phase 4.1: Global singleton store for HITL session context.

CRITICAL: The FiveStepPipeline creates a NEW INSTANCE on every HTTP request.
Therefore, instance fields like `self._task_context` are LOST between requests.

This store solves the problem by persisting the full HITL context (intent_result,
ontology_result, plan_result, composite_result, etc.) keyed by task_id.

Flow:
1. run() emits HITL → saves context to HITLSessionStore[task_id]
2. Frontend submits confirmation → POST /chat/confirm with task_id
3. run_with_confirmation() → loads context from HITLSessionStore[task_id]
4. Continues execution with the SAME context
5. On done/error → clears HITLSessionStore[task_id]
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Default TTL: 30 minutes
DEFAULT_TTL_SECONDS = 30 * 60


@dataclass
class HITLSessionContext:
    """Full context needed to resume a HITL task after user confirmation."""

    task_id: str
    session_id: str = ""

    # Full pipeline state from run() at the point of HITL emission
    intent_result: Optional[Any] = None
    ontology_result: Optional[Any] = None
    plan_result: Optional[Any] = None
    composite_result: Optional[Any] = None
    hitl_request: Optional[Any] = None
    pending_hitl_task: Optional[Dict[str, Any]] = None

    # Semantic contract from Step 1
    semantic_contract: Optional[Any] = None

    # Original user input
    original_user_input: str = ""

    # Pipeline parameters
    agent: Optional[Any] = None  # Reference to agent

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    expires_at: str = ""

    # Phase info: which phase triggered HITL
    # "slot_fill" = missing required slots
    # "ontology_confirm" = LLM inferred object
    # "create_confirm" = create operation requires confirmation
    hitl_phase: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.expires_at:
            self.expires_at = (datetime.now() + timedelta(seconds=DEFAULT_TTL_SECONDS)).isoformat()

    def is_expired(self) -> bool:
        try:
            return datetime.now() > datetime.fromisoformat(self.expires_at)
        except (ValueError, TypeError):
            return True

    def touch(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()
        # Extend expiry on access
        self.expires_at = (datetime.now() + timedelta(seconds=DEFAULT_TTL_SECONDS)).isoformat()


class HITLSessionStore:
    """Thread-safe global store for HITL session contexts.

    Singleton — single instance across all pipeline instances and HTTP requests.
    """

    _instance: Optional["HITLSessionStore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        if self._initialized:
            return
        self._initialized = True
        self._ttl = ttl_seconds
        self._sessions: Dict[str, HITLSessionContext] = {}

    def save(self, task_id: str, ctx: HITLSessionContext) -> None:
        """Save or update a HITL session context."""
        with self._lock:
            self._sessions[task_id] = ctx
            print(f"[HITLSessionStore] Saved session {task_id} | phase={ctx.hitl_phase} | expires={ctx.expires_at}")

    def load(self, task_id: str) -> Optional[HITLSessionContext]:
        """Load a HITL session context. Returns None if not found or expired."""
        with self._lock:
            ctx = self._sessions.get(task_id)
            if ctx is None:
                print(f"[HITLSessionStore] Session {task_id} not found")
                return None
            if ctx.is_expired():
                print(f"[HITLSessionStore] Session {task_id} expired, removing")
                del self._sessions[task_id]
                return None
            ctx.touch()
            print(f"[HITLSessionStore] Loaded session {task_id} | phase={ctx.hitl_phase}")
            return ctx

    def clear(self, task_id: str) -> None:
        """Remove a HITL session context after completion."""
        with self._lock:
            if task_id in self._sessions:
                del self._sessions[task_id]
                print(f"[HITLSessionStore] Cleared session {task_id}")

    def cleanup(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [tid for tid, ctx in self._sessions.items() if ctx.is_expired()]
            for tid in expired:
                del self._sessions[tid]
            if expired:
                print(f"[HITLSessionStore] Cleaned up {len(expired)} expired sessions")
            return len(expired)


# ─── Convenience functions ────────────────────────────────────────────────────

_store: Optional[HITLSessionStore] = None


def get_hitl_session_store() -> HITLSessionStore:
    global _store
    if _store is None:
        _store = HITLSessionStore()
    return _store


def save_hitl_session(
    task_id: str,
    session_id: str,
    intent_result=None,
    ontology_result=None,
    plan_result=None,
    composite_result=None,
    hitl_request=None,
    pending_hitl_task=None,
    semantic_contract=None,
    original_user_input: str = "",
    hitl_phase: str = "",
) -> HITLSessionContext:
    """Convenience: save HITL session context."""
    ctx = HITLSessionContext(
        task_id=task_id,
        session_id=session_id,
        intent_result=intent_result,
        ontology_result=ontology_result,
        plan_result=plan_result,
        composite_result=composite_result,
        hitl_request=hitl_request,
        pending_hitl_task=pending_hitl_task,
        semantic_contract=semantic_contract,
        original_user_input=original_user_input,
        hitl_phase=hitl_phase,
    )
    get_hitl_session_store().save(task_id, ctx)
    return ctx


def load_hitl_session(task_id: str) -> Optional[HITLSessionContext]:
    """Convenience: load HITL session context."""
    return get_hitl_session_store().load(task_id)


def clear_hitl_session(task_id: str) -> None:
    """Convenience: clear HITL session context."""
    get_hitl_session_store().clear(task_id)
