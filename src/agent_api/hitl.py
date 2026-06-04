"""
HITL Resume — POST /hitl/resume

Resumes a pending HITL task after user confirmation, cancellation, or slot editing.

Per the API Manifest MVP protocol, this endpoint handles:
- confirm  : User confirmed — proceed to invoke
- cancel   : User cancelled — abort and clean up
- edit    : User wants to edit parameters — return current params
"""

from typing import Any, Dict, Optional

from ..intent_recognition.pending_task_store import (
    PendingTask,
    delete_pending_task,
    get_task_store,
    save_pending_task,
)
from .dry_run import get_dry_run_store
from .invoke import invoke


def resume_hitl(
    task_id: str,
    action: str,
    payload: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resume a pending HITL task.

    Parameters
    ----------
    task_id:
        The pending task ID (from confirm_request event's taskId field).
    action:
        One of: "confirm", "cancel", "edit", "supplement".
    payload:
        Action-specific payload:
        - confirm: {"confirmed": true, "input": {...}}
        - cancel: {"reason": "..."}
        - edit: {"input": {...}}
        - supplement: {"slots": {...}}
    session_id:
        Optional session ID for tracing.

    Returns
    -------
    Dict
        Result of the HITL resume operation.
    """
    store = get_task_store()
    task = store.load(task_id)

    if action == "cancel":
        return _handle_cancel(task_id, task, payload)

    if action == "confirm":
        return _handle_confirm(task_id, task, payload, session_id)

    if action == "edit":
        return _handle_edit(task_id, task, payload)

    if action == "supplement":
        return _handle_supplement(task_id, task, payload, store)

    return {
        "success": False,
        "error": f"Unknown action: {action}. "
        "Supported actions: confirm, cancel, edit, supplement",
    }


# ---------------------------------------------------------------------------
# Action Handlers
# ---------------------------------------------------------------------------

def _handle_cancel(
    task_id: str,
    task: Optional[PendingTask],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle user cancellation."""
    reason = payload.get("reason", "用户主动取消")
    task_store = get_task_store()

    if task:
        task_store.delete(task_id)

    # Also clear any associated dry-run
    dry_run_id = payload.get("dry_run_id")
    if dry_run_id:
        get_dry_run_store().delete(dry_run_id)

    return {
        "success": True,
        "action": "cancelled",
        "task_id": task_id,
        "reason": reason,
        "message": "操作已取消",
    }


def _handle_confirm(
    task_id: str,
    task: Optional[PendingTask],
    payload: Dict[str, Any],
    session_id: Optional[str],
) -> Dict[str, Any]:
    """Handle user confirmation — invoke the tool."""
    confirmed = payload.get("confirmed", True)

    if not confirmed:
        return _handle_cancel(task_id, task, payload)

    input_params = payload.get("input", {})
    dry_run_id = payload.get("dry_run_id")

    # Merge any filled slots from the pending task
    if task and task.filled_slots:
        input_params = {**task.filled_slots, **input_params}

    tool_name = payload.get("tool_name", task.action if task else "")

    # Invoke the tool
    result = invoke(
        tool_name=tool_name,
        input_params=input_params,
        dry_run_id=dry_run_id,
        confirmed=True,
        session_id=session_id,
    )

    # Clean up task store on success
    if result.get("success"):
        get_task_store().delete(task_id)
        if dry_run_id:
            get_dry_run_store().delete(dry_run_id)

    return {
        "success": result.get("success", False),
        "action": "executed",
        "task_id": task_id,
        "result": result,
        "message": result.get("message", "执行完成"),
    }


def _handle_edit(
    task_id: str,
    task: Optional[PendingTask],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle user request to edit parameters — return current state."""
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found or expired",
        }

    return {
        "success": True,
        "action": "edit_requested",
        "task_id": task_id,
        "current_params": task.filled_slots,
        "missing_slots": task.missing_slots,
        "message": "请提供修改后的参数",
    }


def _handle_supplement(
    task_id: str,
    task: Optional[PendingTask],
    payload: Dict[str, Any],
    store: Any,
) -> Dict[str, Any]:
    """Handle user providing/supplementing slot values."""
    if not task:
        return {
            "success": False,
            "error": f"Task {task_id} not found or expired",
        }

    new_slots = payload.get("slots", {})

    # Merge with existing filled slots
    merged = {**task.filled_slots, **new_slots}
    remaining = [
        slot for slot in task.missing_slots
        if merged.get(slot) in (None, "", [])
    ]

    # Update the task
    store.update(
        task_id,
        filled_slots=merged,
        missing_slots=remaining,
        status="slots_partial" if remaining else "slots_complete",
    )

    if not remaining:
        return {
            "success": True,
            "action": "slots_complete",
            "task_id": task_id,
            "filled_slots": merged,
            "message": "所有必填槽位已填充，可以提交确认",
        }

    return {
        "success": True,
        "action": "slots_updated",
        "task_id": task_id,
        "filled_slots": merged,
        "remaining_slots": remaining,
        "message": f"已填充 {len(merged)} 个槽位，还需补充：{', '.join(remaining)}",
    }
