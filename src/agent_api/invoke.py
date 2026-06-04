"""
Formal Invocation — POST /agent/tools/{toolName}/invoke

Executes a tool with validated parameters after dry-run confirmation.
"""

from typing import Any, Dict, Optional

from .dry_run import get_dry_run_store
from .manifest import get_tool_registry


def invoke(
    tool_name: str,
    input_params: Dict[str, Any],
    dry_run_id: Optional[str] = None,
    confirmed: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Formally invoke a tool after dry-run confirmation.

    Parameters
    ----------
    tool_name:
        The tool name to invoke.
    input_params:
        The input parameters.
    dry_run_id:
        Optional dry-run ID from the dry-run step (for audit trail).
    confirmed:
        Whether the user confirmed the action (default True).
    session_id:
        Optional session ID for tracing.

    Returns
    -------
    Dict
        Invocation result with success flag, result data, and message.
    """
    registry = get_tool_registry()
    manifest = registry.get_tool_manifest(tool_name)

    if manifest is None:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
        }

    # ── Check confirmation ────────────────────────────────────────────────
    if not confirmed:
        return {
            "success": False,
            "message": "用户已取消操作",
            "confirmed": False,
        }

    # ── Map tool_name to ProcurementConnector action_id ────────────────────
    action_id = _map_tool_to_action(tool_name)
    if action_id is None:
        return {
            "success": False,
            "error": f"No action mapping found for tool '{tool_name}'",
        }

    # ── Execute via ProcurementConnector ─────────────────────────────────
    from ..procurement import get_procurement_connector

    connector = get_procurement_connector()
    connector_response = connector.call(action_id, _normalize_params(tool_name, input_params))

    # ── Build result ──────────────────────────────────────────────────────
    if connector_response.success:
        # ── Audit log ──────────────────────────────────────────────────
        _log_invocation(
            tool_name=tool_name,
            action_id=action_id,
            input_params=input_params,
            dry_run_id=dry_run_id,
            session_id=session_id,
            success=True,
        )

        return {
            "success": True,
            "result": connector_response.data,
            "message": connector_response.message,
            "tool_name": tool_name,
            "dry_run_id": dry_run_id,
        }
    else:
        _log_invocation(
            tool_name=tool_name,
            action_id=action_id,
            input_params=input_params,
            dry_run_id=dry_run_id,
            session_id=session_id,
            success=False,
            error=connector_response.error,
        )
        return {
            "success": False,
            "error": connector_response.error,
            "message": connector_response.message,
            "tool_name": tool_name,
            "dry_run_id": dry_run_id,
        }


def _map_tool_to_action(tool_name: str) -> Optional[str]:
    """Map tool_name to ProcurementConnector action_id."""
    mapping = {
        "create_purchase_request": "purchase_request/approve",
        "query_purchase_requests": "purchase_requests/list",
        "query_purchase_orders": "purchase_order/query_execution_status",
    }
    return mapping.get(tool_name)


def _normalize_params(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize input params to ProcurementConnector field names."""
    normalized = dict(params)

    # Aliases for field name compatibility
    alias_map: Dict[str, Dict[str, str]] = {
        "query_purchase_requests": {
            "department": "apply_dep",
        },
    }

    aliases = alias_map.get(tool_name, {})
    for from_key, to_key in aliases.items():
        if from_key in normalized and to_key not in normalized:
            normalized[to_key] = normalized.pop(from_key)

    return normalized


def _log_invocation(
    tool_name: str,
    action_id: str,
    input_params: Dict[str, Any],
    dry_run_id: Optional[str],
    session_id: Optional[str],
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Log tool invocation for audit trail."""
    import json
    from datetime import datetime
    from pathlib import Path

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "action_id": action_id,
        "input_params": input_params,
        "dry_run_id": dry_run_id,
        "session_id": session_id,
        "success": success,
        "error": error,
    }

    log_dir = Path("data/audit")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"invoke-{ts}-{tool_name}.json"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Audit logging should never fail the main operation
