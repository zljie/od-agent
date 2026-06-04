"""
Dry-Run Execution — POST /agent/tools/{toolName}/dry-run

Simulates tool execution without persisting data.
Generates a human-readable summary and HITL confirmation card.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# DryRun Store (in-memory, TTL-based)
# ---------------------------------------------------------------------------

DRY_RUN_TTL_SECONDS = 10 * 60  # 10 minutes


@dataclass
class DryRunRecord:
    dry_run_id: str
    tool_name: str
    input_params: Dict[str, Any]
    summary: str
    risk_level: str
    requires_hitl: bool
    hitl_card: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    connector_preview: Optional[Dict[str, Any]] = None

    def is_expired(self) -> bool:
        return time.time() - self.created_at > DRY_RUN_TTL_SECONDS


class DryRunStore:
    """Thread-safe in-memory store for dry-run records."""

    def __init__(self) -> None:
        self._records: Dict[str, DryRunRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: DryRunRecord) -> None:
        with self._lock:
            self._records[record.dry_run_id] = record

    def load(self, dry_run_id: str) -> Optional[DryRunRecord]:
        with self._lock:
            record = self._records.get(dry_run_id)
            if record and record.is_expired():
                del self._records[dry_run_id]
                return None
            return record

    def delete(self, dry_run_id: str) -> bool:
        with self._lock:
            if dry_run_id in self._records:
                del self._records[dry_run_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired records."""
        with self._lock:
            expired = [k for k, r in self._records.items() if r.is_expired()]
            for k in expired:
                del self._records[k]
            return len(expired)


_dry_run_store: Optional[DryRunStore] = None


def get_dry_run_store() -> DryRunStore:
    global _dry_run_store
    if _dry_run_store is None:
        _dry_run_store = DryRunStore()
    return _dry_run_store


# ---------------------------------------------------------------------------
# Dry-Run Execution Logic
# ---------------------------------------------------------------------------

def generate_dry_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:8].upper()
    return f"dryrun-{ts}-{suffix}"


def dry_run(
    tool_name: str,
    input_params: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a dry-run for the given tool and parameters.

    Parameters
    ----------
    tool_name:
        The tool to dry-run (e.g., "create_purchase_request").
    input_params:
        The input parameters.
    session_id:
        Optional session ID for trace linkage.

    Returns
    -------
    Dict
        Dry-run result containing dry_run_id, summary, risk_level,
        requires_hitl, and hitl_card.
    """
    from .manifest import get_tool_registry

    registry = get_tool_registry()
    manifest = registry.get_tool_manifest(tool_name)

    if manifest is None:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
            "dry_run_id": None,
        }

    # ── Build summary text ─────────────────────────────────────────────────
    summary, risk_level, requires_hitl = _build_summary(manifest, input_params)

    # ── Build HITL card ──────────────────────────────────────────────────
    hitl_card = _build_hitl_card(
        tool_name=tool_name,
        manifest=manifest,
        input_params=input_params,
        summary=summary,
        risk_level=risk_level,
    )

    dry_run_id = generate_dry_run_id()

    # ── Save to store ────────────────────────────────────────────────────
    record = DryRunRecord(
        dry_run_id=dry_run_id,
        tool_name=tool_name,
        input_params=input_params,
        summary=summary,
        risk_level=risk_level,
        requires_hitl=requires_hitl,
        hitl_card=hitl_card,
    )
    get_dry_run_store().save(record)

    return {
        "success": True,
        "dry_run_id": dry_run_id,
        "summary": summary,
        "risk_level": risk_level,
        "requires_hitl": requires_hitl,
        "hitl_card": hitl_card,
    }


def _build_summary(manifest: Any, input_params: Dict[str, Any]) -> tuple[str, str, bool]:
    """Build human-readable summary from manifest + input params."""
    tool_name = manifest.name
    schema = manifest.input_schema

    # Map display labels for known fields
    label_map = {
        "department": "申请部门",
        "item_name": "物料",
        "quantity": "数量",
        "delivery_date": "交付日期",
        "po_id": "订单编号",
        "pr_id": "采购需求编号",
        "page": "页码",
        "page_size": "每页条数",
    }

    def label(key: str) -> str:
        field_def = schema.properties.get(key)
        return label_map.get(key) or (field_def.title if field_def else key)

    parts: List[str] = []

    if tool_name == "create_purchase_request":
        dept = input_params.get("department", "未知部门")
        item = input_params.get("item_name", "未知物料")
        qty = input_params.get("quantity", "?")
        date = input_params.get("delivery_date", "未知日期")
        parts.append(f"将为 {dept} 创建采购申请")
        parts.append(f"物料：{item}")
        parts.append(f"数量：{qty}")
        parts.append(f"期望交付日期：{date}")
        summary = "，".join(parts)
        return summary, "medium", True

    elif tool_name == "query_purchase_requests":
        dept = input_params.get("department", "")
        status = input_params.get("status", "")
        page = input_params.get("page", 1)
        page_size = input_params.get("page_size", 50)
        conditions = []
        if dept:
            conditions.append(f"部门={dept}")
        if status:
            conditions.append(f"状态={status}")
        cond_str = "，".join(conditions) if conditions else "全部"
        summary = f"将查询采购需求（{cond_str}，第{page}页，每页{page_size}条）"
        return summary, "low", False

    elif tool_name == "query_purchase_orders":
        po_id = input_params.get("po_id", "未知")
        summary = f"将查询采购订单 {po_id} 的执行情况（审批状态/发货状态/收货状态/发票状态）"
        return summary, "low", False

    else:
        # Generic summary
        filled = [
            f"{label(k)}={v}"
            for k, v in input_params.items()
            if v not in (None, "", [])
        ]
        summary = f"将执行 {tool_name}，参数：{', '.join(filled) if filled else '无'}"
        return summary, "low", False


def _build_hitl_card(
    tool_name: str,
    manifest: Any,
    input_params: Dict[str, Any],
    summary: str,
    risk_level: str,
) -> Dict[str, Any]:
    """Build HITL confirmation card from manifest + input."""
    hitl_card: Dict[str, Any] = {
        "type": "confirm",
        "title": f"请确认：{manifest.title}",
        "message": summary,
        "confirm_action": {
            "label": "确认执行",
            "resume_endpoint": "/hitl/resume",
        },
        "cancel_action": {
            "label": "取消",
            "resume_endpoint": "/hitl/resume",
        },
    }

    # Risk level and affected records for high-risk actions
    if risk_level in ("medium", "high"):
        hitl_card["risk_level"] = risk_level
        hitl_card["confirm_action"][
            "message"
        ] = f"确认执行 {manifest.title}？此操作 {risk_level} 风险。"
        hitl_card["cancel_action"][
            "message"
        ] = "取消当前操作"

    # Add alternatives if any
    if tool_name == "create_purchase_request":
        hitl_card.setdefault("alternatives", [])
        hitl_card["alternatives"].extend(
            [
                {
                    "id": "edit",
                    "label": "修改参数",
                    "description": "修改后再确认",
                },
            ]
        )

    return hitl_card
