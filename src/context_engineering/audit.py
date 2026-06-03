"""
Context Engineering Audit Module
=================================
UI observability layer for the context engineering pipeline.

Emits SSE-compatible dict events describing which context sources were used
at each pipeline step, enabling frontend panels to display context provenance.

Usage::

    from src.context_engineering.audit import ContextAuditLogger, ContextSourceInfo

    logger = ContextAuditLogger(task_id="TASK-123")
    logger.log_context_usage([
        ContextSourceInfo(source_type="ontology", source_name="品类本体",
                          items_used=5, token_estimate=200, trust_level="high"),
    ])
    event = logger.emit_source_event("step1")
    yield json.dumps(event)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.context_engineering.models import TrustLevel


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContextSourceInfo:
    """Describes a single context source used in building the context package."""

    source_type: str
    source_name: str
    items_used: int
    token_estimate: int
    freshness: Optional[str] = None
    trust_level: str = "high"
    excluded_count: int = 0
    exclusion_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "items_used": self.items_used,
            "token_estimate": self.token_estimate,
            "freshness": self.freshness,
            "trust_level": self.trust_level,
            "excluded_count": self.excluded_count,
            "exclusion_reasons": self.exclusion_reasons,
        }


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


class ContextAuditLogger:
    """Logs context source usage for UI observability.

    Emits SSE-compatible dict events that the pipeline can yield alongside
    other step events. Call ``emit_source_event()`` at the end of each step
    to surface which contexts were used.
    """

    def __init__(self, task_id: str = ""):
        self.task_id = task_id
        self._records: List[ContextSourceInfo] = []

    def log_context_usage(self, sources: List[ContextSourceInfo]) -> None:
        """Record context source usage for a step."""
        self._records.extend(sources)

    def emit_source_event(self, step_name: str) -> Dict[str, Any]:
        """Emit an SSE-compatible dict event describing context sources used so far.

        Returns a dict suitable for pipeline SSE yielding::

            {
                "event": "context_sources",
                "step": step_name,
                "task_id": self.task_id,
                "sources": [...list of ContextSourceInfo.to_dict()...],
                "total_sources": N,
                "total_items": M,
                "total_tokens_estimate": K,
            }
        """
        total_tokens = sum(s.token_estimate for s in self._records)
        total_items = sum(s.items_used for s in self._records)
        return {
            "event": "context_sources",
            "step": step_name,
            "task_id": self.task_id,
            "sources": [s.to_dict() for s in self._records],
            "total_sources": len(self._records),
            "total_items": total_items,
            "total_tokens_estimate": total_tokens,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of all context usage across all steps."""
        if not self._records:
            return {"task_id": self.task_id, "steps": [], "total_sources": 0}

        total_tokens = sum(s.token_estimate for s in self._records)
        total_items = sum(s.items_used for s in self._records)

        by_type: Dict[str, Dict[str, Any]] = {}
        for src in self._records:
            if src.source_type not in by_type:
                by_type[src.source_type] = {
                    "source_type": src.source_type,
                    "sources": [],
                    "total_items": 0,
                    "total_tokens": 0,
                }
            entry = by_type[src.source_type]
            entry["sources"].append(src.to_dict())
            entry["total_items"] += src.items_used
            entry["total_tokens"] += src.token_estimate

        return {
            "task_id": self.task_id,
            "total_sources": len(self._records),
            "total_items": total_items,
            "total_tokens_estimate": total_tokens,
            "by_type": list(by_type.values()),
        }

    def clear(self) -> None:
        """Clear all recorded sources."""
        self._records.clear()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _freshness_status(freshness: Optional[str]) -> str:
    """Return 'fresh', 'stale', or 'unknown' based on freshness timestamp."""
    if freshness is None:
        return "unknown"
    try:
        ts = datetime.fromisoformat(freshness.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_seconds = (now - ts).total_seconds()
        return "fresh" if age_seconds <= 300 else "stale"
    except (ValueError, TypeError):
        return "unknown"


def format_context_sources_for_ui(
    sources: List[ContextSourceInfo],
) -> List[Dict[str, Any]]:
    """Format ``ContextSourceInfo`` list for frontend consumption.

    Groups by ``source_type``, adds display name, freshness status badge,
    and per-source token percentage relative to the total.

    Returns a list of dicts suitable for JSON serialization to the frontend.
    """
    if not sources:
        return []

    total_tokens = sum(s.token_estimate for s in sources)

    result: List[Dict[str, Any]] = []
    for src in sources:
        token_pct = (
            (src.token_estimate / total_tokens) if total_tokens > 0 else 0.0
        )
        result.append(
            {
                **src.to_dict(),
                "display_name": src.source_name,
                "status": _freshness_status(src.freshness),
                "token_percentage": round(token_pct, 4),
            }
        )
    return result


def build_context_panel_html(summary: Dict[str, Any]) -> str:
    """Build a small HTML snippet showing a context panel.

    Renders a ``<div class="context-panel">`` with sections per source_type,
    showing source name, item count, token estimate, freshness badge, and
    trust indicator. Uses inline CSS only (no external dependencies).

    Parameters
    ----------
    summary:
        The output of ``ContextAuditLogger.get_summary()`` or a compatible dict
        with ``by_type``, ``total_sources``, ``total_items``,
        ``total_tokens_estimate`` fields.

    Returns
    -------
    str
        A plain HTML string suitable for embedding in a page or streaming.
    """
    by_type = summary.get("by_type", [])
    total_sources = summary.get("total_sources", 0)
    total_items = summary.get("total_items", 0)
    total_tokens = summary.get("total_tokens_estimate", 0)

    source_rows = ""
    for group in by_type:
        group_type = group.get("source_type", "unknown")
        group_items = group.get("total_items", 0)
        group_tokens = group.get("total_tokens", 0)
        sources = group.get("sources", [])

        rows = ""
        for src in sources:
            status = _freshness_status(src.get("freshness"))
            trust = src.get("trust_level", "unknown")
            rows += f"""
            <tr>
              <td>{src.get('source_name', '')}</td>
              <td>{src.get('items_used', 0)}</td>
              <td>{src.get('token_estimate', 0)}</td>
              <td><span class="badge badge-{status}">{status}</span></td>
              <td><span class="trust trust-{trust}">{trust}</span></td>
            </tr>"""

        source_rows += f"""
        <div class="cp-section">
          <div class="cp-section-header">{group_type} <span class="cp-count">{group_items} items · {group_tokens} tokens</span></div>
          <table class="cp-table">
            <thead><tr><th>Source</th><th>Items</th><th>Tokens</th><th>Freshness</th><th>Trust</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!-- context-panel -->
<style>
.context-panel{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;color:#374151;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;max-width:600px;padding:16px}}
.cp-header{{font-weight:600;font-size:14px;margin-bottom:12px;color:#111827;display:flex;justify-content:space-between}}
.cp-stats{{font-size:12px;color:#6b7280;margin-bottom:16px}}
.cp-section{{margin-bottom:14px}}
.cp-section-header{{font-weight:600;padding:6px 8px;background:#f3f4f6;border-radius:4px;margin-bottom:6px;display:flex;justify-content:space-between}}
.cp-count{{font-weight:400;font-size:11px;color:#6b7280}}
.cp-table{{width:100%;border-collapse:collapse;font-size:12px}}
.cp-table th{{text-align:left;padding:4px 8px;background:#f9fafb;color:#6b7280;font-weight:500;border-bottom:1px solid #e5e7eb}}
.cp-table td{{padding:4px 8px;border-bottom:1px solid #f3f4f6}}
.badge{{padding:2px 6px;border-radius:10px;font-size:10px;font-weight:500}}
.badge-fresh{{background:#d1fae5;color:#065f46}}
.badge-stale{{background:#fef3c7;color:#92400e}}
.badge-unknown{{background:#f3f4f6;color:#6b7280}}
.trust{{padding:2px 6px;border-radius:4px;font-size:10px}}
.trust-high{{background:#dbeafe;color:#1e40af}}
.trust-medium{{background:#fef3c7;color:#92400e}}
.trust-low{{background:#fee2e2;color:#991b1b}}
.trust-unknown{{background:#f3f4f6;color:#6b7280}}
</style>
<div class="context-panel">
  <div class="cp-header">Context Sources <span style="color:#6b7280;font-weight:400">{total_sources} sources</span></div>
  <div class="cp-stats">{total_items} items · ~{total_tokens} tokens</div>
  {source_rows}
</div>"""
