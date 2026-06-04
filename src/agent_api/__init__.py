"""
Agent Tool API — AI-Facing API Manifest Service

Implements the Tool Manifest protocol per docs/prd/API_Manifest_后端服务方案.md (MVP).

Modules
-------
manifest   : Tool registry and manifest definitions
validate   : Parameter validation (validate endpoint)
dry_run    : Dry-run execution (dry-run endpoint) + DryRunStore
invoke     : Formal execution (invoke endpoint)
hitl       : HITL resume logic (hitl/resume endpoint)
"""

from .manifest import TOOL_MANIFESTS, AgentToolRegistry, get_tool_registry

__all__ = [
    "TOOL_MANIFESTS",
    "AgentToolRegistry",
    "get_tool_registry",
]
