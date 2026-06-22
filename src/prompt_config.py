"""
Prompt Configuration Loader
===========================
Centralized helper for reading per-step prompt configuration from
``config/agent_config.json`` with graceful default fallback.

The ``prompt_config`` block is structured as groups:

.. code-block:: json

    {
      "prompt_config": {
        "system_prompt": "...",
        "intent_recognition": {
          "light_system_prompt": "...",
          "light_templates_description": "...",
          ...
        },
        "five_step": {
          "step2_ontology_matching_prompt": "...",
          "step3_planner_description_prompt": "...",
          "step5_response_generation_prompt": "..."
        },
        "procurement": {
          "intent_classification_prompt": "...",
          "slot_collection_prompts": {
            "pr_id": "..."
          }
        }
      }
    }

Each helper returns the configured string if present and non-empty,
otherwise falls back to the supplied default. This makes prompt
customization safe: existing hardcoded values continue to work when
no override is configured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_DEFAULT_CACHE: Dict[str, Any] = {}


def _load_config() -> Dict[str, Any]:
    """Read ``prompt_config`` from disk, with caching."""
    if "prompt_config" in _DEFAULT_CACHE:
        return _DEFAULT_CACHE["prompt_config"]

    try:
        # Imported lazily so module import order stays simple
        from .agent import load_agent_config
        cfg = load_agent_config() or {}
        pc = cfg.get("prompt_config") or {}
    except Exception:
        pc = {}
    _DEFAULT_CACHE["prompt_config"] = pc
    return pc


def invalidate_cache() -> None:
    """Drop the in-memory cache (call after config file changes)."""
    _DEFAULT_CACHE.clear()


def get_group(group: str) -> Dict[str, Any]:
    """Return the sub-dictionary for a named group (e.g. ``intent_recognition``)."""
    pc = _load_config()
    if not isinstance(pc, dict):
        return {}
    grp = pc.get(group)
    return grp if isinstance(grp, dict) else {}


def get_field(group: str, field: str, default: str = "") -> str:
    """Return ``prompt_config[group][field]`` or ``default`` if missing/empty."""
    grp = get_group(group)
    val = grp.get(field)
    if isinstance(val, str) and val.strip():
        return val
    return default


def get_main_system_prompt(default: str = "") -> str:
    """Return ``prompt_config.system_prompt`` or fallback default."""
    pc = _load_config()
    if isinstance(pc, dict):
        val = pc.get("system_prompt")
        if isinstance(val, str) and val.strip():
            return val
    return default


def get_slot_collection_prompts(defaults: Dict[str, str]) -> Dict[str, str]:
    """Merge configured slot prompts on top of the supplied defaults.

    Any slot id present in ``prompt_config.procurement.slot_collection_prompts``
    overrides the default. Slot ids in defaults that are not configured are
    preserved.
    """
    grp = get_group("procurement")
    overrides = grp.get("slot_collection_prompts")
    if not isinstance(overrides, dict):
        return dict(defaults)
    merged = dict(defaults)
    for key, val in overrides.items():
        if isinstance(val, str) and val.strip():
            merged[key] = val
    return merged
