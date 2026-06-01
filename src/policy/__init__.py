"""Policy engine: RBAC permission checks + HITL gating for action execution.

Handles:
- Role-based permission: can a user role invoke a given action?
- HITL gating: should this action require human confirmation?
- Write-action auto-escalation

All policy decisions are deterministic (no LLM).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class PolicyConfig:
    """Loaded policy configuration."""

    confidence_gate: float = 0.3
    always_confirm_actions: List[str] = field(default_factory=list)
    write_action_prefixes: List[str] = field(
        default_factory=lambda: ["create", "update", "delete", "block", "release", "approve", "reject"]
    )
    role_permissions: Dict[str, List[str]] = field(default_factory=dict)
    hitl_prompt_template: str = (
        "系统将执行操作「{action_id}」，请确认：\n"
        "参数：{params}\n"
        "回复「确认」继续，或「取消」中止。"
    )


def load_policy_config(path: Optional[str] = None) -> PolicyConfig:
    """Load policy configuration from JSON file."""
    if path is None:
        path = os.environ.get(
            "POLICY_CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "config" / "policy_config.json"),
        )

    defaults = PolicyConfig()
    if not os.path.exists(path):
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return PolicyConfig(
            confidence_gate=cfg.get("confidence_gate", defaults.confidence_gate),
            always_confirm_actions=cfg.get("always_confirm_actions", defaults.always_confirm_actions),
            write_action_prefixes=cfg.get(
                "write_action_prefixes", defaults.write_action_prefixes
            ),
            role_permissions=cfg.get("role_permissions", defaults.role_permissions),
            hitl_prompt_template=cfg.get(
                "hitl_prompt_template", defaults.hitl_prompt_template
            ),
        )
    except Exception:
        return defaults


class PolicyEngine:
    """Deterministic policy enforcement: RBAC + HITL gating.

    All decisions are rule-based. The engine is stateless — it reads the
    PolicyConfig on construction and applies it consistently.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self._cfg = config or PolicyConfig()

    @classmethod
    def from_path(cls, path: Optional[str] = None) -> "PolicyEngine":
        cfg = load_policy_config(path)
        return cls(cfg)

    def can_invoke(self, action_id: str, user_roles: Optional[List[str]] = None) -> bool:
        """Check if any of the user_roles has permission to invoke action_id.

        Args:
            action_id: The action being invoked (e.g. "querySuppliers", "blockSupplier")
            user_roles: List of the current user's roles (e.g. ["admin", "operator"])

        Returns:
            True if at least one role has permission, False otherwise.
            If user_roles is empty/None, defaults to denying the action.
        """
        if not user_roles:
            return False

        # Wildcard permission: "*" means all actions
        for role in user_roles:
            perms = self._cfg.role_permissions.get(role, [])
            if "*" in perms or action_id in perms:
                return True

        return False

    def is_write_action(self, action_id: str) -> bool:
        """Infer whether an action is a write/mutation action by its name prefix."""
        aid_lower = action_id.lower()
        for prefix in self._cfg.write_action_prefixes:
            if aid_lower.startswith(prefix):
                return True
        return False

    def requires_hitl(
        self,
        action_id: str,
        confidence: float = 1.0,
        is_write_action: Optional[bool] = None,
    ) -> bool:
        """Determine if human-in-the-loop confirmation is required.

        HITL is required if ANY of the following is true:
        1. The action is in always_confirm_actions list
        2. The action is a write action (auto-escalation)
        3. Classification confidence is below the confidence gate

        Args:
            action_id: The action being invoked
            confidence: Intent classification confidence [0, 1]
            is_write_action: Override auto-detection; if None, inferred from action_id

        Returns:
            True if HITL is required
        """
        # Check always-confirm list
        if action_id in self._cfg.always_confirm_actions:
            return True

        # Check write action
        write = is_write_action if is_write_action is not None else self.is_write_action(action_id)
        if write:
            return True

        # Check confidence gate
        if confidence < self._cfg.confidence_gate:
            return True

        return False

    def get_hitl_prompt(
        self, action_id: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a HITL confirmation prompt for the user."""
        params_str = (
            json.dumps(params, ensure_ascii=False, indent=2) if params else "(无参数)"
        )
        return self._cfg.hitl_prompt_template.format(
            action_id=action_id, params=params_str
        )

    def filter_allowed_actions(
        self, actions: List[str], user_roles: Optional[List[str]] = None
    ) -> List[str]:
        """Return the subset of actions the user is allowed to invoke."""
        if not user_roles:
            return []
        return [a for a in actions if self.can_invoke(a, user_roles)]

    @property
    def config(self) -> PolicyConfig:
        return self._cfg
