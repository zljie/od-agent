"""TaskValidator: ontology preflight check before action execution.

Ensures that every tool call matches a declaration in the OSIModel:
- The action_id must exist in the model's actions list
- All parameters in the call must be declared in the action's input schema

This is the "ontology as constitution" enforcement layer — any tool call
that doesn't match the ontology is rejected before it reaches the backend.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..semantic.semantic_backend import SemanticBackend


@dataclass
class ValidationResult:
    """Result of an ontology preflight check."""

    ok: bool
    reason: Optional[str] = None

    @classmethod
    def pass_(cls) -> "ValidationResult":
        return cls(ok=True, reason=None)

    @classmethod
    def fail(cls, reason: str) -> "ValidationResult":
        return cls(ok=False, reason=reason)


class TaskValidator:
    """Ontology preflight validator.

    Validates that:
    1. The action_id is declared in the OSIModel
    2. All provided parameters are declared in the action's parameter list
    3. Required parameters are present (optional validation)
    """

    def __init__(self, backend: Optional["SemanticBackend"] = None):
        self._backend = backend

    def set_backend(self, backend: "SemanticBackend") -> None:
        self._backend = backend

    def preflight(self, action_id: str, params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Run preflight validation on an action_id + params pair.

        Args:
            action_id: The action being invoked (e.g. "querySuppliers")
            params: The parameters being passed to the action

        Returns:
            ValidationResult.pass_() if valid, ValidationResult.fail(reason) if not.
        """
        if self._backend is None:
            # No backend configured — skip validation (backward compat)
            return ValidationResult.pass_()

        model = self._backend.model

        # ── Check 1: action exists in ontology ───────────────────────────────
        declared_action = self._get_action_by_name(model, action_id)
        if declared_action is None:
            available = [a.name for a in model.actions]
            return ValidationResult.fail(
                f"Action '{action_id}' is not declared in ontology. "
                f"Available actions: {available if available else '(none)'}"
            )

        # ── Check 2: all params are declared ────────────────────────────────
        if params is None:
            params = {}

        declared_param_names = {p.name for p in declared_action.parameters}
        for key in params.keys():
            if key not in declared_param_names:
                return ValidationResult.fail(
                    f"Parameter '{key}' is not declared on action '{action_id}'. "
                    f"Declared parameters: {list(declared_param_names) if declared_param_names else '(none)'}"
                )

        # ── Check 3: required params are present ─────────────────────────────
        required = {p.name for p in declared_action.parameters if p.required}
        missing = required - params.keys()
        if missing:
            return ValidationResult.fail(
                f"Missing required parameter(s) on action '{action_id}': {list(missing)}"
            )

        return ValidationResult.pass_()

    def _get_action_by_name(self, model: "OSIModel", name: str) -> Optional["ActionLike"]:
        """Find an action by name in the model. Supports partial match."""
        if model is None:
            return None
        name_lower = name.lower()
        # Exact match first
        for action in model.actions:
            if action.name.lower() == name_lower:
                return action
        # Fallback: substring match (e.g. "blockSupplier" matches "block_supplier")
        for action in model.actions:
            if name_lower in action.name.lower() or action.name.lower() in name_lower:
                return action
        return None


# Type annotation to avoid circular import at runtime
class ActionLike:
    """Duck-typed action object compatible with OSIModel.Action."""

    name: str
    parameters: List["ParamLike"]

    def __init__(self, name: str, parameters: List["ParamLike"]):
        self.name = name
        self.parameters = parameters


class ParamLike:
    """Duck-typed parameter object."""

    name: str
    required: bool

    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
