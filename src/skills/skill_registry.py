"""Skill registry: registration, lookup, and manifest management."""

from typing import Any, Dict, List, Optional

from .base import BaseSkill


class SkillRegistry:
    """Registry for all available skills.

    Responsibilities:
        - Register / unregister skills by ID
        - Lookup skills by name/ID
        - Provide skill catalog for system prompts
        - Startup-time consistency checks
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        """Unregister a skill by name."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name (alias: get_skill)."""
        return self._skills.get(name)

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def all(self) -> List[BaseSkill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def is_empty(self) -> bool:
        return len(self._skills) == 0

    def catalog(self) -> List[Dict[str, Any]]:
        """Return a catalog of all skills for system prompt injection."""
        return [{"name": s.name, "description": s.description, "params": getattr(s, "intent_params", {})} for s in self._skills.values()]
