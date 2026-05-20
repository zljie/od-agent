"""Skill Manager - orchestrates skill detection and execution."""

import asyncio
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseSkill
from .math_teacher import MathTeacherSkill

if TYPE_CHECKING:
    from ..agent import CustomerServiceAgent


class SkillManager:
    """Manages skill registration, intent detection, and execution."""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._intent_rules: List[Dict[str, Any]] = []
        self._register_builtin_skills()

    def _register_builtin_skills(self):
        """Register built-in skills."""
        math_skill = MathTeacherSkill()
        self.register_skill(math_skill)

    def register_skill(self, skill: BaseSkill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill

    def unregister_skill(self, name: str) -> bool:
        """Unregister a skill by name."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all_skills(self) -> List[BaseSkill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def get_skills_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all skills for UI display."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "keywords_count": len(s.keywords),
                "priority": s.priority,
            }
            for s in self._skills.values()
        ]

    def load_intent_rules(self, rules: List[Dict[str, Any]]) -> None:
        """Load intent routing rules from config."""
        self._intent_rules = rules

    async def detect_and_execute(self, message: str, agent: Any = None) -> Optional[Dict[str, Any]]:
        """Detect intent and execute corresponding skill.

        Priority:
        1. Intent rules (from config)
        2. Built-in skill matching
        """
        message_lower = message.lower()

        matched_skill = None
        matched_rule = None
        max_priority = -1

        for rule in self._intent_rules:
            priority = rule.get("priority", 10)
            keywords = rule.get("keywords", [])

            for keyword in keywords:
                if keyword.lower() in message_lower and priority > max_priority:
                    max_priority = priority
                    matched_rule = rule
                    matched_skill = self._skills.get(rule.get("handler"))

        if not matched_skill:
            for skill in sorted(self._skills.values(), key=lambda s: s.priority, reverse=True):
                if skill.match(message):
                    matched_skill = skill
                    break

        if matched_skill:
            try:
                result = await matched_skill.execute({"message": message})
                if agent and hasattr(agent, "_active_skill"):
                    agent._active_skill = matched_skill.name
                return {
                    "skill": matched_skill.name,
                    "executed": True,
                    "result": result,
                }
            except Exception as e:
                return {
                    "skill": matched_skill.name,
                    "executed": False,
                    "error": str(e),
                }

        return None

    def detect_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect which skill/intent should handle the message without executing."""
        message_lower = message.lower()

        for rule in sorted(self._intent_rules, key=lambda r: r.get("priority", 10), reverse=True):
            keywords = rule.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    return {
                        "matched": True,
                        "intent": rule.get("name"),
                        "handler": rule.get("handler"),
                        "matched_keyword": keyword,
                    }

        for skill in sorted(self._skills.values(), key=lambda s: s.priority, reverse=True):
            if skill.match(message):
                return {
                    "matched": True,
                    "intent": skill.name,
                    "handler": skill.name,
                    "matched_keyword": "skill_keyword",
                }

        return {"matched": False}


# Global skill manager instance
_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get the global skill manager instance."""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager


def reload_skill_manager() -> SkillManager:
    """Reload and return the skill manager."""
    global _skill_manager
    _skill_manager = SkillManager()
    return _skill_manager
