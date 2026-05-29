"""Skills module for extensible agent capabilities."""

from .base import BaseSkill
from .math_teacher import MathTeacherSkill
from .semantic_skill import SemanticSkill
from .skill_registry import SkillRegistry
from .task_executor import TaskExecutor
from .time_converter import TimeConverterSkill

# Backward-compatible: SkillManager wraps the new components
from .manager import SkillManager, get_skill_manager, reload_skill_manager

__all__ = [
    "BaseSkill",
    "MathTeacherSkill",
    "TimeConverterSkill",
    "SemanticSkill",
    "SkillManager",
    "get_skill_manager",
    "reload_skill_manager",
    "SkillRegistry",
    "TaskExecutor",
]
