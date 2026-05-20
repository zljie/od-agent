"""Base class for agent skills."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSkill(ABC):
    """Abstract base class for all skills."""

    name: str = ""
    description: str = ""
    keywords: List[str] = []
    priority: int = 10

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the skill with given input.

        Args:
            input_data: Dictionary containing skill input parameters.
                - message: The user's message/question
                - context: Optional conversation context

        Returns:
            Dictionary containing:
                - success: Whether the skill executed successfully
                - response: The skill's response text
                - metadata: Optional additional data
        """
        pass

    def match(self, message: str) -> bool:
        """Check if this skill should handle the given message.

        Args:
            message: The user's input message.

        Returns:
            True if this skill should handle the message, False otherwise.
        """
        message_lower = message.lower()
        return any(keyword.lower() in message_lower for keyword in self.keywords)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this skill when it's activated."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "priority": self.priority,
        }
