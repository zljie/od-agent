"""Customer service agent implementation using AgentScope."""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .models import get_model_config

load_dotenv()


SYSTEM_PROMPT = """You are a professional customer service representative for a technology company.
Your name is OD Assistant.

Guidelines:
1. Be friendly, patient, and professional in all interactions
2. Listen carefully to customer concerns and questions
3. Provide accurate and helpful information
4. If you don't know something, admit it honestly and offer to find out
5. Suggest relevant products or services when appropriate
6. Handle complaints with empathy and work toward resolution
7. Keep responses concise but comprehensive
8. Always end with a question to continue the conversation if appropriate

Current product offerings:
- Enterprise AI Solutions
- Cloud Integration Services
- Custom Development
- Technical Support Plans
"""


class DialogueAgent:
    """Customer service dialogue agent powered by AgentScope."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.model_config = model_config or get_model_config().model_kwargs
        self.conversation_history: List[Dict[str, str]] = []

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages including system prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages

    async def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        self.add_message("user", user_input)
        return f"Thank you for your message: '{user_input}'. How can I assist you today?"

    def format_response(self, response: str) -> Dict[str, Any]:
        """Format response for API output."""
        return {
            "response": response,
            "history_length": len(self.conversation_history),
        }


def create_agent() -> DialogueAgent:
    """Create a customer service agent instance."""
    return DialogueAgent()
