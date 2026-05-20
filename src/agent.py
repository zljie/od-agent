"""Customer service agent implementation using AgentScope ReAct Agent."""

import os
from typing import Any, Dict, List, Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
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


class CustomerServiceAgent:
    """Customer service dialogue agent powered by AgentScope ReAct Agent."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.model_config = model_config or get_model_config()
        
        # Initialize OpenAI-compatible model for DeepSeek
        self.model = OpenAIChatModel(
            api_key=self.model_config.api_key,
            model_name=self.model_config.model_name,
            client_kwargs={"base_url": self.model_config.base_url},
            generate_kwargs={
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
            },
            stream=False,  # Disable streaming for simpler response handling
        )
        
        # Initialize AgentScope ReAct Agent
        self.agent = ReActAgent(
            name="OD_Assistant",
            sys_prompt=self.system_prompt,
            model=self.model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
        )

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.agent.memory.clear()

    async def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        # Create user message
        msg = Msg(
            name="user",
            content=user_input,
            role="user",
        )
        
        try:
            # Call AgentScope ReAct Agent
            response = await self.agent(msg)
            return response.content
        except Exception as e:
            error_msg = f"Error calling DeepSeek API: {str(e)}"
            print(error_msg)
            return error_msg


# Singleton instance
_agent: Optional[CustomerServiceAgent] = None


def get_agent() -> CustomerServiceAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        config = get_model_config()
        _agent = CustomerServiceAgent(model_config=config)
    return _agent


def create_agent() -> CustomerServiceAgent:
    """Create a customer service agent instance."""
    return CustomerServiceAgent()
