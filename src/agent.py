"""Customer service agent implementation using AgentScope ReAct Agent."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv

from .models import get_model_config
from .skills import get_skill_manager, reload_skill_manager

load_dotenv()

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a professional customer service representative for a technology company.
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


def get_config_path() -> Path:
    """Get the path to the agent configuration file."""
    return Path(__file__).parent.parent / "config" / "agent_config.json"


def load_agent_config() -> Dict[str, Any]:
    """Load agent configuration from JSON file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_agent_config(config: Dict[str, Any]) -> None:
    """Save agent configuration to JSON file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


class CustomerServiceAgent:
    """Customer service dialogue agent powered by AgentScope ReAct Agent."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        agent_name: str = "OD_Assistant",
    ):
        # Load from config file if not provided
        config = load_agent_config()
        
        self.agent_name = agent_name or config.get("agent_name", "OD_Assistant")
        self.system_prompt = system_prompt or config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        
        # Get model config
        if model_config is None:
            model_cfg = config.get("model_config", {})
            model_config = get_model_config()
            model_config.model_name = model_cfg.get("model_name", model_config.model_name)
            model_config.base_url = model_cfg.get("base_url", model_config.base_url)
            model_config.temperature = model_cfg.get("temperature", model_config.temperature)
            model_config.max_tokens = model_cfg.get("max_tokens", model_config.max_tokens)
        
        # Initialize OpenAI-compatible model for DeepSeek
        self.model = OpenAIChatModel(
            api_key=model_config.api_key if hasattr(model_config, 'api_key') else os.getenv("DEEPSEEK_API_KEY", ""),
            model_name=model_config.model_name if hasattr(model_config, 'model_name') else "deepseek-chat",
            client_kwargs={"base_url": model_config.base_url if hasattr(model_config, 'base_url') else "https://api.deepseek.com/v1"},
            generate_kwargs={
                "temperature": model_config.temperature if hasattr(model_config, 'temperature') else 0.7,
                "max_tokens": model_config.max_tokens if hasattr(model_config, 'max_tokens') else 2000,
            },
            stream=False,  # Disable streaming for simpler response handling
        )
        
        # Initialize AgentScope ReAct Agent
        self.agent = ReActAgent(
            name=self.agent_name,
            sys_prompt=self.system_prompt,
            model=self.model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
        )

        # Initialize skill manager
        self._skill_manager = get_skill_manager()
        self._active_skill: Optional[str] = None
        self._load_intent_rules()

    def _load_intent_rules(self) -> None:
        """Load intent routing rules from config."""
        intent_config_path = Path(__file__).parent.parent / "config" / "intent_routing.json"
        if intent_config_path.exists():
            with open(intent_config_path, "r", encoding="utf-8") as f:
                rules = json.load(f)
                self._skill_manager.load_intent_rules(rules)

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.agent.memory.clear()
        self._active_skill = None

    async def chat(self, user_input: str) -> str:
        """Process user input and return agent response."""
        # Check for skill match first
        skill_result = await self._skill_manager.detect_and_execute(user_input, self)
        if skill_result and skill_result.get("executed"):
            self._active_skill = skill_result.get("skill")
            return skill_result.get("result", {}).get("response", "")

        # Create user message
        msg = Msg(
            name="user",
            content=user_input,
            role="user",
        )
        
        try:
            # Call AgentScope ReAct Agent
            response = await self.agent(msg)
            # Extract text content from response (handle list of content blocks)
            content = response.content
            if isinstance(content, list):
                # Filter for text content and concatenate
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif "content" in item:
                            text_parts.append(str(item["content"]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "".join(text_parts)
            return str(content) if content else ""
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
        config = load_agent_config()
        agent_name = config.get("agent_name", "OD_Assistant")
        _agent = CustomerServiceAgent(agent_name=agent_name)
    return _agent


def reload_agent() -> CustomerServiceAgent:
    """Reload the agent with fresh configuration."""
    global _agent
    _agent = None
    # Also reload skill manager
    reload_skill_manager()
    return get_agent()


def create_agent() -> CustomerServiceAgent:
    """Create a customer service agent instance."""
    return CustomerServiceAgent()
