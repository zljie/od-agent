"""DeepSeek model configuration for AgentScope."""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """Configuration for DeepSeek model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Get model configuration for AgentScope."""
        return {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "client_kwargs": {"base_url": self.base_url},
            "generate_kwargs": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_type": "openai_chat",
            **self.model_kwargs,
        }


def get_model_config() -> ModelConfig:
    """Get model configuration from environment variables."""
    return ModelConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000")),
    )


def create_deepseek_model() -> Dict[str, Any]:
    """Create DeepSeek model configuration for AgentScope."""
    config = get_model_config()
    return config.to_dict()
