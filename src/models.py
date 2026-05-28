"""DeepSeek model configuration for AgentScope."""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .llm_providers import LLMProvider, get_provider, provider_catalog

load_dotenv()


class ModelConfig:
    """Configuration for an LLM model, backed by a provider definition.

    Supports multi-vendor: DeepSeek, Kimi, Minimax, Custom.
    Thinking mode is provider-agnostic — each provider defines its own API format.
    """

    def __init__(
        self,
        provider_id: str = "deepseek",
        model_name: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
        thinking: bool = False,
        thinking_budget: int = 4000,
    ):
        self.provider_id = provider_id
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        self.thinking = thinking
        self.thinking_budget = thinking_budget

    @property
    def provider(self) -> Optional[LLMProvider]:
        return get_provider(self.provider_id)

    def to_dict(self) -> Dict[str, Any]:
        """Build kwargs dict for OpenAIChatModel / AgentScope."""
        provider = self.provider
        base_url = self.base_url or (provider.default_base_url if provider else "")

        generate_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.seed is not None:
            generate_kwargs["seed"] = self.seed

        # Apply thinking params if enabled and provider supports it
        if self.thinking and provider and provider.supports_thinking:
            thinking_params = dict(provider.thinking_param)

            # Inject thinking_budget into extra_body if present
            if "extra_body" in thinking_params:
                eb = dict(thinking_params["extra_body"])
                # Replace reasoning_effort with budget
                eb["reasoning_effort"] = "high"
                thinking_params["extra_body"] = eb

            generate_kwargs.update(thinking_params)

            # Disable params that conflict with thinking mode
            for param in (provider.thinking_disabled_params or []):
                generate_kwargs.pop(param, None)

        return {
            "model_type": "openai_chat",
            "model_name": self.model_name,
            "api_key": self.api_key,
            "client_kwargs": {"base_url": base_url},
            "generate_kwargs": generate_kwargs,
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """JSON-serializable config for API responses."""
        return {
            "provider_id": self.provider_id,
            "model_name": self.model_name,
            "base_url": self.base_url or (self.provider.default_base_url if self.provider else ""),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "thinking": self.thinking,
            "thinking_budget": self.thinking_budget,
        }

    def to_catalog_dict(self) -> Dict[str, Any]:
        """Config for frontend dropdown, including provider metadata."""
        provider = self.provider
        return {
            "provider_id": self.provider_id,
            "model_name": self.model_name,
            "base_url": self.base_url or (provider.default_base_url if provider else ""),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "thinking": self.thinking,
            "thinking_budget": self.thinking_budget,
            "providers": provider_catalog(),
            "supports_thinking": provider.supports_thinking if provider else False,
        }


def get_model_config() -> ModelConfig:
    """Get model config from environment variables (fallback defaults)."""
    seed_val = os.getenv("DEEPSEEK_SEED")
    return ModelConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000")),
        top_p=float(os.getenv("DEEPSEEK_TOP_P", "1.0")),
        top_k=int(os.getenv("DEEPSEEK_TOP_K", "0")),
        presence_penalty=float(os.getenv("DEEPSEEK_PRESENCE_PENALTY", "0.0")),
        frequency_penalty=float(os.getenv("DEEPSEEK_FREQUENCY_PENALTY", "0.0")),
        seed=int(seed_val) if seed_val else None,
    )
