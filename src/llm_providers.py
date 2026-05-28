"""LLM Provider configuration — defines available vendors, their models, and API parameters."""

from typing import Any, Dict, List, Optional


class LLMProvider:
    """A single LLM vendor (DeepSeek, Kimi, Minimax, Custom)."""

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        models: List[Dict[str, Any]],
        default_base_url: str,
        supports_thinking: bool = False,
        thinking_param: Optional[Dict[str, Any]] = None,
        thinking_disabled_params: Optional[List[str]] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.models = models
        self.default_base_url = default_base_url
        self.supports_thinking = supports_thinking
        self.thinking_param = thinking_param or {}
        self.thinking_disabled_params = thinking_disabled_params or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "models": self.models,
            "default_base_url": self.default_base_url,
            "supports_thinking": self.supports_thinking,
            "thinking_param": self.thinking_param,
        }

    def model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.models:
            if m["id"] == model_id:
                return m
        return None


# ── Provider registry ────────────────────────────────────────────────────────────

PROVIDERS: List[LLMProvider] = [
    # DeepSeek — thinking via extra_body={"thinking":{"type":"enabled"}, "reasoning_effort":"high"}
    LLMProvider(
        id="deepseek",
        name="DeepSeek",
        description="DeepSeek 系列模型（V3 / R1），性价比高",
        default_base_url="https://api.deepseek.com/v1",
        supports_thinking=True,
        thinking_param={
            "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        },
        thinking_disabled_params=["temperature", "top_p", "presence_penalty", "frequency_penalty"],
        models=[
            {
                "id": "deepseek-chat",
                "name": "DeepSeek V3 (deepseek-chat)",
                "description": "标准对话模型，快速低成本",
                "supports_thinking": False,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
            },
            {
                "id": "deepseek-reasoner",
                "name": "DeepSeek R1 (deepseek-reasoner)",
                "description": "深度推理模型，支持思考模式（Chain-of-Thought）",
                "supports_thinking": True,
                "default_temperature": 0.7,
                "default_max_tokens": 4000,
            },
            {
                "id": "deepseek-v4-pro",
                "name": "DeepSeek V4 Pro (deepseek-v4-pro)",
                "description": "最新深度推理模型，支持思考模式，精度最高",
                "supports_thinking": True,
                "default_temperature": 0.7,
                "default_max_tokens": 4000,
            },
        ],
    ),

    # Kimi — thinking via extra_body={"thinking":{"type":"enabled"}}
    LLMProvider(
        id="kimi",
        name="Kimi (Moonshot)",
        description="Moonshot Kimi 系列模型，支持长上下文",
        default_base_url="https://api.moonshot.cn/v1",
        supports_thinking=True,
        thinking_param={
            "extra_body": {"thinking": {"type": "enabled"}}
        },
        thinking_disabled_params=["temperature"],
        models=[
            {
                "id": "moonshot-v1-8k",
                "name": "Moonshot V1 8K",
                "description": "8K 上下文，适合短对话",
                "supports_thinking": False,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
            },
            {
                "id": "moonshot-v1-32k",
                "name": "Moonshot V1 32K",
                "description": "32K 上下文",
                "supports_thinking": False,
                "default_temperature": 0.7,
                "default_max_tokens": 4000,
            },
            {
                "id": "moonshot-v1-128k",
                "name": "Moonshot V1 128K",
                "description": "128K 超长上下文",
                "supports_thinking": False,
                "default_temperature": 0.7,
                "default_max_tokens": 4000,
            },
        ],
    ),

    # Minimax
    LLMProvider(
        id="minimax",
        name="Minimax",
        description="Minimax 海螺模型",
        default_base_url="https://api.minimax.chat/v1",
        supports_thinking=False,
        models=[
            {
                "id": "MiniMax-Text-01",
                "name": "MiniMax Text 01",
                "description": "MiniMax 高性能文本模型",
                "supports_thinking": False,
                "default_temperature": 0.7,
                "default_max_tokens": 2000,
            },
        ],
    ),

    # Custom / Other — user provides base_url and model_name manually
    LLMProvider(
        id="custom",
        name="Custom",
        description="自定义 API 地址和模型名称（兼容 OpenAI 格式）",
        default_base_url="https://api.openai.com/v1",
        supports_thinking=False,
        models=[],
    ),
]


def get_provider(provider_id: str) -> Optional[LLMProvider]:
    for p in PROVIDERS:
        if p.id == provider_id:
            return p
    return None


def provider_catalog() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in PROVIDERS]
