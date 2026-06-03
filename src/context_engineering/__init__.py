"""Context Engineering Layer for BeBISO Agent.

Retrieves and manages candidate context for the 5-step execution pipeline.

Modules:
- retriever: Context retrieval from all sources before filtering
- selector: Context selection based on step-specific relevance
- policy_engine: Policy checks (trust, freshness, security)
- assembler: Assembles LLMContextPackage from selected context
- models: Data models (LLMContextPackage, L1-L9 layers, enums)
"""

from src.context_engineering.models import (
    ContextPurpose,
    LLMContextPackage,
    TokenBudget,
    TrustLevel,
)
from src.context_engineering.policy_engine import ContextPolicyEngine
from src.context_engineering.retriever import (
    ContextRetriever,
    RetrievedContextItem,
    RetrievalResult,
    retrieve_context_for_step,
)
from src.context_engineering.selector import ContextSelector
from src.context_engineering.assembler import ContextAssembler

__all__ = [
    "ContextRetriever",
    "ContextSelector",
    "ContextPolicyEngine",
    "ContextAssembler",
    "ContextPurpose",
    "LLMContextPackage",
    "TokenBudget",
    "TrustLevel",
    "RetrievedContextItem",
    "RetrievalResult",
    "retrieve_context_for_step",
]
