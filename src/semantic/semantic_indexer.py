"""Semantic indexer for NL2GraphQL RAG enhancement.

Builds a vector index over OSI ai_context data to enable semantic search
from natural language queries to the nearest GraphQL operations.

Architecture:
  OSI ai_context → embeddings → vector index
                          ↓
  User NL query → semantic search → nearest GraphQL operation → MCP tool call

This is a GraphQL-native RAG: instead of retrieving documents for an LLM
to reason over, it retrieves the exact GraphQL operation needed to satisfy
the user's natural language intent.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .osi_model import AIContext, DataSet, OSIModel


@dataclass
class IndexedOperation:
    """A single indexed GraphQL operation with its semantic metadata."""

    operation_id: str
    operation_type: str
    dataset: str
    field_name: str
    natural_language_hints: List[str]
    synonyms: List[str]
    examples: List[str]
    description: str
    graphql_fragment: str

    @property
    def search_corpus(self) -> str:
        """All text that will be indexed for semantic search."""
        parts = [self.description, self.field_name, self.dataset]
        parts.extend(self.natural_language_hints)
        parts.extend(self.synonyms)
        parts.extend(self.examples)
        return " ".join(filter(None, parts))


@dataclass
class SemanticSearchResult:
    """A semantic search result with relevance score."""

    operation: IndexedOperation
    score: float
    matched_hint: Optional[str] = None


def cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SimpleEmbeddingFunction:
    """A lightweight TF-IDF-style embedding for environments without heavy ML deps.

    Uses Chinese word segmentation (jieba) when available, with character n-gram
    fingerprints as a robust fallback. For production, replace with
    OpenAI/text-embedding-3-small or a local model.
    """

    def __init__(self, ngram_size: int = 3, n_dims: int = 256):
        self._ngram_size = ngram_size
        self._n_dims = n_dims
        self._use_jieba = False
        try:
            import jieba
            jieba.setLogLevel(20)
            self._jieba = jieba
            self._use_jieba = True
        except ImportError:
            self._jieba = None

    def encode(self, text: str) -> List[float]:
        """Encode text into a fixed-dimension vector using token hashing.

        Uses jieba word segmentation (when available) + MMH3 hashing to produce
        a fixed-dimension vector regardless of vocabulary size. This avoids
        the dimension mismatch problem that occurs with raw TF-IDF vocabularies.
        """
        if self._use_jieba:
            tokens = [t.strip() for t in self._jieba.cut(text.lower()) if t.strip()]
        else:
            tokens = self._char_ngrams(text.lower())

        if not tokens:
            return [0.0] * self._n_dims

        vec = [0.0] * self._n_dims
        for token in tokens:
            idx = self._hash_token(token) % self._n_dims
            vec[idx] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _hash_token(self, token: str) -> int:
        try:
            import mmh3
            return mmh3.hash(token, signed=False)
        except ImportError:
            return hash(token) % (2**31)

    def _char_ngrams(self, text: str) -> List[str]:
        result = []
        for i in range(len(text) - self._ngram_size + 1):
            result.append(text[i : i + self._ngram_size])
        return result


class SemanticIndexer:
    """Builds and queries a semantic index over OSI model GraphQL operations.

    Provides RAG-style retrieval: natural language query → nearest operation.
    The index is built from ai_context (instructions, synonyms, examples)
    and GraphQL schema metadata.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        top_k: int = 3,
    ):
        self._embed_fn = embed_fn or SimpleEmbeddingFunction().encode
        self._top_k = top_k
        self._operations: Dict[str, IndexedOperation] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def index_model(self, model: OSIModel) -> int:
        """Index all datasets and operations from an OSI model.

        Returns the number of operations indexed.
        """
        self._operations.clear()
        self._embeddings.clear()

        for ds in model.datasets:
            self._index_dataset(ds)

        for action in model.actions:
            self._index_action(action)

        for op in self._operations.values():
            self._embeddings[op.operation_id] = self._embed_fn(op.search_corpus)

        return len(self._operations)

    def _index_dataset(self, ds: DataSet) -> None:
        ai_ctx = ds.ai_context

        op = IndexedOperation(
            operation_id=self._op_id("query", ds.name, ""),
            operation_type="query",
            dataset=ds.name,
            field_name=ds.name,
            natural_language_hints=[ds.description] if ds.description else [],
            synonyms=ai_ctx.synonyms if ai_ctx else [],
            examples=[ex for ex in (ai_ctx.examples if ai_ctx else []) if isinstance(ex, str)],
            description=ds.description or "",
            graphql_fragment=f"query {{ {ds.name.lower()}s {{ ...{ds.name}Fields }} }}",
        )
        self._operations[op.operation_id] = op

        for field_def in ds.fields:
            field_op = IndexedOperation(
                operation_id=self._op_id("query", ds.name, field_def.name),
                operation_type="query",
                dataset=ds.name,
                field_name=field_def.name,
                natural_language_hints=[field_def.description] if field_def.description else [],
                synonyms=_extract_synonyms(ai_ctx, field_def.name) if ai_ctx else [],
                examples=[],
                description=field_def.description or "",
                graphql_fragment=f"query {{ {ds.name.lower()}s {{ {field_def.name} }} }}",
            )
            self._embeddings[field_op.operation_id] = self._embed_fn(field_op.search_corpus)
            self._operations[field_op.operation_id] = field_op

    def _index_action(self, action: "Action") -> None:
        ai_ctx = action.ai_context
        op = IndexedOperation(
            operation_id=self._op_id("mutation", action.dataset, action.name),
            operation_type="mutation",
            dataset=action.dataset,
            field_name=action.name,
            natural_language_hints=[action.description] if action.description else [],
            synonyms=ai_ctx.synonyms if ai_ctx else [],
            examples=[ex for ex in (ai_ctx.examples if ai_ctx else []) if isinstance(ex, str)],
            description=action.description or "",
            graphql_fragment=f"mutation {{ {action.name}(...) {{ ...{action.name}PayloadFields }} }}",
        )
        self._embeddings[op.operation_id] = self._embed_fn(op.search_corpus)
        self._operations[op.operation_id] = op

    def search(self, query: str, top_k: Optional[int] = None) -> List[SemanticSearchResult]:
        """Search for the nearest GraphQL operation to a natural language query.

        Args:
            query: Natural language user query (e.g. "查一下这个供应商的采购情况")
            top_k: Number of results to return (default: self._top_k)

        Returns:
            List of SemanticSearchResult sorted by relevance score descending.
        """
        if not self._operations:
            return []

        k = top_k or self._top_k
        query_emb = self._embed_fn(query.lower())

        scored: List[Tuple[SemanticSearchResult, float]] = []
        for op_id, op_emb in self._embeddings.items():
            score = cosine_sim(query_emb, op_emb)
            scored.append((SemanticSearchResult(operation=self._operations[op_id], score=score), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]

    def _op_id(self, op_type: str, dataset: str, field: str) -> str:
        key = f"{op_type}:{dataset}:{field}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def get_operation(self, operation_id: str) -> Optional[IndexedOperation]:
        return self._operations.get(operation_id)


def _extract_synonyms(ai_ctx: AIContext, field_name: str) -> List[str]:
    if not ai_ctx or not ai_ctx.synonyms:
        return []
    return [s for s in ai_ctx.synonyms if field_name.lower() in s.lower()]
