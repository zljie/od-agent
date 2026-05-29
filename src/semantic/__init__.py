"""Semantic Ontology module — OSI YAML models and GraphQL/MCP generation.

Architecture (per docs/2026-05-28_本体-GraphQL-AI-语义原生后端概念.md):
┌──────────────────────────────────────────────────────────────────────┐
│                    AI Agent (Claude / GPT / Cursor Agent)            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ MCP 协议
┌──────────────────────────────▼───────────────────────────────────────┐
│                    MCP Server Layer                                    │
│         (Apollo MCP Server / Cosmo MCP)                                │
│     GraphQL Operations → AI-callable Tools                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ GraphQL Queries / Mutations
┌──────────────────────────────▼───────────────────────────────────────┐
│                    GraphQL API Layer                                  │
│       (Apollo Router / Cosmo Router)                                  │
│     Schema = Business Object API Contract                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ OSI semantic_model → GraphQL SDL
┌──────────────────────────────▼───────────────────────────────────────┐
│                    OSI / Ontology Layer                               │
│  Datasets / Relationships / Metrics / Actions /                      │
│  Rules / ai_context                                                  │
└──────────────────────────────────────────────────────────────────────┘
"""

from .osi_model import (
    OSIModel,
    DataSet,
    Relationship,
    Metric,
    Action,
    Rule,
    AIContext,
    FieldDefinition,
)

__all__ = [
    "OSIModel",
    "DataSet",
    "Relationship",
    "Metric",
    "Action",
    "Rule",
    "AIContext",
    "FieldDefinition",
]
