"""OSI (Open Semantic Interchange) data models.

Corresponds to the semantic_model.yaml format described in the concept doc:
- datasets: core business objects (Supplier, PurchaseOrder, Material, etc.)
- relationships: how datasets relate to each other
- metrics: business KPIs derived from datasets
- behavior.actions: operations that can be performed
- behavior.rules: business rules / governance constraints
- ai_context: NL2GraphQL augmentation (instructions, synonyms, examples)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldType(str, Enum):
    """GraphQL-compatible scalar types."""

    ID = "ID"
    STRING = "String"
    INT = "Int"
    FLOAT = "Float"
    BOOLEAN = "Boolean"
    DATETIME = "DateTime"
    JSON = "JSON"
    ENUM = "Enum"


@dataclass
class FieldDefinition:
    """A field within a dataset.

    Maps to a GraphQL field with optional enum, relation, or AI context.
    """

    name: str
    gql_type: str
    description: str = ""
    required: bool = False
    is_list: bool = False
    enum_values: Optional[List[str]] = None
    enum_name: Optional[str] = None
    relation_target: Optional[str] = None
    ai_hint: Optional[str] = None


@dataclass
class DataSet:
    """A business entity / dataset in the OSI model.

    Maps directly to a GraphQL Object Type.
    ai_context.instructions becomes the type's description.
    """

    name: str
    description: str = ""
    fields: List[FieldDefinition] = field(default_factory=list)
    ai_context: Optional["AIContext"] = None

    def primary_key_field(self) -> Optional[FieldDefinition]:
        for f in self.fields:
            if f.gql_type == "ID":
                return f
        return self.fields[0] if self.fields else None


@dataclass
class Relationship:
    """How two datasets relate to each other.

    Maps to GraphQL field references (e.g. supplier.purchaseOrders).
    """

    from_dataset: str
    to_dataset: str
    relation_type: str = "MANY_TO_ONE"
    description: str = ""
    via_field: Optional[str] = None
    ai_hint: Optional[str] = None


class MetricAggregation(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"


@dataclass
class Metric:
    """A derived business metric.

    Maps to a GraphQL Query field that computes an aggregate.
    """

    name: str
    description: str = ""
    unit: Optional[str] = None
    dataset: str = ""
    aggregation: Optional[MetricAggregation] = None
    filter_fields: List[str] = field(default_factory=list)
    ai_context: Optional["AIContext"] = None


@dataclass
class ActionParameter:
    name: str
    gql_type: str
    description: str = ""
    required: bool = True
    enum_values: Optional[List[str]] = None


@dataclass
class Action:
    """An operation that can be performed on a dataset.

    Maps to a GraphQL Mutation type.
    The status machine (Active/Blocked/Suspended) maps to enum constraints.
    """

    name: str
    dataset: str
    description: str = ""
    parameters: List[ActionParameter] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    post_conditions: List[str] = field(default_factory=list)
    status_transitions: Optional[List[str]] = None
    ai_context: Optional["AIContext"] = None


@dataclass
class Rule:
    """A business rule or governance constraint.

    Maps to GraphQL schema descriptions (runtime constraint hints) and
    becomes the ai_context.instructions field that MCP exposes to AI agents.
    """

    name: str
    description: str = ""
    severity: str = "WARNING"
    dataset: Optional[str] = None
    condition: str = ""
    enforcement: str = "WARN"
    ai_context: Optional["AIContext"] = None


@dataclass
class AIContext:
    """NL2GraphQL augmentation for a dataset or field.

    ai_context.instructions: business-semantic rules the AI must follow
    ai_context.synonyms: alternate names for the entity (供应商 = vendor = supplier)
    ai_context.examples: natural-language → GraphQL query patterns

    These are converted into GraphQL field/type descriptions and MCP tool metadata.
    """

    instructions: str = ""
    synonyms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_description(self) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        if self.synonyms:
            parts.append(f"同义词: {', '.join(self.synonyms)}")
        return "\n\n".join(parts)

    def to_gql_description(self) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        if self.synonyms:
            parts.append(f"同义词: {', '.join(self.synonyms)}")
        return '"""\n' + "\n\n".join(parts) + '\n"""'


@dataclass
class OSIModel:
    """The complete OSI semantic model for a business domain.

    Loaded from semantic_model.yaml. Used by GraphQLGenerator to produce
    a GraphQL SDL, and by SemanticIndexer to build the RAG vector index.
    """

    version: str = "1.0"
    domain: str = ""
    description: str = ""
    datasets: List[DataSet] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)

    def dataset_map(self) -> Dict[str, DataSet]:
        return {ds.name: ds for ds in self.datasets}

    def relationship_map(self) -> Dict[str, List[Relationship]]:
        m: Dict[str, List[Relationship]] = {}
        for r in self.relationships:
            m.setdefault(r.from_dataset, []).append(r)
            m.setdefault(r.to_dataset, []).append(r)
        return m

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "OSIModel":
        """Build an OSIModel from a parsed YAML dict."""
        datasets = [DataSet(**ds) for ds in data.get("datasets", [])]
        relationships = [Relationship(**r) for r in data.get("relationships", [])]
        metrics = [Metric(**m) for m in data.get("metrics", [])]
        actions = [Action(**a) for a in data.get("behavior", {}).get("actions", [])]
        rules = [Rule(**r) for r in data.get("behavior", {}).get("rules", [])]
        return cls(
            version=data.get("version", "1.0"),
            domain=data.get("domain", ""),
            description=data.get("description", ""),
            datasets=datasets,
            relationships=relationships,
            metrics=metrics,
            actions=actions,
            rules=rules,
        )
