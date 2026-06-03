"""
Intent Topology Module
======================
Phase 1 of BeBISO intent recognition upgrade.
Maps procurement intents to executable paths through the ontology graph.

Exports:
    PathType: Enum for intent path execution status
    IntentPath: A single intent path node in the topology graph
    IntentTemplate: Template definitions for intent patterns
    IntentTopology: Complete topology with all paths and indexes
    IntentTopologyGenerator: Generates topology from ontology
    IntentTopologyLoader: Singleton loader / cache for IntentTopology
"""

from .models import (
    PathType,
    IntentPath,
    IntentTopology,
    INTENT_TEMPLATES,
)
from .generator import IntentTopologyGenerator
from .loader import IntentTopologyLoader, get_topology

__all__ = [
    "PathType",
    "IntentPath",
    "IntentTopology",
    "INTENT_TEMPLATES",
    "IntentTopologyGenerator",
    "IntentTopologyLoader",
    "get_topology",
]
