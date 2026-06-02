"""
采购管理模块
"""

from .ontology_loader import (
    ProcurementOntology,
    load_ontology_from_yaml,
    get_procurement_ontology,
    reload_ontology,
    OntologyQuery,
)
from .connector import ProcurementConnector, get_procurement_connector, ConnectorResponse
from .intents import PROCUREMENT_INTENTS, ProcurementIntentRouter
from .intent_router import IntentMatch, IntentRouter

__all__ = [
    # Ontology
    'ProcurementOntology',
    'load_ontology_from_yaml',
    'get_procurement_ontology',
    'reload_ontology',
    'OntologyQuery',
    # Connector
    'ProcurementConnector',
    'get_procurement_connector',
    'ConnectorResponse',
    # Intents
    'PROCUREMENT_INTENTS',
    'ProcurementIntentRouter',
    'IntentMatch',
    'IntentRouter',
]
