"""
Intent Topology Loader
====================
Singleton loader for IntentTopology with caching and text-based candidate filtering.

Usage::

    from src.intent_topology import IntentTopologyLoader, get_topology

    topology = get_topology()                        # cached singleton
    topology = IntentTopologyLoader.reload()         # force reload
    candidates = IntentTopologyLoader.find_candidate_paths(
        "帮我查一下采购需求", topology
    )
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.procurement.ontology_loader import get_procurement_ontology

from .generator import IntentTopologyGenerator
from .models import IntentPath, IntentTopology, PathType


# ----------------------------------------------------------------------
# Singleton Cache
# ----------------------------------------------------------------------

class IntentTopologyLoader:
    """Singleton loader / cache for ``IntentTopology``.

    Thread-unsafe but suitable for single-process use in an Agent runtime.
    """

    _instance: Optional[IntentTopology] = None
    _loaded_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @classmethod
    def get_topology(cls, force_reload: bool = False) -> IntentTopology:
        """Return the cached IntentTopology, generating it if needed.

        Parameters
        ----------
        force_reload:
            If True, discard the cache and regenerate from the ontology.
        """
        if cls._instance is None or force_reload:
            ontology = get_procurement_ontology()
            generator = IntentTopologyGenerator(ontology)
            cls._instance = generator.generate()
        return cls._instance

    @classmethod
    def reload(cls) -> IntentTopology:
        """Force reload the topology from the current ontology."""
        return cls.get_topology(force_reload=True)

    @classmethod
    def clear(cls) -> None:
        """Clear the cached topology (mainly for testing)."""
        cls._instance = None
        cls._loaded_at = None

    # ------------------------------------------------------------------
    # Candidate path matching
    # ------------------------------------------------------------------

    @staticmethod
    def find_candidate_paths(
        text: str,
        topology: IntentTopology,
        top_k: int = 5,
    ) -> List[IntentPath]:
        """Find candidate IntentPaths that may match the user input.

        Performs lightweight text matching against:
        - IntentPath.id
        - IntentPath.object (dataset name and label)
        - IntentPath.action (action ID)
        - IntentPath.dimensions
        - IntentPath.description
        - IntentPath.example_utterances

        Parameters
        ----------
        text:
            Raw user input text (will be lowercased and whitespace-normalised).
        topology:
            The IntentTopology to search within.
        top_k:
            Maximum number of candidates to return.

        Returns
        -------
        List[IntentPath]
            Top-K candidates sorted by estimated relevance score.
        """
        if not text or not text.strip():
            return []

        import re

        norm = _normalize_for_match(text)

        scores: List[tuple] = []
        for path in topology.paths:
            score = _score_path(path, norm)
            if score > 0:
                scores.append((score, path))

        # Sort descending by score, then by confidence
        scores.sort(key=lambda x: (x[0], x[1].confidence), reverse=True)
        return [path for _, path in scores[:top_k]]

    @staticmethod
    def find_candidate_paths_by_type(
        text: str,
        topology: IntentTopology,
        path_type: PathType,
        top_k: int = 3,
    ) -> List[IntentPath]:
        """Find candidate IntentPaths filtered by path type."""
        typed_topology = IntentTopology(
            version=topology.version,
            paths=[
                p for p in topology.paths if p.path_type == path_type
            ],
            object_action_map=topology.object_action_map,
            rule_index=topology.rule_index,
        )
        return IntentTopologyLoader.find_candidate_paths(
            text, typed_topology, top_k=top_k
        )


# ----------------------------------------------------------------------
# Internal scoring helpers
# ----------------------------------------------------------------------

def _normalize_for_match(text: str) -> str:
    """Lightweight normalisation for keyword matching."""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    # Remove common punctuation
    text = re.sub(r"[。，、！？；：""''【】（）]", "", text)
    return text


def _score_path(path: IntentPath, norm_text: str) -> float:
    """Compute a relevance score [0.0, 1.0] for a path against normalized text."""
    score = 0.0

    # --- Exact object name match (highest weight) ---
    if path.object in norm_text or path.object.replace("_", "") in norm_text:
        score += 0.5

    # --- Action ID match ---
    action_key = path.action.split("/")[-1] if "/" in path.action else path.action
    if action_key in norm_text:
        score += 0.4

    # --- Dataset label / synonym match ---
    # Check path.object label-ish substrings
    for segment in path.id.split("."):
        if len(segment) > 2 and segment in norm_text:
            score += 0.3

    # --- Example utterance match ---
    for ex in path.example_utterances:
        ex_norm = _normalize_for_match(ex)
        if ex_norm in norm_text or norm_text in ex_norm:
            score += 0.25
        elif any(seg in ex_norm for seg in norm_text.split("_") if len(seg) > 2):
            score += 0.1

    # --- Dimension match ---
    for dim in path.dimensions:
        if dim in norm_text or dim.replace("_", "") in norm_text:
            score += 0.15

    # --- Description match ---
    desc_norm = _normalize_for_match(path.description)
    # Check for shared significant terms (>=3 chars)
    norm_tokens = set(norm_text)
    desc_tokens = set(desc_norm)
    shared = norm_tokens & desc_tokens
    if len(shared) >= 3:
        score += 0.1

    return min(score, 1.0)


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

def get_topology(force_reload: bool = False) -> IntentTopology:
    """Module-level shortcut for IntentTopologyLoader.get_topology."""
    return IntentTopologyLoader.get_topology(force_reload=force_reload)
