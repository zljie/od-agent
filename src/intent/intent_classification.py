"""Intent classification result and related data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IntentCandidate:
    """A candidate intent with its confidence score."""

    intent_type: str
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)


@dataclass
class IntentClassification:
    """Result of intent classification.

    Attributes:
        intent_type: The matched intent type (e.g. "date_range_diff")
        confidence: Confidence score [0.0, 1.0]
        entities: Extracted entity slots (e.g. {"start": "上周五", "end": "今天"})
        candidates: Top-N candidate intents for ambiguity detection
    """

    intent_type: str
    confidence: float
    entities: Dict[str, str] = field(default_factory=dict)
    candidates: List[IntentCandidate] = field(default_factory=list)

    def is_unknown(self) -> bool:
        return self.intent_type == "UNKNOWN" or self.confidence <= 0.0

    def is_ambiguous(self, gap: float = 0.15, min_confidence: float = 0.35) -> bool:
        """Check if top two candidates are too close (ambiguous).

        Args:
            gap: Minimum gap between top-2 confidence scores
            min_confidence: Both candidates must meet this floor
        """
        if len(self.candidates) < 2:
            return False
        top = self.candidates[0]
        second = self.candidates[1]
        return (
            top.confidence >= min_confidence
            and second.confidence >= min_confidence
            and (top.confidence - second.confidence) < gap
        )


# Module-level singleton — assigned after class so dataclass __init__ is unaffected
IntentClassification.UNKNOWN = IntentClassification(
    intent_type="UNKNOWN", confidence=0.0, entities={}, candidates=[]
)
