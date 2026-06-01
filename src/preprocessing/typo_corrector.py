"""L3 TypoCorrector: Chinese text error correction.

Uses pycorrector (Qwen3-4B-CTC backend) with confidence-threshold gating.
Only auto-corrects when confidence >= threshold; lower-confidence corrections
are recorded but not applied.

Key principle (from 8层链路 design):
  - Correction Precision (误改率) > Correction Recall (漏改率)
  - raw_input is NEVER overwritten — corrections are always recorded separately
"""

from typing import Any, Dict, List, Optional, Tuple

from .cleaner import CorrectionRecord


class TypoCorrector:
    """Chinese text error corrector with configurable confidence threshold.

    Uses pycorrector.correct() internally. When pycorrector is unavailable,
    falls back to a no-op (returns text unchanged, no corrections).

    Threshold strategy:
        confidence >= 0.95  → auto-correct (high precision required)
        0.70 <= confidence < 0.95 → candidate pool (pass to LLM or show user)
        confidence < 0.70  → skip (too uncertain)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        candidate_threshold: float = 0.70,
        use_pycorrrector: bool = True,
    ):
        """
        Args:
            confidence_threshold: Only auto-correct if model confidence >= this.
            candidate_threshold:   Below this, skip entirely. Between threshold and
                                   candidate_threshold, add to candidate pool.
            use_pycorrrector:     If False, disable correction entirely (pass-through).
        """
        self._auto_threshold = confidence_threshold
        self._candidate_threshold = candidate_threshold
        self._use_pycorrrector = use_pycorrrector
        self._pycorrector = None
        self._available = False

        if self._use_pycorrrector:
            self._init_pycorrrector()

    def _init_pycorrrector(self) -> None:
        """Lazy-load pycorrector on first use."""
        try:
            import pycorrector
            self._pycorrector = pycorrector
            self._available = True
        except ImportError:
            self._pycorrector = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def correct(self, text: str) -> str:
        """Correct text and return the corrected version (auto-correct only).

        Low-confidence corrections are NOT applied; call correct_with_details()
        to get the full correction list.
        """
        corrected, _ = self.correct_with_details(text)
        return corrected

    def correct_with_details(
        self, text: str
    ) -> Tuple[str, List[CorrectionRecord]]:
        """Correct text and return corrected text + full correction records.

        Returns:
            (corrected_text, list_of_correction_records)

        Corrections are split into:
            - auto_corrected: confidence >= auto_threshold
            - candidate: confidence between candidate_threshold and auto_threshold
            - skipped: below candidate_threshold
        """
        if not text or not self._available:
            return text, []

        try:
            corrected_raw, detail_list = self._pycorrector.correct(text)
        except Exception:
            return text, []

        records: List[CorrectionRecord] = []
        applied_corrections: List[Tuple[int, int, str]] = []

        for item in detail_list:
            # pycorrector detail format:
            # {'position': int, 'correction': (wrong, right), 'error_type': str}
            pos = item.get("position", 0)
            correction = item.get("correction", (None, None))
            wrong, right = correction[0], correction[1]
            error_type = item.get("error_type", "unknown")
            confidence = item.get("confidence", 0.95)  # pycorrector may not provide this

            if wrong is None or right is None:
                continue

            rec = CorrectionRecord(
                original=wrong,
                corrected=right,
                position=pos,
                correction_type=error_type,
                confidence=confidence,
                method="pycorrector",
            )
            records.append(rec)

            # Apply only if confidence is high enough
            if confidence >= self._auto_threshold:
                applied_corrections.append((pos, len(wrong), right))

        # Apply corrections to text (work from end to start to preserve positions)
        corrected_text = text
        for pos, wlen, replacement in sorted(applied_corrections, key=lambda x: -x[0]):
            corrected_text = corrected_text[:pos] + replacement + corrected_text[pos + wlen :]

        return corrected_text, records

    def correct_candidates_only(self, text: str) -> Tuple[str, List[CorrectionRecord]]:
        """Return corrections in the candidate pool without applying them.

        Use this when you want to show users what COULD be corrected
        but let them decide whether to apply.
        """
        if not text or not self._available:
            return text, []

        try:
            _, detail_list = self._pycorrector.correct(text)
        except Exception:
            return text, []

        records: List[CorrectionRecord] = []
        for item in detail_list:
            pos = item.get("position", 0)
            correction = item.get("correction", (None, None))
            wrong, right = correction[0], correction[1]
            error_type = item.get("error_type", "unknown")
            confidence = item.get("confidence", 0.95)

            if wrong is None or right is None:
                continue

            rec = CorrectionRecord(
                original=wrong,
                corrected=right,
                position=pos,
                correction_type=error_type,
                confidence=confidence,
                method="pycorrector",
            )

            # Only include in candidate pool if between thresholds
            if self._candidate_threshold <= confidence < self._auto_threshold:
                records.append(rec)

        return text, records  # text is unchanged
