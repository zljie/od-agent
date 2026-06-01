"""L1 BasicCleaner: deterministic text normalization without LLM.

Handles:
- Whitespace normalization
- Full/half-width character conversion
- Punctuation standardization
- Repeated punctuation compression
- Language detection hint
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class CorrectionRecord:
    """A single correction made during preprocessing."""

    original: str
    corrected: str
    position: int
    correction_type: str
    confidence: float = 1.0
    method: str = "rule"


@dataclass
class PreprocessingResult:
    """Result from the full preprocessing pipeline."""

    raw_input: str
    normalized_input: str
    corrections: List[CorrectionRecord] = field(default_factory=list)
    protected_terms: List[str] = field(default_factory=list)
    language: str = "zh-CN"
    risk_level: str = "low"

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0


# Chinese fullwidth → halfwidth mapping for common ASCII characters
_FULLWIDTH_OFFSET = ord("\uff01") - ord("!")


def _is_fullwidth(char: str) -> bool:
    cp = ord(char)
    return 0xFF01 <= cp <= 0xFF5E


def _fullwidth_to_halfwidth(char: str) -> str:
    """Convert a fullwidth character to its halfwidth equivalent."""
    cp = ord(char)
    if 0xFF01 <= cp <= 0xFF5E:
        return chr(cp - _FULLWIDTH_OFFSET)
    if cp == 0x3000:  # fullwidth space
        return " "
    return char


def _detect_language(text: str) -> str:
    """Simple language detection based on character ranges.

    Returns: "zh-CN", "en", "mixed"
    """
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isalpha() and not ("\u4e00" <= c <= "\u9fff"))
    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return "unknown"

    if chinese / total > 0.3:
        return "zh-CN"
    if latin / total > 0.7:
        return "en"
    return "mixed"


class BasicCleaner:
    """Deterministic input cleaner for chatbot preprocessing (L1).

    All transformations are rule-based, require no ML/LLM, and are
    designed to be lossless for business-meaningful characters.
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        normalize_punctuation: bool = True,
        compress_repeated_punct: bool = True,
        convert_fullwidth: bool = True,
    ):
        self._normalize_whitespace = normalize_whitespace
        self._normalize_punctuation = normalize_punctuation
        self._compress_repeated_punct = compress_repeated_punct
        self._convert_fullwidth = convert_fullwidth

    def clean(self, text: str) -> str:
        """Apply all enabled cleaning steps to text.

        Returns the cleaned text. Does not modify the original.
        """
        if not text:
            return text

        if self._normalize_whitespace:
            text = self._normalize_ws(text)

        if self._convert_fullwidth:
            text = self._convert_fw_chars(text)

        if self._normalize_punctuation:
            text = self._normalize_punct(text)

        if self._compress_repeated_punct:
            text = self._compress_punct(text)

        return text.strip()

    def _normalize_ws(self, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/tabs, strip line ends."""
        # Collapse runs of whitespace to single space
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize line endings
        text = re.sub(r"[\r\n]+", "\n", text)
        # Remove trailing/leading whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _convert_fw_chars(self, text: str) -> str:
        """Convert fullwidth ASCII characters to halfwidth."""
        return "".join(_fullwidth_to_halfwidth(c) for c in text)

    # Punctuation normalization: maps variant forms to canonical forms
    _PUNCT_MAP: Dict[str, str] = {
        "\u2018": "'",  # ' → '
        "\u2019": "'",  # ' → '
        "\u201c": '"',  # " → "
        "\u201d": '"',  # " → "
        "\u2013": "-",  # en dash → hyphen
        "\u2014": "-",  # em dash → hyphen
        "\u3001": ",",  # Chinese comma → ASCII comma
        "\u3002": ".",  # Chinese period → ASCII period
        "\uff0c": ",",  # Fullwidth comma → ASCII comma
        "\uff0e": ".",  # Fullwidth period → ASCII period
        "\u300a": "《",  # left guillemet
        "\u300b": "》",  # right guillemet
        "\u3008": "<",  # left angle bracket
        "\u3009": ">",  # right angle bracket
    }

    def _normalize_punct(self, text: str) -> str:
        """Normalize variant punctuation to canonical ASCII forms."""
        for src, dst in self._PUNCT_MAP.items():
            text = text.replace(src, dst)
        return text

    # Repeated punctuation patterns: collapse to max N repeats
    _PUNCT_REPEAT_PATTERNS: List[Tuple[str, int]] = [
        (r"[!?\.。]{3,}", 2),  # !!! or ！！！ → !!
        (r"[,~]{3,}", 2),      # ~~~ → ~~
        (r"[-_=]{3,}", 2),     # --- → --
    ]

    def _compress_punct(self, text: str) -> str:
        """Compress runs of repeated punctuation to a maximum length."""
        for pattern, max_count in self._PUNCT_REPEAT_PATTERNS:
            text = re.sub(pattern, lambda m: m.group(0)[0] * max_count, text)
        return text

    def detect_language(self, text: str) -> str:
        """Return language code: zh-CN, en, mixed, or unknown."""
        return _detect_language(text)
