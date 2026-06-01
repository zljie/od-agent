"""L2 TermProtector: business term whitelist protection before typo correction.

Business terms (e.g. BeBIOS, OpenClaw, Dify) must not be modified by
the typo corrector. This layer replaces them with placeholders BEFORE
correction and restores them AFTER.

Supports:
- Exact-match term protection
- Case-sensitive / case-insensitive modes
- Hot-reload from JSON config
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default term whitelist — supplement from config/term_protection.json
_DEFAULT_TERMS = [
    # Tech/product names
    "BeBIOS", "BeEver", "OpenClaw", "Dify",
    # Domain terms (procurement)
    "M-001", "PO-001", "PR-001",
    # Common acronyms that get "corrected"
    "API", "SQL", "HTTP", "URL", "JSON", "XML", "GPT", "LLM",
]


class TermProtector:
    """Protects business terms from being modified by downstream processors.

    Strategy: replace terms with sequential placeholders __TP_0__, __TP_1__, ...
    before correction, then restore them after.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        terms: Optional[List[str]] = None,
        placeholder_prefix: str = "__TP_",
        placeholder_suffix: str = "__",
        case_sensitive: bool = False,
    ):
        self._prefix = placeholder_prefix
        self._suffix = placeholder_suffix
        self._case_sensitive = case_sensitive
        self._mapping: Dict[str, str] = {}  # placeholder → original term
        self._terms: List[str] = list(_DEFAULT_TERMS)
        self._term_pattern: Optional[re.Pattern] = None

        if terms:
            self._terms.extend(terms)

        if config_path:
            self._load_from_file(config_path)

        self._build_pattern()

    def _load_from_file(self, path: str) -> None:
        """Load terms from a JSON config file."""
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            terms = cfg.get("terms", [])
            if terms:
                self._terms.extend(terms)
            # Override placeholder format if specified
            self._prefix = cfg.get("placeholder_prefix", self._prefix)
            self._suffix = cfg.get("placeholder_suffix", self._suffix)
            self._case_sensitive = cfg.get("case_sensitive", self._case_sensitive)
        except Exception:
            pass

    def _build_pattern(self) -> None:
        """Compile a single regex that matches all terms (longest first)."""
        # Sort by length descending to avoid partial matches (e.g. "BeBIOS" before "BIOS")
        sorted_terms = sorted(set(self._terms), key=len, reverse=True)
        if not sorted_terms:
            self._term_pattern = re.compile(r"(?!x)x")  # never matches
            return

        # Escape each term and join with alternation
        escaped = [re.escape(t) for t in sorted_terms]
        pattern_str = "|".join(escaped)
        flags = 0 if self._case_sensitive else re.IGNORECASE
        self._term_pattern = re.compile(pattern_str, flags)

    def protect(self, text: str) -> str:
        """Replace all protected terms with placeholders.

        Returns the text with terms replaced. Use restore() to get them back.
        """
        self._mapping.clear()
        idx = 0

        def replacer(m: re.Match) -> str:
            nonlocal idx
            original = m.group(0)
            placeholder = f"{self._prefix}{idx}{self._suffix}"
            self._mapping[placeholder] = original
            idx += 1
            return placeholder

        return self._term_pattern.sub(replacer, text)

    def protect_with_mapping(
        self, text: str
    ) -> Tuple[str, List[Tuple[int, int, str]], List[str]]:
        """Replace terms with placeholders and return position mapping.

        Returns:
            (protected_text, mapping, protected_term_list)
            mapping: [(start, end, original_term), ...]  # positions in protected_text
        """
        self._mapping.clear()
        idx = 0
        mapping: List[Tuple[int, int, str]] = []
        protected_terms: List[str] = []

        def replacer(m: re.Match) -> str:
            nonlocal idx
            original = m.group(0)
            placeholder = f"{self._prefix}{idx}{self._suffix}"
            self._mapping[placeholder] = original
            mapping.append((m.start(), m.end(), original))
            protected_terms.append(original)
            idx += 1
            return placeholder

        protected = self._term_pattern.sub(replacer, text)
        return protected, mapping, protected_terms

    def restore(self, text: str) -> str:
        """Restore all placeholders back to original terms."""
        for placeholder, original in self._mapping.items():
            text = text.replace(placeholder, original)
        return text

    def get_mapping(self) -> Dict[str, str]:
        """Return the current placeholder → original term mapping."""
        return dict(self._mapping)

    def add_terms(self, terms: List[str]) -> None:
        """Dynamically add terms and rebuild the pattern."""
        self._terms.extend(terms)
        self._build_pattern()

    def add_terms_from_ontology(self, sem_skill: Any, max_synonyms: int = 50) -> int:
        """Load business synonyms from the SemanticSkill's ontology ai_context.

        Iterates all datasets in the OSIModel and collects synonyms from ai_context,
        adding them to the protected term list.

        Returns the number of synonyms added.
        """
        if not hasattr(sem_skill, "backend") or sem_skill.backend is None:
            return 0

        try:
            model = sem_skill.backend.model
        except Exception:
            return 0

        added = 0
        for ds in model.datasets:
            ai_ctx = getattr(ds, "ai_context", None)
            if ai_ctx and hasattr(ai_ctx, "synonyms"):
                for syn in ai_ctx.synonyms[:max_synonyms]:
                    if syn and syn.strip() and syn not in self._terms:
                        self._terms.append(syn.strip())
                        added += 1

        if added > 0:
            self._build_pattern()

        return added

    @property
    def term_count(self) -> int:
        return len(self._terms)
