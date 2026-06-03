from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class RedactionResult:
    """Result of a redaction operation."""
    redacted_text: str
    matches: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return len(self.matches)


class ContextRedactor:
    """Redact sensitive data from text, dicts, and context packages."""

    DEFAULT_PATTERNS: List[Dict[str, Any]] = [
        {
            "name": "supplier_price",
            "pattern": r"(供应商|supplier|报价)[：:\s]*([¥$]?\d+(?:\.\d{1,2})?)",
            "replacement": r"\1: [价格已脱敏]",
            "description": "Supplier prices",
            "mode": "mask",
        },
        {
            "name": "mobile_phone",
            "pattern": r"1[3-9]\d{9}",
            "replacement": "[手机号已脱敏]",
            "description": "Mobile numbers",
            "mode": "mask",
        },
        {
            "name": "id_card",
            "pattern": r"\d{15}|\d{18}|\d{17}X",
            "replacement": "[身份证号已脱敏]",
            "description": "ID card numbers",
            "mode": "mask",
        },
        {
            "name": "bank_account",
            "pattern": r"\d{12,19}",
            "replacement": "[银行账号已脱敏]",
            "description": "Bank account numbers",
            "mode": "mask",
        },
        {
            "name": "email",
            "pattern": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            "replacement": "[邮箱已脱敏]",
            "description": "Email addresses",
            "mode": "mask",
        },
        {
            "name": "salary",
            "pattern": r"(工资|薪资|薪酬)[：:\s]*([¥$]?\d+(?:\.\d{1,2})?(?:元|k|K)?)",
            "replacement": r"\1: [薪资已脱敏]",
            "description": "Salary information",
            "mode": "mask",
        },
    ]

    def __init__(self, custom_patterns: Optional[List[Dict[str, Any]]] = None):
        """Initialize with default patterns merged with custom patterns."""
        self._patterns: List[Dict[str, Any]] = []

        for pattern in self.DEFAULT_PATTERNS:
            self._patterns.append(pattern.copy())

        if custom_patterns:
            for custom in custom_patterns:
                self.add_pattern(
                    name=custom["name"],
                    pattern=custom["pattern"],
                    replacement=custom.get("replacement", "[已脱敏]"),
                    description=custom.get("description", ""),
                )

    def add_pattern(
        self,
        name: str,
        pattern: str,
        replacement: str,
        description: str = "",
    ) -> None:
        """Add a custom pattern at runtime."""
        self.remove_pattern(name)
        self._patterns.append({
            "name": name,
            "pattern": pattern,
            "replacement": replacement,
            "description": description,
            "mode": "mask",
        })

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern by name."""
        self._patterns = [p for p in self._patterns if p["name"] != name]

    def _apply_pattern(
        self,
        text: str,
        pattern: Dict[str, Any],
        dry_run: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply a single pattern to text, return (new_text, matches)."""
        compiled = re.compile(pattern["pattern"])
        matches: List[Dict[str, Any]] = []

        if dry_run:
            for match in compiled.finditer(text):
                matches.append({
                    "pattern_name": pattern["name"],
                    "original": match.group(0),
                    "replacement": pattern["replacement"],
                })
            return text, matches

        def replacer(match: re.Match) -> str:
            original = match.group(0)
            matches.append({
                "pattern_name": pattern["name"],
                "original": original,
                "replacement": pattern["replacement"],
            })
            return pattern["replacement"]

        new_text = compiled.sub(replacer, text)
        return new_text, matches

    def redact_text(self, text: str, dry_run: bool = False) -> RedactionResult:
        """Apply all patterns to text, return RedactionResult."""
        all_matches: List[Dict[str, Any]] = []
        result_text = text

        for pattern in self._patterns:
            result_text, matches = self._apply_pattern(result_text, pattern, dry_run)
            all_matches.extend(matches)

        return RedactionResult(
            redacted_text=result_text,
            matches=all_matches,
        )

    def redact_dict(
        self,
        data: Dict,
        dry_run: bool = False,
    ) -> Tuple[Dict, RedactionResult]:
        """Recursively apply redact_text to all string values in dict."""
        all_matches: List[Dict[str, Any]] = []

        def redact_recursive(obj: Any) -> Any:
            if isinstance(obj, str):
                result = self.redact_text(obj, dry_run)
                all_matches.extend(result.matches)
                return result.redacted_text
            elif isinstance(obj, dict):
                return {k: redact_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(redact_recursive(item) for item in obj)
            else:
                return obj

        redacted_data = redact_recursive(data)
        return redacted_data, RedactionResult(
            redacted_text="",
            matches=all_matches,
        )

    def redact_context_package(
        self,
        package: Any,
        dry_run: bool = False,
    ) -> Tuple[Any, RedactionResult]:
        """Redact a context package with .to_dict() method or a plain dict."""
        all_matches: List[Dict[str, Any]] = []

        if hasattr(package, 'to_dict'):
            raw_dict = package.to_dict()
        elif isinstance(package, dict):
            raw_dict = package
        elif hasattr(package, '__dict__'):
            raw_dict = {
                k: v for k, v in package.__dict__.items()
                if not k.startswith('_')
            }
        else:
            raw_dict = package

        def redact_recursive(obj: Any) -> Any:
            if isinstance(obj, str):
                result = self.redact_text(obj, dry_run)
                all_matches.extend(result.matches)
                return result.redacted_text
            elif isinstance(obj, dict):
                return {k: redact_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(redact_recursive(item) for item in obj)
            else:
                return obj

        redacted_dict = redact_recursive(raw_dict)

        if hasattr(package, 'to_dict'):
            try:
                return type(package)(**redacted_dict), RedactionResult(
                    redacted_text="",
                    matches=all_matches,
                )
            except (TypeError, ValueError):
                return redacted_dict, RedactionResult(
                    redacted_text="",
                    matches=all_matches,
                )
        elif isinstance(package, dict):
            return redacted_dict, RedactionResult(
                redacted_text="",
                matches=all_matches,
            )
        elif hasattr(package, '__dataclass_fields__'):
            try:
                return type(package)(**redacted_dict), RedactionResult(
                    redacted_text="",
                    matches=all_matches,
                )
            except (TypeError, ValueError):
                return redacted_dict, RedactionResult(
                    redacted_text="",
                    matches=all_matches,
                )
        else:
            return redacted_dict, RedactionResult(
                redacted_text="",
                matches=all_matches,
            )
