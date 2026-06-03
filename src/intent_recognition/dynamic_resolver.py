"""
Dynamic Slot Resolver
====================
Layer 7 of the BeBISO 7-layer intent recognition framework.

Resolves ambiguous or underspecified terms in the user input by querying
master data sources. Handles the following term types:

- "organization": department / organisational unit → ``DepartmentExtractor``
- "supplier": vendor name or ID → LLM-based master data lookup
- "material": material / product description → LLM-based lookup
- "user": employee name or ID → LLM-based lookup

Resolution outcomes:
- ``RESOLVED``: exactly one candidate found → use directly
- ``AMBIGUOUS``: multiple candidates found → surface to user or pick highest confidence
- ``UNRESOLVED``: no candidates found → flag as unresolved
"""

from typing import Any, Dict, List, Optional

from src.procurement.slot_extractors import DepartmentExtractor

from .models import (
    LLMLightResult,
    OntologyMatchResult,
    PreprocessingResult,
    ResolutionStatus,
    ResolvedValue,
)


class DynamicSlotResolver:
    """Resolves dynamic terms from user input against master data.

    Reuses ``DepartmentExtractor`` for organisation/department resolution
    (which has a static ``DEPARTMENTS`` mapping) and falls back to LLM
    for other term types (supplier, material, user).
    """

    DEFAULT_THINKING_BUDGET: int = 400

    def __init__(self, thinking_budget: int = DEFAULT_THINKING_BUDGET):
        self._thinking_budget = thinking_budget
        self._dept_extractor = DepartmentExtractor()
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, dynamic_term: str, term_type: str) -> ResolvedValue:
        """Resolve a single dynamic term to a canonical master-data value.

        Parameters
        ----------
        dynamic_term:
            The ambiguous or underspecified term from the user's input.
        term_type:
            One of ``"organization"``, ``"supplier"``, ``"material"``, ``"user"``.

        Returns
        -------
        ResolvedValue
            Contains the resolution status, candidate list, selected value,
            and confidence.
        """
        if term_type == "organization":
            return self._resolve_organization(dynamic_term)
        return self._resolve_via_llm(dynamic_term, term_type)

    def resolve_batch(
        self, tasks: List[Dict[str, Any]]
    ) -> List[ResolvedValue]:
        """Resolve multiple dynamic terms in a single pass.

        Parameters
        ----------
        tasks:
            List of dicts each containing ``"term"`` and ``"term_type"`` keys.

        Returns
        -------
        List[ResolvedValue]
            One ``ResolvedValue`` per task, in the same order.
        """
        results = []
        # Group by type: organisation terms can be resolved in-batch via extractor
        org_tasks = [(i, t) for i, t in enumerate(tasks) if t.get("term_type") == "organization"]
        llm_tasks = [(i, t) for i, t in enumerate(tasks) if t.get("term_type") != "organization"]

        # Pre-allocate result list
        results: List[Optional[ResolvedValue]] = [None] * len(tasks)

        # Resolve organisations via the extractor (static mapping — no LLM needed)
        for idx, task in org_tasks:
            results[idx] = self._resolve_organization(task["term"])

        # Resolve non-organisations via batched LLM
        if llm_tasks:
            llm_results = self._resolve_llm_batch(llm_tasks)
            for (idx, _), resolved in zip(llm_tasks, llm_results):
                results[idx] = resolved

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Organisation / department resolution
    # ------------------------------------------------------------------

    def _resolve_organization(self, term: str) -> ResolvedValue:
        """Resolve an organisation / department term via the static extractor."""
        slot_value = self._dept_extractor.extract(term)

        if slot_value is not None:
            return ResolvedValue(
                term=term,
                term_type="organization",
                status=ResolutionStatus.RESOLVED,
                candidates=[{"value": slot_value.value, "display": slot_value.display_value}],
                selected={"value": slot_value.value, "display": slot_value.display_value},
                confidence=slot_value.confidence,
            )

        # No direct keyword match — fall back to LLM for fuzzy resolution
        return self._resolve_via_llm(term, "organization")

    # ------------------------------------------------------------------
    # LLM-based resolution (supplier / material / user / fuzzy org)
    # ------------------------------------------------------------------

    def _resolve_via_llm(
        self, dynamic_term: str, term_type: str
    ) -> ResolvedValue:
        """Resolve a term via LLM master-data query."""
        try:
            model = self._get_model()
            prompt = self._build_resolution_prompt(dynamic_term, term_type)
            response = model.generate(prompt, thinking_budget=self._thinking_budget)
            return self._parse_llm_response(dynamic_term, term_type, response)
        except Exception as e:
            return ResolvedValue(
                term=dynamic_term,
                term_type=term_type,
                status=ResolutionStatus.UNRESOLVED,
                candidates=[],
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _resolve_llm_batch(
        self, tasks: List[tuple[int, Dict[str, Any]]]
    ) -> List[ResolvedValue]:
        """Resolve multiple non-org terms in one LLM call."""
        try:
            model = self._get_model()
            prompt = self._build_batch_prompt(tasks)
            response = model.generate(prompt, thinking_budget=self._thinking_budget)
            return self._parse_batch_response(tasks, response)
        except Exception as e:
            # Fall back to individual resolution on error
            return [
                self._resolve_via_llm(task["term"], task["term_type"])
                for _, task in tasks
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily load the default LLM model."""
        if self._model is None:
            from src.models import get_default_model
            self._model = get_default_model()
        return self._model

    def _build_resolution_prompt(
        self, dynamic_term: str, term_type: str
    ) -> str:
        type_hints = {
            "supplier": "供应商名称或ID（如'3M'、'华为'、'供应商编号'）",
            "material": "物料名称或描述（如'办公椅'、'A4纸'）",
            "user": "员工姓名或工号",
            "organization": "部门或组织单元名称",
        }
        hint = type_hints.get(term_type, "业务对象")
        return f"""你是企业采购系统的主数据查询助手。

给定一个动态术语，查询其对应的标准主数据记录。

## 待查询术语
- 术语: "{dynamic_term}"
- 类型: {term_type}（{hint}）

## 要求
1. 尝试从已知的主数据中匹配最可能的记录。
2. 如果有多个可能的匹配，记录为"候选"。
3. 如果没有匹配，返回空列表。
4. 每个匹配的记录需要包含：value（标准值）、display（显示名称）、confidence（置信度0-1）。

## 输出格式（仅返回JSON，不要包含其他内容）
{{
  "status": "resolved|ambiguous|unresolved",
  "candidates": [
    {{
      "value": "标准值",
      "display": "显示名称",
      "confidence": 0.0-1.0
    }}
  ],
  "selected": {{
    "value": "标准值",
    "display": "显示名称"
  }}
}}
"""

    def _build_batch_prompt(
        self, tasks: List[tuple[int, Dict[str, Any]]]
    ) -> str:
        lines = []
        for i, (_, task) in enumerate(tasks):
            lines.append(
                f"[{i+1}] 术语: \"{task['term']}\", 类型: {task['term_type']}"
            )
        return f"""你是企业采购系统的主数据查询助手。

批量查询多个动态术语对应的标准主数据记录。

## 待查询列表
{chr(10).join(lines)}

## 要求
1. 为每个术语独立查询主数据。
2. 返回JSON数组，每个元素对应一个术语的查询结果。
3. 每个元素包含：status、candidates、selected字段（同单个查询格式）。

## 输出格式（仅返回JSON数组，不要包含其他内容）
[
  {{"status": "resolved|ambiguous|unresolved", "candidates": [...], "selected": {{}}}},
  ...
]
"""

    def _parse_llm_response(
        self, dynamic_term: str, term_type: str, response: str
    ) -> ResolvedValue:
        """Parse a single LLM resolution response."""
        import json
        import re

        json_match = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", response)
        text = json_match.group() if json_match else response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return ResolvedValue(
                term=dynamic_term,
                term_type=term_type,
                status=ResolutionStatus.UNRESOLVED,
                candidates=[],
                confidence=0.0,
                metadata={"raw_response": response[:500]},
            )

        status_str = data.get("status", "unresolved")
        if status_str == "resolved":
            status = ResolutionStatus.RESOLVED
        elif status_str == "ambiguous":
            status = ResolutionStatus.AMBIGUOUS
        else:
            status = ResolutionStatus.UNRESOLVED

        candidates = data.get("candidates", [])
        selected = data.get("selected")
        confidence = 0.0
        if candidates:
            confidence = max(c.get("confidence", 0.0) for c in candidates)
        elif selected:
            confidence = selected.get("confidence", 0.0)

        return ResolvedValue(
            term=dynamic_term,
            term_type=term_type,
            status=status,
            candidates=candidates,
            selected=selected,
            confidence=confidence,
        )

    def _parse_batch_response(
        self, tasks: List[tuple[int, Dict[str, Any]]], response: str
    ) -> List[ResolvedValue]:
        """Parse a batch LLM resolution response."""
        import json
        import re

        json_match = re.search(r"\[[\s\S]*\]", response)
        text = json_match.group() if json_match else response.strip()

        try:
            data_list = json.loads(text)
        except json.JSONDecodeError:
            return [
                self._resolve_via_llm(task["term"], task["term_type"])
                for _, task in tasks
            ]

        results = []
        for i, (_, task) in enumerate(tasks):
            data = data_list[i] if i < len(data_list) else {}
            results.append(
                self._parse_single_batch_item(task, data)
            )
        return results

    def _parse_single_batch_item(
        self, task: Dict[str, Any], data: Dict[str, Any]
    ) -> ResolvedValue:
        """Parse a single item from a batch response."""
        status_str = data.get("status", "unresolved")
        if status_str == "resolved":
            status = ResolutionStatus.RESOLVED
        elif status_str == "ambiguous":
            status = ResolutionStatus.AMBIGUOUS
        else:
            status = ResolutionStatus.UNRESOLVED

        candidates = data.get("candidates", [])
        selected = data.get("selected")
        confidence = 0.0
        if candidates:
            confidence = max(c.get("confidence", 0.0) for c in candidates)

        return ResolvedValue(
            term=task["term"],
            term_type=task["term_type"],
            status=status,
            candidates=candidates,
            selected=selected,
            confidence=confidence,
        )
