"""
HITL Clarification Engine
========================
Layer 6 of the BeBISO 7-layer intent recognition framework.

Invoked when the Layer-5 ABC fusion score falls below the 0.75 threshold,
indicating insufficient confidence even after deep reasoning.

The engine generates a clarification request (HITLRequest) that asks the user
to disambiguate between candidate intent paths. It uses a dedicated LLM prompt
(Section 18.3 of BeBISO spec) when available, and falls back to template-based
generation when the LLM is unavailable.

Falls back to template-based generation when no LLM is available.
"""

from typing import Any, Dict, List, Optional

from src.intent_topology import IntentPath

from .models import ClarificationOption, HITLRequest


class HITLClarificationEngine:
    """Generates human-in-the-loop clarification requests for low-confidence intents.

    The engine produces a structured ``HITLRequest`` containing:
    - A natural-language clarification question
    - 2-5 concrete options mapping to intent paths
    - A recommended default option
    """

    DEFAULT_THINKING_BUDGET: int = 600

    def __init__(self, thinking_budget: int = DEFAULT_THINKING_BUDGET):
        self._thinking_budget = thinking_budget
        self._model = None
        self._confirmation_manager = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        known_facts: List[str],
        unclear_points: List[str],
        candidate_paths: List[IntentPath],
        ranked_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> HITLRequest:
        """Generate a clarification request for human disambiguation.

        Parameters
        ----------
        known_facts:
            Things the system already understands from the user's input.
        unclear_points:
            Ambiguous or unresolved aspects that need clarification.
        candidate_paths:
            ``IntentPath`` nodes from the ontology topology that are
            plausible matches for the user's intent.
        ranked_candidates:
            Optional ranked intent candidates from Layer-4 deep reasoning.

        Returns
        -------
        HITLRequest
            Structured request containing the question, options, and defaults.
        """
        try:
            return self._generate_with_llm(
                known_facts, unclear_points, candidate_paths, ranked_candidates
            )
        except Exception as e:
            return self._generate_fallback(
                known_facts, unclear_points, candidate_paths, ranked_candidates
            )

    # ------------------------------------------------------------------
    # LLM-based generation
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily load the default LLM model."""
        if self._model is None:
            from src.models import get_default_model
            self._model = get_default_model()
        return self._model

    def _generate_with_llm(
        self,
        known_facts: List[str],
        unclear_points: List[str],
        candidate_paths: List[IntentPath],
        ranked_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> HITLRequest:
        """Use LLM to generate a natural clarification question and options."""
        model = self._get_model()

        paths_text = self._format_paths(candidate_paths)
        candidates_text = self._format_candidates(ranked_candidates or [])
        prompt = self._build_prompt(
            known_facts, unclear_points, paths_text, candidates_text
        )

        response = model.generate(prompt, thinking_budget=self._thinking_budget)
        return self._parse_llm_response(response, candidate_paths, ranked_candidates)

    def _build_prompt(
        self,
        known_facts: List[str],
        unclear_points: List[str],
        paths_text: str,
        candidates_text: str,
    ) -> str:
        """Build the HITL clarification prompt per Section 18.3."""
        return f"""你是企业业务助手的澄清问题生成器。

给定已知信息和候选意图路径，生成一个自然的澄清问题，并提供多个具体选项供用户选择。

## 已理解的信息
{chr(10).join(f"- {f}" for f in known_facts)}

## 需要澄清的问题点
{chr(10).join(f"- {p}" for p in unclear_points)}

## 候选意图路径
{paths_text}

## 深度推理候选（可选）
{candidates_text}

## 要求
1. 生成一个清晰、自然的中文澄清问题。
2. 生成2-5个具体选项，每个选项对应一个明确的意图路径。
3. 每个选项的label应该简洁明确（如"查询采购需求"）。
4. 每个选项的description应该解释这个选择的具体含义。
5. 标记一个推荐的默认选项（recommended_default）。

## 输出格式（仅返回JSON，不要包含其他内容）
{{
  "question": "您的澄清问题",
  "options": [
    {{
      "option_id": "opt_1",
      "label": "选项标签",
      "description": "选项解释",
      "intent_path_id": "对应的IntentPath.id"
    }}
  ],
  "recommended_default": "opt_1"
}}
"""

    def _parse_llm_response(
        self,
        response: str,
        candidate_paths: List[IntentPath],
        ranked_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> HITLRequest:
        """Parse the LLM JSON response into a HITLRequest."""
        import json
        import re

        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            text = json_match.group()
        else:
            text = response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._generate_fallback(
                [], [], candidate_paths, ranked_candidates
            )

        # Map options back to IntentPath ids where possible
        path_id_map = {p.id: p for p in candidate_paths}

        options = []
        for i, opt in enumerate(data.get("options", [])[:5]):
            path_id = opt.get("intent_path_id", "")
            if path_id not in path_id_map and candidate_paths:
                # Fall back to a path by index
                idx = min(i, len(candidate_paths) - 1)
                path_id = candidate_paths[idx].id

            options.append(
                ClarificationOption(
                    option_id=opt.get("option_id", f"opt_{i+1}"),
                    label=opt.get("label", ""),
                    description=opt.get("description", ""),
                    intent_path_id=path_id,
                    params=opt.get("params", {}),
                )
            )

        return HITLRequest(
            question=data.get("question", ""),
            options=options,
            recommended_default=data.get("recommended_default"),
            context={
                "llm_generated": True,
                "num_candidates": len(candidate_paths),
            },
        )

    # ------------------------------------------------------------------
    # Fallback template-based generation
    # ------------------------------------------------------------------

    def _generate_fallback(
        self,
        known_facts: List[str],
        unclear_points: List[str],
        candidate_paths: List[IntentPath],
        ranked_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> HITLRequest:
        """Generate a template-based clarification request when LLM fails."""
        # Build the question from known facts and unclear points
        if known_facts:
            known_part = "、".join(known_facts[:2])
        else:
            known_part = "您的需求"

        if unclear_points:
            unclear_part = "和".join(unclear_points[:2])
        else:
            unclear_part = "具体意图"

        question = f"我已理解您想{known_part}，但{unclear_part}还不够明确。请问您是想："

        # Generate options from candidate paths
        options = []
        for i, path in enumerate(candidate_paths[:5]):
            options.append(
                ClarificationOption(
                    option_id=f"opt_{i+1}",
                    label=self._path_to_label(path),
                    description=path.description or f"执行 {path.action}",
                    intent_path_id=path.id,
                )
            )

        # Use ranked candidates if available
        if ranked_candidates and not options:
            for i, cand in enumerate(ranked_candidates[:5]):
                options.append(
                    ClarificationOption(
                        option_id=f"opt_{i+1}",
                        label=cand.get("intent_name", cand.get("intent_id", "")),
                        description=f"置信度 {cand.get('confidence', 0):.2f}",
                        intent_path_id=cand.get("intent_id", ""),
                    )
                )

        return HITLRequest(
            question=question,
            options=options,
            recommended_default=options[0].option_id if options else None,
            context={"llm_generated": False},
        )

    def _path_to_label(self, path: IntentPath) -> str:
        """Derive a human-readable option label from an IntentPath."""
        # Use the action as the label, stripped of the object prefix
        action = path.action.split("/")[-1] if "/" in path.action else path.action
        action_map = {
            "list": f"查询{path.object}",
            "get_by_id": f"查询{path.object}详情",
            "create": f"创建{path.object}",
            "update": f"更新{path.object}",
            "delete": f"删除{path.object}",
        }
        return action_map.get(action, f"{action} {path.object}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_paths(self, paths: List[IntentPath]) -> str:
        """Format IntentPath list for the prompt."""
        if not paths:
            return "（无候选路径）"
        lines = []
        for p in paths[:10]:
            dims = ", ".join(p.dimensions) if p.dimensions else "无"
            lines.append(
                f"- id={p.id}, action={p.action}, object={p.object}, "
                f"dimensions=[{dims}], description={p.description}"
            )
        return "\n".join(lines)

    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format ranked candidates for the prompt."""
        if not candidates:
            return "（无深度推理候选）"
        lines = []
        for i, c in enumerate(candidates[:3]):
            lines.append(f"- [{i+1}] intent_id={c.get('intent_id','')}, "
                         f"confidence={c.get('confidence',0):.2f}, "
                         f"reason={c.get('reason','')}")
        return "\n".join(lines)
