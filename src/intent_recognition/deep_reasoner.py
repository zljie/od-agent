"""
Deep Intent Reasoner
====================
Layer 4 of the BeBISO 7-layer intent recognition framework.

Invoked when the Layer-3 AB fusion score falls below the 0.75 threshold,
indicating insufficient confidence to proceed with lightweight signals alone.

The reasoner uses a heavy LLM (thinking_budget=1200) to perform deep semantic
analysis, cross-referencing the user's input against the full ontology context,
dialogue history, and the partial results from Layers 0-2.

Output (Section 18.2 of BeBISO spec):
    - ranked_candidates: up to 3 ranked intent candidates
    - missing_info: slots or context still needed
    - recommended_decision: "execute" | "hitl" | "retry"
    - C_deep_score: confidence score from deep reasoning
"""

import json
import re
import time as _time
from typing import Any, Dict, List, Optional

from ..prompt_config import get_field
from .models import (
    DeepReasoningResult,
    LLMLightResult,
    OntologyMatchResult,
    PreprocessingResult,
)


def _log(level: str, msg: str) -> None:
    print(f"[DeepIntentReasoner][{level}] {msg}")


_DEEP_REASONING_PROMPT_DEFAULT = """你是企业业务意图深度推理器。

给定用户输入和前置推理层的结果，请进行深度语义分析，推断用户真实意图，并给出置信度评分。

## 用户原始输入
{raw_input}

## Layer-0 预处理结果
{preprocessing_text}

## Layer-1 轻量化 LLM 推理结果
{llm_light_text}

## Layer-2 本体拓扑匹配结果
{ontology_text}

## 对话上下文
{dialog_text}

## 完整本体上下文（供参考）
以下是企业采购全量本体定义，请结合此上下文进行深度推理：
{full_ontology_context}

## 推理要求
1. 综合分析以上所有信息，推断用户最可能的意图（最多返回3个候选，按置信度从高到低排序）。
2. 识别当前还缺失的关键信息（missing_info）。
3. 给出推荐决策：execute（可直接执行）、hitl（需要人工确认）、retry（需要重试）。
4. 给出一个0-1之间的深度置信度评分C（C_deep_score）。

## 输出格式（仅返回JSON，不要包含其他内容）
{{
  "ranked_candidates": [
    {{
      "intent_id": "...",
      "intent_name": "...",
      "confidence": 0.0-1.0,
      "params": {{}},
      "reason": "..."
    }}
  ],
  "missing_info": ["..."],
  "recommended_decision": "execute|hitl|retry",
  "C_deep_score": 0.0-1.0
}}
"""


def _deep_reasoning_prompt() -> str:
    return get_field("intent_recognition", "deep_reasoning_prompt", _DEEP_REASONING_PROMPT_DEFAULT)


class DeepIntentReasoner:
    """Deep reasoning engine powered by the heavy LLM.

    Uses a thinking budget of 1200 tokens to perform thorough analysis of
    ambiguous or low-confidence inputs.
    """

    DEFAULT_THINKING_BUDGET: int = 1200

    def __init__(self, thinking_budget: int = DEFAULT_THINKING_BUDGET):
        self._thinking_budget = thinking_budget
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        raw_input: str,
        preprocessing: PreprocessingResult,
        llm_light: LLMLightResult,
        ontology_match: OntologyMatchResult,
        full_ontology_context: str,
        dialog_context: Optional[Dict[str, Any]] = None,
    ) -> DeepReasoningResult:
        """Perform deep reasoning over the user input.

        Parameters
        ----------
        raw_input:
            The original user message.
        preprocessing:
            Layer-0 preprocessing result (normalised input, corrections, etc.).
        llm_light:
            Layer-1 light LLM result.
        ontology_match:
            Layer-2 ontology topology match result.
        full_ontology_context:
            Serialised string of the full procurement ontology (objects,
            actions, relationships, rules) for the LLM to reason over.
        dialog_context:
            Optional dialogue history and session state to inform disambiguation.

        Returns
        -------
        DeepReasoningResult
            Contains ranked candidates, missing info, recommended decision,
            and the deep confidence score (C).
        """
        prompt = self._build_prompt(
            raw_input=raw_input,
            preprocessing=preprocessing,
            llm_light=llm_light,
            ontology_match=ontology_match,
            full_ontology_context=full_ontology_context,
            dialog_context=dialog_context or {},
        )

        try:
            response = self._call_model(prompt)
            result = self._parse_response(response)
            _log("INFO", f"Layer-4深度推理结果 | candidates={len(result.ranked_candidates)} | C_score={result.confidence_c:.4f} | decision={result.recommended_decision}")
            return result
        except Exception as e:
            _log("ERROR", f"深度推理异常: {e}")
            return DeepReasoningResult(
                ranked_candidates=[],
                missing_info=["deep_reasoning_failed"],
                recommended_decision="retry",
                confidence_c=0.0,
                reasoning_trace=f"Deep reasoning error: {e}",
                metadata={"error": str(e)},
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily load the default LLM model."""
        if self._model is None:
            from src.models import get_default_model
            self._model = get_default_model()
        return self._model

    def _call_model(self, prompt: str) -> str:
        """Invoke the LLM with the reasoning prompt."""
        _log("INFO", f"LLM调用开始 | thinking_budget={self._thinking_budget} | prompt_len={len(prompt)}")
        start = _time.time()
        model = self._get_model()
        response = model.generate(prompt, thinking_budget=self._thinking_budget)
        elapsed = (_time.time() - start) * 1000
        _log("INFO", f"LLM调用完成 | elapsed_ms={elapsed:.1f} | response_len={len(response)}")
        _log("DEBUG", f"LLM原始响应: {response[:300]}...")
        return response

    def _build_prompt(
        self,
        raw_input: str,
        preprocessing: PreprocessingResult,
        llm_light: LLMLightResult,
        ontology_match: OntologyMatchResult,
        full_ontology_context: str,
        dialog_context: Dict[str, Any],
    ) -> str:
        """Build the deep-reasoning prompt per Section 18.2."""

        # Serialize the layer results into readable text
        preprocessing_text = self._format_preprocessing(preprocessing)
        llm_light_text = self._format_llm_light(llm_light)
        ontology_text = self._format_ontology_match(ontology_match)
        dialog_text = self._format_dialog_context(dialog_context)

        template = _deep_reasoning_prompt()
        return template.format(
            raw_input=raw_input,
            preprocessing_text=preprocessing_text,
            llm_light_text=llm_light_text,
            ontology_text=ontology_text,
            dialog_text=dialog_text,
            full_ontology_context=full_ontology_context,
        )

    def _format_preprocessing(self, r: PreprocessingResult) -> str:
        parts = [
            f"原始输入: {r.original_input}",
            f"标准化输入: {r.normalized_input}",
        ]
        if r.corrections:
            parts.append(f"纠错: {r.corrections}")
        if r.temporal_anchors:
            parts.append(f"时间锚点: {r.temporal_anchors}")
        if r.masked_terms:
            parts.append(f"掩码术语: {r.masked_terms}")
        return "\n".join(parts)

    def _format_llm_light(self, r: LLMLightResult) -> str:
        parts = [
            f"主意图: {r.primary_intent}",
            f"次意图: {r.secondary_intents}",
            f"对象匹配: {[m.object_name for m in r.object_matches]}",
            f"动作匹配: {[m.action_id for m in r.action_matches]}",
            f"维度匹配: {[m.dimension_name for m in r.dimension_matches]}",
            f"置信度A: {r.confidence_a}",
            f"缺失槽位: {r.missing_slots}",
        ]
        if r.raw_reasoning:
            parts.append(f"推理过程: {r.raw_reasoning}")
        return "\n".join(parts)

    def _format_ontology_match(self, r: OntologyMatchResult) -> str:
        parts = [
            f"置信度B: {r.confidence_b}",
            f"匹配路径: {r.matched_path_ids}",
            f"匹配对象: {[m.object_name for m in [r.object_match] if m] if r.object_match else []}",
            f"匹配动作: {[m.action_id for m in [r.action_match] if m] if r.action_match else []}",
            f"缺失槽位: {r.missing_slots}",
            f"原因: {r.reason}",
        ]
        return "\n".join(parts)

    def _format_dialog_context(self, ctx: Dict[str, Any]) -> str:
        if not ctx:
            return "（无历史对话上下文）"
        parts = []
        for key, value in ctx.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _parse_response(self, response: str) -> DeepReasoningResult:
        """Parse JSON from the LLM response, with robust error handling."""
        # Try to extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            text = json_match.group()
        else:
            text = response.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return DeepReasoningResult(
                ranked_candidates=[],
                missing_info=["json_parse_error"],
                recommended_decision="retry",
                confidence_c=0.0,
                reasoning_trace=response[:500],
                metadata={"raw_response": response[:1000]},
            )

        ranked = []
        for i, cand in enumerate(data.get("ranked_candidates", [])[:3]):
            ranked.append({
                "intent_id": cand.get("intent_id", ""),
                "intent_name": cand.get("intent_name", ""),
                "confidence": float(cand.get("confidence", 0.0)),
                "params": cand.get("params", {}),
                "reason": cand.get("reason", ""),
            })

        return DeepReasoningResult(
            ranked_candidates=ranked,
            missing_info=data.get("missing_info", []),
            recommended_decision=data.get("recommended_decision", "retry"),
            confidence_c=float(data.get("C_deep_score", 0.0)),
            reasoning_trace=data.get("reasoning_trace", ""),
            metadata=data,
        )
