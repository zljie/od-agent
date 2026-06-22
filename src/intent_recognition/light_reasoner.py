"""
Layer 1: Light Intent Reasoner
===============================
Fast LLM-based intent template, object, and action extraction.

Responsibilities:
1. Build structured prompt from preprocessing result + ontology summary
2. Call light LLM (thinking_budget=500) for template/object/action extraction
3. Parse JSON response with error handling
4. Calculate A_score from confidence breakdown

Key design principles:
- Fast inference: only uses thinking_budget=500
- Structured output: always asks for JSON with typed fields
- Error resilient: gracefully handles malformed responses
"""

import json
import re
import time as _time
from typing import Any, Dict, List, Optional

from ..prompt_config import get_field
from .models import LLMLightResult, PreprocessingResult


def _log(level: str, msg: str) -> None:
    print(f"[LightIntentReasoner][{level}] {msg}")


# ---------------------------------------------------------------------------
# Prompt templates (from Section 18.1 of BeBISO design)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_DEFAULT = """你是企业业务意图识别器（Light Intent Reasoner）。
给定用户输入和上下文信息，你的任务是从结构化本体中匹配最可能的意图模板、对象和操作。

## 核心能力
1. 快速意图分类（轻量级推理，thinking_budget=500）
2. 对象实体识别（采购需求、询价单、报价单、采购订单等）
3. 操作动作识别（查询、创建、审批、比价、提交等）
4. 槽位提取（从输入中识别关键参数）
5. 歧义检测（标记不确定性）

## 约束
- 只返回结构化JSON，不返回其他文字
- 如果输入不明确，选择最可能的意图并标记歧义
- 置信度评分范围 [0.0, 1.0]
"""

_INTENT_TEMPLATES_DESCRIPTION_DEFAULT = """
## 可用意图模板
- create_object: 创建对象（create 操作，用于新增采购需求、新增订单等）
- query_object: 查询对象（read, query 操作）
- process_object: 处理对象（process, update 操作）
- create_from_object: 基于对象创建新对象（create 操作）
- delete_object: 删除对象（delete 操作）
- submit_approval: 提交审批（submit_approval 操作）
- compare_objects: 比较对象（query 操作）
- recommend_object: 推荐对象（query 操作）
- generate_order: 生成订单（create 操作）
- analyze_risk: 风险分析（query 操作）
"""

_ONTOLOGY_OBJECTS_DESCRIPTION_DEFAULT = """
## 本体对象（常见采购对象）
- purchase_requests: 采购需求
- inquiries: 询价单
- quotations: 报价单
- purchase_orders: 采购订单
- contracts: 合同
- suppliers: 供应商
- materials: 物料
- approval_flows: 审批流
"""

_ONTOLOGY_ACTIONS_DESCRIPTION_DEFAULT = """
## 本体动作
- list: 列出/查询列表
- get: 获取单个详情
- create: 创建
- update: 更新/修改
- submit: 提交
- approve: 审批通过
- reject: 审批驳回
- compare: 比价/比较
- recommend: 推荐
- generate: 生成
"""

_FILTER_FIELDS_DESCRIPTION_DEFAULT = """
## 常用过滤字段（用于从用户输入中提取的值）
- material: 物料名称或编码（如 "A4打印纸", "联想笔记本电脑"）
- quantity: 采购数量（如 "10箱", "100个", "50台"）
- delivery_date: 需求日期（如 "下周五", "2026-06-15"）
- apply_dep: 申请部门（如 "销售部", "采购部", "IT部"）
- pr_id: 采购需求编号
- inquiry_id: 询价单编号
- quotation_id: 报价单编号
- po_id: 采购订单编号
- vendor_id: 供应商编号
- material_id: 物料编号
- purchase_type: 采购类型（标准/紧急/生产）
- flow_status: 流程状态
- date_range: 日期范围
"""


def _system_prompt() -> str:
    return get_field("intent_recognition", "light_system_prompt", _SYSTEM_PROMPT_DEFAULT)


def _intent_templates_description() -> str:
    return get_field(
        "intent_recognition",
        "light_templates_description",
        _INTENT_TEMPLATES_DESCRIPTION_DEFAULT,
    )


def _ontology_objects_description() -> str:
    return get_field(
        "intent_recognition",
        "light_objects_description",
        _ONTOLOGY_OBJECTS_DESCRIPTION_DEFAULT,
    )


def _ontology_actions_description() -> str:
    return get_field(
        "intent_recognition",
        "light_actions_description",
        _ONTOLOGY_ACTIONS_DESCRIPTION_DEFAULT,
    )


def _build_user_prompt(
    raw_input: str,
    preprocessing: PreprocessingResult,
    ontology_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user prompt for light intent reasoning."""
    # Extract key info from preprocessing
    tokens = preprocessing.tokens
    action_terms = preprocessing.candidate_terms.get("action_terms", [])
    object_terms = preprocessing.candidate_terms.get("object_terms", [])
    dynamic_terms = preprocessing.candidate_terms.get("dynamic_terms", [])

    # Build token summary
    token_summary = ", ".join(tokens[:20])  # Limit to first 20 tokens
    action_summary = ", ".join(action_terms) if action_terms else "未识别到动作词"
    object_summary = ", ".join(object_terms) if object_terms else "未识别到对象词"
    dynamic_summary = ", ".join(dynamic_terms) if dynamic_terms else "无"

    # Build ontology context if provided
    ontology_context = ""
    if ontology_summary:
        objects = ontology_summary.get("objects", [])
        actions = ontology_summary.get("actions", [])
        if objects:
            ontology_context += f"\n\n## 本体中的对象\n" + "\n".join(f"- {o}" for o in objects[:20])
        if actions:
            ontology_context += f"\n\n## 本体中的动作\n" + "\n".join(f"- {a}" for a in actions[:20])

    prompt = f"""## 用户输入
"{raw_input}"

## 分词结果
{token_summary}

## 识别的动作词候选
{action_summary}

## 识别的对象词候选
{object_summary}

## 识别的动态实体词
{dynamic_summary}
{ontology_context}

## 输出要求
请以JSON格式返回以下字段：

```json
{{
  "intent_template": "意图模板ID（如 process_object, query_object）",
  "object_candidate": "识别的对象名（如 purchase_requests）",
  "object_label": "对象的中文标签（如 采购需求）",
  "action_candidate": "识别的动作（如 process, query, approve）",
  "filters": {{
    "material": "从输入中提取的物料名称或编码",
    "quantity": "从输入中提取的数量",
    "apply_dep": "从输入中提取的部门名称",
    "delivery_date": "从输入中提取的日期"
  }},
  "ambiguities": ["歧义说明1", "歧义说明2"],
  "candidate_next_goals": ["可能的下一个意图1", "可能的下一个意图2"],
  "missing_info": ["可能缺失的信息1", "可能缺失的信息2"],
  "confidence_breakdown": {{
    "intent_score": 0.0-1.0,
    "object_score": 0.0-1.0,
    "action_score": 0.0-1.0,
    "slot_score": 0.0-1.0
  }}
}}
```

注意事项：
1. intent_template 必须从上述模板列表中选择
2. object_candidate 应与本体对象名匹配
3. action_candidate 应该是标准动作词
4. filters 只提取有明确值的字段
5. confidence_breakdown 各项分数应该合理反映识别置信度

请直接返回JSON，不要包含其他文字：
```

重要提示：filters字段必须从用户输入中提取已明确提供的信息，不要留空！
- "为销售部门新增10箱A4打印纸" → filters={{"material": "A4打印纸", "quantity": "10箱", "apply_dep": "销售部"}}
- "添加50台联想笔记本" → filters={{"material": "联想笔记本", "quantity": "50台"}}
"""
    return prompt


def _calculate_a_score(confidence_breakdown: Dict[str, float]) -> float:
    """Calculate overall A_score from confidence breakdown.

    Weighted combination:
    - intent_score: 30% (most important)
    - object_score: 25%
    - action_score: 25%
    - slot_score: 20% (less important as slots can be collected later)
    """
    weights = {
        "intent_score": 0.30,
        "object_score": 0.25,
        "action_score": 0.25,
        "slot_score": 0.20,
    }

    total_score = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        if key in confidence_breakdown:
            score = confidence_breakdown[key]
            # Clamp to [0.0, 1.0]
            score = max(0.0, min(1.0, score))
            total_score += score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(total_score / total_weight, 4)


def _parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response with robust error handling.

    Handles:
    - JSON wrapped in code fences
    - JSON with leading/trailing whitespace
    - Partial JSON (attempts to extract valid JSON substring)
    - Malformed JSON (tries common fixes)
    """
    if not response:
        return None

    # Strip common wrappers
    cleaned = response.strip()

    # Remove code fences if present
    if cleaned.startswith("```"):
        # Remove opening fence
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Try direct JSON parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from within the response
    # Look for {...} pattern
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, cleaned, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try fixing common JSON issues
    # 1. Single quotes to double quotes (simplified)
    try:
        fixed = cleaned.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


class LightIntentReasoner:
    """Light LLM-based intent reasoner for Layer 1.

    Uses a fast LLM call (thinking_budget=500) to extract:
    - Intent template
    - Object candidate
    - Action candidate
    - Filters / extracted slots
    - Confidence breakdown

    Attributes
    ----------
    model:
        Lazy-loaded default LLM model.
    ontology_summary:
        Optional pre-loaded ontology summary for prompt injection.
    """

    def __init__(self, ontology_summary: Optional[Dict[str, Any]] = None):
        """Initialize the light intent reasoner.

        Parameters
        ----------
        ontology_summary:
            Optional pre-computed ontology summary to inject into prompts.
            Should contain keys: "objects", "actions", "dimensions".
        """
        self.model = None
        self.ontology_summary = ontology_summary

    def _get_model(self):
        """Lazy-load the default LLM model."""
        if self.model is None:
            from ..models import get_default_model

            self.model = get_default_model()
        return self.model

    def reason(
        self,
        text: str,
        preprocessing: PreprocessingResult,
        ontology_summary: Optional[Dict[str, Any]] = None,
    ) -> LLMLightResult:
        """Run light intent reasoning on the preprocessed input.

        Parameters
        ----------
        text:
            Original user input text.
        preprocessing:
            Result from Layer 0 preprocessor.
        ontology_summary:
            Optional ontology summary to inject into prompt.
            Overrides the instance-level ontology_summary if provided.

        Returns
        -------
        LLMLightResult
            Structured result with intent template, object, action, filters,
            and confidence breakdown.
        """
        # Use provided ontology_summary or fall back to instance-level
        effective_summary = ontology_summary or self.ontology_summary or {}

        # Build prompt
        user_prompt = _build_user_prompt(text, preprocessing, effective_summary)

        full_prompt = (
            f"{_system_prompt()}\n"
            f"{_intent_templates_description()}\n"
            f"{_ontology_objects_description()}\n"
            f"{_ontology_actions_description()}\n"
            f"{_FILTER_FIELDS_DESCRIPTION_DEFAULT}\n\n"
            f"{user_prompt}"
        )

        # Call LLM
        model = self._get_model()
        _log("INFO", f"LLM调用开始 | input_len={len(text)} | thinking_budget=500")
        llm_start = _time.time()
        try:
            response = model.generate(full_prompt, thinking_budget=500)
        except Exception as e:
            _log("ERROR", f"LLM调用失败: {e}")
            return self._create_error_result(text, preprocessing)
        llm_elapsed = (_time.time() - llm_start) * 1000
        _log("INFO", f"LLM调用完成 | elapsed_ms={llm_elapsed:.1f} | response_len={len(response)}")
        _log("DEBUG", f"LLM原始响应: {response[:300]}...")

        # Parse response
        parsed = _parse_llm_response(response)
        if parsed is None:
            _log("WARN", f"LLM响应JSON解析失败 | response_preview={response[:200]}")
            return self._create_error_result(text, preprocessing)

        # Extract and validate fields
        intent_template = parsed.get("intent_template", "")
        object_candidate = parsed.get("object_candidate", "")
        object_label = parsed.get("object_label", "")
        action_candidate = parsed.get("action_candidate", "")
        filters = parsed.get("filters", {})
        ambiguities = parsed.get("ambiguities", [])
        next_goals = parsed.get("candidate_next_goals", [])
        missing_info = parsed.get("missing_info", [])
        confidence_breakdown = parsed.get("confidence_breakdown", {})

        # Validate confidence_breakdown
        required_scores = ["intent_score", "object_score", "action_score", "slot_score"]
        for key in required_scores:
            if key not in confidence_breakdown:
                confidence_breakdown[key] = 0.5  # Default to 0.5 if missing
            else:
                confidence_breakdown[key] = float(confidence_breakdown[key])

        # Calculate A_score
        a_score = _calculate_a_score(confidence_breakdown)

        result = LLMLightResult.from_light_reasoner_output(
            intent_template=intent_template,
            object_candidate=object_candidate,
            object_label=object_label,
            action_candidate=action_candidate,
            filters=filters,
            ambiguities=ambiguities if isinstance(ambiguities, list) else [],
            candidate_next_goals=next_goals if isinstance(next_goals, list) else [],
            missing_info=missing_info if isinstance(missing_info, list) else [],
            confidence_breakdown=confidence_breakdown,
            A_score=a_score,
        )
        _log("INFO", f"Layer-1识别结果 | intent={intent_template} | object={object_candidate} | filters={filters} | A_score={a_score:.4f}")
        return result

    def _create_error_result(
        self, text: str, preprocessing: PreprocessingResult
    ) -> LLMLightResult:
        """Create an error result when LLM call or parsing fails.

        Returns a fallback result with low confidence.
        """
        # Try to extract something from preprocessing as fallback
        action_terms = preprocessing.candidate_terms.get("action_terms", [])
        object_terms = preprocessing.candidate_terms.get("object_terms", [])

        action_candidate = action_terms[0] if action_terms else ""
        object_candidate = object_terms[0] if object_terms else ""

        return LLMLightResult.from_light_reasoner_output(
            intent_template="",
            object_candidate=object_candidate,
            object_label="",
            action_candidate=action_candidate,
            filters={},
            ambiguities=["LLM调用失败，使用fallback结果"],
            candidate_next_goals=[],
            missing_info=["intent_template", "object_label"],
            confidence_breakdown={
                "intent_score": 0.0,
                "object_score": 0.0,
                "action_score": 0.0,
                "slot_score": 0.0,
            },
            A_score=0.0,
        )

    def set_ontology_summary(self, summary: Dict[str, Any]) -> None:
        """Update the ontology summary used in prompts.

        Parameters
        ----------
        summary:
            Dictionary with keys: objects, actions, dimensions.
        """
        self.ontology_summary = summary
