"""Step 2: Ontology Object Resolution for the 5-step pipeline."""

from typing import Dict, List, Optional, Any
from .models import OntologyResolveResult
from ..procurement.ontology_loader import get_procurement_ontology, OntologyQuery


class Step2OntologyResolver:
    """Step 2: Ontology Object Resolution.

    Resolves the matched ontology object based on intent and user input.
    Uses the YAML-based procurement ontology as the single source of truth.
    """

    def __init__(self):
        self._ontology = get_procurement_ontology()
        self._query: Optional[OntologyQuery] = None

    def _get_query(self) -> OntologyQuery:
        """Get the ontology query interface."""
        if self._query is None:
            self._query = OntologyQuery(self._ontology)
        return self._query

    def resolve(self, user_input: str, intent_result) -> OntologyResolveResult:
        """Resolve ontology object for the given intent.

        Args:
            user_input: The original user message
            intent_result: Result from Step 1

        Returns:
            OntologyResolveResult with matched object details
        """
        import asyncio

        query = self._get_query()

        # Strategy 1: Find object by the object_term from Step1
        matched_object = None
        if intent_result.object_term:
            matched_object = query.find_object_by_term(intent_result.object_term)

        # Strategy 2: Find object by intent action ID
        if not matched_object and intent_result.intent:
            matched_object = query.find_object_by_intent(intent_result.intent)

        # Strategy 3: LLM-powered fallback (for complex cases)
        # This returns (dataset, reasoning, alternatives)
        llm_reasoning = None
        llm_alternatives = []
        if not matched_object:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # In async context - use sync fallback or run in executor
                matched_object, llm_reasoning, llm_alternatives = self._llm_resolve_fallback_sync(
                    user_input, intent_result, query
                )
            except RuntimeError:
                # No running loop - use sync version
                matched_object, llm_reasoning, llm_alternatives = self._llm_resolve_fallback_sync(
                    user_input, intent_result, query
                )

        if matched_object:
            # Check if this was LLM-inferred
            llm_inferred = llm_reasoning is not None
            result = self._build_success_result(matched_object, intent_result, query)
            result.llm_inferred = llm_inferred
            result.llm_reasoning = llm_reasoning
            result.alternatives = llm_alternatives
            return result

        # Fallback when no ontology match
        return self._fallback_resolution(intent_result)

    def _build_success_result(self, dataset, intent_result, query: OntologyQuery) -> OntologyResolveResult:
        """Build successful resolution result."""
        object_type = dataset.name
        object_label = dataset.label

        # Get attributes from the dataset
        attributes = query.get_entity_attributes(object_type)

        # Get available actions for this entity
        actions = query.get_actions_for_entity(object_type)
        action_list = [
            {
                "id": a.id,
                "name": a.name,
                "kind": a.kind,
                "operation": a.operation,
                "description": a.description,
            }
            for a in actions
        ]

        # Extract keywords from user input
        keywords = self._extract_keywords(intent_result.object_term, dataset)

        # Assess completeness
        completeness, gaps = self._assess_completeness(dataset, intent_result)

        return OntologyResolveResult(
            object_type=object_type,
            object_label=object_label,
            hit_keywords=keywords,
            attributes=attributes,
            available_actions=action_list,
            ontology_completeness=completeness,
            gaps=gaps,
        )

    def _llm_resolve_fallback_sync(
        self, 
        user_input: str, 
        intent_result,
        query: OntologyQuery
    ) -> tuple:
        """LLM-powered fallback for complex object resolution.
        
        Uses a thinking-enabled LLM call to reason about which ontology
        object best matches the user's query.
        
        Returns:
            tuple: (matched_object, llm_reasoning, alternatives)
        """
        try:
            from ..models import get_default_model
            
            # Build context with ontology summary
            ontology_context = query.ontology.to_system_prompt_context()
            
            prompt = f"""## 任务
基于以下业务本体和用户输入，进行深度语义推理，找出最匹配的业务对象。

## 用户输入
{user_input}

## Step1 意图识别结果
- 意图: {intent_result.intent} ({intent_result.intent_label})
- 对象术语: {intent_result.object_term}
- 操作类型: {intent_result.operation_type}

## 采购业务本体
{ontology_context}

## 推理过程
请按以下步骤思考：

1. **分析用户意图**: 用户说的"对象术语"是什么？采购计划/请购单/PR 这些都指向采购需求
2. **理解业务语义**: 采购计划、请购单、PR 都是 purchase_requests 的不同叫法
3. **检查本体覆盖**: 从本体中找到语义最接近的数据集
4. **输出推理结果**: 

请按以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{
    "matched_name": "purchase_requests",  // 匹配的 dataset name
    "reasoning": "用户说的'采购计划'在本体系中对应'采购需求'（purchase_requests），因为...",  // 推理过程
    "confidence": 0.95,  // 置信度 0-1
    "alternatives": [  // 备选方案
        {{"name": "purchase_inquiries", "reason": "如果用户实际想询价..."}}
    ]
}}

如果无法确定匹配，输出：
{{
    "matched_name": null,
    "reasoning": "无法确定匹配...",
    "confidence": 0.0,
    "alternatives": []
}}

"""
            model = get_default_model()
            response = model.generate(prompt, thinking_budget=800)
            
            # Parse the JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                matched_name = result.get("matched_name")
                reasoning = result.get("reasoning", "")
                confidence = result.get("confidence", 0.5)
                alternatives = result.get("alternatives", [])
                
                if matched_name and confidence >= 0.6:
                    dataset = query.ontology.get_entity(matched_name)
                    if dataset:
                        alt_list = [
                            {"name": a.get("name", ""), "reason": a.get("reason", "")}
                            for a in alternatives
                        ]
                        return dataset, reasoning, alt_list
            
        except Exception as e:
            print(f"[Step2] LLM fallback failed: {e}")
        
        return None, None, []

    def _extract_keywords(self, object_term: str, dataset) -> List[str]:
        """Extract keywords from user input and dataset."""
        keywords = []
        
        # Add dataset keywords
        if hasattr(dataset, 'keywords'):
            keywords.extend(dataset.keywords[:5])
        
        # Add dataset synonyms
        if hasattr(dataset, 'synonyms'):
            keywords.extend(dataset.synonyms[:3])
        
        return list(set(keywords)) if keywords else [object_term or "业务查询"]

    def _assess_completeness(self, dataset, intent_result) -> tuple:
        """Assess ontology completeness and detect gaps."""
        gaps = []
        
        # Check for essential fields
        essential_fields = ['status', 'delete_flag', 'pr_id']
        existing_fields = {f.name for f in dataset.fields}
        
        for field_name in essential_fields:
            if field_name not in existing_fields:
                gaps.append({
                    "gap_type": "missing_attribute",
                    "description": f"本体对象 '{dataset.name}' 缺少属性 '{field_name}'",
                    "suggestion": f"建议在 {dataset.name} 中补充 {field_name} 字段"
                })

        # Assess completeness
        if len(gaps) == 0:
            completeness = "full"
        elif len(gaps) <= 2:
            completeness = "partial"
        else:
            completeness = "insufficient"

        return completeness, gaps

    def _fallback_resolution(self, intent_result) -> OntologyResolveResult:
        """Fallback when no ontology match is found."""
        return OntologyResolveResult(
            object_type="unknown",
            object_label=intent_result.object_term or "未知对象",
            hit_keywords=[],
            attributes=[],
            available_actions=[],
            ontology_completeness="insufficient",
            gaps=[{
                "gap_type": "missing_object",
                "description": f"未找到 '{intent_result.object_term}' 对应的本体对象",
                "suggestion": "请确认本体中已定义该业务对象，或意图识别结果是否准确"
            }],
        )
